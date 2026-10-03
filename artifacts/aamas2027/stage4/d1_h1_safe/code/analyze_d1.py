#!/usr/bin/env python3
"""D1-H1-SAFE-v1 analysis + report (post-hoc exploratory).

Consumes the 22 validated cells written by run_d1.py and produces:
D1_SUMMARY.csv, D1_SEED_CAPACITY.csv, D1_REGIME_LEVEL.csv,
D1_CONFLICT_TELEMETRY.csv, D1_REFILL_TELEMETRY.csv, D1_LOCAL_FORK_SUMMARY.csv,
statistics.json, D1_DIAGNOSTIC_REPORT.md.

The MASK_GATE thresholds are engineering heuristics fixed BEFORE looking at
the results (1% / -1% / 5% of frozen BF macro), and BF reference values are
loaded from frozen Stage 3 tables -- never hard-coded.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CODE_DIR = Path(__file__).resolve().parent
D1_ROOT = CODE_DIR.parents[0]
WT = CODE_DIR.parents[4]
B0_CODE = WT / "artifacts/aamas2027/stage3_5/b0_diagnostic/code"
H1_CODE = WT / "artifacts/aamas2027/stage3_5/h1_multiseed/code"
for p in (str(WT), str(B0_CODE), str(H1_CODE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import b0_lib as b0  # noqa: E402
import run_h1_multi as h1m  # noqa: E402
import run_d1  # noqa: E402

SEEDS = run_d1.SEEDS
REGIMES = b0.REGIMES
STEPS_PER_CELL = 12 * b0.EPISODES_PER_REGIME * b0.T  # 2,400,000


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def require(ok: bool, msg: str) -> None:
    b0.require(ok, msg)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_cell(arm: str, C: int, seed: Optional[int]) -> Dict[str, Any]:
    dst = run_d1.cell_path(arm, C, seed)
    with (dst / "episodes.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    tel = json.loads((dst / "telemetry.json").read_text())
    result = json.loads((dst / "result.json").read_text())
    return dict(rows=rows, telemetry=tel, result=result, path=dst)


def regime_money(rows: List[Dict[str, str]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for reg in REGIMES:
        rr = [r for r in rows if r["regime"] == reg]
        require(len(rr) == b0.EPISODES_PER_REGIME, f"coverage {reg}")
        out[reg] = statistics.mean(float(r["money"]) for r in rr)
    return out


def regime_metric(rows: List[Dict[str, str]], field: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for reg in REGIMES:
        rr = [r for r in rows if r["regime"] == reg]
        out[reg] = statistics.mean(float(r[field]) for r in rr)
    return out


def macro(rows: List[Dict[str, str]]) -> float:
    return statistics.mean(regime_money(rows).values())


def cell_counts(tel: Dict[str, Any], regime: str) -> Dict[str, int]:
    return tel["counts"][regime]


def all12_counts(tel: Dict[str, Any]) -> Dict[str, int]:
    keys = list(tel["counts"][REGIMES[0]].keys())
    return {k: sum(int(tel["counts"][r][k]) for r in REGIMES) for k in keys}


def fork_all12(tel: Dict[str, Any]) -> Dict[str, float]:
    keys = list(tel["forks"][REGIMES[0]].keys())
    return {k: float(sum(tel["forks"][r][k] for r in REGIMES)) for k in keys}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze() -> Dict[str, Any]:
    t0 = time.time()
    val_path = D1_ROOT / "D1_VALIDATION.json"
    val = json.loads(val_path.read_text()) if val_path.is_file() else {"status": "MISSING"}

    data: Dict[Tuple[str, int, Optional[int]], Dict[str, Any]] = {}
    macros: Dict[Tuple[str, int, Optional[int]], float] = {}
    for arm in ("U", "M"):
        for C in b0.CAPACITIES:
            for seed in SEEDS:
                d = load_cell(arm, C, seed)
                data[(arm, C, seed)] = d
                macros[(arm, C, seed)] = macro(d["rows"])
    for C in b0.CAPACITIES:
        d = load_cell("BF", C, None)
        data[("BF", C, None)] = d
        macros[("BF", C, None)] = macro(d["rows"])

    # Frozen BF reference (authority for gate denominators)
    bf_ref = {C: h1m.load_bf_reference(C)["macro"] for C in b0.CAPACITIES}

    stats: Dict[str, Any] = dict(
        generated_utc=utc(),
        post_hoc_exploratory=True,
        confirmatory=False,
        note=(
            "Post-hoc exploratory mechanism diagnostic. NEW12-v1 was already "
            "used for method selection. Paired t stats are raw, unadjusted, "
            "n=5 (df=4). Local fork numbers are ONE-STEP LOCAL EXPOSURE ONLY, "
            "not recoverable value, not an optimality theorem."
        ),
        validation_status=val.get("status"),
        bf_macro={str(C): bf_ref[C] for C in b0.CAPACITIES},
    )

    # --- per-seed M-U deltas ---------------------------------------------
    seed_rows: List[Dict[str, Any]] = []
    paired: Dict[str, Any] = {}
    for C in b0.CAPACITIES:
        deltas = []
        for seed in SEEDS:
            u = macros[("U", C, seed)]
            m = macros[("M", C, seed)]
            deltas.append(m - u)
            seed_rows.append(dict(C=C, seed=seed, U_money=u, M_money=m, delta_M_minus_U=m - u))
        tt = h1m.one_sample_t(deltas)
        paired[str(C)] = dict(
            deltas={str(s): macros[("M", C, s)] - macros[("U", C, s)] for s in SEEDS},
            n_positive=sum(d > 0 for d in deltas),
            n_negative=sum(d < 0 for d in deltas),
            n_zero=sum(d == 0 for d in deltas),
            **tt,
        )
    stats["paired_M_minus_U"] = paired

    # --- MASK_GATE ---------------------------------------------------------
    bf800, bf1200 = bf_ref[800], bf_ref[1200]
    gate = {
        "rule": (
            "PASS iff (1) mean(M-U) >= +1% of frozen BF macro at >=1 capacity, "
            "(2) mean(M-U) >= -1% of BF at the other capacity, and "
            "(3) no seed loses >5% of BF relative to its own U baseline."
        ),
        "thresholds": {
            "C800_plus_1pct": 0.01 * bf800,
            "C1200_plus_1pct": 0.01 * bf1200,
            "C800_minus_1pct": -0.01 * bf800,
            "C1200_minus_1pct": -0.01 * bf1200,
            "C800_seed_loss_5pct_of_BF": 0.05 * bf800,
            "C1200_seed_loss_5pct_of_BF": 0.05 * bf1200,
        },
    }
    mean800 = paired["800"]["mean"]
    mean1200 = paired["1200"]["mean"]
    cond1 = (mean800 >= gate["thresholds"]["C800_plus_1pct"]) or (
        mean1200 >= gate["thresholds"]["C1200_plus_1pct"]
    )
    cond2 = (mean800 >= gate["thresholds"]["C800_minus_1pct"]) and (
        mean1200 >= gate["thresholds"]["C1200_minus_1pct"]
    )
    worst_seed_loss = {}
    cond3 = True
    for C in b0.CAPACITIES:
        limit = 0.05 * bf_ref[C]
        for seed in SEEDS:
            loss = macros[("U", C, seed)] - macros[("M", C, seed)]  # positive = M loses
            worst_seed_loss[f"C{C}_S{seed}"] = loss
            if loss > limit:
                cond3 = False
    gate.update(
        mean_delta_C800=mean800,
        mean_delta_C1200=mean1200,
        condition_at_least_one_capacity_plus_1pct=bool(cond1),
        condition_other_capacity_at_worst_minus_1pct=bool(cond2),
        condition_no_seed_loses_gt_5pct=bool(cond3),
        worst_seed_loss=worst_seed_loss,
        PASS=bool(cond1 and cond2 and cond3),
    )
    stats["mask_gate"] = gate

    # --- conflict / refill / fork telemetry per cell ----------------------
    conflict_rows: List[Dict[str, Any]] = []
    refill_rows: List[Dict[str, Any]] = []
    fork_rows: List[Dict[str, Any]] = []
    seed_cap_rows: List[Dict[str, Any]] = []

    conflict_meta: Dict[str, Any] = {}
    for arm in ("U", "M"):
        for C in b0.CAPACITIES:
            for seed in SEEDS:
                tel = data[(arm, C, seed)]["telemetry"]
                tot = all12_counts(tel)
                fk = fork_all12(tel)
                key = f"{arm}-C{C}-S{seed}"
                # per-regime rows + ALL12 roll-up
                for reg in REGIMES + ["ALL12"]:
                    c = tot if reg == "ALL12" else cell_counts(tel, reg)
                    steps = c["steps"]
                    conflict_rows.append(
                        dict(
                            arm=arm, C=C, seed=seed, regime=reg, steps=steps,
                            settle_bf_feasible=c["settle_bf_feasible"],
                            settle_bf_noop=c["settle_bf_noop"],
                            conflict_raw=c["conflict_raw"],
                            conflict_prefeasible=c["conflict_prefeasible"],
                            conflict_executed=c["conflict_executed"],
                            lost_settlement=c["lost_settlement"],
                            conflict_rate=c["conflict_raw"] / steps,
                            prefeasible_conflict_rate=c["conflict_prefeasible"] / steps,
                            executed_conflict_rate=c["conflict_executed"] / steps,
                            lost_settlement_rate=c["lost_settlement"] / steps,
                            flush_requested_real=c["flush_requested_real"],
                            flush_executed=c["flush_executed"],
                            flush_noop=c["flush_noop"],
                            flush_unusable=c["flush_unusable"],
                            mask_changed_choice=c["mask_changed_choice"],
                            accepted=c["accepted"],
                            insufficient_drops=c["insufficient_drops"],
                            oversize_drops=c["oversize_drops"],
                        )
                    )
                # refill row
                refill_rows.append(
                    dict(
                        arm=arm, C=C, seed=seed,
                        refill_completions=tot["refill_completions"],
                        refilled_value=round(
                            sum(tel["sums"][r]["refilled_value"] for r in REGIMES), 6
                        ),
                        episodes_refilled_never_used=tot["episodes_refilled_never_used"],
                        episodes_terminal_pending=tot["episodes_terminal_pending"],
                        starvation_zero_usable_steps=tot["starvation_zero_usable_steps"],
                        starvation_zero_feasible_steps=tot["starvation_zero_feasible_steps"],
                        ttr_hist_json=json.dumps(
                            _merge_hists(tel["ttr_hist"]), sort_keys=True
                        ),
                        hard_starvation_hist_json=json.dumps(
                            _merge_hists(tel["hard_starvation_hist"]), sort_keys=True
                        ),
                        feasible_starvation_hist_json=json.dumps(
                            _merge_hists(tel["feasible_starvation_hist"]), sort_keys=True
                        ),
                    )
                )
                # fork row
                n = int(fk["fork_events"])
                fork_rows.append(
                    dict(
                        arm=arm, C=C, seed=seed, fork_events=n,
                        tx_sum=round(fk["tx_sum"], 6),
                        settled_A_sum=round(fk["settled_A_sum"], 6),
                        settled_B_sum=round(fk["settled_B_sum"], 6),
                        flushes_A_sum=int(fk["flushes_A_sum"]),
                        flushes_B_sum=int(fk["flushes_B_sum"]),
                        money_A_minus_B_sum=round(fk["exposure_A_minus_B_sum"], 6),
                        mean_money_A_minus_B=(
                            fk["exposure_A_minus_B_sum"] / n if n else 0.0
                        ),
                        B_accept_rate=(fk["B_accepted_sum"] / n if n else 0.0),
                    )
                )
                # seed/capacity summary
                delta = macros[("M", C, seed)] - macros[("U", C, seed)]
                seed_cap_rows.append(
                    dict(
                        C=C, seed=seed,
                        U_money=macros[("U", C, seed)],
                        M_money=macros[("M", C, seed)],
                        delta_M_minus_U=delta,
                        delta_pct_of_BF=100.0 * delta / bf_ref[C],
                        U_conflict_raw=all12_counts(data[("U", C, seed)]["telemetry"])["conflict_raw"],
                        M_conflict_raw=tot["conflict_raw"],
                        U_conflict_prefeasible=all12_counts(
                            data[("U", C, seed)]["telemetry"])["conflict_prefeasible"],
                        M_conflict_prefeasible=tot["conflict_prefeasible"],
                        U_conflict_executed=all12_counts(
                            data[("U", C, seed)]["telemetry"])["conflict_executed"],
                        M_conflict_executed=tot["conflict_executed"],
                        U_lost=all12_counts(data[("U", C, seed)]["telemetry"])["lost_settlement"],
                        M_lost=tot["lost_settlement"],
                        M_mask_changed=tot["mask_changed_choice"],
                    )
                )
                conflict_meta[key] = dict(
                    conflict_prefeasible=tot["conflict_prefeasible"],
                    conflict_executed=tot["conflict_executed"],
                    lost=tot["lost_settlement"],
                    fork_events=int(fk["fork_events"]),
                    fork_exposure_sum=fk["exposure_A_minus_B_sum"],
                )
    stats["conflict_meta"] = conflict_meta

    # --- BF telemetry rows (2 cells) --------------------------------------
    for C in b0.CAPACITIES:
        tel = data[("BF", C, None)]["telemetry"]
        tot = all12_counts(tel)
        for reg in REGIMES + ["ALL12"]:
            c = tot if reg == "ALL12" else cell_counts(tel, reg)
            steps = c["steps"]
            conflict_rows.append(
                dict(
                    arm="BF", C=C, seed="", regime=reg, steps=steps,
                    settle_bf_feasible=c["settle_bf_feasible"],
                    settle_bf_noop=c["settle_bf_noop"],
                    conflict_raw=c["conflict_raw"],
                    conflict_prefeasible=c["conflict_prefeasible"],
                    conflict_executed=c["conflict_executed"],
                    lost_settlement=c["lost_settlement"],
                    conflict_rate=c["conflict_raw"] / steps,
                    prefeasible_conflict_rate=c["conflict_prefeasible"] / steps,
                    executed_conflict_rate=c["conflict_executed"] / steps,
                    lost_settlement_rate=c["lost_settlement"] / steps,
                    flush_requested_real=c["flush_requested_real"],
                    flush_executed=c["flush_executed"],
                    flush_noop=c["flush_noop"],
                    flush_unusable=c["flush_unusable"],
                    mask_changed_choice=c["mask_changed_choice"],
                    accepted=c["accepted"],
                    insufficient_drops=c["insufficient_drops"],
                    oversize_drops=c["oversize_drops"],
                )
            )
        refill_rows.append(
            dict(
                arm="BF", C=C, seed="",
                refill_completions=tot["refill_completions"],
                refilled_value=round(
                    sum(tel["sums"][r]["refilled_value"] for r in REGIMES), 6
                ),
                episodes_refilled_never_used=tot["episodes_refilled_never_used"],
                episodes_terminal_pending=tot["episodes_terminal_pending"],
                starvation_zero_usable_steps=tot["starvation_zero_usable_steps"],
                starvation_zero_feasible_steps=tot["starvation_zero_feasible_steps"],
                ttr_hist_json=json.dumps(_merge_hists(tel["ttr_hist"]), sort_keys=True),
                hard_starvation_hist_json=json.dumps(
                    _merge_hists(tel["hard_starvation_hist"]), sort_keys=True
                ),
                feasible_starvation_hist_json=json.dumps(
                    _merge_hists(tel["feasible_starvation_hist"]), sort_keys=True
                ),
            )
        )

    # --- money delta per prefeasible-conflict exposure --------------------
    exposure_rows: Dict[str, Any] = {}
    for C in b0.CAPACITIES:
        for seed in SEEDS:
            d = macros[("M", C, seed)] - macros[("U", C, seed)]
            n_u = conflict_meta[f"U-C{C}-S{seed}"]["conflict_prefeasible"]
            exposure_rows[f"C{C}_S{seed}"] = dict(
                cell_money_delta=d,
                U_prefeasible_conflicts=n_u,
                delta_per_U_prefeasible_conflict=(d / n_u if n_u else None),
                U_fork_exposure_sum=conflict_meta[f"U-C{C}-S{seed}"]["fork_exposure_sum"],
            )
    stats["delta_per_conflict_exposure"] = exposure_rows

    # --- regime-level table ------------------------------------------------
    regime_rows: List[Dict[str, Any]] = []
    for arm in ("U", "M"):
        for C in b0.CAPACITIES:
            for seed in SEEDS:
                rows = data[(arm, C, seed)]["rows"]
                rm = regime_money(rows)
                rsett = regime_metric(rows, "settled")
                rflush = regime_metric(rows, "flushes")
                tel = data[(arm, C, seed)]["telemetry"]
                for reg in REGIMES:
                    c = cell_counts(tel, reg)
                    regime_rows.append(
                        dict(
                            arm=arm, C=C, seed=seed, regime=reg,
                            money=rm[reg], settled=rsett[reg], flushes=rflush[reg],
                            conflicts=c["conflict_raw"],
                            conflicts_prefeasible=c["conflict_prefeasible"],
                            conflicts_executed=c["conflict_executed"],
                            lost_settlement=c["lost_settlement"],
                            mask_changed=c["mask_changed_choice"],
                        )
                    )
    for C in b0.CAPACITIES:
        rows = data[("BF", C, None)]["rows"]
        rm = regime_money(rows)
        rsett = regime_metric(rows, "settled")
        rflush = regime_metric(rows, "flushes")
        for reg in REGIMES:
            regime_rows.append(
                dict(arm="BF", C=C, seed="", regime=reg, money=rm[reg],
                     settled=rsett[reg], flushes=rflush[reg],
                     conflicts=0, conflicts_prefeasible=0, conflicts_executed=0,
                     lost_settlement=0, mask_changed=0)
            )

    # --- classification ----------------------------------------------------
    classification = classify(stats, data, macros, bf_ref)
    stats["classification"] = classification

    # --- summary rows ------------------------------------------------------
    summary_rows: List[Dict[str, Any]] = []
    for C in b0.CAPACITIES:
        p = paired[str(C)]
        summary_rows.append(
            dict(
                C=C,
                bf_macro=bf_ref[C],
                U_mean=statistics.mean(macros[("U", C, s)] for s in SEEDS),
                M_mean=statistics.mean(macros[("M", C, s)] for s in SEEDS),
                mean_delta_M_minus_U=p["mean"],
                sd_delta=p["sd"],
                se_delta=p["se"],
                ci95_lo=p["ci95_lo"],
                ci95_hi=p["ci95_hi"],
                t=p["t"],
                df=p["df"],
                p_two_sided_raw=p["p_two_sided_raw"],
                n_positive=p["n_positive"],
                n_negative=p["n_negative"],
                plus_1pct_BF=0.01 * bf_ref[C],
                minus_1pct_BF=-0.01 * bf_ref[C],
            )
        )

    # --- elapsed / write ---------------------------------------------------
    stats["analysis_elapsed_seconds"] = time.time() - t0
    _write_csv(D1_ROOT / "D1_SUMMARY.csv", summary_rows)
    _write_csv(D1_ROOT / "D1_SEED_CAPACITY.csv", seed_cap_rows)
    _write_csv(D1_ROOT / "D1_REGIME_LEVEL.csv", regime_rows)
    _write_csv(D1_ROOT / "D1_CONFLICT_TELEMETRY.csv", conflict_rows)
    _write_csv(D1_ROOT / "D1_REFILL_TELEMETRY.csv", refill_rows)
    _write_csv(D1_ROOT / "D1_LOCAL_FORK_SUMMARY.csv", fork_rows)
    (D1_ROOT / "statistics.json").write_text(json.dumps(stats, indent=2, sort_keys=True))

    # report is written by report_d1.py (kept separate for readability)
    from write_report_d1 import write_report

    write_report(stats, data, macros, bf_ref)

    return stats


def _merge_hists(per_regime: Dict[str, Dict[str, int]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for reg in REGIMES:
        for k, v in per_regime[reg].items():
            out[k] = out.get(k, 0) + int(v)
    return out


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Heuristic classification (engineering rules; fixed before seeing results)
# ---------------------------------------------------------------------------

def classify(
    stats: Dict[str, Any],
    data: Dict[Tuple[str, int, Optional[int]], Dict[str, Any]],
    macros: Dict[Tuple[str, int, Optional[int]], float],
    bf_ref: Dict[int, float],
) -> Dict[str, Any]:
    gate_pass = bool(stats["mask_gate"]["PASS"])
    p800 = stats["paired_M_minus_U"]["800"]
    p1200 = stats["paired_M_minus_U"]["1200"]

    # aggregate executed-conflict rates across cells
    exec_rates = {"800": [], "1200": []}
    pre_rates = {"800": [], "1200": []}
    lost_tot = {"800": 0, "1200": 0}
    pre_tot = {"800": 0, "1200": 0}
    for C in b0.CAPACITIES:
        for seed in SEEDS:
            tel = data[("U", C, seed)]["telemetry"]
            tot = all12_counts(tel)
            steps = STEPS_PER_CELL
            exec_rates[str(C)].append(tot["conflict_executed"] / steps)
            pre_rates[str(C)].append(tot["conflict_prefeasible"] / steps)
            lost_tot[str(C)] += tot["lost_settlement"]
            pre_tot[str(C)] += tot["conflict_prefeasible"]

    def effect_cap(p: Dict[str, Any], C: int) -> str:
        mean, ci_lo = p["mean"], p["ci95_lo"]
        bf = bf_ref[C]
        if mean >= 0.01 * bf and p["n_positive"] >= 4:
            return "strong_positive"
        if mean >= 0.005 * bf and p["n_positive"] >= 4:
            return "moderate_positive"
        if mean > 0:
            return "weak_positive"
        if mean <= -0.01 * bf:
            return "negative"
        return "null"

    eff = {800: effect_cap(p800, 800), 1200: effect_cap(p1200, 1200)}

    # DIRECT_CONFLICT_IMPORTANCE
    if gate_pass:
        direct = "HIGH"
    elif "strong_positive" in eff.values() or "moderate_positive" in eff.values():
        direct = "MODERATE"
    elif any(e.startswith("weak_positive") for e in eff.values()):
        direct = "LOW"
    else:
        # how often does the raw same-wallet conflict actually execute and
        # void a currently feasible settlement?
        mean_exec_rate = statistics.mean(exec_rates["800"] + exec_rates["1200"])
        direct = "NEGLIGIBLE" if mean_exec_rate < 1e-5 else "LOW"

    # STATE_DISTRIBUTION_MISMATCH_EVIDENCE:
    # the mask changes choices at raw-conflict states; if money does not
    # respond despite non-trivial executed-conflict exposure, the frozen
    # flush head's value depends on state paths that the mask disrupts --
    # i.e. evidence for M2 rather than pure M1.
    mean_pre_rate = statistics.mean(pre_rates["800"] + pre_rates["1200"])
    best_mean = max(p800["mean"], p1200["mean"])
    worst_mean = min(p800["mean"], p1200["mean"])
    # per-capacity "coherent gain" test: mean delta >= 0.5% of THAT capacity's BF
    coherent_gain = (
        p800["mean"] >= 0.005 * bf_ref[800]
        or p1200["mean"] >= 0.005 * bf_ref[1200]
    )
    if not gate_pass and mean_pre_rate >= 1e-3 and not coherent_gain:
        # conflicts common, but masking yields no coherent gain
        mismatch = "STRONG" if worst_mean < 0 else "MODERATE"
    elif not gate_pass and mean_pre_rate >= 1e-4 and best_mean <= 0:
        mismatch = "MODERATE"
    elif gate_pass:
        mismatch = "WEAK"
    elif best_mean > 0:
        mismatch = "WEAK"
    else:
        mismatch = "NONE"

    # recommendations
    if gate_pass:
        a_plus = "YES"
        stop_branch = "YES"
    elif direct in ("MODERATE", "HIGH") or mismatch in ("STRONG", "MODERATE"):
        a_plus = "PI_REVIEW"
        stop_branch = "NO"
    else:
        a_plus = "NO"
        stop_branch = "YES"

    return dict(
        effects_by_capacity={str(C): eff[C] for C in b0.CAPACITIES},
        mean_executed_conflict_rate_C800=statistics.mean(exec_rates["800"]),
        mean_executed_conflict_rate_C1200=statistics.mean(exec_rates["1200"]),
        mean_prefeasible_conflict_rate_C800=statistics.mean(pre_rates["800"]),
        mean_prefeasible_conflict_rate_C1200=statistics.mean(pre_rates["1200"]),
        lost_per_prefeasible_conflict_C800=(
            lost_tot["800"] / pre_tot["800"] if pre_tot["800"] else 0.0
        ),
        lost_per_prefeasible_conflict_C1200=(
            lost_tot["1200"] / pre_tot["1200"] if pre_tot["1200"] else 0.0
        ),
        DIRECT_CONFLICT_IMPORTANCE=direct,
        STATE_DISTRIBUTION_MISMATCH_EVIDENCE=mismatch,
        RECOMMEND_A_PLUS_PILOT=a_plus,
        RECOMMEND_STOP_IMPROVEMENT_BRANCH=stop_branch,
        rule_doc=(
            "Engineering heuristics fixed pre-results: HIGH if MASK_GATE passes; "
            "MODERATE if >=0.5% BF mean gain with >=4/5 seeds positive at a "
            "capacity; LOW for weak/null positive; NEGLIGIBLE only when null "
            "AND executed-conflict rate <1e-5. Mismatch STRONG when "
            "prefeasible-conflict rate >=1e-3 but best mean gain <0.5% BF; "
            "MODERATE at rate >=1e-4 with no positive mean."
        ),
    )


def cmd_analyze(_args: Any) -> None:
    stats = analyze()
    g = stats["mask_gate"]
    cls = stats["classification"]
    print("D1 ANALYSIS complete")
    for C in b0.CAPACITIES:
        p = stats["paired_M_minus_U"][str(C)]
        print(
            f"  C{C}: mean M-U {p['mean']:+.3f} [{p['ci95_lo']:+.3f},"
            f" {p['ci95_hi']:+.3f}] t={p['t']:.3f} p={p['p_two_sided_raw']:.4f}"
            f" pos/neg {p['n_positive']}/{p['n_negative']}"
        )
    print(f"  MASK_GATE PASS={g['PASS']} (c1={g['condition_at_least_one_capacity_plus_1pct']}"
          f" c2={g['condition_other_capacity_at_worst_minus_1pct']}"
          f" c3={g['condition_no_seed_loses_gt_5pct']})")
    print(f"  DIRECT_CONFLICT_IMPORTANCE={cls['DIRECT_CONFLICT_IMPORTANCE']}")
    print(f"  STATE_DISTRIBUTION_MISMATCH_EVIDENCE={cls['STATE_DISTRIBUTION_MISMATCH_EVIDENCE']}")
    print(f"  RECOMMEND_A_PLUS_PILOT={cls['RECOMMEND_A_PLUS_PILOT']}")
    print(f"  RECOMMEND_STOP_IMPROVEMENT_BRANCH={cls['RECOMMEND_STOP_IMPROVEMENT_BRANCH']}")


if __name__ == "__main__":
    cmd_analyze(None)
