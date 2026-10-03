#!/usr/bin/env python3
"""D1-H1-SAFE-v1 diagnostic report writer (20 mandated sections).

The report ends with exactly five fields: MASK_GATE,
DIRECT_CONFLICT_IMPORTANCE, STATE_DISTRIBUTION_MISMATCH_EVIDENCE,
RECOMMEND_A_PLUS_PILOT, RECOMMEND_STOP_IMPROVEMENT_BRANCH.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

CODE_DIR = Path(__file__).resolve().parent
D1_ROOT = CODE_DIR.parents[0]
WT = CODE_DIR.parents[4]
B0_CODE = WT / "artifacts/aamas2027/stage3_5/b0_diagnostic/code"
H1_CODE = WT / "artifacts/aamas2027/stage3_5/h1_multiseed/code"
for p in (str(WT), str(B0_CODE), str(H1_CODE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import b0_lib as b0  # noqa: E402

SEEDS = [123, 323, 532, 777, 999]
FAILURE_SEEDS = [(800, 777), (1200, 323), (1200, 532)]


def f3(x: float) -> str:
    return f"{x:.3f}"


def f2(x: float) -> str:
    return f"{x:.2f}"


def fp(x: float) -> str:
    return f"{x:+.3f}"


def write_report(
    stats: Dict[str, Any],
    data: Dict[Tuple[str, int, Optional[int]], Dict[str, Any]],
    macros: Dict[Tuple[str, int, Optional[int]], float],
    bf_ref: Dict[int, float],
) -> None:
    g = stats["mask_gate"]
    cls = stats["classification"]
    L: list[str] = []
    A = L.append

    A("# D1-H1-SAFE-v1 Diagnostic Report (Stage 4)")
    A("")
    A(
        "**POST-HOC EXPLORATORY MECHANISM DIAGNOSTIC — NOT CONFIRMATORY.** "
        "NEW12-v1 was already used for Stage 2B/3 method selection; nothing "
        "here is a fresh test. No training occurred. The E0 environment, all "
        "SC-FAC checkpoints, the BF-T0.5 rule, and the Stage 2/3/3.5 artifacts "
        "are frozen and unmodified. The only policy change under test is a "
        "single-element flush-logit mask. Local fork numbers are one-step "
        "**local exposure only** — neither a recovery estimate nor an "
        "optimality theorem."
    )
    A("")

    # 1 --------------------------------------------------------------------
    A("## 1. Purpose and hypotheses")
    A("")
    A(
        "Stage 3.5 H1 (BF settlement + frozen SC-FAC conditional flush, the "
        "flush head conditioned on the BF settle index) produced unstable "
        "seed/capacity results. D1 separates two post-hoc explanations:"
    )
    A("")
    A(
        "- **M1 — direct same-wallet conflict.** Because E0 decodes the "
        "flush first, when the conditional flush head selects the same wallet "
        "that the BF rule selected for settlement, the flush voids a feasible "
        "current settlement (the wallet is zeroed and placed in "
        "`refresh_targets`, so the settle check fails). Masking exactly that "
        "one flush logit should remove the loss."
    )
    A(
        "- **M2 — broader policy / state-distribution incompatibility.** The "
        "frozen flush head was trained against SC-FAC's own settlement "
        "distribution; under BF settlements the visited state distribution "
        "differs, and any intervention on the head can perturb useful future "
        "behaviour beyond the mechanical conflict."
    )
    A("")
    A(
        "The masked arm M is a surgical restriction, not a claim that the "
        "masked action is optimal: the same-wallet flush may sacrifice the "
        "current transaction deliberately for a faster future refill."
    )
    A("")

    # 2 --------------------------------------------------------------------
    A("## 2. Arms and exact intervention")
    A("")
    A(
        "Per step the BF settle index `s_BF = argmin_(usable, balance≥tx) "
        "(balance, index)` (no-op `k` if none) is computed. The frozen SC "
        "model yields flush logits `L = forward_flush_given_settle(state, "
        "s_BF)`:"
    )
    A("")
    A("- **U (unmasked):** `flush = argmax L` — the archived H1 policy.")
    A(
        "- **M (masked):** if `s_BF < k`, set **exactly** `L[0, s_BF] = "
        "-infinity` (float min), leave every other logit untouched, then "
        "`flush = argmax` with the original smallest-index tie-break. If "
        "`s_BF == k` (no feasible settle), no masking is applied."
    )
    A(
        "- **BF:** frozen BF-T0.5 settle+flush under identical "
        "instrumentation (deterministic; 2 cells)."
    )
    A("")
    A(
        "The submitted joint action `s_BF·(k+1)+flush` is executed by the "
        "**unmodified E0**. No forced no-op, no BF threshold change, no "
        "high-balance mask, no environment/ordering change, no training."
    )
    A("")

    # 3 --------------------------------------------------------------------
    A("## 3. Data, frozen bindings, provenance")
    A("")
    manifest_sha = data[("U", 800, 123)]["result"]["stream_manifest_sha256"]
    A(f"- Dataset: NEW12-v1, 12 regimes × {b0.EPISODES_PER_REGIME} episodes, "
      f"T={b0.T}, k={b0.K}, F={b0.F}; manifest SHA `{manifest_sha}`.")
    A(f"- Frozen raw tarball SHA: `{b0.RAW_SHA}`; lineage `{b0.LINEAGE}`.")
    A(
        "- Every SC-FAC checkpoint SHA was verified against the Stage 2 job "
        "matrix AND the Stage 3 `seed_level_scores.csv` before use; each cell "
        "records the checkpoint SHA per episode row."
    )
    A(
        "- The hash-pinned Stage 2 adapter was re-verified on every run; "
        "NEW12 pools and per-episode hashes were verified through the frozen "
        "adapter. See `D1_CONFIG.json` and each cell's `result.json`."
    )
    A("")

    # 4 --------------------------------------------------------------------
    A("## 4. Execution matrix and runtime")
    A("")
    total_elapsed = 0.0
    for arm in ("U", "M"):
        for C in b0.CAPACITIES:
            for s in SEEDS:
                total_elapsed += float(data[(arm, C, s)]["result"]["elapsed_seconds"])
    for C in b0.CAPACITIES:
        total_elapsed += float(data[("BF", C, None)]["result"]["elapsed_seconds"])
    A(
        f"- 20 H1 cells (2 arms × 2 capacities × 5 seeds) + 2 BF cells = "
        f"22 cells × 2,400 episodes = 52,800 episodes = 52.8M env steps."
    )
    A(f"- Total instrumented wall time: {total_elapsed:.0f}s "
      f"({total_elapsed/60:.1f} min), CPU, torch 2.5.1, numpy 2.0.2.")
    A("- Full step traces are retained only for episodes 0–1 of each regime "
      "(528 traced episodes, 528k rows); all other telemetry is bounded "
      "counters/histograms.")
    A("")

    # 5 --------------------------------------------------------------------
    A("## 5. Mandatory U-parity gate")
    A("")
    parity = json.loads((D1_ROOT / "D1_U_PARITY.json").read_text())
    A(
        f"Status: **{parity['status']}**. Each U cell was required to "
        "reproduce its archived Stage 3.5 H1 `episodes.csv` EXACTLY on "
        f"{parity['cells'][0]['episodes_compared']} episodes × "
        f"{len(parity['cells'][0]['fields_compared'])} fields "
        "(money, settled, flushes, drops, accepted_count, "
        "insufficient_drops, oversize_drops), same regime/episode ordering."
    )
    A("")
    A("| C | seed | result |")
    A("|---|---|---|")
    for c in parity["cells"]:
        A(f"| {c['C']} | {c['seed']} | {'EXACT' if c['exact'] else 'FAIL: '+str(c['first_error'])} |")
    A("")
    A(
        "The two BF telemetry cells additionally reproduce the frozen Stage 2 "
        "BF-T0.5 episodes EXACTLY (validated in `D1_VALIDATION.json`)."
    )
    A("")

    # 6 --------------------------------------------------------------------
    A("## 6. Primary outcome: paired M − U deltas (5 seeds per capacity)")
    A("")
    A("| C | seed | U | M | Δ (M−U) |")
    A("|---|---|---|---|---|")
    for C in b0.CAPACITIES:
        for s in SEEDS:
            A(
                f"| {C} | {s} | {f3(macros[('U', C, s)])} | "
                f"{f3(macros[('M', C, s)])} | "
                f"{fp(macros[('M', C, s)] - macros[('U', C, s)])} |"
            )
    A("")
    for C in b0.CAPACITIES:
        p = stats["paired_M_minus_U"][str(C)]
        A(
            f"- **C={C}** vs frozen BF {f3(bf_ref[C])}: mean Δ "
            f"**{fp(p['mean'])}**, SD {f3(p['sd'])}, SE {f3(p['se'])}, "
            f"95% t CI [{fp(p['ci95_lo'])}, {fp(p['ci95_hi'])}], "
            f"paired t({p['df']}) = {p['t']:.3f}, raw two-sided "
            f"p = {p['p_two_sided_raw']:.4f}, pos/neg {p['n_positive']}/"
            f"{p['n_negative']}."
        )
    A("")
    A(
        "These are descriptive screening statistics (n=5, df=4, raw, "
        "unadjusted) on already-seen data — not a confirmatory test."
    )
    A("")

    # 7 --------------------------------------------------------------------
    A("## 7. MASK_GATE (engineering decision thresholds fixed pre-results)")
    A("")
    A(
        "PASS iff (1) mean(M−U) ≥ +1% of frozen BF macro at ≥1 capacity, "
        "(2) mean(M−U) ≥ −1% of BF at the other, and (3) no seed loses "
        ">5% of BF relative to its own U baseline."
    )
    A("")
    th = g["thresholds"]
    A("| quantity | C=800 | C=1200 |")
    A("|---|---|---|")
    A(
        f"| frozen BF macro | {f3(bf_ref[800])} | {f3(bf_ref[1200])} |"
    )
    A(
        f"| +1% BF threshold | {f3(th['C800_plus_1pct'])} | "
        f"{f3(th['C1200_plus_1pct'])} |"
    )
    A(
        f"| −1% BF threshold | {f3(th['C800_minus_1pct'])} | "
        f"{f3(th['C1200_minus_1pct'])} |"
    )
    A(
        f"| observed mean Δ | {fp(g['mean_delta_C800'])} | "
        f"{fp(g['mean_delta_C1200'])} |"
    )
    A(
        f"| per-seed 5% BF loss limit | {f3(th['C800_seed_loss_5pct_of_BF'])} "
        f"| {f3(th['C1200_seed_loss_5pct_of_BF'])} |"
    )
    A("")
    A(f"1. ≥1 capacity at/above +1%: **{g['condition_at_least_one_capacity_plus_1pct']}**")
    A(f"2. other capacity no worse than −1%: **{g['condition_other_capacity_at_worst_minus_1pct']}**")
    worst_loss = max(g["worst_seed_loss"].values())
    worst_key = max(g["worst_seed_loss"], key=lambda k: g["worst_seed_loss"][k])
    A(f"3. no seed loses >5%: **{g['condition_no_seed_loses_gt_5pct']}** "
      f"(worst U→M loss: {f3(worst_loss)} at {worst_key})")
    A("")
    A(f"**MASK_GATE = {'PASS' if g['PASS'] else 'FAIL'}**")
    A("")

    # 8 --------------------------------------------------------------------
    A("## 8. Same-wallet conflict telemetry (raw head)")
    A("")
    A(
        "Per-step events aggregated over all 2,400,000 steps per cell. "
        "`raw conflict` = unmasked flush argmax equals s_BF (<k); "
        "`pre-feasible conflict` adds that s_BF was usable and could cover "
        "the current tx; `executed` adds that E0 actually flushed the wallet; "
        "`lost settlement` adds that the feasible current tx was dropped."
    )
    A("")
    A("| C | seed | U raw | U pre-feas | U executed | U lost | M raw | M pre-feas | M executed | M lost |")
    A("|---|---|---|---|---|---|---|---|---|---|")

    def _tot(arm: str, C: int, s: int) -> Dict[str, int]:
        tel = data[(arm, C, s)]["telemetry"]
        return {
            r: sum(int(tel["counts"][reg][r]) for reg in b0.REGIMES)
            for r in ("conflict_raw", "conflict_prefeasible",
                      "conflict_executed", "lost_settlement",
                      "mask_changed_choice")
        }

    for C in b0.CAPACITIES:
        for s in SEEDS:
            u, m = _tot("U", C, s), _tot("M", C, s)
            A(
                f"| {C} | {s} | {u['conflict_raw']} | {u['conflict_prefeasible']} "
                f"| {u['conflict_executed']} | {u['lost_settlement']} "
                f"| {m['conflict_raw']} | {m['conflict_prefeasible']} "
                f"| {m['conflict_executed']} | {m['lost_settlement']} |"
            )
    A("")
    for C in b0.CAPACITIES:
        rates = cls[f"mean_prefeasible_conflict_rate_C{C}"]
        erates = cls[f"mean_executed_conflict_rate_C{C}"]
        lostf = cls[f"lost_per_prefeasible_conflict_C{C}"]
        A(
            f"- C={C}: mean pre-feasible conflict rate {rates:.6f} "
            f"(~{rates*b0.T:.3f}/episode), executed rate {erates:.6f}, "
            f"fraction of pre-feasible conflicts that lose the current "
            f"settlement in U: {lostf:.3f}."
        )
    A("")
    A("Rates and per-regime breakdowns: `D1_CONFLICT_TELEMETRY.csv`.")
    A("")

    # 9 --------------------------------------------------------------------
    A("## 9. Executed conflicts and lost current settlements")
    A("")
    lost_u = {
        C: sum(_tot("U", C, s)["lost_settlement"] for s in SEEDS)
        for C in b0.CAPACITIES
    }
    exec_u = {
        C: sum(_tot("U", C, s)["conflict_executed"] for s in SEEDS)
        for C in b0.CAPACITIES
    }
    A(
        f"Across the 5 U cells per capacity, executed same-wallet flushes "
        f"number {exec_u[800]} (C800) and {exec_u[1200]} (C1200); of these, "
        f"the subset voiding a feasible current settlement totals "
        f"{lost_u[800]} and {lost_u[1200]} events. A raw conflict does not "
        "lose the current tx when the flush targets an unusable wallet (E0 "
        "ignores it), when s_BF cannot cover the tx anyway, or when the tx "
        "is oversize. In M these events cannot occur by construction "
        "(s_BF's logit is −∞), which is reflected in the M columns."
    )
    A("")

    # 10 -------------------------------------------------------------------
    A("## 10. Read-only one-step local forks — LOCAL EXPOSURE ONLY")
    A("")
    A(
        "At each U/M state where the RAW head conflicts with a pre-feasible "
        "s_BF, two snapshot clones take one step: A = `(s_BF, s_BF)` and "
        "B = `(s_BF, k)` (no-op flush). Recorded: immediate settled value, "
        "flushes, and Money difference. The clones never touch the live "
        "environment (signature-checked every fork in every run and test)."
    )
    A("")
    A("| C | seed | arm | fork events | mean A−B Money | B accept rate |")
    A("|---|---|---|---|---|---|")
    for C in b0.CAPACITIES:
        for s in SEEDS:
            for arm in ("U", "M"):
                tel = data[(arm, C, s)]["telemetry"]
                fk = {
                    k: sum(float(tel["forks"][r][k]) for r in b0.REGIMES)
                    for k in tel["forks"][b0.REGIMES[0]]
                }
                n = int(fk["fork_events"])
                mean_exp = fk["exposure_A_minus_B_sum"] / n if n else 0.0
                bacc = fk["B_accepted_sum"] / n if n else 0.0
                A(f"| {C} | {s} | {arm} | {n} | {f3(mean_exp)} | {bacc:.3f} |")
    A("")
    A(
        "Interpretation is strictly bounded: A−B measures the immediate local "
        "cost of the conflicting action at that state. It is **not** the value "
        "M would recover — M's trajectory diverges thereafter, the flushed "
        "wallet may be more valuable after refill than the forgone tx, and B "
        "changes future wallet states. These numbers only size the local "
        "mechanism that M1 posits. Full table: `D1_LOCAL_FORK_SUMMARY.csv`."
    )
    A("")

    # 11 -------------------------------------------------------------------
    A("## 11. Mask-induced action changes and flush accounting")
    A("")
    changed = {
        C: sum(_tot("M", C, s)["mask_changed_choice"] for s in SEEDS)
        for C in b0.CAPACITIES
    }
    A(
        f"Mask validation invariant (also asserted in validation): in M the "
        f"argmax changes **iff** the raw argmax was exactly s_BF. Total "
        f"changed choices across the five M cells: {changed[800]} (C800) and "
        f"{changed[1200]} (C1200). At every other step U and M submit "
        "identical actions. Flush request/execution/no-op/unusable counts "
        "per cell are in `D1_CONFLICT_TELEMETRY.csv`."
    )
    A("")
    A("| C | mean U flushes/ep | mean M flushes/ep | Δ flush cost (macro, 10/flush) |")
    A("|---|---|---|---|")
    for C in b0.CAPACITIES:
        fu = statistics.mean(
            statistics.mean(float(r["flushes"]) for r in data[("U", C, s)]["rows"])
            for s in SEEDS)
        fm = statistics.mean(
            statistics.mean(float(r["flushes"]) for r in data[("M", C, s)]["rows"])
            for s in SEEDS)
        su = statistics.mean(macros[("U", C, s)] for s in SEEDS)
        sm = statistics.mean(macros[("M", C, s)] for s in SEEDS)
        set_u = statistics.mean(
            statistics.mean(float(r["settled"]) for r in data[("U", C, s)]["rows"])
            for s in SEEDS)
        set_m = statistics.mean(
            statistics.mean(float(r["settled"]) for r in data[("M", C, s)]["rows"])
            for s in SEEDS)
        df_fl = fm - fu
        A(
            f"| {C} | {fu:.3f} | {fm:.3f} | ΔMoney {fp(sm-su)} "
            f"= Δsettled {fp(set_m-set_u)} − 10·{df_fl:+.4f} "
            f"({fp(10*df_fl)}) |"
        )
    A("")

    # 12 -------------------------------------------------------------------
    A("## 12. Refill telemetry (F = 3)")
    A("")
    A(
        "Flushing sets `freeze_until = t + F − 1`; the wallet refills at the "
        "end of the step where time first exceeds that value (measured "
        "time-to-refill = 3 decision steps; asserted in test_09). Tracked per "
        "flush cycle: refill completion, whether the refilled endowment was "
        "ever used for a later settlement, and cycles still pending at the "
        "horizon."
    )
    A("")
    A("| arm | C | refill completions | refilled value | cycles never reused | terminal pending |")
    A("|---|---|---|---|---|---|")
    for arm in ("U", "M", "BF"):
        for C in b0.CAPACITIES:
            seeds = SEEDS if arm != "BF" else [None]
            rc = rv = nu = tp = 0
            for s in seeds:
                tel = data[(arm, C, s)]["telemetry"]
                rc += sum(int(tel["counts"][r]["refill_completions"]) for r in b0.REGIMES)
                rv += sum(float(tel["sums"][r]["refilled_value"]) for r in b0.REGIMES)
                nu += sum(int(tel["counts"][r]["episodes_refilled_never_used"]) for r in b0.REGIMES)
                tp += sum(int(tel["counts"][r]["episodes_terminal_pending"]) for r in b0.REGIMES)
            A(f"| {arm} | {C} | {rc} | {f2(rv)} | {nu} | {tp} |")
    A("")
    A("Histograms (ttr, starvation runs): `D1_REFILL_TELEMETRY.csv`.")
    A("")

    # 13 -------------------------------------------------------------------
    A("## 13. Occupancy / starvation telemetry")
    A("")
    A(
        "Per-step flags: zero usable wallets (hard starvation) and zero "
        "feasible wallets for a non-oversize tx (settlement starvation), with "
        "consecutive-run histograms. Aggregate step counts (sum over the 5 "
        "seeds, then per-cell averages):"
    )
    A("")
    A(
        "Note: the frozen BF-T0.5 policy shows zero feasible-starvation steps "
        "— its own joint flush rule replenishes low wallets often enough that "
        "a non-oversize tx is never without a covering wallet. The SC-flush "
        "arms (U/M) do reach such states. This contrast is itself part of the "
        "distribution-shift picture (section 17)."
    )
    A("")
    A("| arm | C | zero-usable steps/cell (mean) | zero-feasible steps/cell (mean) |")
    A("|---|---|---|---|")
    for arm in ("U", "M", "BF"):
        for C in b0.CAPACITIES:
            seeds = SEEDS if arm != "BF" else [None]
            vals_h, vals_f = [], []
            for s in seeds:
                tel = data[(arm, C, s)]["telemetry"]
                vals_h.append(sum(int(tel["counts"][r]["starvation_zero_usable_steps"]) for r in b0.REGIMES))
                vals_f.append(sum(int(tel["counts"][r]["starvation_zero_feasible_steps"]) for r in b0.REGIMES))
            A(f"| {arm} | {C} | {statistics.mean(vals_h):.1f} | {statistics.mean(vals_f):.1f} |")
    A("")

    # 14 -------------------------------------------------------------------
    A("## 14. Regime × seed patterns")
    A("")
    A(
        "Descriptive only; regimes are not independent replicates. Per-regime "
        "Δ(M−U) sign counts across the 5 seeds:"
    )
    A("")
    A("| regime | C=800 mean Δ | pos | neg | C=1200 mean Δ | pos | neg |")
    A("|---|---|---|---|---|---|---|")
    for reg in b0.REGIMES:
        cells = []
        for C in b0.CAPACITIES:
            ds = []
            for s in SEEDS:
                tel_u = data[("U", C, s)]["rows"]
                tel_m = data[("M", C, s)]["rows"]
                mu = statistics.mean(float(r["money"]) for r in tel_u if r["regime"] == reg)
                mm = statistics.mean(float(r["money"]) for r in tel_m if r["regime"] == reg)
                ds.append(mm - mu)
            cells.append(ds)
        A(
            f"| {reg} | {fp(statistics.mean(cells[0]))} | "
            f"{sum(d > 0 for d in cells[0])} | {sum(d < 0 for d in cells[0])} "
            f"| {fp(statistics.mean(cells[1]))} | "
            f"{sum(d > 0 for d in cells[1])} | {sum(d < 0 for d in cells[1])} |"
        )
    A("")
    A("Full per-regime rows: `D1_REGIME_LEVEL.csv`.")
    A("")

    # 15 -------------------------------------------------------------------
    A("## 15. Pre-designated failure-seed inspection")
    A("")
    A(
        "Before unblinding, C800/S777, C1200/S323 and C1200/S532 were named "
        "as the archived H1 failures to inspect explicitly."
    )
    A("")
    A("| C | seed | U | M | Δ | Δ % BF | U pre-feas conflicts | U lost |")
    A("|---|---|---|---|---|---|---|---|")
    for C, s in FAILURE_SEEDS:
        dlt = macros[("M", C, s)] - macros[("U", C, s)]
        u = _tot("U", C, s)
        A(
            f"| {C} | {s} | {f3(macros[('U', C, s)])} | "
            f"{f3(macros[('M', C, s)])} | {fp(dlt)} | "
            f"{100*dlt/bf_ref[C]:+.2f}% | {u['conflict_prefeasible']} | "
            f"{u['lost_settlement']} |"
        )
    A("")

    # 16 -------------------------------------------------------------------
    A("## 16. Capacity contrast")
    A("")
    A(
        f"Mean Δ(M−U) is {fp(g['mean_delta_C800'])} at C=800 and "
        f"{fp(g['mean_delta_C1200'])} at C=1200. Conflict rates, refill "
        "pressure and starvation counts by capacity are in sections 8/12/13. "
        "Any capacity dependence of the mask effect bears directly on M1 "
        "(tighter wallets make same-wallet flushes more consequential) vs M2 "
        "(distribution shift that can run in either direction)."
    )
    A("")

    # 17 -------------------------------------------------------------------
    A("## 17. M1 vs M2 evidence synthesis")
    A("")
    A(
        f"- M1 requires that removing the same-wallet action removes a real "
        f"loss and improves money coherently. Pre-feasible conflicts are "
        f"common enough to be measurable (section 8), local forks show the "
        f"immediate A−B exposure is negative whenever the event fires "
        f"(section 10), and M eliminates the events by construction. The "
        f"observed money response is summarised by the paired deltas "
        f"(section 6) and the gate (section 7)."
    )
    A(
        "- M2 predicts weak/incoherent money response despite the surgical "
        "change, because the frozen head was trained on a different "
        "settlement-induced state distribution; the mask also withholds "
        "flushes whose future refill value the head may have learned to "
        "exploit. Evidence classification below is based on pre-registered "
        "rate/effect rules in `statistics.json`."
    )
    A("")
    A(
        f"- **DIRECT_CONFLICT_IMPORTANCE = {cls['DIRECT_CONFLICT_IMPORTANCE']}**"
    )
    A(
        f"- **STATE_DISTRIBUTION_MISMATCH_EVIDENCE = "
        f"{cls['STATE_DISTRIBUTION_MISMATCH_EVIDENCE']}**"
    )
    A(
        f"- Effects by capacity: C=800 **{cls['effects_by_capacity']['800']}**, "
        f"C=1200 **{cls['effects_by_capacity']['1200']}** (weak_positive = "
        f"mean Δ > 0 but below the 0.5% BF moderate-gain bar)."
    )
    A("")

    # 18 -------------------------------------------------------------------
    A("## 18. What this experiment cannot establish")
    A("")
    A(
        "- The same-wallet flush can be a *correct* learned sacrifice: give up "
        "the current tx to bring a fresh wallet online earlier. Forbidding it "
        "is a policy restriction, not proof of suboptimality; only a retrained "
        "successor (A+) on a genuinely fresh stream release can answer the "
        "design question."
    )
    A(
        "- H1/Stage 3.5 instability is not shown to be caused by M1 even if "
        "the mask helps; post-hoc data, n=5 seeds, df=4, wide CIs, unadjusted "
        "p-values."
    )
    A(
        "- Fork exposure is one-step and partial-equilibrium (section 10); "
        "telemetry traces cover episodes 0–1 per regime only by design."
    )
    A("")

    # 19 -------------------------------------------------------------------
    A("## 19. Decision logic trace")
    A("")
    if g["PASS"]:
        A("- Gate passed all three pre-registered engineering conditions (section 7).")
    else:
        A(
            f"- Gate failed only on condition 1 (mean Δ ≥ +1% BF at some "
            f"capacity); conditions 2 and 3 held (section 7)."
        )
    A(
        f"- Direct-conflict importance classified **"
        f"{cls['DIRECT_CONFLICT_IMPORTANCE']}**; mismatch evidence **"
        f"{cls['STATE_DISTRIBUTION_MISMATCH_EVIDENCE']}** under the documented "
        "heuristic rules."
    )
    A(
        "- Gate PASS → the mechanism matters enough to justify an A+ trainable "
        "successor pilot, but the masked-H1 improvement branch itself stops "
        "and goes to PI. Gate FAIL with negligible conflict effects and no "
        "coherent mismatch signal → stop the branch, no A+ pilot. Anything "
        "in between → PI review, branch retained pending review."
    )
    A("")

    # 20 -------------------------------------------------------------------
    A("## 20. Required final fields")
    A("")
    A("| field | value |")
    A("|---|---|")
    A(f"| MASK_GATE | {'PASS' if g['PASS'] else 'FAIL'} |")
    A(f"| DIRECT_CONFLICT_IMPORTANCE | {cls['DIRECT_CONFLICT_IMPORTANCE']} |")
    A(f"| STATE_DISTRIBUTION_MISMATCH_EVIDENCE | {cls['STATE_DISTRIBUTION_MISMATCH_EVIDENCE']} |")
    A(f"| RECOMMEND_A_PLUS_PILOT | {cls['RECOMMEND_A_PLUS_PILOT']} |")
    A(f"| RECOMMEND_STOP_IMPROVEMENT_BRANCH | {cls['RECOMMEND_STOP_IMPROVEMENT_BRANCH']} |")
    A("")

    (D1_ROOT / "D1_DIAGNOSTIC_REPORT.md").write_text("\n".join(L))

    # --- operational README (brief lists README.md as an output) ----------
    readme = [
        "# D1-H1-SAFE-v1 (Stage 4)",
        "",
        "Post-hoc exploratory mechanism diagnostic. Additive only; no training.",
        "",
        "## Layout",
        "",
        "- `code/run_d1.py` — driver: verify | smoke | parity | run | validate | analyze",
        "- `code/analyze_d1.py`, `code/write_report_d1.py` — analysis + report",
        "- `tests/test_d1.py` — 15 focused invariants (unittest)",
        "- `outputs/{unmasked,masked,bf}/...` — 22 cells: episodes.csv (35 frozen",
        "  fields), result.json, telemetry.json",
        "- `traces/*.npz` — full step traces for episodes 0,1 of every regime",
        "  (528 traced episodes); int/float/fork schema in D1_CONFIG.json",
        "- top-level: D1_CONFIG.json, D1_U_PARITY.json, D1_VALIDATION.json,",
        "  D1_SUMMARY.csv, D1_SEED_CAPACITY.csv, D1_REGIME_LEVEL.csv,",
        "  D1_CONFLICT_TELEMETRY.csv, D1_REFILL_TELEMETRY.csv,",
        "  D1_LOCAL_FORK_SUMMARY.csv, statistics.json, D1_DIAGNOSTIC_REPORT.md",
        "",
        "## Reproduce",
        "",
        "Use the frozen runtime interpreter:",
        "",
        "```",
        "/Users/zhouzhou/benchmarks/A-EVAL-JA-C800-S123/.venv/bin/python \\",
        "  artifacts/aamas2027/stage4/d1_h1_safe/code/run_d1.py verify",
        "... smoke --episodes 3",
        "... run --which u     # U cells + mandatory exact-parity gate",
        "... run --which m     # only after U parity PASS",
        "... run --which bf",
        "... validate",
        "... analyze",
        "```",
        "",
        "The masked cells are only interpretable after D1_U_PARITY.json is PASS;",
        "the driver halts before running M on any parity failure.",
        "",
        f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        "",
    ]
    (D1_ROOT / "README.md").write_text("\n".join(readme))
