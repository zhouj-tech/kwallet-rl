#!/usr/bin/env python3
"""Stage 3.5 B0 hybrid diagnostic runner (post-hoc exploratory).

Commands:
  verify    Frozen-binding integrity checks only (no simulation).
  parity    Replay frozen SC-FAC seed123 (SC/SC) and frozen BF-T0.5 (BF/BF)
            and require EXACT equality with frozen Stage 2 episodes.
  smoke     Tiny H1/H2 run (a few episodes) + action-validity/traces.
  pilot     One full 12-regime x 200-episode hybrid evaluation.
  analyze   Build B0_PILOT_SUMMARY.csv / B0_PER_REGIME.csv / report from
            pilot outputs plus frozen SC123 and BF references.
  all       verify -> parity (both C) -> four pilots -> analyze.

Nothing frozen is edited, retrained, or overwritten. Pilot directories that
already exist are never replaced.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import b0_lib as b0  # noqa: E402

OUT = b0.B0_ROOT / "outputs"
SMOKE_DIR = OUT / "smoke"
PARITY_DIR = OUT / "parity"
PILOT_DIR = OUT / "pilots"

COMPARE_FIELDS = [
    "settled",
    "drops",
    "oversize_drops",
    "insufficient_drops",
    "flushes",
    "utilization",
    "avg_tx_value",
    "drop_rate",
    "value_accept_ratio",
    "count_accept_ratio",
    "total_requested_value",
    "total_tx_count",
    "accepted_count",
    "eval_money_p",
    "eval_money_tau",
    "eval_money",
    "money",
]


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class Session:
    def __init__(self) -> None:
        self.adapter = b0.import_frozen_adapter()
        b0.register_adapter(self.adapter)
        b0.configure_caches()
        self.adapter.verify_sources(b0.FROZEN_REPO)
        self.sc, self.ja = b0.import_frozen_modules(self.adapter)
        self.streams, self.manifest_sha = b0.load_new12_streams(self.adapter)
        self._cfg: Dict[int, Dict[str, Any]] = {}
        self._cp: Dict[int, Path] = {}
        self._cp_sha: Dict[int, str] = {}
        self._ri_sha: Dict[int, str] = {}
        self._agent: Dict[int, Any] = {}
        self._ja_cfg: Dict[int, Dict[str, Any]] = {}

    def cfg(self, C: int) -> Dict[str, Any]:
        if C not in self._cfg:
            cfg, cp, cp_sha, ri_sha = b0.load_sc_bundle(C, self.adapter)
            self._cfg[C], self._cp[C], self._cp_sha[C], self._ri_sha[C] = cfg, cp, cp_sha, ri_sha
        return self._cfg[C]

    def cp_sha(self, C: int) -> str:
        self.cfg(C)
        return self._cp_sha[C]

    def ri_sha(self, C: int) -> str:
        self.cfg(C)
        return self._ri_sha[C]

    def agent(self, C: int):
        if C not in self._agent:
            cfg = self.cfg(C)
            _env, agent = b0.build_sc_agent(self.sc, cfg, self._cp[C])
            self._agent[C] = agent
        return self._agent[C]

    def ja_cfg(self, C: int) -> Dict[str, Any]:
        if C not in self._ja_cfg:
            self._ja_cfg[C] = b0.load_ja_transfer_config(C)
        return self._ja_cfg[C]

    def pool(self, regime: str):
        return next(s for s in self.streams if s["regime"] == regime)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def cmd_verify(_args: argparse.Namespace) -> None:
    t0 = time.time()
    approval = b0.raw_approval()
    print(f"raw tarball SHA : {b0.sha256(b0.RAW_TARBALL)}")
    print(f"expected        : {b0.RAW_SHA}")
    b0.require(b0.sha256(b0.RAW_TARBALL) == b0.RAW_SHA, "Raw tarball SHA mismatch")
    b0.require(approval["git_head"] == b0.LINEAGE, "Raw approval lineage mismatch")

    sess = Session()
    for C in b0.CAPACITIES:
        cfg, cp, cp_sha, ri_sha = b0.load_sc_bundle(C, sess.adapter)
        print(f"SC-FAC seed123 C{C}: checkpoint {cp.name} {cp_sha[:16]}...")
        print(
            f"  cfg C={cfg['env']['C']} k={cfg['env']['k']} F={cfg['env']['F']} "
            f"T={cfg['env']['T']} shaping={cfg['env']['enable_shaping']} "
            f"mode={cfg['model_mode']} money=({cfg['reward']['money_p']},"
            f"{cfg['reward']['money_tau']})"
        )
        ja = b0.load_ja_transfer_config(C)
        b0.require(int(ja["env"]["C"]) == C, "JA transfer config capacity mismatch")
    print(f"NEW12 streams: {len(sess.streams)} pools, manifest {sess.manifest_sha[:16]}...")
    print("VERIFY PASS (%.1fs)" % (time.time() - t0))


# ---------------------------------------------------------------------------
# Frozen parity
# ---------------------------------------------------------------------------

def exact_episode_match(raw: List[Dict[str, float]], frozen: List[Dict[str, str]]) -> List[str]:
    errs: List[str] = []
    if len(raw) != len(frozen):
        return [f"length {len(raw)} != frozen {len(frozen)}"]
    for i, (got, exp) in enumerate(zip(raw, frozen)):
        for field in COMPARE_FIELDS:
            # 'money' is the CSV alias of eval_money added at serialization.
            got_field = "eval_money" if field == "money" else field
            g = float(got[got_field])
            e = float(exp[field])
            if g != e:
                errs.append(f"ep{i}.{field}: got {g!r} frozen {e!r}")
                if len(errs) > 10:
                    return errs
    return errs


def run_parity_for_capacity(sess: Session, C: int, regimes: List[str], episodes: int) -> Dict[str, Any]:
    cfg = sess.cfg(C)
    agent = sess.agent(C)
    frozen_sc_rows = b0.load_frozen_episodes(b0.FROZEN_JOB_IDS[("SC", C)])
    frozen_bf_rows = b0.load_frozen_episodes(b0.FROZEN_JOB_IDS[("BF", C)])
    checks: Dict[str, Any] = {}

    for reg in regimes:
        stream = sess.pool(reg)
        pool = stream["pool"][:episodes]
        frozen_rows_sc = sorted(
            (r for r in frozen_sc_rows if r["regime"] == reg and int(r["episode_index"]) < episodes),
            key=lambda r: int(r["episode_index"]),
        )
        frozen_rows_bf = sorted(
            (r for r in frozen_bf_rows if r["regime"] == reg and int(r["episode_index"]) < episodes),
            key=lambda r: int(r["episode_index"]),
        )

        # 1) SC settle + SC flush via the hybrid evaluator == frozen SC-FAC.
        sc_res = b0.evaluate_hybrid_pool(
            sess.sc, cfg, agent, pool, "SC", "SC", rule_fidelity_check=False
        )
        sess.adapter.validate_result(sc_res, count=episodes)
        sc_err = exact_episode_match(sc_res["raw_results"], frozen_rows_sc)

        # 2) Frozen BF rule through the exact frozen adapter code path (JA env).
        # NOTE: frozen rule_evaluation hardcodes num_episodes=200 in its return
        # dict, so the full result validator applies only to 200-episode runs;
        # short runs validate each episode individually instead.
        ja_res = sess.adapter.rule_evaluation(sess.ja, sess.ja_cfg(C), pool, reg)
        bf_on_sc = sess.adapter.rule_evaluation(sess.sc, cfg, pool, reg)
        if episodes == b0.EPISODES_PER_REGIME:
            sess.adapter.validate_result(ja_res)
            sess.adapter.validate_result(bf_on_sc)
        else:
            for row in ja_res["raw_results"] + bf_on_sc["raw_results"]:
                sess.adapter.validate_episode(row)
        ja_err = exact_episode_match(ja_res["raw_results"], frozen_rows_bf)

        # 3) Same BF rule on the SC config's E0 instance (dynamics equivalence).
        bf_sc_err = exact_episode_match(bf_on_sc["raw_results"], frozen_rows_bf)

        checks[reg] = {
            "sc_sc_exact": not sc_err,
            "bf_bf_ja_env_exact": not ja_err,
            "bf_bf_sc_env_exact": not bf_sc_err,
            "sc_sc_first_error": sc_err[0] if sc_err else None,
            "bf_ja_first_error": ja_err[0] if ja_err else None,
            "bf_sc_env_first_error": bf_sc_err[0] if bf_sc_err else None,
            "episodes_compared": episodes,
        }
        print(
            f"  C{C} {reg:5s} SC/SC={'PASS' if not sc_err else 'FAIL'} "
            f"BF/BF(JA env)={'PASS' if not ja_err else 'FAIL'} "
            f"BF/BF(SC env)={'PASS' if not bf_sc_err else 'FAIL'}"
        )
    all_pass = all(
        v["sc_sc_exact"] and v["bf_bf_ja_env_exact"] and v["bf_bf_sc_env_exact"]
        for v in checks.values()
    )
    return dict(capacity=C, pass_=all_pass, regimes=checks, episodes_per_regime=episodes)


def cmd_parity(args: argparse.Namespace) -> None:
    sess = Session()
    regimes = b0.REGIMES if args.regimes is None else args.regimes
    episodes = args.episodes
    result = dict(started_utc=utc(), runtime=runtime_banner(), capacities=[])
    for C in (b0.CAPACITIES if args.capacity is None else args.capacity):
        print(f"== Parity C={C}, {len(regimes)} regimes x {episodes} episodes ==")
        cap = run_parity_for_capacity(sess, C, regimes, episodes)
        result["capacities"].append(cap)
        if not cap["pass_"]:
            write_json(PARITY_DIR / f"parity_C{C}_FAILED.json", result)
            raise SystemExit(f"PARITY FAILURE at C={C}; frozen artifacts left untouched")
    result["completed_utc"] = utc()
    result["status"] = "PASS"
    if args.save:
        write_json(PARITY_DIR / f"parity_ep{episodes}.json", result)
        print(f"parity saved -> {PARITY_DIR / f'parity_ep{episodes}.json'}")
    print("PARITY PASS: hybrid evaluator reproduces frozen SC-FAC and BF-T0.5 exactly")


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------

def cmd_smoke(args: argparse.Namespace) -> None:
    sess = Session()
    n = args.episodes
    out = dict(started_utc=utc(), episodes_per_regime=n, capacities=[])
    for C in b0.CAPACITIES:
        cfg = sess.cfg(C)
        agent = sess.agent(C)
        pool = sess.pool("US")["pool"][:n]
        cap = dict(C=C, hybrids={})
        for hybrid in b0.HYBRIDS:
            settle_mode, flush_mode = ("BF", "SC") if hybrid == b0.H1 else ("SC", "BF")
            res = b0.evaluate_hybrid_pool(
                sess.sc,
                cfg,
                agent,
                pool,
                settle_mode,
                flush_mode,
                record=True,
                rule_fidelity_check=True,
            )
            rows = res["raw_results"]
            cond_ok = verify_conditioning_in_traces(sess, res["traces"], hybrid, pool, cfg, C)
            money_ids = [
                abs(r["eval_money"] - (r["settled"] - 10 * r["flushes"])) <= 1e-9 for r in rows
            ]
            cap["hybrids"][hybrid] = dict(
                episodes=len(rows),
                money_identity_all=all(money_ids),
                conditioning_trace_ok=cond_ok,
                mean_money=statistics.mean(r["eval_money"] for r in rows),
                first_trace=res["traces"][:5],
            )
            print(
                f"  smoke C{C} {hybrid}: money_identity={all(money_ids)} "
                f"conditioning={cond_ok} meanMoney={cap['hybrids'][hybrid]['mean_money']:.2f}"
            )
        out["capacities"].append(cap)
    out["completed_utc"] = utc()
    out["status"] = "PASS"
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    write_json(SMOKE_DIR / f"smoke_ep{n}.json", out)
    b0.require(
        all(
            h["money_identity_all"] and h["conditioning_trace_ok"]
            for c in out["capacities"]
            for h in c["hybrids"].values()
        ),
        "Smoke validation failed",
    )
    print("SMOKE PASS")


def verify_conditioning_in_traces(
    sess: Session, traces: List[Dict[str, Any]], hybrid: str, pool, cfg, C: int
) -> bool:
    """Replay recorded traces and prove which settle index conditioned the flush."""
    import torch

    env = sess.sc.make_env(cfg, max_steps=b0.T)
    agent = sess.agent(C)
    for ep in range(len(pool)):
        state = env.reset(tx_stream=pool[ep])
        ep_traces = [t for t in traces if t["episode"] == ep]
        for t in ep_traces:
            s_bf = b0.bf_settle(env)
            state_t = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            with torch.no_grad():
                sl, _ = agent.model.forward_settle_value(state_t)
                s_sc = int(torch.argmax(sl, dim=-1).item())
                f_given_bf = int(
                    torch.argmax(
                        agent.model.forward_flush_given_settle(
                            state_t, torch.tensor([s_bf], dtype=torch.long, device=agent.device)
                        ),
                        dim=-1,
                    ).item()
                )
                f_given_sc = int(
                    torch.argmax(
                        agent.model.forward_flush_given_settle(
                            state_t, torch.tensor([s_sc], dtype=torch.long, device=agent.device)
                        ),
                        dim=-1,
                    ).item()
                )
            if hybrid == b0.H1:
                if t["settle"] != s_bf or t["flush"] != f_given_bf:
                    return False
                if s_bf != s_sc and t["sc_flush"] != f_given_sc:
                    return False
            else:
                f_bf = b0.bf_flush(env, s_sc)
                if t["settle"] != s_sc or t["flush"] != f_bf:
                    return False
            state = env.step(t["action"])[0]
    return True


# ---------------------------------------------------------------------------
# Pilots
# ---------------------------------------------------------------------------

def pilot_paths(hybrid: str, C: int) -> Path:
    return PILOT_DIR / f"{hybrid}_C{C}_S{b0.SC_SEED}"


def write_pilot_outputs(
    sess: Session,
    hybrid: str,
    C: int,
    per_regime: Dict[str, Dict[str, Any]],
    elapsed: float,
    started: float,
) -> Path:
    job_id = b0.PILOT_JOB[(hybrid, C)]
    fields = b0.frozen_episode_fields()
    episode_rows: List[Dict[str, Any]] = []
    summaries = {}
    means = []
    for stream in sess.streams:
        reg = stream["regime"]
        res = per_regime[reg]
        summaries[reg] = res["summary"]
        means.append(res["summary"]["eval_money"]["mean"])
        for i, metric in enumerate(res["raw_results"]):
            row = dict(
                schema_version=1,
                dataset_id="NEW12-v1",
                job_id=job_id,
                family="B0",
                method=b0.HYBRID_METHOD_LABEL[hybrid],
                C=C,
                k=b0.K,
                F=b0.F,
                T=b0.T,
                training_seed=b0.SC_SEED,
                condition_mode="full",
                cohort="stage3_5_b0_posthoc_diagnostic",
                regime=reg,
                episode_index=i,
                episode_seed=stream["episode_seeds"][i],
                episode_sha256=stream["episode_hashes"][i],
                pool_sha256=stream["sha256"],
                checkpoint_sha256=sess.cp_sha(C),
            )
            # 'money' is the frozen CSV alias of the live metric 'eval_money'.
            metric_alias = {"money": "eval_money"}
            for key in fields:
                if key in row:
                    continue
                if key.startswith("gate_"):
                    continue
                src = metric_alias.get(key, key)
                b0.require(
                    src in metric,
                    f"Frozen episode field {key!r} missing from live metrics",
                )
                row[key] = metric[src]
            episode_rows.append(row)

    macro = statistics.mean(means)
    result = dict(
        schema_version=1,
        b0_stage="stage3_5_b0_hybrid_diagnostic",
        post_hoc_exploratory=True,
        confirmatory=False,
        dataset_id="NEW12-v1",
        job_id=job_id,
        hybrid=hybrid,
        hybrid_description=b0.HYBRID_DESC[hybrid],
        settle_policy="BF-T0.5" if hybrid == b0.H1 else "SC-FAC",
        flush_policy="SC-FAC-conditioned-on-selected-settle"
        if hybrid == b0.H1
        else "BF-T0.5-excluding-selected-settle",
        config=sess.cfg(C),
        checkpoint_sha256=sess.cp_sha(C),
        run_info_sha256=sess.ri_sha(C),
        source_hashes=sess.adapter.SOURCE_HASHES,
        adapter_sha256=b0.sha256(b0.FROZEN_REPO / "tools/aamas_stage2/adapter.py"),
        stream_manifest_sha256=sess.manifest_sha,
        frozen_raw_tarball_sha256=b0.sha256(b0.RAW_TARBALL),
        frozen_lineage=b0.LINEAGE,
        regime_summaries=summaries,
        row_count=len(episode_rows),
        macro12_money=macro,
        validation_status="PASS",
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        completed_utc=utc(),
        elapsed_seconds=elapsed,
        runtime=runtime_banner(),
    )
    dst = pilot_paths(hybrid, C)
    b0.require(not dst.exists(), f"Refusing to overwrite existing pilot output: {dst}")
    dst.mkdir(parents=True)

    with (dst / "episodes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(episode_rows)
    write_json(dst / "result.json", result)
    return dst


def run_pilot(sess: Session, hybrid: str, C: int, require_parity: bool) -> Path:
    if require_parity:
        gate = PARITY_DIR / "parity_ep200.json"
        if not gate.is_file():
            print("Full parity gate not found; running it before pilot ...")
            gate_args = argparse.Namespace(
                capacity=[C], regimes=None, episodes=200, save=True
            )
            cmd_parity(gate_args)
        else:
            doc = json.loads(gate.read_text())
            ok = any(
                c["capacity"] == C
                and c["episodes_per_regime"] == 200
                and all(
                    v["sc_sc_exact"] and v["bf_bf_ja_env_exact"] and v["bf_bf_sc_env_exact"]
                    for v in c["regimes"].values()
                )
                for c in doc["capacities"]
            )
            b0.require(ok, f"Cached parity does not PASS for C={C}")
            print(f"cached full parity PASS for C={C}")

    settle_mode, flush_mode = ("BF", "SC") if hybrid == b0.H1 else ("SC", "BF")
    cfg = sess.cfg(C)
    agent = sess.agent(C)
    started = time.time()
    per_regime: Dict[str, Dict[str, Any]] = {}
    for stream in sess.streams:
        t0 = time.time()
        per_regime[stream["regime"]] = b0.evaluate_hybrid_pool(
            sess.sc,
            cfg,
            agent,
            stream["pool"],
            settle_mode,
            flush_mode,
            record=False,
            rule_fidelity_check=True,
        )
        m = per_regime[stream["regime"]]["summary"]["eval_money"]["mean"]
        print(f"  {hybrid} C{C} {stream['regime']:5s} money={m:11.3f} ({time.time()-t0:.0f}s)")
    elapsed = time.time() - started
    dst = write_pilot_outputs(sess, hybrid, C, per_regime, elapsed, started)
    macro = statistics.mean(
        per_regime[r]["summary"]["eval_money"]["mean"] for r in b0.REGIMES
    )
    print(f"PILOT {hybrid} C{C} macro12_money={macro:.4f} -> {dst}  ({elapsed:.0f}s)")
    return dst


def cmd_pilot(args: argparse.Namespace) -> None:
    sess = Session()
    jobs = (
        [(h, C) for h in b0.HYBRIDS for C in b0.CAPACITIES]
        if args.all
        else [(args.hybrid, args.C)]
    )
    for hybrid, C in jobs:
        run_pilot(sess, hybrid, C, require_parity=not args.skip_parity)


def runtime_banner() -> Dict[str, Any]:
    import numpy as np
    import torch

    return dict(
        python_executable=sys.executable,
        python_version=platform.python_version(),
        platform=platform.platform(),
        numpy_version=np.__version__,
        torch_version=torch.__version__,
        torch_threads=torch.get_num_threads(),
        device="cpu",
    )


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def load_pilot_episodes(hybrid: str, C: int) -> List[Dict[str, str]]:
    dst = pilot_paths(hybrid, C)
    with (dst / "episodes.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def regime_means(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    metrics = [
        "money",
        "settled",
        "flushes",
        "accepted_count",
        "insufficient_drops",
        "oversize_drops",
        "drops",
        "total_requested_value",
        "value_accept_ratio",
        "count_accept_ratio",
    ]
    for reg in b0.REGIMES:
        rr = [r for r in rows if r["regime"] == reg]
        out[reg] = {m: statistics.mean(float(r[m]) for r in rr) for m in metrics}
    return out


def cmd_analyze(_args: argparse.Namespace) -> None:
    analysis_start = time.time()
    refs = b0.load_stage3_reference()

    # Cross-check frozen Stage 3 tables against the frozen Stage 2 raw episodes.
    raw_check = {}
    for label, method in [("SC", "SC-FAC"), ("BF", "BF-T0.5")]:
        for C in b0.CAPACITIES:
            vec = b0.episodes_to_vectors(b0.load_frozen_episodes(b0.FROZEN_JOB_IDS[(label, C)]))
            raw_macro = b0.macro_from_vectors(vec)
            tab = refs[(method, C)]
            b0.require(
                math.isclose(raw_macro, tab["macro"], rel_tol=0, abs_tol=1e-9),
                f"Stage2/Stage3 macro mismatch {method} C{C}",
            )
            for reg in b0.REGIMES:
                b0.require(
                    math.isclose(
                        statistics.mean(vec[reg]["money"]),
                        tab["per_regime_money"][reg],
                        rel_tol=0,
                        abs_tol=1e-9,
                    ),
                    f"Stage2/Stage3 regime mismatch {method} C{C} {reg}",
                )
            raw_check[f"{method}_C{C}"] = raw_macro

    summary_rows: List[Dict[str, Any]] = []
    regime_rows_out: List[Dict[str, Any]] = []
    decision: Dict[str, Any] = {h: {"delta_sc_by_C": {}, "recovery_by_C": {}} for h in b0.HYBRIDS}

    for hybrid in b0.HYBRIDS:
        for C in b0.CAPACITIES:
            rows = load_pilot_episodes(hybrid, C)
            means = regime_means(rows)
            macro = statistics.mean(means[r]["money"] for r in b0.REGIMES)
            sc_ref = refs[("SC-FAC", C)]
            bf_ref = refs[("BF-T0.5", C)]
            gap = bf_ref["macro"] - sc_ref["macro"]
            d_sc = macro - sc_ref["macro"]
            d_bf = macro - bf_ref["macro"]
            recovery = d_sc / gap if gap != 0 else None

            sc_settled = statistics.mean(
                float(sc_ref["regime_rows"][r]["settled"]) for r in b0.REGIMES
            )
            sc_flushes = statistics.mean(
                float(sc_ref["regime_rows"][r]["flushes"]) for r in b0.REGIMES
            )
            bf_settled = statistics.mean(
                float(bf_ref["regime_rows"][r]["settled"]) for r in b0.REGIMES
            )
            bf_flushes = statistics.mean(
                float(bf_ref["regime_rows"][r]["flushes"]) for r in b0.REGIMES
            )
            hy_settled = statistics.mean(means[r]["settled"] for r in b0.REGIMES)
            hy_flushes = statistics.mean(means[r]["flushes"] for r in b0.REGIMES)

            above_sc = sum(means[r]["money"] > sc_ref["per_regime_money"][r] for r in b0.REGIMES)
            above_bf = sum(means[r]["money"] >= bf_ref["per_regime_money"][r] for r in b0.REGIMES)

            summary_rows.append(
                dict(
                    hybrid=hybrid,
                    hybrid_description=b0.HYBRID_DESC[hybrid],
                    C=C,
                    hybrid_money=macro,
                    SC123_money=sc_ref["macro"],
                    BF_money=bf_ref["macro"],
                    delta_vs_SC=d_sc,
                    delta_vs_BF=d_bf,
                    SC_BF_gap=gap,
                    gap_recovery=recovery,
                    settled=hy_settled,
                    flushes=hy_flushes,
                    flush_cost=10 * hy_flushes,
                    accepted_count=statistics.mean(means[r]["accepted_count"] for r in b0.REGIMES),
                    insufficient_drops=statistics.mean(
                        means[r]["insufficient_drops"] for r in b0.REGIMES
                    ),
                    oversize_drops=statistics.mean(
                        means[r]["oversize_drops"] for r in b0.REGIMES
                    ),
                    SC_settled=sc_settled,
                    SC_flushes=sc_flushes,
                    BF_settled=bf_settled,
                    BF_flushes=bf_flushes,
                    delta_settled_vs_SC=hy_settled - sc_settled,
                    delta_flushcost_vs_SC=10 * (hy_flushes - sc_flushes),
                    delta_settled_vs_BF=hy_settled - bf_settled,
                    delta_flushcost_vs_BF=10 * (hy_flushes - bf_flushes),
                    regimes_above_SC=above_sc,
                    regimes_at_or_above_BF=above_bf,
                )
            )
            decision[hybrid]["delta_sc_by_C"][C] = d_sc
            decision[hybrid]["recovery_by_C"][C] = recovery

            for reg in b0.REGIMES:
                regime_rows_out.append(
                    dict(
                        hybrid=hybrid,
                        C=C,
                        regime=reg,
                        hybrid_money=means[reg]["money"],
                        SC123_money=sc_ref["per_regime_money"][reg],
                        BF_money=bf_ref["per_regime_money"][reg],
                        delta_vs_SC=means[reg]["money"] - sc_ref["per_regime_money"][reg],
                        delta_vs_BF=means[reg]["money"] - bf_ref["per_regime_money"][reg],
                        hybrid_settled=means[reg]["settled"],
                        hybrid_flushes=means[reg]["flushes"],
                        SC_settled=float(sc_ref["regime_rows"][reg]["settled"]),
                        SC_flushes=float(sc_ref["regime_rows"][reg]["flushes"]),
                        BF_settled=float(bf_ref["regime_rows"][reg]["settled"]),
                        BF_flushes=float(bf_ref["regime_rows"][reg]["flushes"]),
                        hybrid_accepted=means[reg]["accepted_count"],
                        hybrid_insufficient_drops=means[reg]["insufficient_drops"],
                        hybrid_oversize_drops=means[reg]["oversize_drops"],
                    )
                )

    # Screening heuristics (NOT significance thresholds).
    for hybrid in b0.HYBRIDS:
        d = decision[hybrid]
        rule_a = all(d["delta_sc_by_C"][C] > 0 for C in b0.CAPACITIES)
        rule_b = any(
            d["delta_sc_by_C"][C] > 200 or (d["recovery_by_C"][C] or 0) > 0.30
            for C in b0.CAPACITIES
        )
        d["rule_A_improves_both_capacities"] = rule_a
        d["rule_B_gt200_or_gt30pct_any_capacity"] = rule_b
        d["promising_for_expansion"] = rule_a or rule_b

    write_csv(b0.B0_ROOT / "B0_PILOT_SUMMARY.csv", summary_rows)
    write_csv(b0.B0_ROOT / "B0_PER_REGIME.csv", regime_rows_out)
    write_json(
        b0.B0_ROOT / "outputs" / "B0_ANALYSIS.json",
        dict(
            status="complete",
            post_hoc_exploratory=True,
            confirmatory=False,
            frozen_reference_macro_money=raw_check,
            screening=decision,
            summary=summary_rows,
            analysis_elapsed_seconds=time.time() - analysis_start,
            completed_utc=utc(),
        ),
    )
    write_report(summary_rows, regime_rows_out, decision, raw_check)
    print("ANALYZE complete:")
    for r in summary_rows:
        print(
            f"  {r['hybrid']} C{r['C']}: money={r['hybrid_money']:.2f} "
            f"dSC={r['delta_vs_SC']:+.2f} dBF={r['delta_vs_BF']:+.2f} "
            f"recovery={r['gap_recovery']:.3f}"
        )
    for h in b0.HYBRIDS:
        print(f"  {h} promising_for_expansion = {decision[h]['promising_for_expansion']}")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_report(
    summary: List[Dict[str, Any]],
    per_regime: List[Dict[str, Any]],
    decision: Dict[str, Any],
    raw_check: Dict[str, float],
) -> None:
    def f3(x: Any) -> str:
        return "NA" if x is None else f"{x:.3f}"

    lines: List[str] = []
    lines.append("# B0 Hybrid Diagnostic Report (Stage 3.5)")
    lines.append("")
    lines.append(
        "**POST-HOC EXPLORATORY DIAGNOSTIC — not confirmatory.** No model was "
        "retrained; no frozen Stage 2/Stage 3 artifact was modified. The two "
        "hybrids only re-combine the frozen BF-T0.5 rule components with the "
        "frozen SC-FAC seed=123 policy on the frozen NEW12-v1 streams."
    )
    lines.append("")
    lines.append("## 1. Pilot table")
    lines.append("")
    header = "| Hybrid | C | Hybrid Money | SC123 Money | BF Money | Δ vs SC | Δ vs BF | Gap recovery |"
    sep = "|" + "---|" * 7
    lines.extend([header, sep])
    for r in summary:
        lines.append(
            f"| {r['hybrid']} | {r['C']} | {r['hybrid_money']:.3f} | "
            f"{r['SC123_money']:.3f} | {r['BF_money']:.3f} | {r['delta_vs_SC']:+.3f} | "
            f"{r['delta_vs_BF']:+.3f} | {f3(r['gap_recovery'])} |"
        )
    lines.append("")
    lines.append(
        "Gap recovery = (M_hybrid − M_SC123) / (M_BF − M_SC123). "
        "0 = none recovered; 1 = reached BF; >1 = exceeded BF."
    )
    lines.append("")
    lines.append("## 2. Settled-value / flush-cost decomposition")
    lines.append("")
    lines.append(
        "| Hybrid | C | hybrid settled | hybrid flushes | flush cost | "
        "Δsettled vs SC | Δflushcost vs SC | Δsettled vs BF | Δflushcost vs BF |"
    )
    lines.append("|" + "---|" * 9)
    for r in summary:
        lines.append(
            f"| {r['hybrid']} | {r['C']} | {r['settled']:.3f} | {r['flushes']:.3f} | "
            f"{r['flush_cost']:.3f} | {r['delta_settled_vs_SC']:+.3f} | "
            f"{r['delta_flushcost_vs_SC']:+.3f} | {r['delta_settled_vs_BF']:+.3f} | "
            f"{r['delta_flushcost_vs_BF']:+.3f} |"
        )
    lines.append("")
    lines.append("Per-pilot means (across 12 regimes of 200 episodes):")
    lines.append("")
    lines.append(
        "| Hybrid | C | accepted/ep | insufficient drops | oversize drops | SC settled | "
        "SC flushes | BF settled | BF flushes | regimes > SC | regimes ≥ BF |"
    )
    lines.append("|" + "---|" * 11)
    for r in summary:
        lines.append(
            f"| {r['hybrid']} | {r['C']} | {r['accepted_count']:.2f} | "
            f"{r['insufficient_drops']:.3f} | {r['oversize_drops']:.3f} | "
            f"{r['SC_settled']:.3f} | {r['SC_flushes']:.3f} | {r['BF_settled']:.3f} | "
            f"{r['BF_flushes']:.3f} | {r['regimes_above_SC']} | {r['regimes_at_or_above_BF']} |"
        )
    lines.append("")
    lines.append("## 3. Per-regime directional summary")
    lines.append("")
    for hybrid in b0.HYBRIDS:
        for C in b0.CAPACITIES:
            rr = [r for r in per_regime if r["hybrid"] == hybrid and r["C"] == C]
            pos = [r for r in rr if r["delta_vs_SC"] > 0]
            lines.append(
                f"- {hybrid} C={C}: {len(pos)}/12 regimes above SC123; "
                f"best Δ vs SC {max(x['delta_vs_SC'] for x in rr):+.2f} "
                f"({max(rr, key=lambda x: x['delta_vs_SC'])['regime']}), "
                f"worst {min(x['delta_vs_SC'] for x in rr):+.2f} "
                f"({min(rr, key=lambda x: x['delta_vs_SC'])['regime']})."
            )
    lines.append("")
    lines.append("Full per-regime numbers: `B0_PER_REGIME.csv`.")
    lines.append("")
    lines.append("## 4. Screening decision (heuristics, not significance)")
    lines.append("")
    for h in b0.HYBRIDS:
        d = decision[h]
        lines.append(f"### {h}")
        lines.append("")
        lines.append(f"- Rule A (improve over SC123 at BOTH capacities): **{d['rule_A_improves_both_capacities']}**")
        lines.append(
            f"- Rule B (> +200 Money or > 30% gap recovery at either capacity): "
            f"**{d['rule_B_gt200_or_gt30pct_any_capacity']}**"
        )
        lines.append(
            f"- Δ vs SC: C800 {d['delta_sc_by_C'][800]:+.3f}, "
            f"C1200 {d['delta_sc_by_C'][1200]:+.3f}; recovery: "
            f"C800 {f3(d['recovery_by_C'][800])}, C1200 {f3(d['recovery_by_C'][1200])}"
        )
        lines.append(
            f"- **Flagged promising for expansion (PI approval required for any "
            f"extra seeds): {d['promising_for_expansion']}**"
        )
        lines.append("")
    lines.append("## 5. Provenance and caveats")
    lines.append("")
    lines.append(
        "- Frozen references loaded from Stage 3 tables and cross-checked "
        "directly against the frozen Stage 2 raw episodes "
        f"(tarball SHA {b0.RAW_SHA[:16]}…)."
    )
    lines.append(
        f"- SC123 macro references (raw, equal to Stage 3): "
        f"C800 {raw_check['SC-FAC_C800']:.6f}, C1200 {raw_check['SC-FAC_C1200']:.6f}; "
        f"BF: C800 {raw_check['BF-T0.5_C800']:.6f}, C1200 {raw_check['BF-T0.5_C1200']:.6f}."
    )
    lines.append(
        "- Before pilots, the B0 evaluator reproduced every frozen SC123 and "
        "BF-T0.5 episode metric EXACTLY (see outputs/parity/parity_ep200.json)."
    )
    lines.append(
        "- H1 may select a flush wallet equal to the BF settlement wallet; when "
        "that happens E0 processes the flush first and the settlement becomes "
        "infeasible for that step. This is a genuine interaction effect, "
        "preserved by submitting the joint action to the unmodified E0."
    )
    lines.append(
        "- One deterministic pilot per cell; no uncertainty estimates, no "
        "multiple seeds, no new significance claims."
    )
    lines.append("")
    (b0.B0_ROOT / "B0_DIAGNOSTIC_REPORT.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("verify").set_defaults(func=cmd_verify)

    sp = sub.add_parser("parity")
    sp.add_argument("--capacity", type=int, nargs="*", choices=b0.CAPACITIES)
    sp.add_argument("--regimes", nargs="*", choices=b0.REGIMES)
    sp.add_argument("--episodes", type=int, default=200)
    sp.add_argument("--save", action="store_true")
    sp.set_defaults(func=cmd_parity)

    ss = sub.add_parser("smoke")
    ss.add_argument("--episodes", type=int, default=3)
    ss.set_defaults(func=cmd_smoke)

    pp = sub.add_parser("pilot")
    pp.add_argument("--hybrid", choices=b0.HYBRIDS)
    pp.add_argument("--C", type=int, choices=b0.CAPACITIES)
    pp.add_argument("--all", action="store_true")
    pp.add_argument("--skip-parity", action="store_true")
    pp.set_defaults(func=cmd_pilot)

    sub.add_parser("analyze").set_defaults(func=cmd_analyze)

    ap = sub.add_parser("all")
    ap.set_defaults(func=cmd_all)

    args = p.parse_args()
    args.func(args)


def cmd_all(_args: argparse.Namespace) -> None:
    wall = time.time()
    cmd_verify(argparse.Namespace())
    cmd_parity(argparse.Namespace(capacity=None, regimes=None, episodes=200, save=True))
    cmd_smoke(argparse.Namespace(episodes=3))
    sess = Session()
    for hybrid in b0.HYBRIDS:
        for C in b0.CAPACITIES:
            run_pilot(sess, hybrid, C, require_parity=False)  # parity just ran
    cmd_analyze(argparse.Namespace())
    print(f"B0 ALL COMPLETE in {(time.time()-wall)/60:.1f} minutes")


if __name__ == "__main__":
    main()
