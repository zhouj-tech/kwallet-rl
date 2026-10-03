#!/usr/bin/env python3
"""D1-H1-SAFE-v1 driver (K-Wallet AAMAS 2027, Stage 4).

ADDITIVE ONLY. No training, no frozen-artifact edits, no binary backups.

Cells (22):
  * unmasked U: exact archived H1 policy (BF settle + frozen SC conditional
    flush) -- must reproduce the archived Stage 3.5 H1 episode CSVs EXACTLY.
  * masked M: same policy, but when s_BF < k the flush logit at index s_BF is
    set to -infinity before the deterministic argmax. Nothing else changes.
  * BF: frozen BF-T0.5 rule with identical instrumentation (2 cells).

Commands: verify | smoke | parity | run | validate | analyze
Post-hoc exploratory diagnostic -- NOT confirmatory.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- paths ----------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parent
D1_ROOT = CODE_DIR.parents[0]                       # .../stage4/d1_h1_safe
STAGE4_ROOT = CODE_DIR.parents[1]                   # .../aamas2027/stage4
WT = CODE_DIR.parents[4]                            # worktree root
B0_CODE = WT / "artifacts/aamas2027/stage3_5/b0_diagnostic/code"
H1_CODE = WT / "artifacts/aamas2027/stage3_5/h1_multiseed/code"
for p in (str(WT), str(B0_CODE), str(H1_CODE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import b0_lib as b0  # noqa: E402
import run_h1_multi as h1m  # noqa: E402
from tools.aamas_stage4 import d1_core as d1  # noqa: E402

OUT = D1_ROOT / "outputs"
TRACES_DIR = D1_ROOT / "traces"
SMOKE_DIR = D1_ROOT / "smoke"

SEEDS = h1m.SEEDS                       # [123, 323, 532, 777, 999]
ARM_DIR = {"U": "unmasked", "M": "masked", "BF": "bf"}
ARM_METHOD = {
    "U": "D1-H1-BFsettle-SCflush-unmasked",
    "M": "D1-H1-BFsettle-SCflush-masked-settlelogit",
    "BF": "D1-BF-T0.5-telemetry",
}
ARM_DESC = {
    "U": "Archived H1, instrumentation-only (parity arm)",
    "M": "H1 with flush_logit[s_BF]=-inf when s_BF<k (single-logit mask)",
    "BF": "Frozen BF-T0.5 rule under D1 instrumentation",
}
FAMILY = "STAGE4_D1_D1H1SAFE"
COHORT = "stage4_d1_h1_safe_posthoc"

COMPARE_FIELDS = h1m.COMPARE_FIELDS


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def require(ok: bool, msg: str) -> None:
    b0.require(ok, msg)


# ---------------------------------------------------------------------------
# Session (frozen modules, streams, per-seed bundles; plus the JA/BF module)
# ---------------------------------------------------------------------------

class D1Session(h1m.MSSession):
    def __init__(self) -> None:
        super().__init__()
        _sc, self.ja = b0.import_frozen_modules(self.adapter)
        self._bf_cfg: Dict[int, Tuple[Dict[str, Any], str]] = {}

    def bf_config(self, C: int) -> Tuple[Dict[str, Any], str]:
        """Frozen BF-T0.5 transfer config (JA seed123) + run_info sha256."""
        if C not in self._bf_cfg:
            # Resolve exactly as b0.load_ja_transfer_config, keeping the sha.
            matches = [
                r
                for r in b0.csv_rows(b0.ARTIFACT_MANIFEST)
                if r.get("artifact_type") == "run_info"
                and r.get("method") == "JA-PPO"
                and r.get("C") == str(C)
                and r.get("training_seed") == "123"
            ]
            require(len(matches) == 1, f"Ambiguous JA transfer run_info C{C}")
            rec = matches[0]
            p = b0.FROZEN_ARTIFACTS / rec["transfer_relative_path"]
            require(p.is_file(), f"Missing JA transfer run_info: {p}")
            sha = b0.sha256(p)
            require(sha == rec["sha256"], "JA transfer run_info SHA mismatch")
            self._bf_cfg[C] = (b0.read_json(p)["config"], sha)
        return self._bf_cfg[C]


# ---------------------------------------------------------------------------
# Instrumented evaluator (metric bookkeeping mirrors b0.evaluate_hybrid_pool)
# ---------------------------------------------------------------------------

def evaluate_d1_pool(
    module: Any,
    cfg: Dict[str, Any],
    agent: Optional[Any],
    pool: Any,
    arm: str,
    telemetry: d1.CellTelemetry,
    regime: str,
    max_steps: int = b0.T,
    n_episodes: Optional[int] = None,
    adapter: Optional[Any] = None,
) -> Dict[str, Any]:
    require(arm in ("U", "M", "BF"), f"unknown arm {arm}")
    env = module.make_env(cfg, max_steps=max_steps)
    raw_results: List[Dict[str, float]] = []
    n_total = min(b0.EPISODES_PER_REGIME, pool.shape[0])
    n_episodes = n_total if n_episodes is None else min(n_episodes, n_total)

    for ep in range(n_episodes):
        state = env.reset(tx_stream=pool[ep])
        tracker = d1.RefillTracker(env.k)
        post_events = d1.make_post_observer(tracker)
        total_requested_value = 0.0
        total_tx_count = 0
        accepted_count = 0
        episode_gates: List[float] = []
        hard_run = 0
        feas_run = 0

        for t in range(max_steps):
            total_requested_value += float(env.current_tx)
            total_tx_count += 1

            s = int(b0.bf_settle(env))
            pre = d1.observe_pre(env, s)

            if arm == "BF":
                dec = d1.bf_decision(env, s, b0.bf_flush)
                if adapter is not None:
                    require(
                        b0.bf_joint_action(env) == adapter.bf_action(env),
                        "Factorized BF helpers diverged from frozen adapter.bf_action",
                    )
            else:
                dec = d1.d1_h1_action(env, agent, state, arm, s)

            # Read-only local fork at RAW same-wallet conflict states.
            fork_row = None
            if (
                dec["raw_f"] == dec["s_bf"]
                and dec["s_bf"] < env.k
                and pre["prefeasible"]
            ):
                sig_before = d1.env_state_signature(env)
                fork_row = d1.local_fork(env, dec["s_bf"])
                require(
                    d1.env_state_signature(env) == sig_before,
                    "local fork mutated the live environment",
                )

            starve_hard = int(pre["n_usable"] == 0)
            # feasible-settlement starvation ignores unavoidable oversize tx
            starve_feas = int((not pre["oversize"]) and pre["n_feasible"] == 0)
            if starve_hard:
                hard_run += 1
            elif hard_run:
                telemetry.add_starvation_run(regime, "hard", hard_run)
                hard_run = 0
            if starve_feas:
                feas_run += 1
            elif feas_run:
                telemetry.add_starvation_run(regime, "feas", feas_run)
                feas_run = 0

            state, _reward, done, info = env.step(dec["action"])
            episode_gates.append(0.0)
            if info.get("accepted", False):
                accepted_count += 1

            if int(info["flushes_this_step"]) == 1:
                tracker.on_flush(int(info["flush_choice"]), t)
            events = post_events(env, pre["pending"])
            telemetry.begin_step_refill_value(env.wallet_size, len(events))
            telemetry.record_step(
                regime, ep, pre, dec, info, events, tracker,
                (starve_hard, starve_feas), fork_row,
            )

            if done:
                break

        if hard_run:
            telemetry.add_starvation_run(regime, "hard", hard_run)
        if feas_run:
            telemetry.add_starvation_run(regime, "feas", feas_run)
        telemetry.close_episode(regime, tracker)

        require(total_tx_count == max_steps, "Unexpected episode length")
        metrics = env.get_metrics()
        metrics["value_accept_ratio"] = (
            metrics["settled"] / total_requested_value if total_requested_value > 0 else 0.0
        )
        metrics["count_accept_ratio"] = (
            accepted_count / total_tx_count if total_tx_count > 0 else 0.0
        )
        metrics["total_requested_value"] = total_requested_value
        metrics["total_tx_count"] = total_tx_count
        metrics["accepted_count"] = accepted_count
        metrics["gate_mean"] = float(np_mean(episode_gates))
        metrics["gate_std"] = float(np_std(episode_gates))
        metrics["gate_min"] = float(min(episode_gates)) if episode_gates else 0.0
        metrics["gate_max"] = float(max(episode_gates)) if episode_gates else 0.0
        module.add_eval_money_metrics(metrics, cfg)

        for key, value in metrics.items():
            require(
                isinstance(value, (int, float)) and not isinstance(value, bool),
                f"Non-numeric metric {key}",
            )
            require(math.isfinite(float(value)), f"Non-finite metric {key}")
        require(
            math.isclose(
                metrics["eval_money"],
                metrics["settled"] - 10.0 * metrics["flushes"],
                rel_tol=0,
                abs_tol=1e-9,
            ),
            "Money identity failure",
        )
        require(
            int(metrics["drops"]) + int(metrics["accepted_count"]) == max_steps,
            "Count identity failure",
        )
        raw_results.append(metrics)

    summary = module.summarize_episode_metrics(raw_results)
    module.add_reward_summary_metadata(summary, cfg)
    return dict(num_episodes=n_episodes, summary=summary, raw_results=raw_results)


def np_mean(xs: List[float]) -> float:
    import numpy as np

    return float(np.mean(xs)) if xs else 0.0


def np_std(xs: List[float]) -> float:
    import numpy as np

    return float(np.std(xs)) if xs else 0.0


# ---------------------------------------------------------------------------
# Cell layout / provenance
# ---------------------------------------------------------------------------

def cell_path(arm: str, C: int, seed: Optional[int] = None) -> Path:
    if arm == "BF":
        return OUT / ARM_DIR[arm] / f"C{C}"
    return OUT / ARM_DIR[arm] / f"C{C}_S{seed}"


def trace_path(arm: str, C: int, seed: Optional[int] = None) -> Path:
    if arm == "BF":
        return TRACES_DIR / f"{ARM_DIR[arm]}_C{C}.npz"
    return TRACES_DIR / f"{ARM_DIR[arm]}_C{C}_S{seed}.npz"


def job_id_of(arm: str, C: int, seed: Optional[int] = None) -> str:
    if arm == "BF":
        return f"D1-BF-C{C}"
    return f"D1-{arm}-H1-C{C}-S{seed}"


def intervention_spec(arm: str) -> Dict[str, Any]:
    return dict(
        arm=arm,
        description=ARM_DESC[arm],
        settle_policy="BF-T0.5 (frozen factorized rule)",
        flush_policy=(
            "BF-T0.5 (frozen)"
            if arm == "BF"
            else "SC-FAC conditional flush head, conditioned on BF settle"
        ),
        mask=(
            None
            if arm in ("U", "BF")
            else dict(
                rule="if s_BF < k: flush_logits[0, s_BF] = -infinity; argmax unchanged otherwise",
                masked_logits=1,
                tie_break="torch.argmax smallest index (unchanged)",
                noop_flush_index=24,
                no_mask_when_settle_noop=True,
            )
        ),
        environment="E0 unmodified; action=(settle*(k+1)+flush)",
        other_interventions="none",
    )


# ---------------------------------------------------------------------------
# Run one cell
# ---------------------------------------------------------------------------

def run_cell(
    sess: D1Session,
    arm: str,
    C: int,
    seed: Optional[int] = None,
) -> Path:
    dst = cell_path(arm, C, seed)
    require(not dst.exists(), f"Refusing to overwrite existing cell output: {dst}")

    started = time.time()
    telemetry = d1.CellTelemetry(b0.K, b0.REGIMES)

    if arm == "BF":
        module = sess.ja
        cfg, cfg_sha = sess.bf_config(C)
        agent = None
        ckpt_sha = ""
        cfg_seed = 123
        adapter = sess.adapter
    else:
        require(seed is not None, "H1 cells require a seed")
        bnd = sess.bundle(C, seed)
        module = sess.sc
        cfg = bnd["cfg"]
        cfg_sha = bnd["ri_sha"]
        agent = sess.agent(C, seed)
        ckpt_sha = bnd["cp_sha"]
        cfg_seed = seed
        adapter = None

    per_regime: Dict[str, Dict[str, Any]] = {}
    for stream in sess.streams:
        t0 = time.time()
        res = evaluate_d1_pool(
            module,
            cfg,
            agent,
            stream["pool"],
            arm,
            telemetry,
            stream["regime"],
            adapter=adapter,
        )
        per_regime[stream["regime"]] = res
        m = res["summary"]["eval_money"]["mean"]
        print(
            f"  D1 {arm} C{C} {('S%s' % seed) if seed is not None else '':4s} "
            f"{stream['regime']:5s} money={m:11.3f} ({time.time()-t0:.0f}s)",
            flush=True,
        )

    means = [per_regime[r]["summary"]["eval_money"]["mean"] for r in b0.REGIMES]
    macro = statistics.mean(means)
    elapsed = time.time() - started
    job_id = job_id_of(arm, C, seed)

    # --- frozen-schema episode rows (same 35 fields as Stage 3.5) ---------
    fields = b0.frozen_episode_fields()
    episode_rows: List[Dict[str, Any]] = []
    summaries: Dict[str, Any] = {}
    for stream in sess.streams:
        reg = stream["regime"]
        res = per_regime[reg]
        summaries[reg] = res["summary"]
        for i, metric in enumerate(res["raw_results"]):
            row = dict(
                schema_version=1,
                dataset_id="NEW12-v1",
                job_id=job_id,
                family=FAMILY,
                method=ARM_METHOD[arm],
                C=C,
                k=b0.K,
                F=b0.F,
                T=b0.T,
                training_seed=("" if arm == "BF" else seed),
                condition_mode=("n/a" if arm == "BF" else "full"),
                cohort=COHORT,
                regime=reg,
                episode_index=i,
                episode_seed=stream["episode_seeds"][i],
                episode_sha256=stream["episode_hashes"][i],
                pool_sha256=stream["sha256"],
                checkpoint_sha256=(cfg_sha if arm == "BF" else ckpt_sha),
            )
            for key in fields:
                if key in row or key.startswith("gate_"):
                    continue
                src = "eval_money" if key == "money" else key
                require(src in metric, f"Frozen episode field {key!r} missing from metrics")
                row[key] = metric[src]
            require(set(row.keys()) == set(fields), "Episode row schema mismatch")
            episode_rows.append(row)

    result = dict(
        schema_version=1,
        stage="stage4_d1_h1_safe",
        post_hoc_exploratory=True,
        confirmatory=False,
        dataset_id="NEW12-v1",
        job_id=job_id,
        arm=arm,
        intervention=intervention_spec(arm),
        C=C,
        training_seed=("" if arm == "BF" else seed),
        config=cfg,
        config_seed=cfg_seed,
        run_info_sha256=cfg_sha,
        checkpoint_sha256=ckpt_sha,
        frozen_sc_job_id=(None if arm == "BF" else h1m.frozen_sc_job_id(C, seed)),
        frozen_source_hashes=sess.adapter.SOURCE_HASHES,
        adapter_sha256=b0.sha256(b0.FROZEN_REPO / "tools/aamas_stage2/adapter.py"),
        stream_manifest_sha256=sess.manifest_sha,
        frozen_raw_tarball_sha256=b0.sha256(b0.RAW_TARBALL),
        frozen_lineage=b0.LINEAGE,
        regime_summaries=summaries,
        row_count=len(episode_rows),
        episodes_per_regime=b0.EPISODES_PER_REGIME,
        macro12_money=macro,
        validation_status="PASS",
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        completed_utc=utc(),
        elapsed_seconds=elapsed,
        runtime=sess.runtime_banner(),
    )

    dst.mkdir(parents=True)
    with (dst / "episodes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(episode_rows)
    (dst / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    (dst / "telemetry.json").write_text(
        json.dumps(telemetry.summary(), indent=2, sort_keys=True)
    )

    # traces (npz): keys {REG}_{ep}_i / _f / _fork
    tpath = trace_path(arm, C, seed)
    tpath.parent.mkdir(parents=True, exist_ok=True)
    kv: Dict[str, Any] = {}
    import numpy as np

    for reg in b0.REGIMES:
        for ep in d1.TRACED_EPISODES:
            bucket = telemetry.traces[reg].get(ep)
            if bucket is None:
                continue
            kv[f"{reg}_{ep}_i"] = np.asarray(bucket["I"], dtype=np.int16)
            kv[f"{reg}_{ep}_f"] = np.asarray(bucket["F"], dtype=np.float32)
            kv[f"{reg}_{ep}_fork"] = (
                np.asarray(bucket["fork"], dtype=np.float32)
                if bucket["fork"]
                else np.zeros((0, len(d1.TRACE_FORK_COLUMNS)), dtype=np.float32)
            )
    np.savez(tpath, **kv)

    print(
        f"CELL D1 {arm} C{C} {('S%s' % seed) if seed is not None else ''} "
        f"macro12_money={macro:.4f} -> {dst} ({elapsed:.0f}s)",
        flush=True,
    )
    return dst


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def cmd_verify(_args: argparse.Namespace) -> None:
    t0 = time.time()
    approval = b0.raw_approval()
    require(b0.sha256(b0.RAW_TARBALL) == b0.RAW_SHA, "Raw tarball SHA mismatch")
    require(approval["git_head"] == b0.LINEAGE, "Raw approval lineage mismatch")
    sess = D1Session()
    for C in b0.CAPACITIES:
        for seed in SEEDS:
            bnd = sess.bundle(C, seed)
            require(bnd["cfg"]["env"]["C"] == C and bnd["cfg"]["seed"] == seed, "cfg binding")
            require(
                bnd["cfg"]["env"]["k"] == b0.K
                and bnd["cfg"]["env"]["F"] == b0.F
                and bnd["cfg"]["env"]["T"] == b0.T,
                "env shape",
            )
        cfg_bf, sha_bf = sess.bf_config(C)
        require(cfg_bf["env"]["C"] == C, "BF cfg binding")
    # archived Stage 3.5 H1 references must all be readable
    for C in b0.CAPACITIES:
        for seed in SEEDS:
            rows = h1m.load_h1_episodes(C, seed)
            require(len(rows) == 2400, f"archived H1 coverage C{C} S{seed}")
    print(f"VERIFY PASS in {time.time()-t0:.1f}s")


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------

def cmd_smoke(args: argparse.Namespace) -> None:
    t0 = time.time()
    n = args.episodes
    sess = D1Session()
    if SMOKE_DIR.exists():
        import shutil

        shutil.rmtree(SMOKE_DIR)
    checks: List[str] = []

    cfg = sess.bundle(800, 123)["cfg"]
    agent = sess.agent(800, 123)
    pool = sess.streams[0]["pool"]  # US

    # 1. observer on/off identity: plain B0 evaluator vs instrumented U
    plain = b0.evaluate_hybrid_pool(sess.sc, cfg, agent, pool[:n], "BF", "SC")
    tel_u = d1.CellTelemetry(b0.K, ["US"])
    inst = evaluate_d1_pool(
        sess.sc, cfg, agent, pool[:n], "U", tel_u, "US", n_episodes=n
    )
    for i in range(n):
        for fld in ("eval_money", "settled", "flushes", "drops", "accepted_count"):
            require(
                float(plain["raw_results"][i][fld]) == float(inst["raw_results"][i][fld]),
                f"observer identity failure ep{i} {fld}",
            )
    checks.append(f"instrumented U == plain B0 H1 evaluator on {n} episodes")

    # 2. masked arm: single-logit mask semantics
    tel_m = d1.CellTelemetry(b0.K, ["US"])
    evm = evaluate_d1_pool(
        sess.sc, cfg, agent, pool[:n], "M", tel_m, "US", n_episodes=n
    )
    cc = tel_m.counts["US"]
    require(
        cc["mask_changed_choice"] == cc["conflict_raw"],
        "mask changes choice iff raw argmax was s_BF",
    )
    # trace-level: chosen flush can never be s_BF (s_BF<k) in the M arm
    for ep in d1.TRACED_EPISODES:
        b = tel_m.traces["US"].get(ep)
        if b is None:
            continue
        I = b["I"]
        for row in I:
            s_bf, chosen = row[1], row[5]
            if s_bf < b0.K:
                require(chosen != s_bf, "masked arm flushed settle wallet")
    checks.append("mask: choice changes iff raw conflict; settle wallet never flushed")

    # 3. BF arm identical to frozen adapter rule + frozen frozen-output metrics
    cfg_bf, _sha = sess.bf_config(800)
    tel_b = d1.CellTelemetry(b0.K, ["US"])
    evb = evaluate_d1_pool(
        sess.ja, cfg_bf, None, pool[:n], "BF", tel_b, "US",
        n_episodes=n, adapter=sess.adapter,
    )
    frozen_bf = b0.load_frozen_episodes(b0.FROZEN_JOB_IDS[("BF", 800)])
    frows = sorted(
        (r for r in frozen_bf if r["regime"] == "US"),
        key=lambda r: int(r["episode_index"]),
    )
    for i in range(n):
        for fld in ("money", "settled", "flushes", "drops"):
            got = evb["raw_results"][i]["eval_money"] if fld == "money" else evb["raw_results"][i][fld]
            require(float(got) == float(frows[i][fld]), f"BF smoke mismatch ep{i} {fld}")
    checks.append(f"instrumented BF == frozen adapter + frozen outputs on {n} episodes")

    # 4. trace shapes
    for tel, name in ((tel_u, "U"), (tel_m, "M"), (tel_b, "BF")):
        for ep in d1.TRACED_EPISODES:
            b = tel.traces["US"].get(ep)
            if b is None or ep >= n:
                continue
            require(len(b["I"]) == b0.T and len(b["F"]) == b0.T, f"trace length {name} ep{ep}")
            require(
                len(b["I"][0]) == len(d1.TRACE_INT_COLUMNS),
                f"trace int schema {name}",
            )
            require(
                len(b["F"][0]) == len(d1.TRACE_FLOAT_COLUMNS),
                f"trace float schema {name}",
            )
    checks.append("trace shapes/schema correct for traced smoke episodes")

    doc = dict(
        status="PASS",
        checks=checks,
        n_episodes_per_check=n,
        elapsed_seconds=time.time() - t0,
        completed_utc=utc(),
        u_macro_smoke=statistics.mean(
            r["eval_money"] for r in inst["raw_results"]
        ),
        m_macro_smoke=statistics.mean(
            r["eval_money"] for r in evm["raw_results"]
        ),
        bf_macro_smoke=statistics.mean(
            r["eval_money"] for r in evb["raw_results"]
        ),
    )
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    (SMOKE_DIR / "SMOKE.json").write_text(json.dumps(doc, indent=2, sort_keys=True))
    print(f"SMOKE PASS in {time.time()-t0:.1f}s")
    for c in checks:
        print(f"  - {c}")


# ---------------------------------------------------------------------------
# U parity gate (mandatory): exact replay vs archived Stage 3.5 H1 CSVs
# ---------------------------------------------------------------------------

def archived_rows(C: int, seed: int) -> List[Dict[str, str]]:
    rows = h1m.load_h1_episodes(C, seed)
    idx = {(r["regime"], int(r["episode_index"])): r for r in rows}
    ordered: List[Dict[str, str]] = []
    for reg in b0.REGIMES:
        for ep in range(b0.EPISODES_PER_REGIME):
            ordered.append(idx[(reg, ep)])
    return ordered


def parity_cell(C: int, seed: int) -> Dict[str, Any]:
    dst = cell_path("U", C, seed)
    require((dst / "episodes.csv").is_file(), f"Missing U cell (run it first): {dst}")
    with (dst / "episodes.csv").open(newline="") as f:
        got_rows = list(csv.DictReader(f))
    require(len(got_rows) == 12 * b0.EPISODES_PER_REGIME, f"U row count C{C} S{seed}")
    got_idx = {(r["regime"], int(r["episode_index"])): r for r in got_rows}
    want = archived_rows(C, seed)
    errors: List[str] = []
    n_compared = 0
    for w in want:
        key = (w["regime"], int(w["episode_index"]))
        g = got_idx.get(key)
        if g is None:
            errors.append(f"missing {key}")
            continue
        for fld in COMPARE_FIELDS:
            if float(g[fld]) != float(w[fld]):
                errors.append(f"{key}.{fld}: {g[fld]!r} != {w[fld]!r}")
                break
        n_compared += 1
    return dict(
        C=C,
        seed=seed,
        exact=not errors,
        episodes_compared=n_compared,
        fields_compared=list(COMPARE_FIELDS),
        first_error=(errors[0] if errors else None),
        n_errors=len(errors),
        u_cell=str(dst),
        archived_source=str(h1m.cell_dir(C, seed) if seed != 123 else h1m.b0_h1_dir(C)),
    )


def cmd_parity(_args: argparse.Namespace, write_doc: bool = True) -> Dict[str, Any]:
    t0 = time.time()
    per_cell = []
    for C in b0.CAPACITIES:
        for seed in SEEDS:
            res = parity_cell(C, seed)
            per_cell.append(res)
            print(
                f"  parity U C{C} S{seed}: {'EXACT' if res['exact'] else 'FAIL'} "
                f"({res['episodes_compared']} eps x {len(COMPARE_FIELDS)} fields)"
            )
            if not res["exact"]:
                print(f"    FIRST ERROR: {res['first_error']}")
    status = "PASS" if all(c["exact"] for c in per_cell) else "FAIL"
    doc = dict(
        status=status,
        gate="U_PARITY",
        cells=per_cell,
        elapsed_seconds=time.time() - t0,
        completed_utc=utc(),
    )
    if write_doc:
        (D1_ROOT / "D1_U_PARITY.json").write_text(
            json.dumps(doc, indent=2, sort_keys=True)
        )
    print(f"U PARITY {status} in {time.time()-t0:.1f}s")
    return doc


# ---------------------------------------------------------------------------
# run (U -> mandatory parity gate -> M -> BF)
# ---------------------------------------------------------------------------

def write_config(sess: D1Session) -> None:
    cfg_doc = dict(
        schema_version=1,
        experiment="D1-H1-SAFE-v1",
        stage="stage4_d1_h1_safe",
        post_hoc_exploratory=True,
        confirmatory=False,
        created_utc=utc(),
        matrix=dict(
            arms=["U", "M"],
            seeds=SEEDS,
            capacities=b0.CAPACITIES,
            bf_telemetry_cells=2,
            cells_total=22,
            episodes_per_cell=2400,
            total_episodes=22 * 2400,
            total_steps=22 * 2400 * b0.T,
        ),
        env=dict(dataset="NEW12-v1", regimes=b0.REGIMES, k=b0.K, F=b0.F, T=b0.T,
                 episodes_per_regime=b0.EPISODES_PER_REGIME),
        arms=dict(
            U=intervention_spec("U"),
            M=intervention_spec("M"),
            BF=intervention_spec("BF"),
        ),
        trace_schema=dict(
            int_columns=d1.TRACE_INT_COLUMNS,
            float_columns=d1.TRACE_FLOAT_COLUMNS,
            fork_columns=d1.TRACE_FORK_COLUMNS,
            traced_episodes_per_regime=list(d1.TRACED_EPISODES),
            traced_episodes_total=22 * 12 * len(d1.TRACED_EPISODES),
            hist_edges=list(d1.HIST_EDGES),
        ),
        parity_gate=dict(
            name="U_PARITY",
            compares_against="archived Stage 3.5 H1 episodes.csv",
            fields=list(COMPARE_FIELDS),
            rule="STOP all D1 interpretation on any failure",
        ),
        frozen=dict(
            raw_tarball_sha256=b0.RAW_SHA,
            lineage=b0.LINEAGE,
            new12_manifest_sha256=sess.manifest_sha,
            frozen_repo=str(b0.FROZEN_REPO),
        ),
        runtime=sess.runtime_banner(),
        constraints=[
            "no training / no A+ / no retraining",
            "no edits to frozen Stage 2/3/3.5 artifacts",
            "additive under tools/aamas_stage4 and artifacts/aamas2027/stage4",
            "mask changes exactly one flush logit; environment unmodified",
            "local forks are read-only one-step snapshots (LOCAL EXPOSURE ONLY)",
        ],
    )
    (D1_ROOT / "D1_CONFIG.json").write_text(json.dumps(cfg_doc, indent=2, sort_keys=True))


def cmd_run(args: argparse.Namespace) -> None:
    sess = D1Session()
    write_config(sess)

    which = args.which  # 'all' | 'u' | 'm' | 'bf'

    # --- U cells ----------------------------------------------------------
    if which in ("all", "u"):
        for C in b0.CAPACITIES:
            for seed in SEEDS:
                dst = cell_path("U", C, seed)
                if dst.exists():
                    print(f"  skip existing U {dst}")
                    continue
                run_cell(sess, "U", C, seed)

    # --- mandatory U parity gate before any M interpretation --------------
    if which in ("all", "u", "m"):
        parity = cmd_parity(args, write_doc=True)
        if parity["status"] != "PASS":
            raise b0.B0Error(
                "U PARITY FAILED -- D1 halted; masked cells must not be interpreted"
            )

    # --- M cells ----------------------------------------------------------
    if which in ("all", "m"):
        for C in b0.CAPACITIES:
            for seed in SEEDS:
                dst = cell_path("M", C, seed)
                if dst.exists():
                    print(f"  skip existing M {dst}")
                    continue
                run_cell(sess, "M", C, seed)

    # --- BF telemetry cells ----------------------------------------------
    if which in ("all", "bf"):
        for C in b0.CAPACITIES:
            dst = cell_path("BF", C)
            if dst.exists():
                print(f"  skip existing BF {dst}")
                continue
            run_cell(sess, "BF", C)


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

def _load_cell_rows(arm: str, C: int, seed: Optional[int]) -> List[Dict[str, str]]:
    dst = cell_path(arm, C, seed)
    with (dst / "episodes.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def cmd_validate(_args: argparse.Namespace) -> Dict[str, Any]:
    t0 = time.time()
    checks: List[str] = []
    errors: List[str] = []

    cells = [
        (arm, C, seed)
        for arm in ("U", "M")
        for C in b0.CAPACITIES
        for seed in SEEDS
    ] + [("BF", C, None) for C in b0.CAPACITIES]

    # 1. presence / PASS / 2400 rows
    for arm, C, seed in cells:
        dst = cell_path(arm, C, seed)
        try:
            require(dst.is_dir(), f"missing cell {dst}")
            require((dst / "episodes.csv").is_file(), f"missing episodes {dst}")
            require((dst / "result.json").is_file(), f"missing result {dst}")
            require((dst / "telemetry.json").is_file(), f"missing telemetry {dst}")
            result = json.loads((dst / "result.json").read_text())
            require(result["validation_status"] == "PASS", f"result not PASS {dst}")
            require(result["row_count"] == 2400, f"row_count {dst}")
            rows = _load_cell_rows(arm, C, seed)
            require(len(rows) == 2400, f"episodes rows {dst}")
            require(trace_path(arm, C, seed).is_file(), f"missing trace {dst}")
        except b0.B0Error as e:
            errors.append(str(e))
    if not errors:
        checks.append("all 22 cells present: episodes.csv/result.json/telemetry.json/traces, 2400 rows, PASS")

    # 2. per-row identities + coverage
    if not errors:
        for arm, C, seed in cells:
            rows = _load_cell_rows(arm, C, seed)
            seen = set()
            for r in rows:
                key = (r["regime"], int(r["episode_index"]))
                if key in seen:
                    errors.append(f"duplicate pair {key} {arm} C{C} {seed}")
                    break
                seen.add(key)
                try:
                    require(
                        abs(float(r["money"]) - (float(r["settled"]) - 10.0 * float(r["flushes"])))
                        <= 1e-9,
                        f"money identity {key}",
                    )
                    require(
                        int(r["accepted_count"]) + int(r["drops"]) == b0.T,
                        f"count identity {key}",
                    )
                    for f in ("settled", "flushes", "drops", "accepted_count",
                              "insufficient_drops", "oversize_drops", "money"):
                        require(math.isfinite(float(r[f])), f"non-finite {f} {key}")
                except b0.B0Error as e:
                    errors.append(f"{arm} C{C} {seed}: {e}")
                    break
            if len(seen) != 2400:
                errors.append(f"coverage {arm} C{C} {seed}: {len(seen)}")
    if not errors:
        checks.append("no duplicate/missing regime-episode pairs; Money + count identities; finite")

    # 3. trace files: exactly 24 traced episodes/cell, 1000 rows, schema
    if not errors:
        import numpy as np

        for arm, C, seed in cells:
            tp = trace_path(arm, C, seed)
            try:
                z = np.load(tp)
                keys = set(z.files)
                expect = set()
                for reg in b0.REGIMES:
                    for ep in d1.TRACED_EPISODES:
                        expect |= {f"{reg}_{ep}_i", f"{reg}_{ep}_f", f"{reg}_{ep}_fork"}
                require(keys == expect, f"trace keys {tp.name}")
                for reg in b0.REGIMES:
                    for ep in d1.TRACED_EPISODES:
                        I = z[f"{reg}_{ep}_i"]
                        F = z[f"{reg}_{ep}_f"]
                        fk = z[f"{reg}_{ep}_fork"]
                        require(I.shape == (b0.T, len(d1.TRACE_INT_COLUMNS)), f"trace I shape {tp.name}")
                        require(F.shape == (b0.T, len(d1.TRACE_FLOAT_COLUMNS)), f"trace F shape {tp.name}")
                        require(fk.ndim == 2 and fk.shape[1] == len(d1.TRACE_FORK_COLUMNS),
                                f"trace fork shape {tp.name}")
            except (b0.B0Error, OSError) as e:
                errors.append(f"trace {arm} C{C} {seed}: {e}")
                break
    if not errors:
        checks.append("traces: 528 traced episodes (22 cells x 24), T=1000, fixed schema, forks 7 cols")

    # 4. U parity document
    parity_doc_path = D1_ROOT / "D1_U_PARITY.json"
    if parity_doc_path.is_file():
        pdoc = json.loads(parity_doc_path.read_text())
        if pdoc.get("status") != "PASS":
            errors.append("D1_U_PARITY.json status != PASS")
    else:
        errors.append("missing D1_U_PARITY.json")
    if not errors:
        checks.append("U parity gate: 10 cells EXACT vs archived Stage 3.5 H1")

    # 5. M-specific mask invariants
    if not errors:
        for C in b0.CAPACITIES:
            for seed in SEEDS:
                tel = json.loads((cell_path("M", C, seed) / "telemetry.json").read_text())
                for reg in b0.REGIMES:
                    c = tel["counts"][reg]
                    if c["mask_changed_choice"] != c["conflict_raw"]:
                        errors.append(
                            f"M mask invariant C{C} S{seed} {reg}: "
                            f"changed {c['mask_changed_choice']} != conflicts {c['conflict_raw']}"
                        )
                        break
                if errors:
                    break
    if not errors:
        checks.append("M invariant: mask changes argmax iff raw flush argmax was s_BF")

    # 6. BF exact vs frozen Stage 2 BF episodes
    if not errors:
        for C in b0.CAPACITIES:
            rows = _load_cell_rows("BF", C, None)
            frozen = b0.load_frozen_episodes(b0.FROZEN_JOB_IDS[("BF", C)])
            fidx = {(r["regime"], int(r["episode_index"])): r for r in frozen}
            for r in rows:
                f = fidx[(r["regime"], int(r["episode_index"]))]
                for fld in COMPARE_FIELDS:
                    if float(r[fld]) != float(f[fld]):
                        errors.append(f"BF mismatch C{C} {r['regime']} ep{r['episode_index']} {fld}")
                        break
                if errors:
                    break
    if not errors:
        checks.append("BF telemetry cells EXACT vs frozen Stage 2 BF-T0.5 episodes")

    # 7. checkpoint/stream/raw hashes
    if not errors:
        try:
            adapter = b0.import_frozen_adapter()
            adapter.verify_sources(b0.FROZEN_REPO)
            _streams, manifest_sha = b0.load_new12_streams(adapter)
            approval = b0.raw_approval()
            require(b0.sha256(b0.RAW_TARBALL) == b0.RAW_SHA, "raw tarball")
            require(manifest_sha == approval["new12_manifest_sha256"], "new12 manifest")
            for arm, C, seed in [c for c in cells if c[0] in ("U", "M")]:
                result = json.loads((cell_path(arm, C, seed) / "result.json").read_text())
                bnd = h1m.load_sc_bundle_seed(C, seed, adapter)
                require(result["checkpoint_sha256"] == bnd["cp_sha"], f"ckpt sha {arm} C{C} S{seed}")
                rows = _load_cell_rows(arm, C, seed)
                require({r["checkpoint_sha256"] for r in rows} == {bnd["cp_sha"]},
                        f"episodes ckpt {arm} C{C} S{seed}")
                require({r["pool_sha256"] for r in rows} ==
                        {s["sha256"] for s in _streams},
                        f"pool shas {arm} C{C} S{seed}")
        except b0.B0Error as e:
            errors.append(str(e))
    if not errors:
        checks.append("frozen raw/adapter/sources/NEW12/checkpoint/stream hashes verified")

    # 8. no training artifacts / no unexpected files
    if not errors:
        allowed_suffix = {".csv", ".json", ".npz", ".py", ".md", ".log"}
        skip_dirs = {"__pycache__", "_cache"}
        for p in D1_ROOT.rglob("*"):
            rel = p.relative_to(D1_ROOT)
            if any(part in skip_dirs for part in rel.parts):
                continue
            if p.is_dir():
                continue
            if p.suffix in (".pt", ".pth", ".ckpt", ".bin"):
                errors.append(f"training-like artifact present: {rel}")
            elif p.suffix not in allowed_suffix and "smoke" not in rel.parts:
                errors.append(f"unexpected file type: {rel}")
    if not errors:
        checks.append("no training/checkpoint artifacts; only csv/json/npz/py/md outputs")

    status = "PASS" if not errors else "FAIL"
    doc = dict(
        status=status,
        checks=checks,
        errors=errors,
        cells=[f"{arm}_C{C}" + (f"_S{seed}" if seed is not None else "")
               for arm, C, seed in cells],
        elapsed_seconds=time.time() - t0,
        completed_utc=utc(),
    )
    (D1_ROOT / "D1_VALIDATION.json").write_text(json.dumps(doc, indent=2, sort_keys=True))
    print(f"VALIDATE {status} in {time.time()-t0:.1f}s")
    for c in checks:
        print(f"  - {c}")
    for e in errors[:10]:
        print(f"  ERROR: {e}")
    return doc


# ---------------------------------------------------------------------------
# analyze (delegated to the sibling module; written after runs are launched)
# ---------------------------------------------------------------------------

def cmd_analyze(args: argparse.Namespace) -> None:
    import analyze_d1

    analyze_d1.cmd_analyze(args)


def main() -> None:
    p = argparse.ArgumentParser(description="D1-H1-SAFE-v1 driver")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify")
    sp = sub.add_parser("smoke")
    sp.add_argument("--episodes", type=int, default=3)
    sub.add_parser("parity")
    rp = sub.add_parser("run")
    rp.add_argument("--which", choices=["all", "u", "m", "bf"], default="all")
    sub.add_parser("validate")
    sub.add_parser("analyze")
    args = p.parse_args()
    {
        "verify": cmd_verify,
        "smoke": cmd_smoke,
        "parity": cmd_parity,
        "run": cmd_run,
        "validate": cmd_validate,
        "analyze": cmd_analyze,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
