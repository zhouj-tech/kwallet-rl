#!/usr/bin/env python3
"""H1 multi-seed exploratory expansion (K-Wallet AAMAS 2027, Stage 3.5).

ADDITIVE ONLY. Reuses the validated B0 machinery (b0_lib) unchanged:

* H1 = frozen BF-T0.5 settlement + frozen SC-FAC conditional flush head
  conditioned on the BF settlement index, executed by the unmodified E0.
* Only the SC-FAC checkpoint changes across cells: seeds 323/532/777/999 are
  evaluated NEW; seed 123 is REUSED from the validated B0 pilot outputs
  (never recomputed, never overwritten).

Post-hoc exploratory diagnostic -- NOT confirmatory, NOT a new method claim.
No training, no tuning, no frozen artifact is modified.

Commands: verify | run | validate | analyze | all
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# --- import the frozen B0 library unchanged -------------------------------
B0_CODE = Path(__file__).resolve().parents[2] / "b0_diagnostic" / "code"
if str(B0_CODE) not in sys.path:
    sys.path.insert(0, str(B0_CODE))
import b0_lib as b0  # noqa: E402

MS_ROOT = Path(__file__).resolve().parents[1]
OUT = MS_ROOT / "outputs"

SEEDS = [123, 323, 532, 777, 999]
NEW_SEEDS = [323, 532, 777, 999]
N_SEEDS = len(SEEDS)
# Reduced per-seed provenance replay (SC/SC exact vs frozen) before each new
# H1 cell; full per-pool stream hash verification plus checkpoint SHA pinning
# remain the primary bindings (see validate).
PARITY_REGIMES = ["US", "TPLS", "PLB"]

COMPARE_FIELDS = [
    "money",
    "settled",
    "flushes",
    "drops",
    "accepted_count",
    "insufficient_drops",
    "oversize_drops",
]

METHOD_LABEL = "H1-BFsettle-SCflush"
COHORT = "stage3_5_h1_multiseed_posthoc"


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def require(ok: bool, msg: str) -> None:
    b0.require(ok, msg)


# ---------------------------------------------------------------------------
# Small numerical statistics (no scipy in the frozen runtime)
# ---------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Numerical Recipes)."""
    maxit, eps, fpmin = 500, 3e-15, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: int) -> float:
    """CDF of Student's t distribution via the incomplete beta function."""
    x = df / (df + t * t)
    ib = _betai(df / 2.0, 0.5, x)
    return 1.0 - 0.5 * ib if t >= 0.0 else 0.5 * ib


def t_ppf(p: float, df: int) -> float:
    """Inverse CDF by bisection (monotone CDF); p in (0, 1)."""
    require(0.0 < p < 1.0, "t_ppf domain")
    lo, hi = -1000.0, 1000.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def t_two_sided_p(t: float, df: int) -> float:
    if math.isinf(t):
        return 0.0
    return 2.0 * (1.0 - t_cdf(abs(t), df))


def sample_stats(values: List[float]) -> Dict[str, Any]:
    """mean / sample SD / SE / 95% t CI / min / max for a small sample."""
    n = len(values)
    require(n >= 2, "need at least two values")
    mean = statistics.mean(values)
    sd = statistics.stdev(values)  # sample SD, ddof=1
    se = sd / math.sqrt(n)
    tcrit = t_ppf(0.975, n - 1)
    return dict(
        n=n,
        mean=mean,
        sd=sd,
        se=se,
        tcrit=tcrit,
        ci95_lo=mean - tcrit * se,
        ci95_hi=mean + tcrit * se,
        min=min(values),
        max=max(values),
    )


def one_sample_t(deltas: List[float]) -> Dict[str, Any]:
    """One-sample / paired t-test of mean(delta) against 0 (raw two-sided p)."""
    s = sample_stats(deltas)
    if s["se"] > 0:
        t = s["mean"] / s["se"]
    else:
        t = math.inf if s["mean"] > 0 else (-math.inf if s["mean"] < 0 else 0.0)
    out = dict(s)
    out["t"] = t
    out["df"] = s["n"] - 1
    out["p_two_sided_raw"] = t_two_sided_p(t, s["n"] - 1)
    return out


# ---------------------------------------------------------------------------
# Frozen per-seed SC-FAC bundle resolution (mirrors b0.load_sc_bundle)
# ---------------------------------------------------------------------------

def _remap_artifacts(server_path: str) -> Path:
    prefix = "/data/sijia/aamas2027_artifacts/"
    require(server_path.startswith(prefix), f"Unexpected artifact path: {server_path}")
    return b0.FROZEN_ARTIFACTS / server_path[len(prefix):]


def frozen_sc_job_id(C: int, seed: int) -> str:
    return f"A-EVAL-SC-C{C}-S{seed}"


def h1_job_id(C: int, seed: int) -> str:
    return f"H1MS-H1-C{C}-S{seed}"


def stage3_seed_row(C: int, seed: int) -> Dict[str, str]:
    rows = [
        r
        for r in b0.csv_rows(b0.STAGE3_DIR / "seed_level_scores.csv")
        if r["method"] == "SC-FAC" and int(r["C"]) == C and r["training_seed"] == str(seed)
    ]
    require(len(rows) == 1, f"Stage3 seed row not unique: SC-FAC C{C} S{seed}")
    return rows[0]


def tar_member_bytes(name: str) -> bytes:
    with tarfile.open(b0.RAW_TARBALL, "r:gz") as tf:
        f = tf.extractfile(f"{b0.BUNDLE}/{name}")
        require(f is not None, f"Missing tarball member {name}")
        return f.read()


def load_sc_bundle_seed(C: int, seed: int, adapter: Any) -> Dict[str, Any]:
    """Frozen SC-FAC checkpoint + config for (C, seed), fully hash/field bound."""
    job = b0.frozen_job_rows()[frozen_sc_job_id(C, seed)]
    require(
        job["method"] == "SC-FAC" and int(job["C"]) == C and int(job["seed"]) == seed,
        f"Job matrix binding wrong for C{C} S{seed}",
    )
    require(job["condition_mode"] == "full", "SC-FAC condition_mode must be full")
    cp = _remap_artifacts(job["checkpoint_source"])
    ri = _remap_artifacts(job["run_info_source"])
    require(cp.is_file() and ri.is_file(), f"Missing checkpoint/run_info C{C} S{seed}")
    cp_sha = b0.sha256(cp)
    require(cp_sha == job["checkpoint_sha256"], f"Checkpoint SHA mismatch C{C} S{seed}")
    srow = stage3_seed_row(C, seed)
    require(
        srow["checkpoint_sha256"] == cp_sha,
        f"Stage3/job-matrix checkpoint SHA mismatch C{C} S{seed}",
    )
    ri_doc = b0.read_json(ri)
    cfg = ri_doc["config"]
    adapter.check_config(cfg, job)  # C, k, F, T, seed, device, reward, arch, cond
    return dict(
        cfg=cfg,
        cp=cp,
        cp_sha=cp_sha,
        ri_sha=b0.sha256(ri),
        job=job,
        stage3_row=srow,
    )


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class MSSession:
    def __init__(self) -> None:
        self.adapter = b0.import_frozen_adapter()
        b0.register_adapter(self.adapter)
        b0.configure_caches()
        self.adapter.verify_sources(b0.FROZEN_REPO)
        self.sc, _ja = b0.import_frozen_modules(self.adapter)
        self.streams, self.manifest_sha = b0.load_new12_streams(self.adapter)
        self._bundles: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self._agents: Dict[Tuple[int, int], Any] = {}

    def bundle(self, C: int, seed: int) -> Dict[str, Any]:
        key = (C, seed)
        if key not in self._bundles:
            self._bundles[key] = load_sc_bundle_seed(C, seed, self.adapter)
        return self._bundles[key]

    def agent(self, C: int, seed: int) -> Any:
        key = (C, seed)
        if key not in self._agents:
            bnd = self.bundle(C, seed)
            _env, agent = b0.build_sc_agent(self.sc, bnd["cfg"], bnd["cp"])
            self._agents[key] = agent
        return self._agents[key]

    def runtime_banner(self) -> Dict[str, Any]:
        import numpy as np
        import torch

        return dict(
            python_executable=sys.executable,
            python_version=platform_python(),
            numpy_version=np.__version__,
            torch_version=torch.__version__,
            torch_threads=torch.get_num_threads(),
            device="cpu",
        )


def platform_python() -> str:
    import platform

    return platform.python_version()


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def cmd_verify(_args: argparse.Namespace) -> None:
    t0 = time.time()
    approval = b0.raw_approval()
    require(b0.sha256(b0.RAW_TARBALL) == b0.RAW_SHA, "Raw tarball SHA mismatch")
    require(approval["git_head"] == b0.LINEAGE, "Raw approval lineage mismatch")
    print(f"raw tarball SHA OK {b0.RAW_SHA[:16]}... lineage {b0.LINEAGE[:12]}")

    sess = MSSession()
    for C in b0.CAPACITIES:
        for seed in SEEDS:
            bnd = sess.bundle(C, seed)
            cfg = bnd["cfg"]
            require(cfg["env"]["C"] == C and cfg["seed"] == seed, "config binding")
            require(
                cfg["env"]["k"] == b0.K and cfg["env"]["F"] == b0.F and cfg["env"]["T"] == b0.T,
                "env shape",
            )
            require(
                cfg["conditional"]["settle_embed_dim"] == 32
                and cfg["conditional"]["conditional_hidden_size"] == 256,
                "conditional head shape",
            )
            print(
                f"  SC-FAC C{C} S{seed}: ckpt {bnd['cp_sha'][:12]}... "
                f"run_info {bnd['ri_sha'][:12]}... config OK"
            )
    # frozen episodes_sha256 cross-check for every frozen SC reference
    for C in b0.CAPACITIES:
        for seed in SEEDS:
            srow = stage3_seed_row(C, seed)
            blob = tar_member_bytes(f"output/{frozen_sc_job_id(C, seed)}/episodes.csv")
            require(
                hashlib.sha256(blob).hexdigest() == srow["episodes_sha256"],
                f"Frozen episodes SHA mismatch C{C} S{seed}",
            )
    print(f"frozen SC episodes_sha256 OK for all {N_SEEDS} seeds x 2 capacities")
    print(f"NEW12 manifest SHA OK {sess.manifest_sha[:16]}...")
    print(f"VERIFY PASS in {time.time()-t0:.1f}s")


# ---------------------------------------------------------------------------
# Per-seed reduced SC/SC provenance parity (exact replay vs frozen)
# ---------------------------------------------------------------------------

def sc_parity_for_seed(sess: MSSession, C: int, seed: int) -> Dict[str, Any]:
    """Replay frozen SC-FAC (SC settle + SC flush) on PARITY_REGIMES and require
    EXACT per-episode equality with the frozen Stage 2 episodes. Proves the
    loaded checkpoint is exactly the one that produced the frozen reference."""
    frozen = b0.load_frozen_episodes(frozen_sc_job_id(C, seed))
    cfg = sess.bundle(C, seed)["cfg"]
    agent = sess.agent(C, seed)
    per_regime: Dict[str, Any] = {}
    for stream in sess.streams:
        reg = stream["regime"]
        if reg not in PARITY_REGIMES:
            continue
        res = b0.evaluate_hybrid_pool(sess.sc, cfg, agent, stream["pool"], "SC", "SC")
        frows = sorted(
            (r for r in frozen if r["regime"] == reg), key=lambda r: int(r["episode_index"])
        )
        require(len(frows) == b0.EPISODES_PER_REGIME, f"frozen coverage {reg}")
        errs: List[str] = []
        for i, (metric, frow) in enumerate(zip(res["raw_results"], frows)):
            for field in COMPARE_FIELDS:
                got = metric["eval_money"] if field == "money" else metric[field]
                if float(got) != float(frow[field]):
                    errs.append(f"ep{i}.{field}: {got!r} != {frow[field]!r}")
                    break
        per_regime[reg] = dict(exact=not errs, first_error=errs[0] if errs else None)
        if errs:
            raise b0.B0Error(f"SC/SC parity failed C{C} S{seed} {reg}: {errs[0]}")
    return dict(regimes=per_regime, parity_regimes=list(PARITY_REGIMES), exact=True)


# ---------------------------------------------------------------------------
# run: 8 new H1 cells (seed123 is reused from B0, never recomputed)
# ---------------------------------------------------------------------------

def cell_dir(C: int, seed: int) -> Path:
    return OUT / f"C{C}_S{seed}"


def b0_h1_dir(C: int) -> Path:
    return b0.B0_ROOT / "outputs" / "pilots" / f"H1_C{C}_S{b0.SC_SEED}"


def run_cell(sess: MSSession, C: int, seed: int) -> Path:
    require(seed in NEW_SEEDS, "run_cell is only for the 8 NEW seeds")
    dst = cell_dir(C, seed)
    require(not dst.exists(), f"Refusing to overwrite existing cell output: {dst}")

    bnd = sess.bundle(C, seed)
    cfg, agent = bnd["cfg"], sess.agent(C, seed)

    parity = sc_parity_for_seed(sess, C, seed)
    print(f"  parity C{C} S{seed}: SC/SC exact on {PARITY_REGIMES}")

    started = time.time()
    per_regime: Dict[str, Dict[str, Any]] = {}
    for stream in sess.streams:
        t0 = time.time()
        per_regime[stream["regime"]] = b0.evaluate_hybrid_pool(
            sess.sc, cfg, agent, stream["pool"], "BF", "SC", rule_fidelity_check=True
        )
        m = per_regime[stream["regime"]]["summary"]["eval_money"]["mean"]
        print(
            f"  H1 C{C} S{seed} {stream['regime']:5s} money={m:11.3f} ({time.time()-t0:.0f}s)"
        )
    elapsed = time.time() - started

    fields = b0.frozen_episode_fields()
    job_id = h1_job_id(C, seed)
    episode_rows: List[Dict[str, Any]] = []
    means: List[float] = []
    summaries = {}
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
                family="STAGE3_5_H1MS",
                method=METHOD_LABEL,
                C=C,
                k=b0.K,
                F=b0.F,
                T=b0.T,
                training_seed=seed,
                condition_mode="full",
                cohort=COHORT,
                regime=reg,
                episode_index=i,
                episode_seed=stream["episode_seeds"][i],
                episode_sha256=stream["episode_hashes"][i],
                pool_sha256=stream["sha256"],
                checkpoint_sha256=bnd["cp_sha"],
            )
            metric_alias = {"money": "eval_money"}
            for key in fields:
                if key in row or key.startswith("gate_"):
                    continue
                src = metric_alias.get(key, key)
                require(src in metric, f"Frozen episode field {key!r} missing from metrics")
                row[key] = metric[src]
            require(set(row.keys()) == set(fields), "Episode row schema mismatch")
            episode_rows.append(row)

    macro = statistics.mean(means)
    result = dict(
        schema_version=1,
        stage="stage3_5_h1_multiseed",
        post_hoc_exploratory=True,
        confirmatory=False,
        dataset_id="NEW12-v1",
        job_id=job_id,
        hybrid="H1",
        hybrid_description=b0.HYBRID_DESC[b0.H1],
        settle_policy="BF-T0.5",
        flush_policy="SC-FAC-conditioned-on-BF-settle",
        C=C,
        training_seed=seed,
        config=cfg,
        checkpoint_sha256=bnd["cp_sha"],
        run_info_sha256=bnd["ri_sha"],
        frozen_sc_job_id=frozen_sc_job_id(C, seed),
        frozen_source_hashes=sess.adapter.SOURCE_HASHES,
        adapter_sha256=b0.sha256(b0.FROZEN_REPO / "tools/aamas_stage2/adapter.py"),
        stream_manifest_sha256=sess.manifest_sha,
        frozen_raw_tarball_sha256=b0.sha256(b0.RAW_TARBALL),
        frozen_lineage=b0.LINEAGE,
        provenance_parity=parity,
        regime_summaries=summaries,
        row_count=len(episode_rows),
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
    print(f"CELL H1 C{C} S{seed} macro12_money={macro:.4f} -> {dst} ({elapsed:.0f}s)")
    return dst


def snapshot_seed123_reference() -> Path:
    """Hash the reused B0 seed123 H1 outputs so validate can prove they were
    not modified by the multi-seed expansion."""
    snap_path = OUT / "seed123_reference_sha.json"
    if snap_path.exists():
        return snap_path
    snap = {}
    for C in b0.CAPACITIES:
        for name in ("episodes.csv", "result.json"):
            p = b0_h1_dir(C) / name
            require(p.is_file(), f"Missing B0 seed123 reference {p}")
            snap[str(p)] = b0.sha256(p)
    OUT.mkdir(parents=True, exist_ok=True)
    snap_path.write_text(json.dumps(snap, indent=2, sort_keys=True))
    return snap_path


def cmd_run(args: argparse.Namespace) -> None:
    snapshot_seed123_reference()
    sess = MSSession()
    cells = (
        [(C, s) for C in b0.CAPACITIES for s in NEW_SEEDS]
        if args.all
        else [(args.C, args.seed)]
    )
    for C, seed in cells:
        run_cell(sess, C, seed)


# ---------------------------------------------------------------------------
# Episode loading shared by validate/analyze
# ---------------------------------------------------------------------------

def load_h1_episodes(C: int, seed: int) -> List[Dict[str, str]]:
    if seed == b0.SC_SEED:
        path = b0_h1_dir(C) / "episodes.csv"
    else:
        path = cell_dir(C, seed) / "episodes.csv"
    require(path.is_file(), f"Missing H1 episodes for C{C} S{seed}: {path}")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


REGIME_METRICS = [
    "money",
    "settled",
    "flushes",
    "accepted_count",
    "insufficient_drops",
    "oversize_drops",
]


def regime_means_from_rows(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for reg in b0.REGIMES:
        rr = [r for r in rows if r["regime"] == reg]
        require(len(rr) == b0.EPISODES_PER_REGIME, f"coverage {reg}")
        out[reg] = {m: statistics.mean(float(r[m]) for r in rr) for m in REGIME_METRICS}
    return out


def macro_of(means: Dict[str, Dict[str, float]]) -> float:
    return statistics.mean(means[r]["money"] for r in b0.REGIMES)


# ---------------------------------------------------------------------------
# validate (section 11)
# ---------------------------------------------------------------------------

def cmd_validate(_args: argparse.Namespace) -> Dict[str, Any]:
    t0 = time.time()
    checks: List[str] = []

    # 1. all 8 new cells exist with 2400 rows + PASS result
    for C in b0.CAPACITIES:
        for seed in NEW_SEEDS:
            d = cell_dir(C, seed)
            ep, rj = d / "episodes.csv", d / "result.json"
            require(ep.is_file() and rj.is_file(), f"missing cell C{C} S{seed}")
            result = json.loads(rj.read_text())
            require(result["validation_status"] == "PASS", f"cell not PASS C{C} S{seed}")
            require(result["row_count"] == 2400, f"row count C{C} S{seed}")
            rows = load_h1_episodes(C, seed)
            require(len(rows) == 2400, f"episodes rows C{C} S{seed}")
    checks.append("8 new cells present, 2400 rows each, validation PASS")

    # 2. seed123 reused output unchanged
    snap = json.loads((OUT / "seed123_reference_sha.json").read_text())
    for path, sha in snap.items():
        require(b0.sha256(Path(path)) == sha, f"seed123 reference modified: {path}")
    checks.append("seed123 B0 H1 outputs unchanged (sha256 match)")

    # 3. final matrix 5 seeds x 2 capacities loadable
    macros = {}
    for C in b0.CAPACITIES:
        for seed in SEEDS:
            means = regime_means_from_rows(load_h1_episodes(C, seed))
            macros[(C, seed)] = macro_of(means)
    checks.append("final H1 matrix 5 seeds x 2 capacities loadable")

    # 4. checkpoint seed/C bindings (new cells: full chain job->sha->stage3)
    for C in b0.CAPACITIES:
        for seed in NEW_SEEDS:
            result = json.loads((cell_dir(C, seed) / "result.json").read_text())
            bnd = load_sc_bundle_seed(C, seed, b0.import_frozen_adapter())
            require(result["checkpoint_sha256"] == bnd["cp_sha"], f"ckpt binding C{C} S{seed}")
            require(result["config"]["seed"] == seed, f"config seed C{C} S{seed}")
            require(result["config"]["env"]["C"] == C, f"config C C{C} S{seed}")
            require(int(result["C"]) == C and int(result["training_seed"]) == seed, "result binding")
            require(result["provenance_parity"]["exact"], f"parity C{C} S{seed}")
            ep_rows = load_h1_episodes(C, seed)
            require(
                {r["checkpoint_sha256"] for r in ep_rows} == {bnd["cp_sha"]},
                f"episodes ckpt sha C{C} S{seed}",
            )
    checks.append("checkpoint seed/C bindings verified (job matrix + stage3 + result + episodes)")

    # 5. no duplicate/missing regime-episode pairs; 6. Money identity
    for C in b0.CAPACITIES:
        for seed in SEEDS:
            rows = load_h1_episodes(C, seed)
            seen = set()
            for r in rows:
                key = (r["regime"], int(r["episode_index"]))
                require(key not in seen, f"duplicate {key} C{C} S{seed}")
                seen.add(key)
                require(
                    abs(float(r["money"]) - (float(r["settled"]) - 10.0 * float(r["flushes"])))
                    <= 1e-9,
                    f"Money identity {key} C{C} S{seed}",
                )
                require(
                    int(r["accepted_count"]) + int(r["drops"]) == b0.T,
                    f"count identity {key} C{C} S{seed}",
                )
            require(len(seen) == 12 * b0.EPISODES_PER_REGIME, f"coverage C{C} S{seed}")
    checks.append("no duplicate/missing regime-episode pairs; Money identity holds everywhere")

    # 7. frozen source hashes; 8. NEW12 hashes unchanged
    approval = b0.raw_approval()
    require(b0.sha256(b0.RAW_TARBALL) == b0.RAW_SHA, "raw tarball changed")
    adapter = b0.import_frozen_adapter()  # itself pins adapter SHA vs approval
    adapter.verify_sources(b0.FROZEN_REPO)
    _streams, manifest_sha = b0.load_new12_streams(adapter)  # per-pool sha enforced
    require(manifest_sha == approval["new12_manifest_sha256"], "NEW12 manifest changed")
    for C in b0.CAPACITIES:
        for seed in SEEDS:
            srow = stage3_seed_row(C, seed)
            blob = tar_member_bytes(f"output/{frozen_sc_job_id(C, seed)}/episodes.csv")
            require(
                hashlib.sha256(blob).hexdigest() == srow["episodes_sha256"],
                f"frozen SC episodes changed C{C} S{seed}",
            )
    checks.append("frozen adapter/source/raw/NEW12/SC-episode hashes all verified")

    doc = dict(
        status="PASS",
        checks=checks,
        macros={f"C{C}_S{s}": macros[(C, s)] for C in b0.CAPACITIES for s in SEEDS},
        elapsed_seconds=time.time() - t0,
        completed_utc=utc(),
    )
    (OUT / "H1_VALIDATION.json").write_text(json.dumps(doc, indent=2, sort_keys=True))
    print(f"VALIDATE PASS ({time.time()-t0:.1f}s)")
    for c in checks:
        print(f"  - {c}")
    return doc


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

def load_frozen_sc_reference(C: int, seed: int) -> Dict[str, Any]:
    """Frozen SC-FAC reference for (C, seed): Stage 3 table cross-checked
    against the raw tarball episodes at tol 1e-9 (never hardcoded)."""
    srow = stage3_seed_row(C, seed)
    raw = b0.load_frozen_episodes(frozen_sc_job_id(C, seed))
    means = regime_means_from_rows(raw)
    require(
        math.isclose(macro_of(means), float(srow["macro12_money"]), rel_tol=0, abs_tol=1e-9),
        f"Stage3/raw macro mismatch SC C{C} S{seed}",
    )
    for reg in b0.REGIMES:
        require(
            math.isclose(
                means[reg]["money"], float(srow[f"money_{reg}"]), rel_tol=0, abs_tol=1e-9
            ),
            f"Stage3/raw regime mismatch SC C{C} S{seed} {reg}",
        )
    return dict(macro=float(srow["macro12_money"]), regime_means=means, stage3_row=srow)


def load_bf_reference(C: int) -> Dict[str, Any]:
    rows = [
        r
        for r in b0.csv_rows(b0.STAGE3_DIR / "seed_level_scores.csv")
        if r["method"] == "BF-T0.5" and int(r["C"]) == C
    ]
    require(len(rows) == 1 and rows[0]["training_seed"] == "", "BF row not unique")
    raw = b0.load_frozen_episodes(b0.FROZEN_JOB_IDS[("BF", C)])
    means = regime_means_from_rows(raw)
    require(
        math.isclose(macro_of(means), float(rows[0]["macro12_money"]), rel_tol=0, abs_tol=1e-9),
        f"Stage3/raw macro mismatch BF C{C}",
    )
    return dict(macro=float(rows[0]["macro12_money"]), regime_means=means)


def fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def fmt_signed(x: float, nd: int = 3) -> str:
    return f"{x:+.{nd}f}"


def cmd_analyze(_args: argparse.Namespace) -> None:
    t0 = time.time()
    require((OUT / "H1_VALIDATION.json").is_file(), "Run validate before analyze")
    val = json.loads((OUT / "H1_VALIDATION.json").read_text())
    require(val["status"] == "PASS", "validation not PASS")

    # ---- load everything -------------------------------------------------
    h1: Dict[Tuple[int, int], Dict[str, Any]] = {}
    sc: Dict[Tuple[int, int], Dict[str, Any]] = {}
    bf: Dict[int, Dict[str, Any]] = {}
    for C in b0.CAPACITIES:
        bf[C] = load_bf_reference(C)
        for seed in SEEDS:
            means = regime_means_from_rows(load_h1_episodes(C, seed))
            h1[(C, seed)] = dict(macro=macro_of(means), regime_means=means)
            sc[(C, seed)] = load_frozen_sc_reference(C, seed)

    stats: Dict[str, Any] = dict(
        generated_utc=utc(),
        post_hoc_exploratory=True,
        confirmatory=False,
        seeds=SEEDS,
        capacities=b0.CAPACITIES,
        episodes_per_regime=b0.EPISODES_PER_REGIME,
        regimes=b0.REGIMES,
        note=(
            "Exploratory screening statistics. NEW12-v1 was already used for "
            "method selection; nothing here is a fresh confirmatory test. "
            "p-values are raw, unadjusted, n=5, df=4."
        ),
    )

    # ---- H1 seed-level ---------------------------------------------------
    seed_level_rows: List[Dict[str, Any]] = []
    method_summary_rows: List[Dict[str, Any]] = []
    stats["h1_seed_level"] = {}
    for C in b0.CAPACITIES:
        vals = [h1[(C, s)]["macro"] for s in SEEDS]
        st = sample_stats(vals)
        stats["h1_seed_level"][str(C)] = dict(
            values={str(s): h1[(C, s)]["macro"] for s in SEEDS}, **st
        )
        for s in SEEDS:
            row: Dict[str, Any] = dict(
                C=C,
                seed=s,
                macro12_money=h1[(C, s)]["macro"],
                source="b0_reuse" if s == b0.SC_SEED else "h1ms_new",
                checkpoint_sha256=json.loads(
                    (
                        b0_h1_dir(C) / "result.json"
                        if s == b0.SC_SEED
                        else cell_dir(C, s) / "result.json"
                    ).read_text()
                )["checkpoint_sha256"],
            )
            for reg in b0.REGIMES:
                row[f"money_{reg}"] = h1[(C, s)]["regime_means"][reg]["money"]
            seed_level_rows.append(row)
        method_summary_rows.append(
            dict(
                C=C,
                n_seeds=st["n"],
                h1_mean=st["mean"],
                h1_sd=st["sd"],
                h1_se=st["se"],
                h1_ci95_lo=st["ci95_lo"],
                h1_ci95_hi=st["ci95_hi"],
                h1_min=st["min"],
                h1_max=st["max"],
                sc_mean=statistics.mean(sc[(C, s)]["macro"] for s in SEEDS),
                bf_fixed=bf[C]["macro"],
            )
        )

    # ---- paired H1 vs SC --------------------------------------------------
    paired_rows: List[Dict[str, Any]] = []
    stats["paired_vs_sc"] = {}
    for C in b0.CAPACITIES:
        deltas = [h1[(C, s)]["macro"] - sc[(C, s)]["macro"] for s in SEEDS]
        tt = one_sample_t(deltas)
        stats["paired_vs_sc"][str(C)] = dict(
            deltas={str(s): h1[(C, s)]["macro"] - sc[(C, s)]["macro"] for s in SEEDS},
            n_positive=sum(d > 0 for d in deltas),
            n_negative=sum(d < 0 for d in deltas),
            **tt,
        )
        for s in SEEDS:
            paired_rows.append(
                dict(
                    C=C,
                    seed=s,
                    h1_money=h1[(C, s)]["macro"],
                    sc_money=sc[(C, s)]["macro"],
                    delta=h1[(C, s)]["macro"] - sc[(C, s)]["macro"],
                )
            )
        paired_rows.append(
            dict(
                C=C,
                seed="summary",
                h1_money=tt["mean"] + statistics.mean(sc[(C, s)]["macro"] for s in SEEDS),
                sc_money=statistics.mean(sc[(C, s)]["macro"] for s in SEEDS),
                delta=tt["mean"],
                sd=tt["sd"],
                se=tt["se"],
                ci95_lo=tt["ci95_lo"],
                ci95_hi=tt["ci95_hi"],
                t=tt["t"],
                df=tt["df"],
                p_two_sided_raw=tt["p_two_sided_raw"],
                n_positive=sum(d > 0 for d in deltas),
                n_negative=sum(d < 0 for d in deltas),
            )
        )

    # ---- H1 vs fixed BF ---------------------------------------------------
    vs_bf_rows: List[Dict[str, Any]] = []
    stats["vs_bf"] = {}
    for C in b0.CAPACITIES:
        deltas = [h1[(C, s)]["macro"] - bf[C]["macro"] for s in SEEDS]
        tt = one_sample_t(deltas)
        stats["vs_bf"][str(C)] = dict(
            bf_fixed=bf[C]["macro"],
            deltas={str(s): h1[(C, s)]["macro"] - bf[C]["macro"] for s in SEEDS},
            **tt,
        )
        for s in SEEDS:
            vs_bf_rows.append(
                dict(
                    C=C,
                    seed=s,
                    h1_money=h1[(C, s)]["macro"],
                    bf_money=bf[C]["macro"],
                    delta=h1[(C, s)]["macro"] - bf[C]["macro"],
                )
            )
        vs_bf_rows.append(
            dict(
                C=C,
                seed="summary",
                h1_money=statistics.mean(h1[(C, s)]["macro"] for s in SEEDS),
                bf_money=bf[C]["macro"],
                delta=tt["mean"],
                sd=tt["sd"],
                se=tt["se"],
                ci95_lo=tt["ci95_lo"],
                ci95_hi=tt["ci95_hi"],
                t=tt["t"],
                df=tt["df"],
                p_two_sided_raw=tt["p_two_sided_raw"],
            )
        )

    # ---- gap recovery ------------------------------------------------------
    stats["gap_recovery"] = {}
    for C in b0.CAPACITIES:
        recs = {}
        for s in SEEDS:
            gap = bf[C]["macro"] - sc[(C, s)]["macro"]
            recs[str(s)] = (h1[(C, s)]["macro"] - sc[(C, s)]["macro"]) / gap
        vals = list(recs.values())
        stats["gap_recovery"][str(C)] = dict(
            per_seed=recs,
            mean=statistics.mean(vals),
            median=statistics.median(vals),
            min=min(vals),
            max=max(vals),
            n_gt_0=sum(v > 0 for v in vals),
            n_gt_30pct=sum(v > 0.30 for v in vals),
            n_ge_100pct=sum(v >= 1.0 for v in vals),
        )

    # ---- regime level ------------------------------------------------------
    regime_rows: List[Dict[str, Any]] = []
    stats["regime_analysis"] = {}
    for C in b0.CAPACITIES:
        stats["regime_analysis"][str(C)] = {}
        for reg in b0.REGIMES:
            deltas = [
                h1[(C, s)]["regime_means"][reg]["money"]
                - sc[(C, s)]["regime_means"][reg]["money"]
                for s in SEEDS
            ]
            n_pos = sum(d > 0 for d in deltas)
            cls = (
                "consistently_improved"
                if n_pos == N_SEEDS
                else ("consistently_degraded" if n_pos == 0 else "heterogeneous")
            )
            stats["regime_analysis"][str(C)][reg] = dict(
                mean_delta=statistics.mean(deltas),
                sd_delta=statistics.stdev(deltas),
                min_delta=min(deltas),
                max_delta=max(deltas),
                n_seeds_positive=n_pos,
                classification=cls,
            )
            for i, s in enumerate(SEEDS):
                hm = h1[(C, s)]["regime_means"][reg]
                sm = sc[(C, s)]["regime_means"][reg]
                bm = bf[C]["regime_means"][reg]
                regime_rows.append(
                    dict(
                        C=C,
                        regime=reg,
                        seed=s,
                        h1_money=hm["money"],
                        sc_money=sm["money"],
                        bf_money=bm["money"],
                        delta_vs_sc=deltas[i],
                        delta_vs_bf=hm["money"] - bm["money"],
                        h1_settled=hm["settled"],
                        h1_flushes=hm["flushes"],
                        h1_flush_cost=10 * hm["flushes"],
                        sc_settled=sm["settled"],
                        sc_flushes=sm["flushes"],
                        bf_settled=bm["settled"],
                        bf_flushes=bm["flushes"],
                        h1_accepted=hm["accepted_count"],
                        h1_insufficient_drops=hm["insufficient_drops"],
                        h1_oversize_drops=hm["oversize_drops"],
                    )
                )

    # ---- settled/flush decomposition --------------------------------------
    stats["decomposition"] = {}
    for C in b0.CAPACITIES:
        per_seed = {}
        for s in SEEDS:
            hset = statistics.mean(h1[(C, s)]["regime_means"][r]["settled"] for r in b0.REGIMES)
            hf = statistics.mean(h1[(C, s)]["regime_means"][r]["flushes"] for r in b0.REGIMES)
            sset = statistics.mean(sc[(C, s)]["regime_means"][r]["settled"] for r in b0.REGIMES)
            sf = statistics.mean(sc[(C, s)]["regime_means"][r]["flushes"] for r in b0.REGIMES)
            per_seed[str(s)] = dict(
                h1_settled=hset,
                h1_flush_cost=10 * hf,
                sc_settled=sset,
                sc_flush_cost=10 * sf,
                delta_settled_vs_sc=hset - sset,
                delta_flushcost_vs_sc=10 * hf - 10 * sf,
                delta_settled_vs_bf=hset
                - statistics.mean(bf[C]["regime_means"][r]["settled"] for r in b0.REGIMES),
                delta_flushcost_vs_bf=10 * hf
                - 10 * statistics.mean(bf[C]["regime_means"][r]["flushes"] for r in b0.REGIMES),
            )
        stats["decomposition"][str(C)] = dict(
            per_seed=per_seed,
            mean_delta_settled_vs_sc=statistics.mean(
                v["delta_settled_vs_sc"] for v in per_seed.values()
            ),
            mean_delta_flushcost_vs_sc=statistics.mean(
                v["delta_flushcost_vs_sc"] for v in per_seed.values()
            ),
            mean_delta_settled_vs_bf=statistics.mean(
                v["delta_settled_vs_bf"] for v in per_seed.values()
            ),
            mean_delta_flushcost_vs_bf=statistics.mean(
                v["delta_flushcost_vs_bf"] for v in per_seed.values()
            ),
        )

    # ---- verdict (documented screening heuristic, not significance) -------
    strong_caps = []
    for C in b0.CAPACITIES:
        p = stats["paired_vs_sc"][str(C)]
        if p["mean"] > 0 and p["n_positive"] >= 4:
            strong_caps.append(C)
    if len(strong_caps) == 2:
        verdict = "STRONG SIGNAL FOR NEXT-STAGE DESIGN"
    elif strong_caps or any(
        stats["paired_vs_sc"][str(C)]["mean"] > 0
        and stats["paired_vs_sc"][str(C)]["n_positive"] >= 3
        for C in b0.CAPACITIES
    ):
        verdict = "MIXED SIGNAL"
    else:
        verdict = "NO SIGNAL"
    stats["verdict"] = dict(
        value=verdict,
        rule=(
            "STRONG if mean paired H1-SC delta > 0 with >=4/5 seeds positive at BOTH "
            "capacities; MIXED if that holds at one capacity (>=3/5 positive counts as "
            "partial); else NO SIGNAL. Screening heuristic only."
        ),
        strong_capacities=strong_caps,
    )
    stats["analysis_elapsed_seconds"] = time.time() - t0

    # ---- write CSVs / JSON -------------------------------------------------
    write_csv(MS_ROOT / "H1_SEED_LEVEL_SCORES.csv", seed_level_rows)
    write_csv(MS_ROOT / "H1_METHOD_SUMMARY.csv", method_summary_rows)
    write_csv(MS_ROOT / "H1_PAIRED_VS_SC.csv", paired_rows)
    write_csv(MS_ROOT / "H1_VS_BF.csv", vs_bf_rows)
    write_csv(MS_ROOT / "H1_REGIME_LEVEL.csv", regime_rows)
    (MS_ROOT / "statistics.json").write_text(json.dumps(stats, indent=2, sort_keys=True))

    write_report(stats, h1, sc, bf)

    print("ANALYZE complete:")
    for C in b0.CAPACITIES:
        p = stats["paired_vs_sc"][str(C)]
        g = stats["gap_recovery"][str(C)]
        print(
            f"  C{C}: H1 mean {stats['h1_seed_level'][str(C)]['mean']:.2f} "
            f"(sd {p['sd']:.2f}) | paired d mean {fmt_signed(p['mean'], 2)} "
            f"[{fmt_signed(p['ci95_lo'], 2)}, {fmt_signed(p['ci95_hi'], 2)}] "
            f"t={p['t']:.3f} p={p['p_two_sided_raw']:.4f} | "
            f"recovery mean {g['mean']:.3f} | pos {p['n_positive']}/5"
        )
    print(f"  VERDICT: {verdict}")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
# Report
# ---------------------------------------------------------------------------

def write_report(
    stats: Dict[str, Any],
    h1: Dict[Tuple[int, int], Dict[str, Any]],
    sc: Dict[Tuple[int, int], Dict[str, Any]],
    bf: Dict[int, Dict[str, Any]],
) -> None:
    L: List[str] = []
    L.append("# H1 Multi-Seed Diagnostic Report (Stage 3.5)")
    L.append("")
    L.append(
        "**POST-HOC EXPLORATORY DIAGNOSTIC — not confirmatory.** NEW12-v1 was "
        "already used for Stage 2B/Stage 3 method selection, so this is not a "
        "fresh test set. H1 is not a new method: it re-combines the frozen "
        "BF-T0.5 settlement rule with each frozen SC-FAC checkpoint's "
        "conditional flush head (conditioned on the BF settle index), executed "
        "by the unmodified E0 environment. No training, no tuning, no frozen "
        "artifact modified. p-values are raw, unadjusted, n=5 (df=4), and are "
        "reported as descriptive screening statistics only."
    )
    L.append("")

    # 1. Provenance
    L.append("## 1. Provenance")
    L.append("")
    L.append(
        "- Frozen SC-FAC checkpoints for seeds 123/323/532/777/999 resolved "
        "from the frozen Stage 2 job matrix; every checkpoint SHA256 was "
        "verified against the job matrix AND the frozen Stage 3 "
        "`seed_level_scores.csv` before evaluation."
    )
    L.append(
        "- `adapter.check_config` verified per cell: C, k=24, F=3, T=1000, "
        "training seed, cpu, original reward (money_p=1, money_tau=10), "
        "`conditional_factorized_ac`, condition_mode=full, settle_embed_dim=32, "
        "conditional_hidden_size=256."
    )
    L.append(
        f"- Before each NEW cell, the loaded checkpoint was required to "
        f"reproduce its own frozen SC-FAC NEW12 episodes EXACTLY on the "
        f"provenance regimes {PARITY_REGIMES} (200 episodes each); all passed."
    )
    L.append(
        "- Seed 123 H1 outputs are REUSED unchanged from the validated B0 "
        "pilot (sha256-pinned in `outputs/seed123_reference_sha.json`)."
    )
    L.append(
        f"- Frozen raw tarball SHA {b0.RAW_SHA[:16]}…, lineage {b0.LINEAGE[:12]}…, "
        "NEW12-v1 manifest verified per run."
    )
    L.append("")

    # 2. Final matrix
    L.append("## 2. Final matrix (5 seeds × 2 capacities = 10 cells)")
    L.append("")
    L.append("| C | seed | H1 macro12 Money | source |")
    L.append("|---|---|---|---|")
    for C in b0.CAPACITIES:
        for s in SEEDS:
            src = "B0 reuse" if s == b0.SC_SEED else "new"
            L.append(f"| {C} | {s} | {fmt(h1[(C, s)]['macro'], 4)} | {src} |")
    L.append("")

    # 3. Seed-level H1 results
    L.append("## 3. Seed-level H1 results")
    L.append("")
    L.append(
        "| C | mean | sample SD | SE | 95% t CI | min | max | SC mean | BF (fixed) |"
    )
    L.append("|---|---|---|---|---|---|---|---|---|")
    for C in b0.CAPACITIES:
        st = stats["h1_seed_level"][str(C)]
        sc_mean = statistics.mean(sc[(C, s)]["macro"] for s in SEEDS)
        L.append(
            f"| {C} | {fmt(st['mean'], 2)} | {fmt(st['sd'], 2)} | {fmt(st['se'], 2)} | "
            f"[{fmt(st['ci95_lo'], 2)}, {fmt(st['ci95_hi'], 2)}] | {fmt(st['min'], 2)} | "
            f"{fmt(st['max'], 2)} | {fmt(sc_mean, 2)} | {fmt(bf[C]['macro'], 2)} |"
        )
    L.append("")

    # 4. Paired H1 vs SC
    L.append("## 4. Paired H1 vs original SC-FAC (same training seed)")
    L.append("")
    for C in b0.CAPACITIES:
        p = stats["paired_vs_sc"][str(C)]
        L.append(f"### C={C}")
        L.append("")
        L.append("| seed | H1 | SC | Δ (H1−SC) |")
        L.append("|---|---|---|---|")
        for s in SEEDS:
            d = p["deltas"][str(s)]
            L.append(
                f"| {s} | {fmt(h1[(C, s)]['macro'], 3)} | {fmt(sc[(C, s)]['macro'], 3)} | "
                f"{fmt_signed(d, 3)} |"
            )
        L.append("")
        L.append(
            f"- positive/negative: **{p['n_positive']}/{p['n_negative']}**; "
            f"mean Δ **{fmt_signed(p['mean'], 3)}**, SD {fmt(p['sd'], 3)}, "
            f"SE {fmt(p['se'], 3)}, 95% CI [{fmt_signed(p['ci95_lo'], 3)}, "
            f"{fmt_signed(p['ci95_hi'], 3)}], paired t({p['df']}) = "
            f"{fmt(p['t'], 3)}, raw two-sided p = {p['p_two_sided_raw']:.4f}"
        )
        L.append("")

    # 5. H1 vs BF
    L.append("## 5. H1 vs fixed deterministic BF-T0.5")
    L.append("")
    L.append(
        "BF-T0.5 is a deterministic rule: ONE score per capacity, no seed "
        "uncertainty is fabricated."
    )
    L.append("")
    for C in b0.CAPACITIES:
        v = stats["vs_bf"][str(C)]
        L.append(f"### C={C} (BF = {fmt(v['bf_fixed'], 3)})")
        L.append("")
        L.append("| seed | H1 | Δ (H1−BF) |")
        L.append("|---|---|---|")
        for s in SEEDS:
            L.append(
                f"| {s} | {fmt(h1[(C, s)]['macro'], 3)} | "
                f"{fmt_signed(v['deltas'][str(s)], 3)} |"
            )
        L.append("")
        L.append(
            f"- mean Δ **{fmt_signed(v['mean'], 3)}**, SD {fmt(v['sd'], 3)}, "
            f"SE {fmt(v['se'], 3)}, 95% CI [{fmt_signed(v['ci95_lo'], 3)}, "
            f"{fmt_signed(v['ci95_hi'], 3)}], one-sample t({v['df']}) = "
            f"{fmt(v['t'], 3)}, raw p = {v['p_two_sided_raw']:.4f}"
        )
        L.append("")

    # 6. Gap recovery
    L.append("## 6. Gap recovery  (H1−SC)/(BF−SC) per seed")
    L.append("")
    L.append(
        "Interpret cautiously: the denominator varies with the seed's own SC "
        "score; recovery > 1 can reflect a small SC baseline as much as a "
        "strong H1."
    )
    L.append("")
    L.append("| C | seed | recovery |")
    L.append("|---|---|---|")
    for C in b0.CAPACITIES:
        g = stats["gap_recovery"][str(C)]
        for s in SEEDS:
            L.append(f"| {C} | {s} | {fmt(g['per_seed'][str(s)], 3)} |")
    L.append("")
    L.append("| C | mean | median | min | max | #>0 | #>30% | #>=100% |")
    L.append("|---|---|---|---|---|---|---|---|")
    for C in b0.CAPACITIES:
        g = stats["gap_recovery"][str(C)]
        L.append(
            f"| {C} | {fmt(g['mean'], 3)} | {fmt(g['median'], 3)} | {fmt(g['min'], 3)} | "
            f"{fmt(g['max'], 3)} | {g['n_gt_0']}/5 | {g['n_gt_30pct']}/5 | {g['n_ge_100pct']}/5 |"
        )
    L.append("")

    # 7. Regime-level patterns
    L.append("## 7. Regime-level patterns (descriptive; regimes are NOT independent replicates)")
    L.append("")
    for C in b0.CAPACITIES:
        ra = stats["regime_analysis"][str(C)]
        L.append(f"### C={C}: mean H1−SC per regime (across 5 seeds)")
        L.append("")
        L.append("| regime | mean Δ | SD | min | max | seeds positive | class |")
        L.append("|---|---|---|---|---|---|---|")
        for reg in b0.REGIMES:
            r = ra[reg]
            L.append(
                f"| {reg} | {fmt_signed(r['mean_delta'], 2)} | {fmt(r['sd_delta'], 2)} | "
                f"{fmt_signed(r['min_delta'], 2)} | {fmt_signed(r['max_delta'], 2)} | "
                f"{r['n_seeds_positive']}/5 | {r['classification']} |"
            )
        ci = [r for r in b0.REGIMES if ra[r]["classification"] == "consistently_improved"]
        cd = [r for r in b0.REGIMES if ra[r]["classification"] == "consistently_degraded"]
        hh = [r for r in b0.REGIMES if ra[r]["classification"] == "heterogeneous"]
        L.append("")
        L.append(f"- consistently improved (5/5 seeds positive): {', '.join(ci) or 'none'}")
        L.append(f"- consistently degraded (0/5 positive): {', '.join(cd) or 'none'}")
        L.append(f"- high-heterogeneity (mixed signs): {', '.join(hh) or 'none'}")
        L.append("")

    # 8. Decomposition
    L.append("## 8. Settlement/flush decomposition (macro12 component means)")
    L.append("")
    L.append(
        "| C | mean Δsettled vs SC | mean Δflushcost vs SC | mean Δsettled vs BF | "
        "mean Δflushcost vs BF |"
    )
    L.append("|---|---|---|---|---|")
    for C in b0.CAPACITIES:
        d = stats["decomposition"][str(C)]
        L.append(
            f"| {C} | {fmt_signed(d['mean_delta_settled_vs_sc'], 2)} | "
            f"{fmt_signed(d['mean_delta_flushcost_vs_sc'], 2)} | "
            f"{fmt_signed(d['mean_delta_settled_vs_bf'], 2)} | "
            f"{fmt_signed(d['mean_delta_flushcost_vs_bf'], 2)} |"
        )
    L.append("")
    L.append(
        "Positive Δflushcost means H1 flushes MORE than the comparator (worse); "
        "negative means it saves flush cost. Per-seed values in statistics.json."
    )
    L.append("")

    # 9. Negative / null findings
    L.append("## 9. Negative / null findings")
    L.append("")
    for C in b0.CAPACITIES:
        p = stats["paired_vs_sc"][str(C)]
        v = stats["vs_bf"][str(C)]
        negs = [s for s in SEEDS if p["deltas"][str(s)] <= 0]
        L.append(
            f"- C={C}: seeds without improvement over SC: "
            f"{', '.join(str(s) for s in negs) or 'none'}; "
            f"H1 remains below BF at every seed: "
            f"{all(v['deltas'][str(s)] < 0 for s in SEEDS)} "
            f"(min Δ vs BF {fmt_signed(v['min'], 2)}, max {fmt_signed(v['max'], 2)})."
        )
    L.append("")

    # 10. Did seed123 generalize?
    L.append("## 10. Did seed123 generalize? (exploratory Q1/Q2)")
    L.append("")
    p8 = stats["paired_vs_sc"]["800"]
    p12 = stats["paired_vs_sc"]["1200"]
    L.append(
        f"- C=800: seed123 Δ vs SC was +380.44. Across all 5 seeds: mean "
        f"{fmt_signed(p8['mean'], 2)}, {p8['n_positive']}/5 positive, "
        f"95% CI [{fmt_signed(p8['ci95_lo'], 2)}, {fmt_signed(p8['ci95_hi'], 2)}]."
    )
    L.append(
        f"- C=1200: seed123 Δ vs SC was −4.47 (≈neutral). Across all 5 seeds: "
        f"mean {fmt_signed(p12['mean'], 2)}, {p12['n_positive']}/5 positive, "
        f"95% CI [{fmt_signed(p12['ci95_lo'], 2)}, {fmt_signed(p12['ci95_hi'], 2)}]."
    )
    L.append("")

    # 11. Trainable successor?
    L.append("## 11. Does H1 warrant a trainable BF-settlement + learned-flush successor?")
    L.append("")
    L.append(
        "Screening opinion (not a design commitment): the multi-seed evidence "
        "above indicates whether replacing the learned settlement with the BF "
        "rule transfers a consistent, capacity-dependent gain. Any B1 model "
        "design is explicitly out of scope here and must be proposed by Codex "
        "after these results are frozen and uploaded; a future B1 must be "
        "confirmed on a fresh held-out stream release."
    )
    L.append("")

    # 12. Caveats
    L.append("## 12. Caveats")
    L.append("")
    L.append(
        "- n=5 seeds, df=4: tiny sample, wide CIs; raw p-values are descriptive "
        "only and are not multiplicity-adjusted."
    )
    L.append(
        "- NEW12-v1 is the same stream family used for Stage 2B/3 selection — "
        "results can be optimistic relative to a truly fresh stream."
    )
    L.append(
        "- H1's flush head can flush the BF settlement wallet itself; E0 "
        "processes flush-first, voiding that settlement step (interaction kept "
        "intact by the unmodified environment)."
    )
    L.append(
        "- Provenance parity per new seed covered 3 regimes exactly; the other "
        "9 regimes are bound by checkpoint SHA, config checks, and stream "
        "hashes, not by replay."
    )
    L.append(
        "- Gap-recovery denominators differ across seeds (each seed has its own "
        "SC baseline); interpret the ratio with care."
    )
    L.append("")

    # 13. Artifact inventory
    L.append("## 13. Exact artifact inventory")
    L.append("")
    L.append("```")
    L.append("artifacts/aamas2027/stage3_5/h1_multiseed/")
    L.append("  code/run_h1_multi.py        # verify|run|validate|analyze|all")
    L.append("  code/test_h1_stats.py       # numerical stats unit tests")
    L.append("  outputs/C{C}_S{seed}/       # 8 new cells: episodes.csv + result.json")
    L.append("  outputs/seed123_reference_sha.json")
    L.append("  outputs/H1_VALIDATION.json")
    L.append("  H1_SEED_LEVEL_SCORES.csv")
    L.append("  H1_METHOD_SUMMARY.csv")
    L.append("  H1_PAIRED_VS_SC.csv")
    L.append("  H1_VS_BF.csv")
    L.append("  H1_REGIME_LEVEL.csv")
    L.append("  statistics.json")
    L.append("  H1_MULTI_SEED_REPORT.md")
    L.append("  README.md")
    L.append("```")
    L.append("")

    # 14. Reproduction
    L.append("## 14. Reproduction commands")
    L.append("")
    L.append("```bash")
    L.append("PY=/Users/zhouzhou/benchmarks/A-EVAL-JA-C800-S123/.venv/bin/python")
    L.append("cd /Users/zhouzhou/Desktop/kwallet-aamas-b0")
    L.append("M=artifacts/aamas2027/stage3_5/h1_multiseed")
    L.append("$PY $M/code/run_h1_multi.py verify")
    L.append("$PY $M/code/test_h1_stats.py")
    L.append("$PY $M/code/run_h1_multi.py run --all")
    L.append("$PY $M/code/run_h1_multi.py validate")
    L.append("$PY $M/code/run_h1_multi.py analyze")
    L.append("```")
    L.append("")
    L.append("---")
    L.append("")
    L.append(f"## Screening verdict: **{stats['verdict']['value']}**")
    L.append("")
    L.append(f"_{stats['verdict']['rule']}_")
    L.append("")
    (MS_ROOT / "H1_MULTI_SEED_REPORT.md").write_text("\n".join(L))


# ---------------------------------------------------------------------------

def cmd_all(_args: argparse.Namespace) -> None:
    wall = time.time()
    cmd_verify(argparse.Namespace())
    cmd_run(argparse.Namespace(all=True, C=None, seed=None))
    cmd_validate(argparse.Namespace())
    cmd_analyze(argparse.Namespace())
    print(f"H1 MULTI-SEED ALL COMPLETE in {(time.time()-wall)/60:.1f} minutes")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify").set_defaults(func=cmd_verify)
    rp = sub.add_parser("run")
    rp.add_argument("--C", type=int, choices=b0.CAPACITIES)
    rp.add_argument("--seed", type=int, choices=NEW_SEEDS)
    rp.add_argument("--all", action="store_true")
    rp.set_defaults(func=cmd_run)
    sub.add_parser("validate").set_defaults(func=cmd_validate)
    sub.add_parser("analyze").set_defaults(func=cmd_analyze)
    sub.add_parser("all").set_defaults(func=cmd_all)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
