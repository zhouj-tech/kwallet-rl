#!/usr/bin/env python3
"""B0 (Stage 3.5) hybrid diagnostic support library.

ADDITIVE ONLY. This module never edits, retrains, or regenerates anything:

* the frozen E0 environment and the frozen SC-FAC evaluator are imported
  through the hash-pinned Stage 2 adapter (``tools/aamas_stage2/adapter.py``);
* the frozen SC-FAC seed=123 checkpoints and the frozen NEW12-v1 streams are
  only read and hash-verified;
* the only new logic is the factorization of the frozen BF-T0.5 rule into
  separate settle/flush helpers and an evaluator that combines an externally
  chosen settle action with an externally chosen flush action.

H1 = BF settle + SC conditional flush (flush head conditioned on the BF
settle index).  H2 = SC settle + BF flush (settle wallet excluded).

Post-hoc exploratory diagnostic -- NOT confirmatory, NOT a new method claim.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import statistics
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Paths and frozen protocol constants
# ---------------------------------------------------------------------------

REGIMES = "US TLS LNS TLNS TPLS PLS UB TLB LNB TLNB TPLB PLB".split()
CAPACITIES = [800, 1200]
K = 24
F = 3
T = 1000
EPISODES_PER_REGIME = 200
SC_SEED = 123

# b0_lib.py lives at <wt>/artifacts/aamas2027/stage3_5/b0_diagnostic/code/
B0_ROOT = Path(__file__).resolve().parents[1]
B0_WORKTREE = Path(__file__).resolve().parents[5]

# Frozen external roots (read-only). Overridable only for portability/review.
FROZEN_REPO = Path(os.environ.get("B0_FROZEN_REPO", "/Users/zhouzhou/Desktop/kwallet-rl"))
FROZEN_ARTIFACTS = Path(
    os.environ.get("B0_FROZEN_ARTIFACTS", "/Users/zhouzhou/Desktop/kwallet-aamas-artifacts")
)
NEW12_ROOT = Path(
    os.environ.get(
        "B0_NEW12_ROOT",
        "/Users/zhouzhou/benchmarks/A-EVAL-JA-C800-S123/data/streams/NEW12-v1",
    )
)
RAW_TARBALL = (
    B0_WORKTREE / "artifacts/aamas2027/stage2/raw/KWALLET_AAMAS_STAGE2B_RAW_20261002.tar.gz"
)
STAGE3_DIR = B0_WORKTREE / "artifacts/aamas2027/stage3"
JOB_MATRIX = FROZEN_REPO / "research/aamas2027/STAGE2_JOB_MATRIX.csv"
ARTIFACT_MANIFEST = (
    FROZEN_REPO / "research/aamas2027/artifact_handoff/AAMAS_ARTIFACT_MANIFEST.csv"
)

LINEAGE = "5573ec642f0f28c218f3e6058478f62ab6db6b2b"
RAW_SHA = "576c9baa00c4438543ab9f9c2063f65bb26b1a6f287974a9e8ab91b18189ace3"
BUNDLE = "KWALLET_AAMAS_STAGE2B_RAW_20261002"
FROZEN_JOB_IDS = {
    ("SC", 800): "A-EVAL-SC-C800-S123",
    ("SC", 1200): "A-EVAL-SC-C1200-S123",
    ("BF", 800): "C-EVAL-BFT05-C800",
    ("BF", 1200): "C-EVAL-BFT05-C1200",
}

# Hybrid identifiers
H1 = "H1"  # BF settle, SC conditional flush
H2 = "H2"  # SC settle, BF flush
HYBRIDS = [H1, H2]
HYBRID_DESC = {
    H1: "BF-settle + SC-FLUSH(conditioned on BF settle)",
    H2: "SC-settle + BF-flush(excluding SC settle)",
}
PILOT_JOB = {
    (H1, 800): "B0-H1-C800-S123",
    (H1, 1200): "B0-H1-C1200-S123",
    (H2, 800): "B0-H2-C800-S123",
    (H2, 1200): "B0-H2-C1200-S123",
}
HYBRID_METHOD_LABEL = {
    H1: "B0-H1-BFsettle-SCflush",
    H2: "B0-H2-SCsettle-BFflush",
}

CACHE_DIR = B0_ROOT / "outputs" / "_cache"
FROZEN_EPISODE_FIELDS = None  # populated from the frozen episodes.csv header


class B0Error(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise B0Error(message)


# ---------------------------------------------------------------------------
# Hashing / frozen IO
# ---------------------------------------------------------------------------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def csv_rows(path: Path) -> List[Dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def tar_member_text(name: str) -> str:
    """Read a text member of the frozen raw tarball without extracting it."""
    require(RAW_TARBALL.is_file(), f"Missing frozen raw tarball: {RAW_TARBALL}")
    with tarfile.open(RAW_TARBALL, "r:gz") as tf:
        member = tf.extractfile(f"{BUNDLE}/{name}")
        require(member is not None, f"Missing tar member: {name}")
        return member.read().decode("utf-8")


def raw_approval() -> Dict[str, Any]:
    return json.loads(tar_member_text("MAC_RUNTIME_APPROVED.json"))


def load_frozen_episodes(job_id: str) -> List[Dict[str, str]]:
    """Parse frozen Stage 2 episodes.csv for one job directly from the tarball."""
    text = tar_member_text(f"output/{job_id}/episodes.csv")
    return list(csv.DictReader(io.StringIO(text)))


def frozen_episode_fields() -> List[str]:
    global FROZEN_EPISODE_FIELDS
    if FROZEN_EPISODE_FIELDS is None:
        text = tar_member_text(f"output/{FROZEN_JOB_IDS[('SC', 800)]}/episodes.csv")
        FROZEN_EPISODE_FIELDS = next(csv.reader(io.StringIO(text)))
    return list(FROZEN_EPISODE_FIELDS)


# ---------------------------------------------------------------------------
# Frozen artifact resolution (paths/hashes come from frozen tables, never
# hard-coded comparison numbers)
# ---------------------------------------------------------------------------

def frozen_job_rows() -> Dict[str, Dict[str, str]]:
    rows = csv_rows(JOB_MATRIX)
    return {r["job_id"]: r for r in rows}


def _remap_artifacts(server_path: str) -> Path:
    prefix = "/data/sijia/aamas2027_artifacts/"
    require(server_path.startswith(prefix), f"Unexpected artifact path: {server_path}")
    return FROZEN_ARTIFACTS / server_path[len(prefix):]


def load_sc_bundle(C: int, adapter: Any) -> Tuple[Dict[str, Any], Path, str, str]:
    """Resolve the frozen SC-FAC seed123 checkpoint + config for capacity C."""
    job = frozen_job_rows()[f"A-EVAL-SC-C{C}-S123"]
    cp = _remap_artifacts(job["checkpoint_source"])
    ri = _remap_artifacts(job["run_info_source"])
    require(cp.is_file() and ri.is_file(), f"Missing SC C{C} checkpoint/run_info")
    require(sha256(cp) == job["checkpoint_sha256"], f"SC C{C} checkpoint SHA mismatch")
    cfg = read_json(ri)["config"]
    adapter.check_config(cfg, job)
    return cfg, cp, job["checkpoint_sha256"], sha256(ri)


def load_ja_transfer_config(C: int) -> Dict[str, Any]:
    """BF-T0.5 uses the JA-PPO seed123 C-specific transfer config (frozen)."""
    wanted_method = "basic_ppo"
    wanted = f"basic_ppo_trainMIX12_EQ_C{C}_k24_T1000_F3_seed123"
    matches = [
        r
        for r in csv_rows(ARTIFACT_MANIFEST)
        if r.get("artifact_type") == "run_info"
        and r.get("method") == "JA-PPO"
        and r.get("C") == str(C)
        and r.get("training_seed") == "123"
    ]
    require(len(matches) == 1, f"Ambiguous JA transfer run_info for C{C}: {len(matches)}")
    rec = matches[0]
    p = FROZEN_ARTIFACTS / rec["transfer_relative_path"]
    require(p.is_file(), f"Missing JA transfer run_info: {p}")
    require(sha256(p) == rec["sha256"], "JA transfer run_info SHA mismatch")
    require(wanted in rec["transfer_relative_path"], "Unexpected JA run_info selected")
    return read_json(p)["config"]


# ---------------------------------------------------------------------------
# Frozen module / pool loading -- delegates to the hash-pinned adapter
# ---------------------------------------------------------------------------

def import_frozen_adapter() -> Any:
    sys.dont_write_bytecode = True
    adapter_dir = FROZEN_REPO / "tools" / "aamas_stage2"
    if str(adapter_dir) not in sys.path:
        sys.path.insert(0, str(adapter_dir))
    import adapter  # noqa: WPS433 (intentional frozen import)

    require(sha256(adapter_dir / "adapter.py") == raw_approval()["adapter_sha256"],
            "Frozen adapter SHA differs from raw approval")
    return adapter


def configure_caches() -> None:
    xdg = (CACHE_DIR / "xdg").resolve()
    mpl = (CACHE_DIR / "mpl").resolve()
    xdg.mkdir(parents=True, exist_ok=True)
    mpl.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(xdg)
    os.environ["MPLCONFIGDIR"] = str(mpl)
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")


def import_frozen_modules(adapter: Any) -> Tuple[Any, Any]:
    configure_caches()
    adapter.verify_sources(FROZEN_REPO)
    sc = adapter.import_original(FROZEN_REPO, "SC-FAC")
    ja = adapter.import_original(FROZEN_REPO, "BF-T0.5")
    return sc, ja


def load_new12_streams(adapter: Any) -> Tuple[List[Dict[str, Any]], str]:
    """Verify and load the 12 frozen NEW12-v1 pools via the frozen adapter."""
    approval = raw_approval()
    manifest_path = NEW12_ROOT / "new12_manifest.json"
    manifest_sha = sha256(manifest_path)
    require(
        manifest_sha == approval["new12_manifest_sha256"],
        "NEW12 manifest SHA does not match frozen raw approval",
    )
    manifest = read_json(manifest_path)
    require([p["regime"] for p in manifest["pools"]] == REGIMES, "Wrong NEW12 regime order")
    streams = []
    for p in manifest["pools"]:
        path = NEW12_ROOT / p["filename"]
        pool, hashes = adapter.load_pool(path, p["sha256"], p["episode_hashes"])
        streams.append(
            dict(
                regime=p["regime"],
                pool=pool,
                sha256=p["sha256"],
                episode_seeds=list(p["episode_seeds"]),
                episode_hashes=list(hashes),
            )
        )
    return streams, manifest_sha


# ---------------------------------------------------------------------------
# Frozen BF-T0.5 rule, factorized exactly from adapter.bf_action
# (adapter.py lines 121-127; RULE string in the frozen adapter)
# ---------------------------------------------------------------------------

def bf_settle(env: Any) -> int:
    """s_BF = argmin_(i usable, b_i >= x) (b_i, i); no-op index k if none."""
    usable = [i for i in range(env.k) if env._usable(i)]
    candidates = [i for i in usable if env.wallets[i] >= env.current_tx]
    if not candidates:
        return env.k
    return min(candidates, key=lambda i: (env.wallets[i], i))


def bf_flush(env: Any, settle: int) -> int:
    """f_BF = argmin_(j usable, j != settle, b_j < 0.5*C/k) (b_j, j).

    Strict threshold, index tie-break, settlement wallet excluded.  A no-op
    settlement (k) never matches a wallet index, so flushing remains possible.
    Eligibility is independent of transaction feasibility, so an oversized /
    infeasible request never suppresses a valid flush.
    """
    usable = [i for i in range(env.k) if env._usable(i)]
    bound = 0.5 * (env.C / env.k)
    eligible = [i for i in usable if i != settle and env.wallets[i] < bound]
    if not eligible:
        return env.k
    return min(eligible, key=lambda i: (env.wallets[i], i))


def bf_joint_action(env: Any) -> int:
    """Recomposition; must equal frozen adapter.bf_action(env) everywhere."""
    s = bf_settle(env)
    f = bf_flush(env, s)
    return s * (env.k + 1) + f


# ---------------------------------------------------------------------------
# Frozen SC-FAC agent construction (exact replica of adapter.learned_evaluation)
# ---------------------------------------------------------------------------

def build_sc_agent(module: Any, cfg: Dict[str, Any], checkpoint: Path) -> Tuple[Any, Any]:
    import torch

    env = module.make_env(cfg, max_steps=T)
    agent = module.ConditionalFactorizedPPOAgent(
        config=cfg,
        state_size=env.state_size,
        base_state_size=env.base_state_size,
        k=env.k,
    )
    agent.model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
    )
    # Deliberately do NOT call .eval(): the frozen evaluator acts in the
    # original inference mode. These models contain no dropout/batchnorm.
    return env, agent


# ---------------------------------------------------------------------------
# Hybrid action selection
# ---------------------------------------------------------------------------

def _finite(t: Any) -> None:
    import torch

    require(bool(torch.isfinite(t).all()), "Non-finite policy logits")


def hybrid_action(
    env: Any,
    agent: Any,
    state: np.ndarray,
    settle_mode: str,
    flush_mode: str,
    diagnostics: bool = False,
) -> Tuple[int, int, int, int, int]:
    """Return (settle, flush, joint_action, sc_settle, sc_flush).

    settle_mode / flush_mode are each ``'BF'`` or ``'SC'``.

    SC deterministic decoding is exactly the frozen path: settle = argmax of
    settle logits; flush = argmax of ``forward_flush_given_settle`` given the
    *actually selected* settle index.  For H1 the externally supplied value is
    the BF settle index -- the SC settle head is evaluated only to report the
    counterfactual SC choice, never to choose the submitted settlement.
    """
    import torch

    require(settle_mode in ("BF", "SC"), f"Unknown settle_mode {settle_mode}")
    require(flush_mode in ("BF", "SC"), f"Unknown flush_mode {flush_mode}")
    model = agent.model
    device = agent.device
    k = env.k

    state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        settle_logits, _ = model.forward_settle_value(state_t)
    _finite(settle_logits)
    sc_settle = int(torch.argmax(settle_logits, dim=-1).item())

    # 1) Settlement choice.
    settle = bf_settle(env) if settle_mode == "BF" else sc_settle

    # 2) Flush choice, conditioned on the chosen settlement.
    if flush_mode == "SC":
        settle_t = torch.tensor([settle], dtype=torch.long, device=device)
        with torch.no_grad():
            flush_logits = model.forward_flush_given_settle(state_t, settle_t)
        _finite(flush_logits)
        flush = int(torch.argmax(flush_logits, dim=-1).item())
    else:
        flush = bf_flush(env, settle)

    # Optional counterfactual: SC's native flush under SC's own settle.
    if diagnostics:
        sc_settle_t = torch.tensor([sc_settle], dtype=torch.long, device=device)
        with torch.no_grad():
            sc_flush_logits = model.forward_flush_given_settle(state_t, sc_settle_t)
        sc_flush = int(torch.argmax(sc_flush_logits, dim=-1).item())
    else:
        sc_flush = flush

    require(0 <= settle <= k and 0 <= flush <= k, "Action index out of range")
    return settle, flush, settle * (k + 1) + flush, sc_settle, sc_flush


# ---------------------------------------------------------------------------
# Episode evaluation (bookkeeping mirrors evaluate_agent_on_array exactly)
# ---------------------------------------------------------------------------

def evaluate_hybrid_pool(
    module: Any,
    cfg: Dict[str, Any],
    agent: Any,
    pool: np.ndarray,
    settle_mode: str,
    flush_mode: str,
    max_steps: int = T,
    record: bool = False,
    rule_fidelity_check: bool = False,
) -> Dict[str, Any]:
    """Evaluate one 200-episode NEW12 regime with a hybrid policy."""
    env = module.make_env(cfg, max_steps=max_steps)
    raw_results: List[Dict[str, float]] = []
    traces: List[Dict[str, Any]] = []
    n_episodes = min(EPISODES_PER_REGIME, pool.shape[0])

    for ep in range(n_episodes):
        state = env.reset(tx_stream=pool[ep])
        total_requested_value = 0.0
        total_tx_count = 0
        accepted_count = 0
        episode_gates: List[float] = []

        for t in range(max_steps):
            total_requested_value += float(env.current_tx)
            total_tx_count += 1

            settle, flush, action, sc_settle, sc_flush = hybrid_action(
                env, agent, state, settle_mode, flush_mode, diagnostics=record
            )

            if rule_fidelity_check:
                # Wherever both halves are BF, recomposition must match the
                # frozen adapter rule call exactly on this live env state.
                require(
                    bf_joint_action(env) == _frozen_adapter_bf_action(env),
                    "Factorized BF helpers diverged from frozen adapter.bf_action",
                )

            if record:
                traces.append(
                    dict(
                        episode=ep,
                        step=t,
                        settle=int(settle),
                        flush=int(flush),
                        sc_settle=int(sc_settle),
                        sc_flush=int(sc_flush),
                        action=int(action),
                        tx=int(env.current_tx),
                    )
                )

            state, _, done, info = env.step(action)
            episode_gates.append(0.0)
            if info.get("accepted", False):
                accepted_count += 1
            if done:
                break

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
        metrics["gate_mean"] = float(np.mean(episode_gates)) if episode_gates else 0.0
        metrics["gate_std"] = float(np.std(episode_gates)) if episode_gates else 0.0
        metrics["gate_min"] = float(np.min(episode_gates)) if episode_gates else 0.0
        metrics["gate_max"] = float(np.max(episode_gates)) if episode_gates else 0.0
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
    out = dict(num_episodes=n_episodes, summary=summary, raw_results=raw_results)
    if record:
        out["traces"] = traces
    return out


def _frozen_adapter_bf_action(env: Any) -> int:
    return _ADAPTER.bf_action(env)


_ADAPTER: Any = None


def register_adapter(adapter: Any) -> None:
    global _ADAPTER
    _ADAPTER = adapter


# ---------------------------------------------------------------------------
# Reference tables (frozen Stage 2 raw + frozen Stage 3 derived tables)
# ---------------------------------------------------------------------------

FROZEN_NUMERIC_FIELDS = [
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
    "eval_money",
    "money",
]


def episodes_to_vectors(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, List[float]]]:
    """Group frozen episode rows by regime -> metric -> ordered float vector."""
    grouped: Dict[str, Dict[str, List[float]]] = {}
    for reg in REGIMES:
        regrows = sorted(
            (r for r in rows if r["regime"] == reg), key=lambda r: int(r["episode_index"])
        )
        require(len(regrows) == EPISODES_PER_REGIME, f"Wrong frozen coverage for {reg}")
        grouped[reg] = {
            field: [float(r[field]) for r in regrows] for field in FROZEN_NUMERIC_FIELDS
        }
    return grouped


def macro_from_vectors(vec: Dict[str, Dict[str, List[float]]], field: str = "money") -> float:
    return statistics.mean(statistics.mean(vec[reg][field]) for reg in REGIMES)


def load_stage3_reference() -> Dict[Tuple[str, int], Dict[str, Any]]:
    """{(method,C): {macro, per_regime_money, regime_metrics}} from frozen Stage 3."""
    seed_rows = csv_rows(STAGE3_DIR / "seed_level_scores.csv")
    regime_rows = csv_rows(STAGE3_DIR / "regime_level_scores.csv")
    ref: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for method in ["SC-FAC", "BF-T0.5"]:
        for C in CAPACITIES:
            seed = "" if method == "BF-T0.5" else str(SC_SEED)
            sr = [
                r
                for r in seed_rows
                if r["method"] == method and int(r["C"]) == C and r["training_seed"] == seed
            ]
            require(len(sr) == 1, f"Stage3 seed row not unique: {method} C{C}")
            rr = [
                r
                for r in regime_rows
                if r["method"] == method
                and int(r["C"]) == C
                and r["training_seed"] == seed
                and r["regime"] in REGIMES
            ]
            require(len(rr) == 12, f"Stage3 regime rows missing: {method} C{C}")
            by_reg = {r["regime"]: r for r in rr}
            ref[(method, C)] = dict(
                macro=float(sr[0]["macro12_money"]),
                per_regime_money={reg: float(by_reg[reg]["money"]) for reg in REGIMES},
                regime_rows=by_reg,
            )
    return ref
