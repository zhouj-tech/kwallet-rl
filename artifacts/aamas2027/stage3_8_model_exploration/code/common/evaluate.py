"""Deterministic deployment evaluation pipeline (zero BF calls by construction).

This module is the deployment path: it must never import common.teacher or any
BF helper. Checkpoint loading + argmax policy evaluation on frozen pools or
SCFAC-DESIGN-v3 panels.

Money identity (frozen B0 adapter, mandatory): per episode,
    eval_money == settled_value - 10 * executed_flushes
with money_p = 1.0 and money_tau = 10.0. The identity is asserted for every
episode in every evaluation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .config import (
    ALPHA_DROP,
    BETA_FLUSH,
    C,
    F,
    K,
    MAX_STEPS,
    MAX_TRANSACTION,
    MONEY_TAU,
    episode_fingerprint,
)
from .frozen_env import KWalletEnv
from .ppo import ConditionalFactorizedPPOAgent


def money_identity_ok(settled: float, flushes: int, eval_money: float) -> bool:
    return abs(eval_money - (settled - MONEY_TAU * flushes)) < 1e-6


def make_eval_env(capacity: Optional[float] = None) -> KWalletEnv:
    """Fresh evaluation environment at the requested capacity (default frozen C=800).

    Evaluation episodes run in separate environments; capacity must be threaded
    from the trained policy's run capacity so C1200 policies are evaluated on
    C1200 semantics (wallet size w = 1200/24 = 50)."""
    return KWalletEnv(
        C=C if capacity is None else float(capacity), k=K, F=F,
        max_transaction=MAX_TRANSACTION, max_steps=MAX_STEPS,
        seed=0, enable_shaping=False, alpha_drop=ALPHA_DROP, beta_flush=BETA_FLUSH,
    )


def evaluate_episode(agent: ConditionalFactorizedPPOAgent, env: KWalletEnv,
                     tx_stream: np.ndarray) -> Dict[str, float]:
    """Run one deterministic (argmax) episode; returns per-episode metrics."""
    env.reset(tx_stream=tx_stream)
    done = False
    accepted_count = 0
    total_requested_value = 0.0
    total_tx_count = 0
    while not done:
        settle_action, flush_action = agent.argmax_action(env._get_state())
        action = settle_action * (env.k + 1) + flush_action
        _, _, done, info = env.step(action)
        total_tx_count += 1
        total_requested_value += float(info["tx"])
        if info["accepted"]:
            accepted_count += 1

    m = env.get_metrics()
    settled = float(m["settled"])
    flushes = int(m["flushes"])
    drops = int(m["drops"])
    oversize = int(m["oversize_drops"])
    insufficient = int(m["insufficient_drops"])
    eval_money = settled - MONEY_TAU * flushes
    assert money_identity_ok(settled, flushes, eval_money), (
        f"Money identity violated: {eval_money} != {settled} - {MONEY_TAU}*{flushes}"
    )
    return {
        "settled": settled,
        "drops": drops,
        "oversize_drops": oversize,
        "insufficient_drops": insufficient,
        "flushes": flushes,
        "accepted_count": accepted_count,
        "total_tx_count": total_tx_count,
        "total_requested_value": total_requested_value,
        "value_accept_ratio": settled / total_requested_value if total_requested_value > 0 else 0.0,
        "count_accept_ratio": accepted_count / total_tx_count if total_tx_count > 0 else 0.0,
        "eval_money": eval_money,
        "utilization": float(m["utilization"]),
        "avg_tx_value": float(m["avg_tx_value"]),
        "drop_rate": float(m["drop_rate"]),
    }


def evaluate_agent_on_pool(agent: ConditionalFactorizedPPOAgent, pool: np.ndarray,
                           num_episodes: Optional[int] = None,
                           episode_offset: int = 0,
                           capacity: Optional[float] = None) -> List[Dict[str, float]]:
    """Deterministic eval over consecutive pool rows, one fresh frozen env per episode."""
    n = pool.shape[0] if num_episodes is None else min(num_episodes, pool.shape[0] - episode_offset)
    results: List[Dict[str, float]] = []
    for ep in range(episode_offset, episode_offset + n):
        env = make_eval_env(capacity=capacity)
        results.append(evaluate_episode(agent, env, pool[ep]))
    return results


def panel_summary(per_episode: List[Dict[str, float]]) -> Dict[str, float]:
    """Equal-weight mean/std over a pool's episodes."""
    keys = ["settled", "drops", "flushes", "eval_money", "value_accept_ratio",
            "count_accept_ratio", "oversize_drops", "insufficient_drops"]
    out: Dict[str, float] = {}
    for key in keys:
        vals = np.array([e[key] for e in per_episode], dtype=np.float64)
        out[f"{key}_mean"] = float(vals.mean())
        out[f"{key}_std"] = float(vals.std())
    return out


def evaluate_panel(agent: ConditionalFactorizedPPOAgent,
                   panel_pools: Dict[str, np.ndarray],
                   episodes_per_regime: Optional[int] = None,
                   include_fingerprints: bool = False,
                   capacity: Optional[float] = None) -> Dict[str, Dict]:
    """Evaluate on {regime: pool_array}. Returns per-regime summaries + macro12."""
    per_regime: Dict[str, Dict] = {}
    for regime, pool in panel_pools.items():
        eps = evaluate_agent_on_pool(agent, pool, num_episodes=episodes_per_regime,
                                     capacity=capacity)
        s = panel_summary(eps)
        s["episodes"] = len(eps)
        if include_fingerprints:
            s["episode_fingerprints"] = [episode_fingerprint(pool[i]) for i in range(len(eps))]
        per_regime[regime] = s
    regime_names = list(per_regime.keys())   # snapshot before scalar keys are added
    macro12 = float(np.mean([per_regime[r]["eval_money_mean"] for r in regime_names]))
    per_regime["macro12_mean_money"] = macro12
    per_regime["macro12_value_accept_ratio"] = float(
        np.mean([per_regime[r]["value_accept_ratio_mean"] for r in regime_names])
    )
    return per_regime


def load_panel_from_release(release_root: Path, split: str,
                            regimes: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """Load {regime: pool} for a SCFAC-DESIGN-v3 split (npz per split)."""
    path = release_root / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Release split missing: {path}")
    data = np.load(path)
    pools: Dict[str, np.ndarray] = {}
    for key in data.files:
        if key.endswith("__streams"):
            regime = key[: -len("__streams")]
            if regimes is None or regime in regimes:
                pools[regime] = data[key]
    return pools


def load_checkpoint_into_agent(ckpt_path: Path, model, agent_state: str = "model") -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
