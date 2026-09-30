"""Versioned transaction-pool construction, caching, manifest and hashing.

Builds (or loads cached) pools for the paper protocol:
  * MIX12_EQ training pool  (5000 episodes, as-equal-as-possible across 12)
  * separate MIX12_EQ validation pool
  * one static 200-episode pool per regime (2400 evaluation episodes)

Seed namespaces are separated (train / val / eval) exactly as the source
generator. Every pool is content-hashed and recorded in a manifest so runs are
traceable. Pools are cached under ``data/pools_v1/`` (gitignored) and can be
regenerated deterministically.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .regimes import (
    DEFAULT_GEN_CONFIG,
    GENERATOR_VERSION,
    REGIME_ORDER,
    generate_mixed_equal_pool,
    generate_regime_pool,
)

CACHE_DIR = Path("data/pools_v1")


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(a.tobytes()).hexdigest()


def pool_stats(pool: np.ndarray) -> Dict[str, float]:
    flat = pool.reshape(-1).astype(np.float64)
    return {
        "episodes": int(pool.shape[0]),
        "steps": int(pool.shape[1]),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "p50": float(np.percentile(flat, 50)),
        "p95": float(np.percentile(flat, 95)),
        "p99": float(np.percentile(flat, 99)),
        "max": float(np.max(flat)),
    }


@dataclass
class PoolBundle:
    train: np.ndarray
    val: np.ndarray
    eval_pools: Dict[str, np.ndarray]
    manifest: Dict

    def eval_all(self) -> np.ndarray:
        """Concatenate per-regime eval pools in REGIME_ORDER (2400 x T)."""
        return np.concatenate([self.eval_pools[r] for r in REGIME_ORDER], axis=0)


def build_pools(
    episode_length: int = 1000,
    train_episodes: int = 5000,
    val_episodes: int = 300,
    eval_per_regime: int = 200,
    base_seed: int = 532,
    calibration_sample_size: int = 200_000,
    cache_dir: Optional[Path] = None,
    force_regenerate: bool = False,
) -> PoolBundle:
    cache_dir = CACHE_DIR if cache_dir is None else Path(cache_dir)
    tag = (f"{GENERATOR_VERSION}_T{episode_length}_b{base_seed}_"
           f"tr{train_episodes}_va{val_episodes}_ev{eval_per_regime}")
    run_dir = cache_dir / tag
    manifest_path = run_dir / "manifest.json"

    if not force_regenerate and manifest_path.exists():
        return _load(run_dir)

    train_seed = base_seed + DEFAULT_GEN_CONFIG["train_seed_offset"]
    eval_seed = base_seed + DEFAULT_GEN_CONFIG["eval_seed_offset"]
    val_seed = base_seed + DEFAULT_GEN_CONFIG["val_seed_offset"]

    train, train_masks, train_counts = generate_mixed_equal_pool(
        REGIME_ORDER, train_episodes, episode_length, train_seed,
        calibration_sample_size)
    val, _, _ = generate_mixed_equal_pool(
        REGIME_ORDER, val_episodes, episode_length, val_seed,
        calibration_sample_size)

    eval_pools: Dict[str, np.ndarray] = {}
    for i, regime in enumerate(REGIME_ORDER):
        # static eval pools use the eval namespace; per-regime cursor offset
        pool, _ = generate_regime_pool(
            regime, eval_per_regime, episode_length,
            eval_seed + i * 100_000, calibration_sample_size)
        eval_pools[regime] = pool

    manifest = {
        "generator_version": GENERATOR_VERSION,
        "episode_length": episode_length,
        "base_seed": base_seed,
        "train_seed": train_seed,
        "val_seed": val_seed,
        "eval_seed": eval_seed,
        "train_episodes": train_episodes,
        "val_episodes": val_episodes,
        "eval_per_regime": eval_per_regime,
        "calibration_sample_size": calibration_sample_size,
        "regime_order": REGIME_ORDER,
        "train_counts": train_counts,
        "hashes": {
            "train": sha256_array(train),
            "val": sha256_array(val),
            **{f"eval_{r}": sha256_array(eval_pools[r]) for r in REGIME_ORDER},
        },
        "stats": {
            "train": pool_stats(train),
            "val": pool_stats(val),
            **{f"eval_{r}": pool_stats(eval_pools[r]) for r in REGIME_ORDER},
        },
    }

    _save(run_dir, train, val, eval_pools, manifest)
    return PoolBundle(train=train, val=val, eval_pools=eval_pools, manifest=manifest)


def _save(run_dir: Path, train, val, eval_pools, manifest) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "train.npy", train)
    np.save(run_dir / "val.npy", val)
    for r, p in eval_pools.items():
        np.save(run_dir / f"eval_{r}.npy", p)
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def _load(run_dir: Path) -> PoolBundle:
    with open(run_dir / "manifest.json") as f:
        manifest = json.load(f)
    train = np.load(run_dir / "train.npy")
    val = np.load(run_dir / "val.npy")
    eval_pools = {r: np.load(run_dir / f"eval_{r}.npy") for r in REGIME_ORDER}
    return PoolBundle(train=train, val=val, eval_pools=eval_pools, manifest=manifest)
