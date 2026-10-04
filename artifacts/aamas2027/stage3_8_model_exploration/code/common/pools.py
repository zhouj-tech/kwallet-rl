"""Frozen pool access with mandatory SHA256 verification.

An inaccessible pool or a hash mismatch blocks execution (raises) - it never
falls back to a substitute file.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .config import (
    FROZEN_STATIC_EVAL_POOL_DIR,
    FROZEN_TRAINING_POOL,
    FROZEN_VAL_POOL,
    NEW12_ROOT,
    sha256_file,
)
from .frozen_kwallet_ideaextra_generator import REGIME_ORDER


class PoolVerificationError(RuntimeError):
    pass


def load_frozen_pool(path: Path, expected_sha256: str | None = None,
                     expected_shape: Tuple[int, int] | None = None) -> np.ndarray:
    if not path.exists():
        raise PoolVerificationError(f"Frozen pool missing: {path}")
    if expected_sha256 is not None:
        actual = sha256_file(path)
        if actual != expected_sha256:
            raise PoolVerificationError(
                f"Frozen pool SHA256 mismatch for {path}: expected {expected_sha256}, got {actual}"
            )
    arr = np.load(path)
    if arr.dtype != np.int32:
        raise PoolVerificationError(f"Frozen pool {path} dtype {arr.dtype} != int32")
    if expected_shape is not None and tuple(arr.shape) != tuple(expected_shape):
        raise PoolVerificationError(
            f"Frozen pool {path} shape {arr.shape} != expected {expected_shape}"
        )
    arr = np.ascontiguousarray(arr)
    arr.setflags(write=False)  # read-only, like the frozen adapter
    return arr


def load_training_pool(expected_sha256: str | None = None) -> np.ndarray:
    return load_frozen_pool(FROZEN_TRAINING_POOL, expected_sha256, (5000, 1000))


def load_historical_val_pool(expected_sha256: str | None = None) -> np.ndarray:
    return load_frozen_pool(FROZEN_VAL_POOL, expected_sha256, (300, 1000))


def static_eval_pool_name(regime: str) -> str:
    return f"{regime}_static_eval_T1000.npy"


def load_static_eval_pool(regime: str) -> np.ndarray:
    return load_frozen_pool(FROZEN_STATIC_EVAL_POOL_DIR / static_eval_pool_name(regime), None, (200, 1000))


def load_new12_pool(regime: str) -> np.ndarray:
    return load_frozen_pool(NEW12_ROOT / f"{regime}_NEW12_v1_T1000.npy", None, (200, 1000))


def all_historical_pools() -> List[Tuple[str, np.ndarray]]:
    """Every historical pool used in overlap audits (training, val, 12 static evals, NEW12)."""
    out: List[Tuple[str, np.ndarray]] = [
        ("historical_train_master", load_training_pool()),
        ("historical_val", load_historical_val_pool()),
    ]
    for r in REGIME_ORDER:
        out.append((f"historical_static_eval/{r}", load_static_eval_pool(r)))
        out.append((f"NEW12/{r}", load_new12_pool(r)))
    return out
