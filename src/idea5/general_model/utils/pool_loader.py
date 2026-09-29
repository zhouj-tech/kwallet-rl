from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[4]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "src" / "ideaextra" / "data" / "pools"

REGIME_ORDER = [
    "US", "TLS", "LNS", "TLNS", "TPLS", "PLS",
    "UB", "TLB", "LNB", "TLNB", "TPLB", "PLB",
]

DEFAULT_MIX_EQ_MASTER = (
    "MIX12_EQ_US_TLS_LNS_TLNS_TPLS_PLS_UB_TLB_LNB_TLNB_TPLB_PLB_master_T1000.npy"
)
DEFAULT_MIX_EQ_VAL = (
    "MIX12_EQ_US_TLS_LNS_TLNS_TPLS_PLS_UB_TLB_LNB_TLNB_TPLB_PLB_val_T1000.npy"
)
DEFAULT_STATIC_EVAL_FILES = {regime: f"{regime}_static_eval_T1000.npy" for regime in REGIME_ORDER}


def resolve_data_root(data_root: str | None = None) -> Path:
    if data_root:
        return Path(data_root).expanduser().resolve()
    return DEFAULT_DATA_ROOT


def train_pool_file_for_regime(train_regime: str) -> str:
    if train_regime == "MIX12_EQ":
        return DEFAULT_MIX_EQ_MASTER
    if train_regime in REGIME_ORDER:
        return f"{train_regime}_static_master_T1000.npy"
    raise ValueError(f"Unknown train_regime={train_regime}. Use MIX12_EQ or one of {REGIME_ORDER}")


def resolve_pool_paths(train_regime: str, data_root: str | None = None) -> Dict[str, Any]:
    root = resolve_data_root(data_root)
    if train_regime == "MIX12_EQ":
        train_file = DEFAULT_MIX_EQ_MASTER
        val_file = DEFAULT_MIX_EQ_VAL
    else:
        train_file = train_pool_file_for_regime(train_regime)
        val_file = train_file
    return {
        "data_root": str(root),
        "train_pool_path": str(root / train_file),
        "val_pool_path": str(root / val_file),
        "test_pool_paths": {
            regime: str(root / file_name)
            for regime, file_name in DEFAULT_STATIC_EVAL_FILES.items()
        },
    }


def load_tx_pool(pool_path: str | Path, expected_steps: int) -> np.ndarray:
    path = Path(pool_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Pool file not found: {path}")
    if path.suffix == ".npy":
        tx_pool = np.load(path)
    elif path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            tx_pool = np.asarray(json.load(f), dtype=np.float32)
    else:
        raise ValueError(f"Unsupported pool format: {path}")

    if tx_pool.ndim != 2:
        raise ValueError(f"tx_pool must be 2D [episodes, steps], got shape={tx_pool.shape}")
    if tx_pool.shape[1] != int(expected_steps):
        raise ValueError(f"Expected steps={expected_steps}, got {tx_pool.shape[1]} in {path}")
    if not np.issubdtype(tx_pool.dtype, np.number):
        raise ValueError(f"tx_pool must contain numeric transaction values, got dtype={tx_pool.dtype}")
    if np.any(tx_pool < 0):
        raise ValueError(f"tx_pool contains negative transaction values: {path}")
    return tx_pool.astype(np.float32, copy=False)


def verify_data_integrity(pool_path: str | Path, expected_steps: int, label: str = "") -> bool:
    try:
        tx_pool = load_tx_pool(pool_path, expected_steps)
        pool_hash = hashlib.md5(tx_pool.tobytes()).hexdigest()
        print(f"[data] {label} shape={tx_pool.shape} md5={pool_hash} path={Path(pool_path).resolve()}")
        return True
    except Exception as exc:
        print(f"[data] failed {label}: {exc}")
        return False


def pool_fingerprint(pool_path: str | Path, expected_steps: int) -> Dict[str, Any]:
    tx_pool = load_tx_pool(pool_path, expected_steps)
    return {
        "path": str(Path(pool_path).expanduser().resolve()),
        "shape": [int(x) for x in tx_pool.shape],
        "dtype": str(tx_pool.dtype),
        "min": float(np.min(tx_pool)),
        "max": float(np.max(tx_pool)),
        "mean": float(np.mean(tx_pool)),
        "md5": hashlib.md5(tx_pool.tobytes()).hexdigest(),
    }


def build_pool_fingerprints(paths: Dict[str, Any], expected_steps: int) -> Dict[str, Any]:
    return {
        "train": pool_fingerprint(paths["train_pool_path"], expected_steps),
        "validation": pool_fingerprint(paths["val_pool_path"], expected_steps),
        "eval": {
            regime: pool_fingerprint(pool_path, expected_steps)
            for regime, pool_path in paths["test_pool_paths"].items()
        },
    }


def infer_t_max(train_pool: np.ndarray, explicit_t_max: float | None = None) -> float:
    if explicit_t_max is not None:
        if explicit_t_max <= 0:
            raise ValueError("--T_max must be positive when provided.")
        return float(explicit_t_max)
    return float(np.max(train_pool))


def load_train_val_pools(
    train_pool_path: str | Path,
    val_pool_path: str | Path,
    expected_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    return (
        load_tx_pool(train_pool_path, expected_steps),
        load_tx_pool(val_pool_path, expected_steps),
    )

