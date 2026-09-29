from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[4]
V2_ROOT = THIS_FILE.parents[1]
DEFAULT_DATA_ROOT = V2_ROOT / "data" / "pools"
V1_DATA_ROOT = PROJECT_ROOT / "src" / "ideaextra" / "data" / "pools"

REGIME_ORDER = [
    "US", "TLS", "LNS", "TLNS", "TPLS", "PLS",
    "UB", "TLB", "LNB", "TLNB", "TPLB", "PLB",
]

V1_MIX_EQ_MASTER = (
    "MIX12_EQ_US_TLS_LNS_TLNS_TPLS_PLS_UB_TLB_LNB_TLNB_TPLB_PLB_master_T1000.npy"
)
V1_MIX_EQ_VAL = (
    "MIX12_EQ_US_TLS_LNS_TLNS_TPLS_PLS_UB_TLB_LNB_TLNB_TPLB_PLB_val_T1000.npy"
)
V1_STATIC_EVAL_FILES = {regime: f"{regime}_static_eval_T1000.npy" for regime in REGIME_ORDER}


def resolve_data_root(data_root: str | None = None) -> Path:
    if data_root:
        return Path(data_root).expanduser().resolve()
    return DEFAULT_DATA_ROOT


def resolve_v1_data_root(data_root: str | None = None) -> Path:
    if data_root:
        return Path(data_root).expanduser().resolve()
    return V1_DATA_ROOT


def resolve_v1_pool_paths(v1_data_root: str | None = None) -> Dict[str, Any]:
    root = resolve_v1_data_root(v1_data_root)
    return {
        "data_root": str(root),
        "train_pool_path": str(root / V1_MIX_EQ_MASTER),
        "val_pool_path": str(root / V1_MIX_EQ_VAL),
        "test_pool_paths": {
            regime: str(root / file_name)
            for regime, file_name in V1_STATIC_EVAL_FILES.items()
        },
    }


def resolve_pool_paths(train_regime: str, data_root: str | None = None) -> Dict[str, Any]:
    if train_regime != "MIX12_EQ":
        raise ValueError("general_model_v2 currently supports --train_regime MIX12_EQ.")
    root = resolve_data_root(data_root)
    return {
        "data_root": str(root),
        "train_pool_path": str(root / "MIX12_EQ_two_pool_train_T1000.npz"),
        "val_pool_path": str(root / "MIX12_EQ_two_pool_val_T1000.npz"),
        "test_pool_paths": {
            regime: str(root / f"{regime}_two_pool_test_T1000.npz")
            for regime in REGIME_ORDER
        },
    }


def load_v1_value_pool(pool_path: str | Path, expected_steps: int) -> np.ndarray:
    path = Path(pool_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Pool file not found: {path}")
    values = np.load(path)
    if values.ndim != 2:
        raise ValueError(f"value pool must be 2D [episodes, T], got {values.shape}")
    if values.shape[1] != int(expected_steps):
        raise ValueError(f"Expected T={expected_steps}, got {values.shape[1]} in {path}")
    if np.any(values < 0):
        raise ValueError(f"value pool contains negative transactions: {path}")
    return values.astype(np.float32, copy=False)


def load_two_pool(pool_path: str | Path, expected_steps: int) -> Dict[str, np.ndarray]:
    path = Path(pool_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Two-pool file not found: {path}")
    if path.suffix != ".npz":
        raise ValueError(f"Two-pool pools must be .npz, got {path}")
    with np.load(path) as data:
        values = np.asarray(data["values"], dtype=np.float32)
        types = np.asarray(data["types"], dtype=np.int64)
    if values.ndim != 2 or types.ndim != 2:
        raise ValueError(f"values/types must be 2D, got {values.shape} and {types.shape}")
    if values.shape != types.shape:
        raise ValueError(f"values/types shape mismatch: {values.shape} vs {types.shape}")
    if values.shape[1] != int(expected_steps):
        raise ValueError(f"Expected T={expected_steps}, got {values.shape[1]} in {path}")
    if np.any(values < 0):
        raise ValueError(f"values contain negative transactions: {path}")
    if not np.all(np.isin(types, [0, 1])):
        raise ValueError(f"types must contain only 0 or 1: {path}")
    return {"values": values, "types": types}


def pool_fingerprint(pool_path: str | Path, expected_steps: int) -> Dict[str, Any]:
    pool = load_two_pool(pool_path, expected_steps)
    values = pool["values"]
    types = pool["types"]
    payload_bytes = values.tobytes() + types.tobytes()
    type_A_ratio = float(np.mean(types == 0))
    type_B_ratio = float(np.mean(types == 1))
    return {
        "path": str(Path(pool_path).expanduser().resolve()),
        "values_shape": [int(x) for x in values.shape],
        "types_shape": [int(x) for x in types.shape],
        "values_dtype": str(values.dtype),
        "types_dtype": str(types.dtype),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "type_A_ratio": type_A_ratio,
        "type_B_ratio": type_B_ratio,
        "md5": hashlib.md5(payload_bytes).hexdigest(),
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


def infer_t_max(train_pool: Dict[str, np.ndarray], explicit_t_max: float | None = None) -> float:
    if explicit_t_max is not None:
        if explicit_t_max <= 0:
            raise ValueError("--T_max must be positive when provided.")
        return float(explicit_t_max)
    return float(np.max(train_pool["values"]))


def load_train_val_pools(
    train_pool_path: str | Path,
    val_pool_path: str | Path,
    expected_steps: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    return (
        load_two_pool(train_pool_path, expected_steps),
        load_two_pool(val_pool_path, expected_steps),
    )


def save_json(payload: Dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
