from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

THIS_FILE = Path(__file__).resolve()
IDEA5_ROOT = THIS_FILE.parents[2]
if str(IDEA5_ROOT) not in sys.path:
    sys.path.insert(0, str(IDEA5_ROOT))

from general_model_v2.utils.pool_loader import (
    DEFAULT_DATA_ROOT,
    REGIME_ORDER,
    load_v1_value_pool,
    resolve_v1_pool_paths,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate two-pool transaction pools.")
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--type_prob_A", type=float, default=0.5)
    parser.add_argument("--v1_data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_DATA_ROOT))
    return parser.parse_args()


def type_balance(types: np.ndarray) -> Dict[str, float]:
    return {
        "type_A_ratio": float(np.mean(types == 0)),
        "type_B_ratio": float(np.mean(types == 1)),
    }


def write_two_pool_file(
    values: np.ndarray,
    rng: np.random.Generator,
    type_prob_A: float,
    output_path: Path,
    source_path: str,
) -> Dict[str, Any]:
    types = (rng.random(values.shape) >= float(type_prob_A)).astype(np.int64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, values=values.astype(np.float32), types=types)
    balance = type_balance(types)
    row = {
        "output_path": str(output_path.resolve()),
        "source_v1_pool_path": str(Path(source_path).expanduser().resolve()),
        "values_shape": [int(x) for x in values.shape],
        "types_shape": [int(x) for x in types.shape],
        **balance,
    }
    print(
        f"[pool] {output_path.name} values={values.shape} "
        f"type_A_ratio={balance['type_A_ratio']:.6f} "
        f"type_B_ratio={balance['type_B_ratio']:.6f}"
    )
    return row


def main() -> None:
    args = parse_args()
    if args.T <= 0:
        raise ValueError("--T must be positive.")
    if not 0.0 <= args.type_prob_A <= 1.0:
        raise ValueError("--type_prob_A must be in [0, 1].")

    rng = np.random.default_rng(int(args.seed))
    output_dir = Path(args.output_dir).expanduser().resolve()
    v1_paths = resolve_v1_pool_paths(args.v1_data_root)

    metadata: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "T": int(args.T),
        "type_assignment_seed": int(args.seed),
        "type_prob_A": float(args.type_prob_A),
        "source_v1_data_root": v1_paths["data_root"],
        "output_dir": str(output_dir),
        "pools": {},
    }

    train_values = load_v1_value_pool(v1_paths["train_pool_path"], args.T)
    metadata["pools"]["train"] = write_two_pool_file(
        values=train_values,
        rng=rng,
        type_prob_A=float(args.type_prob_A),
        output_path=output_dir / "MIX12_EQ_two_pool_train_T1000.npz",
        source_path=v1_paths["train_pool_path"],
    )

    val_values = load_v1_value_pool(v1_paths["val_pool_path"], args.T)
    metadata["pools"]["val"] = write_two_pool_file(
        values=val_values,
        rng=rng,
        type_prob_A=float(args.type_prob_A),
        output_path=output_dir / "MIX12_EQ_two_pool_val_T1000.npz",
        source_path=v1_paths["val_pool_path"],
    )

    metadata["pools"]["eval"] = {}
    for regime in REGIME_ORDER:
        source_path = v1_paths["test_pool_paths"][regime]
        values = load_v1_value_pool(source_path, args.T)
        metadata["pools"]["eval"][regime] = write_two_pool_file(
            values=values,
            rng=rng,
            type_prob_A=float(args.type_prob_A),
            output_path=output_dir / f"{regime}_two_pool_test_T1000.npz",
            source_path=source_path,
        )

    metadata_path = output_dir / "two_pool_pool_metadata_T1000.json"
    save_json(metadata, metadata_path)
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
