from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

THIS_FILE = Path(__file__).resolve()
IDEA5_ROOT = THIS_FILE.parents[2]
if str(IDEA5_ROOT) not in sys.path:
    sys.path.insert(0, str(IDEA5_ROOT))

from general_model_v2.utils.io_utils import DEFAULT_RESULT_ROOT, list_cross_regime_files, read_json
from general_model_v2.utils.metrics import confidence_interval_95, flatten_cross_regime_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate two-pool benchmark results.")
    parser.add_argument("--result_root", type=str, default=str(DEFAULT_RESULT_ROOT))
    return parser.parse_args()


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def to_float_or_nan(value: Any) -> float:
    try:
        if value == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def seed_level_rows(per_regime_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in per_regime_rows:
        key = (
            row.get("model", ""),
            row.get("threshold_mode", ""),
            row.get("C", ""),
            row.get("F", ""),
            row.get("T", ""),
            row.get("flush_levels", ""),
            row.get("num_flush_choices", ""),
            row.get("action_size", ""),
            row.get("factorized_policy_output_size", ""),
            row.get("reward_scale", ""),
            row.get("money_tau", ""),
            row.get("train_regime", ""),
            row.get("state_feature_mode", ""),
            row.get("mask_mode", ""),
            row.get("imitation_mode", "none"),
            row.get("imitation_episodes", 0),
            row.get("imitation_epochs", 0),
            row.get("imitation_reg_coef", 0.0),
            row.get("selected_eta", ""),
            row.get("seed", ""),
            row.get("scenario", ""),
        )
        grouped[key].append(row)

    metric_names = [
        "value_accept_ratio",
        "drops",
        "flushes",
        "money",
        "flush_A_count",
        "flush_B_count",
        "drop_A_count",
        "drop_B_count",
        "settled_A_value",
        "settled_B_value",
        "settle_accept_rate",
        "zero_flush_rate",
        "no_flush_rate",
        "full_flush_A_rate",
        "full_flush_B_rate",
    ]
    out: List[Dict[str, Any]] = []
    for key, rows in grouped.items():
        (
            model, threshold_mode, C, F, T, flush_levels, num_flush_choices, action_size,
            factorized_policy_output_size, reward_scale, money_tau,
            train_regime, state_feature_mode, mask_mode,
            imitation_mode, imitation_episodes, imitation_epochs,
            imitation_reg_coef, selected_eta,
            seed, scenario,
        ) = key
        value_accept = np.asarray([to_float_or_nan(r.get("value_accept_ratio", "")) for r in rows])
        row_out: Dict[str, Any] = {
            "model": model,
            "threshold_mode": threshold_mode,
            "C": C,
            "F": F,
            "T": T,
            "flush_levels": flush_levels,
            "num_flush_choices": num_flush_choices,
            "policy_output": factorized_policy_output_size if model != "flat_ppo" else action_size,
            "action_size": action_size,
            "factorized_policy_output_size": factorized_policy_output_size,
            "reward_scale": reward_scale,
            "money_tau": money_tau,
            "train_regime": train_regime,
            "state_feature_mode": state_feature_mode,
            "mask_mode": mask_mode,
            "imitation_mode": imitation_mode,
            "imitation_episodes": imitation_episodes,
            "imitation_epochs": imitation_epochs,
            "imitation_reg_coef": imitation_reg_coef,
            "selected_eta": selected_eta,
            "seed": seed,
            "scenario": scenario,
            "seed_mean_value_accept_ratio": float(np.nanmean(value_accept)),
            "seed_worst_regime_value_accept_ratio": float(np.nanmin(value_accept)),
            "num_test_regimes": len(rows),
        }
        for metric in metric_names:
            values = np.asarray([to_float_or_nan(r.get(metric, "")) for r in rows])
            if not np.all(np.isnan(values)):
                row_out[f"seed_mean_{metric}"] = float(np.nanmean(values))
        out.append(row_out)
    return out


def main_rows(seed_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        key = (
            row.get("model", ""),
            row.get("threshold_mode", ""),
            row.get("C", ""),
            row.get("F", ""),
            row.get("T", ""),
            row.get("flush_levels", ""),
            row.get("num_flush_choices", ""),
            row.get("action_size", ""),
            row.get("factorized_policy_output_size", ""),
            row.get("reward_scale", ""),
            row.get("money_tau", ""),
            row.get("train_regime", ""),
            row.get("state_feature_mode", ""),
            row.get("mask_mode", ""),
            row.get("imitation_mode", "none"),
            row.get("imitation_episodes", 0),
            row.get("imitation_epochs", 0),
            row.get("imitation_reg_coef", 0.0),
            row.get("selected_eta", ""),
        )
        grouped[key].append(row)

    metric_map = {
        "mean_value_accept_ratio": "seed_mean_value_accept_ratio",
        "worst_regime_value_accept_ratio": "seed_worst_regime_value_accept_ratio",
        "drops": "seed_mean_drops",
        "flushes": "seed_mean_flushes",
        "money": "seed_mean_money",
        "flush_A_count": "seed_mean_flush_A_count",
        "flush_B_count": "seed_mean_flush_B_count",
        "drop_A_count": "seed_mean_drop_A_count",
        "drop_B_count": "seed_mean_drop_B_count",
        "settled_A_value": "seed_mean_settled_A_value",
        "settled_B_value": "seed_mean_settled_B_value",
        "settle_accept_rate": "seed_mean_settle_accept_rate",
        "zero_flush_rate": "seed_mean_zero_flush_rate",
        "no_flush_rate": "seed_mean_no_flush_rate",
        "full_flush_A_rate": "seed_mean_full_flush_A_rate",
        "full_flush_B_rate": "seed_mean_full_flush_B_rate",
    }
    out: List[Dict[str, Any]] = []
    for key, rows in grouped.items():
        (
            model, threshold_mode, C, F, T, flush_levels, num_flush_choices, action_size,
            factorized_policy_output_size, reward_scale, money_tau,
            train_regime, state_feature_mode, mask_mode,
            imitation_mode, imitation_episodes, imitation_epochs,
            imitation_reg_coef, selected_eta,
        ) = key
        result: Dict[str, Any] = {
            "model": model,
            "threshold_mode": threshold_mode,
            "C": C,
            "F": F,
            "T": T,
            "flush_levels": flush_levels,
            "num_flush_choices": num_flush_choices,
            "policy_output": rows[0].get("policy_output", ""),
            "action_size": action_size,
            "factorized_policy_output_size": factorized_policy_output_size,
            "reward_scale": reward_scale,
            "money_tau": money_tau,
            "train_regime": train_regime,
            "state_feature_mode": state_feature_mode,
            "mask_mode": mask_mode,
            "imitation_mode": imitation_mode,
            "imitation_episodes": imitation_episodes,
            "imitation_epochs": imitation_epochs,
            "imitation_reg_coef": imitation_reg_coef,
            "selected_eta": selected_eta,
            "n_seeds": len({str(row.get("seed", "")) for row in rows if str(row.get("seed", ""))}),
            "seeds": ";".join(sorted({str(row.get("seed", "")) for row in rows if str(row.get("seed", ""))})),
            "n_runs": len(rows),
        }
        for out_name, seed_name in metric_map.items():
            values = [float(row[seed_name]) for row in rows if seed_name in row and row[seed_name] != ""]
            if not values:
                result[out_name] = ""
                result[f"{out_name}_ci95"] = ""
                continue
            result[out_name] = float(np.mean(values))
            ci = confidence_interval_95(values)
            result[f"{out_name}_ci95"] = "" if ci is None else ci
        out.append(result)
    return out


def main() -> None:
    args = parse_args()
    result_root = Path(args.result_root).expanduser().resolve()
    files = list_cross_regime_files(result_root)
    per_regime: List[Dict[str, Any]] = []
    for path in files:
        payload = read_json(path)
        per_regime.extend(flatten_cross_regime_result(payload, source_path=str(path)))
    seed_rows = seed_level_rows(per_regime)
    main_result_rows = main_rows(seed_rows)
    aggregate_dir = result_root / "aggregates"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    per_regime_path = aggregate_dir / "two_pool_per_regime_results.csv"
    seed_level_path = aggregate_dir / "two_pool_seed_level_results.csv"
    main_path = aggregate_dir / "two_pool_main_results.csv"
    write_csv(per_regime, per_regime_path)
    write_csv(seed_rows, seed_level_path)
    write_csv(main_result_rows, main_path)
    print(f"Wrote {per_regime_path} ({len(per_regime)} rows)")
    print(f"Wrote {seed_level_path} ({len(seed_rows)} rows)")
    print(f"Wrote {main_path} ({len(main_result_rows)} rows)")


if __name__ == "__main__":
    main()
