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

from general_model.utils.io_utils import (
    DEFAULT_RESULT_ROOT,
    list_cross_regime_files,
    read_json,
)
from general_model.utils.metrics import (
    confidence_interval_95,
    flatten_cross_regime_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate general collateral model benchmark results."
    )
    parser.add_argument(
        "--result_root",
        type=str,
        default=str(DEFAULT_RESULT_ROOT),
    )
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


def safe_get(mapping: Dict[str, Any], key: str, default: Any = "") -> Any:
    value = mapping.get(key, default)
    return default if value is None else value


def nested_get(
    mapping: Dict[str, Any],
    path: List[str],
    default: Any = "",
) -> Any:
    cur: Any = mapping
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return default if cur is None else cur


def policy_output_for_model(model: str, flush_levels: Any = 5) -> Any:
    model = str(model)
    try:
        num_flush_choices = int(float(flush_levels))
    except Exception:
        num_flush_choices = 5

    if model == "flat_ppo":
        return 2 * num_flush_choices

    if model in {"factorized_ac", "conditional_factorized_ac"}:
        return 2 + num_flush_choices

    if model in {"fixed_threshold", "grid_threshold"}:
        return ""

    return ""


def output_reduction_for_model(model: str, flush_levels: Any = 5) -> Any:
    policy_output = policy_output_for_model(model, flush_levels)

    if policy_output == "":
        return ""

    try:
        flat_output = float(2 * int(float(flush_levels)))
    except Exception:
        flat_output = 10.0
    return float(1.0 - float(policy_output) / flat_output)


def enrich_row_with_config(
    row: Dict[str, Any],
    payload: Dict[str, Any],
    source_path: str,
) -> Dict[str, Any]:
    config = payload.get("config", {})
    env_cfg = config.get("env", {})
    reward_cfg = config.get("reward", {})
    train_cfg = config.get("train", {})
    data_cfg = config.get("data", {})
    cond_cfg = config.get("conditional", {})
    imitation_cfg = config.get("imitation", {})

    model = (
        row.get("model")
        or payload.get("model_name")
        or config.get("model_name")
        or config.get("model_mode")
        or ""
    )

    enriched = dict(row)

    enriched["source_path"] = source_path

    enriched["model"] = model
    enriched["model_mode"] = (
        row.get("model_mode")
        or payload.get("model_mode")
        or config.get("model_mode")
        or model
    )

    enriched["scenario"] = (
        row.get("scenario")
        or payload.get("scenario")
        or config.get("scenario")
        or ""
    )

    enriched["seed"] = (
        row.get("seed")
        or payload.get("seed")
        or config.get("seed")
        or ""
    )

    enriched["train_regime"] = (
        row.get("train_regime")
        or payload.get("train_regime")
        or data_cfg.get("train_regime")
        or ""
    )

    enriched["C"] = row.get("C", env_cfg.get("C", ""))
    enriched["F"] = row.get("F", env_cfg.get("F", ""))
    enriched["T"] = row.get("T", env_cfg.get("T", ""))
    enriched["T_max"] = row.get("T_max", env_cfg.get("T_max", ""))
    enriched["flush_levels"] = row.get("flush_levels", env_cfg.get("flush_levels", 5))
    enriched["flush_grid"] = row.get("flush_grid", env_cfg.get("flush_grid", "uniform"))
    enriched["num_flush_choices"] = row.get(
        "num_flush_choices",
        env_cfg.get("num_flush_choices", enriched["flush_levels"]),
    )
    enriched["state_feature_mode"] = row.get(
        "state_feature_mode",
        env_cfg.get("state_feature_mode", "base"),
    )
    enriched["mask_mode"] = row.get("mask_mode", env_cfg.get("mask_mode", "none"))
    enriched["reward_scale"] = row.get("reward_scale", train_cfg.get("reward_scale", 1.0))
    enriched["imitation_mode"] = row.get(
        "imitation_mode",
        imitation_cfg.get("imitation_mode", imitation_cfg.get("mode", "none")),
    )
    enriched["imitation_episodes"] = row.get(
        "imitation_episodes",
        imitation_cfg.get("imitation_episodes", imitation_cfg.get("episodes", 0)),
    )
    enriched["imitation_epochs"] = row.get(
        "imitation_epochs",
        imitation_cfg.get("imitation_epochs", imitation_cfg.get("epochs", 0)),
    )
    enriched["threshold_advantage_mode"] = row.get(
        "threshold_advantage_mode",
        config.get("threshold_advantage_mode", "none"),
    )

    enriched["money_p"] = row.get("money_p", reward_cfg.get("money_p", ""))
    enriched["money_tau"] = row.get("money_tau", reward_cfg.get("money_tau", ""))
    enriched["drop_penalty"] = row.get(
        "drop_penalty",
        reward_cfg.get("drop_penalty", ""),
    )

    enriched["hidden_size"] = row.get(
        "hidden_size",
        train_cfg.get("hidden_size", ""),
    )
    enriched["learning_rate"] = row.get(
        "learning_rate",
        train_cfg.get("learning_rate", ""),
    )

    enriched["settle_embed_dim"] = row.get(
        "settle_embed_dim",
        cond_cfg.get("settle_embed_dim", ""),
    )
    enriched["conditional_hidden_size"] = row.get(
        "conditional_hidden_size",
        cond_cfg.get("conditional_hidden_size", ""),
    )

    enriched["threshold_mode"] = (
        row.get("threshold_mode")
        or config.get("threshold_mode")
        or ""
    )

    enriched["selection_method"] = (
        row.get("selection_method")
        or payload.get("selection_method")
        or config.get("selection_method")
        or ""
    )

    enriched["selected_eta"] = (
        row.get("selected_eta")
        or payload.get("selected_eta")
        or imitation_cfg.get("selected_eta")
        or config.get("eta")
        or ""
    )

    policy_output = policy_output_for_model(str(model), enriched["num_flush_choices"])
    output_reduction = output_reduction_for_model(str(model), enriched["num_flush_choices"])

    enriched["policy_output"] = policy_output
    enriched["output_reduction_vs_flat"] = output_reduction
    enriched["output_reduction_pct"] = (
        "" if output_reduction == "" else float(output_reduction) * 100.0
    )

    return enriched


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
            row.get("C", ""),
            row.get("F", ""),
            row.get("T", ""),
            row.get("flush_levels", ""),
            row.get("flush_grid", "uniform"),
            row.get("num_flush_choices", row.get("flush_levels", "")),
            row.get("state_feature_mode", "base"),
            row.get("mask_mode", "none"),
            row.get("reward_scale", 1.0),
            row.get("money_p", ""),
            row.get("money_tau", ""),
            row.get("drop_penalty", ""),
            row.get("train_regime", ""),
            row.get("hidden_size", ""),
            row.get("learning_rate", ""),
            row.get("settle_embed_dim", ""),
            row.get("conditional_hidden_size", ""),
            row.get("threshold_mode", ""),
            row.get("imitation_mode", "none"),
            row.get("imitation_episodes", 0),
            row.get("imitation_epochs", 0),
            row.get("selected_eta", ""),
            row.get("threshold_advantage_mode", "none"),
            row.get("seed", ""),
            row.get("scenario", ""),
        )
        grouped[key].append(row)

    out: List[Dict[str, Any]] = []

    for key, rows in grouped.items():
        (
            model,
            C,
            F,
            T,
            flush_levels,
            flush_grid,
            num_flush_choices,
            state_feature_mode,
            mask_mode,
            reward_scale,
            money_p,
            money_tau,
            drop_penalty,
            train_regime,
            hidden_size,
            learning_rate,
            settle_embed_dim,
            conditional_hidden_size,
            threshold_mode,
            imitation_mode,
            imitation_episodes,
            imitation_epochs,
            selected_eta,
            threshold_advantage_mode,
            seed,
            scenario,
        ) = key

        value_accept = np.asarray(
            [to_float_or_nan(r.get("value_accept_ratio", "")) for r in rows],
            dtype=float,
        )

        drops = np.asarray(
            [to_float_or_nan(r.get("drops", "")) for r in rows],
            dtype=float,
        )

        flushes = np.asarray(
            [to_float_or_nan(r.get("flushes", "")) for r in rows],
            dtype=float,
        )

        money = np.asarray(
            [to_float_or_nan(r.get("money", "")) for r in rows],
            dtype=float,
        )

        policy_discards = np.asarray(
            [to_float_or_nan(r.get("policy_discards", "")) for r in rows],
            dtype=float,
        )

        capacity_drops = np.asarray(
            [to_float_or_nan(r.get("capacity_drops", "")) for r in rows],
            dtype=float,
        )

        diagnostic_arrays = {
            "settle_accept_rate": np.asarray(
                [to_float_or_nan(r.get("settle_accept_rate", "")) for r in rows],
                dtype=float,
            ),
            "settle_reject_rate": np.asarray(
                [to_float_or_nan(r.get("settle_reject_rate", "")) for r in rows],
                dtype=float,
            ),
            "zero_flush_rate": np.asarray(
                [to_float_or_nan(r.get("zero_flush_rate", "")) for r in rows],
                dtype=float,
            ),
            "full_flush_rate": np.asarray(
                [to_float_or_nan(r.get("full_flush_rate", "")) for r in rows],
                dtype=float,
            ),
            "middle_flush_rate": np.asarray(
                [to_float_or_nan(r.get("middle_flush_rate", "")) for r in rows],
                dtype=float,
            ),
            "mean_flush_fraction": np.asarray(
                [to_float_or_nan(r.get("mean_flush_fraction", "")) for r in rows],
                dtype=float,
            ),
            "mean_flush_action": np.asarray(
                [to_float_or_nan(r.get("mean_flush_action", "")) for r in rows],
                dtype=float,
            ),
        }

        source_paths = sorted({str(r.get("source_path", "")) for r in rows if r.get("source_path", "")})

        first = rows[0]
        policy_output = first.get("policy_output", "")
        output_reduction_vs_flat = first.get("output_reduction_vs_flat", "")
        output_reduction_pct = first.get("output_reduction_pct", "")

        row_out: Dict[str, Any] = {
            "model": model,
            "C": C,
            "F": F,
            "T": T,
            "flush_levels": flush_levels,
            "flush_grid": flush_grid,
            "num_flush_choices": num_flush_choices,
            "state_feature_mode": state_feature_mode,
            "mask_mode": mask_mode,
            "reward_scale": reward_scale,
            "money_p": money_p,
            "money_tau": money_tau,
            "drop_penalty": drop_penalty,
            "train_regime": train_regime,
            "hidden_size": hidden_size,
            "learning_rate": learning_rate,
            "settle_embed_dim": settle_embed_dim,
            "conditional_hidden_size": conditional_hidden_size,
            "threshold_mode": threshold_mode,
            "imitation_mode": imitation_mode,
            "imitation_episodes": imitation_episodes,
            "imitation_epochs": imitation_epochs,
            "selected_eta": selected_eta,
            "threshold_advantage_mode": threshold_advantage_mode,
            "seed": seed,
            "scenario": scenario,
            "policy_output": policy_output,
            "output_reduction_vs_flat": output_reduction_vs_flat,
            "output_reduction_pct": output_reduction_pct,
            "seed_mean_value_accept_ratio": float(np.nanmean(value_accept)),
            "seed_worst_regime_value_accept_ratio": float(np.nanmin(value_accept)),
            "seed_mean_drops": float(np.nanmean(drops)),
            "seed_mean_flushes": float(np.nanmean(flushes)),
            "seed_mean_money": float(np.nanmean(money)),
            "num_test_regimes": len(rows),
            "source_paths": ";".join(source_paths),
        }

        if not np.all(np.isnan(policy_discards)):
            row_out["seed_mean_policy_discards"] = float(np.nanmean(policy_discards))

        if not np.all(np.isnan(capacity_drops)):
            row_out["seed_mean_capacity_drops"] = float(np.nanmean(capacity_drops))

        for metric, values in diagnostic_arrays.items():
            if not np.all(np.isnan(values)):
                row_out[f"seed_mean_{metric}"] = float(np.nanmean(values))

        out.append(row_out)

    return out


def main_rows(seed_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)

    for row in seed_rows:
        key = (
            row.get("model", ""),
            row.get("C", ""),
            row.get("F", ""),
            row.get("T", ""),
            row.get("flush_levels", ""),
            row.get("flush_grid", "uniform"),
            row.get("num_flush_choices", row.get("flush_levels", "")),
            row.get("state_feature_mode", "base"),
            row.get("mask_mode", "none"),
            row.get("reward_scale", 1.0),
            row.get("money_p", ""),
            row.get("money_tau", ""),
            row.get("drop_penalty", ""),
            row.get("train_regime", ""),
            row.get("hidden_size", ""),
            row.get("learning_rate", ""),
            row.get("settle_embed_dim", ""),
            row.get("conditional_hidden_size", ""),
            row.get("threshold_mode", ""),
            row.get("imitation_mode", "none"),
            row.get("imitation_episodes", 0),
            row.get("imitation_epochs", 0),
            row.get("selected_eta", ""),
            row.get("threshold_advantage_mode", "none"),
        )
        grouped[key].append(row)

    metric_map = {
        "mean_value_accept_ratio": "seed_mean_value_accept_ratio",
        "worst_regime_value_accept_ratio": "seed_worst_regime_value_accept_ratio",
        "drops": "seed_mean_drops",
        "flushes": "seed_mean_flushes",
        "money": "seed_mean_money",
        "policy_discards": "seed_mean_policy_discards",
        "capacity_drops": "seed_mean_capacity_drops",
        "settle_accept_rate": "seed_mean_settle_accept_rate",
        "settle_reject_rate": "seed_mean_settle_reject_rate",
        "zero_flush_rate": "seed_mean_zero_flush_rate",
        "full_flush_rate": "seed_mean_full_flush_rate",
        "middle_flush_rate": "seed_mean_middle_flush_rate",
        "mean_flush_fraction": "seed_mean_mean_flush_fraction",
        "mean_flush_action": "seed_mean_mean_flush_action",
    }

    out: List[Dict[str, Any]] = []

    for key, rows in grouped.items():
        (
            model,
            C,
            F,
            T,
            flush_levels,
            flush_grid,
            num_flush_choices,
            state_feature_mode,
            mask_mode,
            reward_scale,
            money_p,
            money_tau,
            drop_penalty,
            train_regime,
            hidden_size,
            learning_rate,
            settle_embed_dim,
            conditional_hidden_size,
            threshold_mode,
            imitation_mode,
            imitation_episodes,
            imitation_epochs,
            selected_eta,
            threshold_advantage_mode,
        ) = key

        unique_seeds = sorted({str(row.get("seed", "")) for row in rows if str(row.get("seed", "")) != ""})
        scenarios = sorted({str(row.get("scenario", "")) for row in rows if str(row.get("scenario", "")) != ""})
        source_paths = sorted(
            {
                path
                for row in rows
                for path in str(row.get("source_paths", "")).split(";")
                if path
            }
        )

        first = rows[0]

        result: Dict[str, Any] = {
            "model": model,
            "C": C,
            "F": F,
            "T": T,
            "flush_levels": flush_levels,
            "flush_grid": flush_grid,
            "num_flush_choices": num_flush_choices,
            "state_feature_mode": state_feature_mode,
            "mask_mode": mask_mode,
            "reward_scale": reward_scale,
            "money_p": money_p,
            "money_tau": money_tau,
            "drop_penalty": drop_penalty,
            "train_regime": train_regime,
            "hidden_size": hidden_size,
            "learning_rate": learning_rate,
            "settle_embed_dim": settle_embed_dim,
            "conditional_hidden_size": conditional_hidden_size,
            "threshold_mode": threshold_mode,
            "imitation_mode": imitation_mode,
            "imitation_episodes": imitation_episodes,
            "imitation_epochs": imitation_epochs,
            "selected_eta": selected_eta,
            "threshold_advantage_mode": threshold_advantage_mode,
            "policy_output": first.get("policy_output", ""),
            "output_reduction_vs_flat": first.get("output_reduction_vs_flat", ""),
            "output_reduction_pct": first.get("output_reduction_pct", ""),
            "n_seeds": len(unique_seeds),
            "seeds": ";".join(unique_seeds),
            "n_runs": len(rows),
            "scenarios": ";".join(scenarios),
            "source_paths": ";".join(source_paths),
        }

        for out_name, seed_name in metric_map.items():
            values = [
                float(row[seed_name])
                for row in rows
                if seed_name in row and row[seed_name] != ""
            ]

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

        flattened_rows = flatten_cross_regime_result(
            payload,
            source_path=str(path),
        )

        for row in flattened_rows:
            enriched = enrich_row_with_config(
                row=row,
                payload=payload,
                source_path=str(path),
            )
            per_regime.append(enriched)

    seed_rows = seed_level_rows(per_regime)
    main_result_rows = main_rows(seed_rows)

    aggregate_dir = result_root / "aggregates"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    per_regime_path = aggregate_dir / "general_model_per_regime_results.csv"
    seed_level_path = aggregate_dir / "general_model_seed_level_results.csv"
    main_path = aggregate_dir / "general_model_main_results.csv"

    write_csv(per_regime, per_regime_path)
    write_csv(seed_rows, seed_level_path)
    write_csv(main_result_rows, main_path)

    print(f"Wrote {per_regime_path} ({len(per_regime)} rows)")
    print(f"Wrote {seed_level_path} ({len(seed_rows)} rows)")
    print(f"Wrote {main_path} ({len(main_result_rows)} rows)")


if __name__ == "__main__":
    main()
