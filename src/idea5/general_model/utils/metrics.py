from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List

import numpy as np


def _is_sequence_metric(value: Any) -> bool:
    return isinstance(value, (list, tuple, np.ndarray)) and not isinstance(value, (str, bytes))


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def summarize_episode_metrics(all_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not all_results:
        return {}
    summary: Dict[str, Dict[str, Any]] = {}
    keys = sorted({key for row in all_results for key in row.keys()})
    for metric in keys:
        present_values = [row.get(metric) for row in all_results if metric in row]
        if not present_values:
            continue

        if _is_sequence_metric(present_values[0]):
            arrays = [np.asarray(value, dtype=float) for value in present_values]
            if not arrays or len({arr.shape for arr in arrays}) != 1:
                continue
            arr = np.stack(arrays, axis=0)
            summary[metric] = {
                "mean": _to_jsonable(np.mean(arr, axis=0)),
                "std": _to_jsonable(np.std(arr, axis=0)),
                "min": _to_jsonable(np.min(arr, axis=0)),
                "max": _to_jsonable(np.max(arr, axis=0)),
                "median": _to_jsonable(np.median(arr, axis=0)),
                "values": [_to_jsonable(value) for value in arrays],
            }
        else:
            values = [float(value if value is not None else 0.0) for value in present_values]
            arr = np.asarray(values, dtype=float)
            summary[metric] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "median": float(np.median(arr)),
                "values": values,
            }
    return summary


def safe_summary_mean(summary: Dict[str, Any], metric: str, default: float = 0.0) -> float:
    data = summary.get(metric)
    if isinstance(data, dict) and "mean" in data:
        mean = data["mean"]
        if _is_sequence_metric(mean):
            return float(default)
        return float(mean)
    return float(default)


def safe_summary_raw_mean(summary: Dict[str, Any], metric: str, default: Any = "") -> Any:
    data = summary.get(metric)
    if isinstance(data, dict) and "mean" in data:
        return data["mean"]
    return default


def _append_if_present(target: List[float], summary: Dict[str, Any], metric: str) -> None:
    if metric in summary:
        target.append(safe_summary_mean(summary, metric))


def _mean_or_zero(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def compute_cross_regime_aggregate(test_results: Dict[str, Any]) -> Dict[str, Any]:
    value_accept = []
    drops = []
    flushes = []
    money = []
    settled_value = []
    diagnostics: Dict[str, List[float]] = {
        "settle_accept_rate": [],
        "settle_reject_rate": [],
        "zero_flush_rate": [],
        "full_flush_rate": [],
        "middle_flush_rate": [],
        "mean_flush_fraction": [],
        "mean_flush_action": [],
    }
    list_diagnostics: Dict[str, List[np.ndarray]] = {
        "flush_action_histogram": [],
        "flush_action_prob": [],
    }

    for regime_result in test_results.values():
        summary = regime_result.get("summary", {})
        value_accept.append(safe_summary_mean(summary, "value_accept_ratio"))
        drops.append(safe_summary_mean(summary, "drops"))
        flushes.append(safe_summary_mean(summary, "flushes"))
        money.append(safe_summary_mean(summary, "money"))
        settled_value.append(safe_summary_mean(summary, "settled_value"))

        for metric, values in diagnostics.items():
            _append_if_present(values, summary, metric)

        for metric, values in list_diagnostics.items():
            raw_mean = safe_summary_raw_mean(summary, metric, default="")
            if _is_sequence_metric(raw_mean):
                values.append(np.asarray(raw_mean, dtype=float))

    aggregate: Dict[str, Any] = {
        "mean_value_accept_ratio": float(np.mean(value_accept)) if value_accept else 0.0,
        "worst_regime_value_accept_ratio": float(np.min(value_accept)) if value_accept else 0.0,
        "std_value_accept_ratio_across_regimes": float(np.std(value_accept)) if value_accept else 0.0,
        "mean_drops": float(np.mean(drops)) if drops else 0.0,
        "mean_flushes": float(np.mean(flushes)) if flushes else 0.0,
        "mean_money": float(np.mean(money)) if money else 0.0,
        "worst_regime_money": float(np.min(money)) if money else 0.0,
        "mean_settled_value": float(np.mean(settled_value)) if settled_value else 0.0,
    }

    diagnostic_names = {
        "settle_accept_rate": "mean_settle_accept_rate",
        "settle_reject_rate": "mean_settle_reject_rate",
        "zero_flush_rate": "mean_zero_flush_rate",
        "full_flush_rate": "mean_full_flush_rate",
        "middle_flush_rate": "mean_middle_flush_rate",
        "mean_flush_fraction": "mean_flush_fraction",
        "mean_flush_action": "mean_flush_action",
    }
    for metric, out_name in diagnostic_names.items():
        if diagnostics[metric]:
            aggregate[out_name] = _mean_or_zero(diagnostics[metric])

    for metric, values in list_diagnostics.items():
        if values and len({value.shape for value in values}) == 1:
            aggregate[f"mean_{metric}"] = np.mean(np.stack(values, axis=0), axis=0).tolist()

    return aggregate


def compact_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for metric, data in summary.items():
        if isinstance(data, dict) and "mean" in data:
            out[metric] = {
                "mean": data["mean"],
                "std": data["std"],
                "min": data["min"],
                "max": data["max"],
                "median": data["median"],
            }
        else:
            out[metric] = data
    return out


def confidence_interval_95(values: Iterable[float]) -> float | None:
    arr = np.asarray(list(values), dtype=float)
    if arr.size < 2:
        return None
    return float(1.96 * np.std(arr, ddof=1) / math.sqrt(arr.size))


def flatten_cross_regime_result(payload: Dict[str, Any], source_path: str = "") -> List[Dict[str, Any]]:
    config = payload.get("config", {})
    env_cfg = config.get("env", {})
    reward_cfg = config.get("reward", {})
    train_cfg = config.get("train", {})
    imitation_cfg = config.get("imitation", {})
    rows: List[Dict[str, Any]] = []
    for regime, regime_result in payload.get("test_results", {}).items():
        summary = regime_result.get("summary", {})
        rows.append({
            "source_path": source_path,
            "scenario": payload.get("scenario"),
            "model": payload.get("model_name", payload.get("model_mode")),
            "model_mode": payload.get("model_mode"),
            "train_regime": payload.get("train_regime"),
            "seed": payload.get("seed"),
            "test_regime": regime,
            "C": env_cfg.get("C"),
            "F": env_cfg.get("F"),
            "T": env_cfg.get("T"),
            "T_max": env_cfg.get("T_max"),
            "flush_levels": env_cfg.get("flush_levels", 5),
            "flush_grid": env_cfg.get("flush_grid", "uniform"),
            "num_flush_choices": env_cfg.get("num_flush_choices", env_cfg.get("flush_levels", 5)),
            "state_feature_mode": env_cfg.get("state_feature_mode", "base"),
            "mask_mode": env_cfg.get("mask_mode", "none"),
            "reward_scale": train_cfg.get("reward_scale", 1.0),
            "imitation_mode": imitation_cfg.get(
                "imitation_mode",
                imitation_cfg.get("mode", "none"),
            ),
            "imitation_episodes": imitation_cfg.get(
                "imitation_episodes",
                imitation_cfg.get("episodes", 0),
            ),
            "imitation_epochs": imitation_cfg.get(
                "imitation_epochs",
                imitation_cfg.get("epochs", 0),
            ),
            "selected_eta": imitation_cfg.get(
                "selected_eta",
                payload.get("selected_eta", config.get("eta", "")),
            ),
            "threshold_advantage_mode": config.get("threshold_advantage_mode", "none"),
            "money_p": reward_cfg.get("money_p", payload.get("money_p")),
            "money_tau": reward_cfg.get("money_tau", payload.get("money_tau")),
            "drop_penalty": reward_cfg.get("drop_penalty", payload.get("drop_penalty")),
            "value_accept_ratio": safe_summary_mean(summary, "value_accept_ratio"),
            "settled_value": safe_summary_mean(summary, "settled_value"),
            "total_transaction_value": safe_summary_mean(summary, "total_transaction_value"),
            "drops": safe_summary_mean(summary, "drops"),
            "drop_rate": safe_summary_mean(summary, "drop_rate"),
            "policy_discards": safe_summary_mean(summary, "policy_discards", float("nan")),
            "capacity_drops": safe_summary_mean(summary, "capacity_drops", float("nan")),
            "flushes": safe_summary_mean(summary, "flushes"),
            "money": safe_summary_mean(summary, "money"),
            "total_steps": safe_summary_mean(summary, "total_steps", float("nan")),
            "settle_accept_count": safe_summary_mean(summary, "settle_accept_count", float("nan")),
            "settle_reject_count": safe_summary_mean(summary, "settle_reject_count", float("nan")),
            "settle_accept_rate": safe_summary_mean(summary, "settle_accept_rate", float("nan")),
            "settle_reject_rate": safe_summary_mean(summary, "settle_reject_rate", float("nan")),
            "mean_flush_action": safe_summary_mean(summary, "mean_flush_action", float("nan")),
            "mean_flush_fraction": safe_summary_mean(summary, "mean_flush_fraction", float("nan")),
            "zero_flush_rate": safe_summary_mean(summary, "zero_flush_rate", float("nan")),
            "full_flush_rate": safe_summary_mean(summary, "full_flush_rate", float("nan")),
            "middle_flush_rate": safe_summary_mean(summary, "middle_flush_rate", float("nan")),
            "flush_action_histogram": safe_summary_raw_mean(summary, "flush_action_histogram"),
            "flush_action_prob": safe_summary_raw_mean(summary, "flush_action_prob"),
        })
    return rows
