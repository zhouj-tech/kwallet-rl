from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


OUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUT_DIR.parents[3]
IDEA4_RESULTS = REPO_ROOT / "src" / "idea4" / "ac" / "results"
IDEA3_RESULTS = REPO_ROOT / "src" / "idea3" / "results"

MISSING = "missing"
SOURCE_TYPE = "raw_cross_regime_json"
REGIME_NAMES = {
    "US", "TLS", "LNS", "TLNS", "TPLS", "PLS",
    "UB", "TLB", "LNB", "TLNB", "TPLB", "PLB",
}

TIMELINE_COLUMNS = [
    "timestamp", "result_path", "source_type", "run_status", "is_rerun", "run_group_key",
    "scenario", "model_mode", "direction", "parent_direction", "experiment_tag",
    "primary_metric_used", "comparison_reference",
    "seed", "C", "k", "F", "T", "train_regime", "reward_mode",
    "maskA_mode", "maskA_soft_penalty", "mask_settle_capacity",
    "mask_flush_empty_wallet", "mask_same_wallet_conflict",
    "gate_min", "gate_max", "gate_temperature", "gate_target", "gate_reg_coef",
    "enable_dual_critic", "dual_critic_coef",
    "aux_risk_coef", "aux_risk_window",
    "mean_value_accept_ratio", "worst_regime_value_accept_ratio",
    "std_value_accept_ratio_across_regimes", "mean_drops", "mean_flushes",
    "mean_eval_money",
    "gate_mean", "gate_std", "gate_min_eval", "gate_max_eval",
    "gate_near_lower_rate", "gate_near_upper_rate",
    "value_disagreement", "value_capacity_mean", "value_risk_mean",
]

COMPARISON_GROUP_COLUMNS = [
    "C", "k", "F", "T", "train_regime", "seed", "direction", "model_mode",
    "gate_min", "gate_max", "gate_temperature", "gate_target", "gate_reg_coef",
    "maskA_mode", "maskA_soft_penalty", "enable_dual_critic", "dual_critic_coef",
]

METRIC_COLUMNS = [
    "mean_value_accept_ratio", "worst_regime_value_accept_ratio",
    "std_value_accept_ratio_across_regimes", "mean_drops", "mean_flushes",
    "mean_eval_money",
]


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def clean(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def safe_float(value: Any) -> float | None:
    if value is None or value == "" or value == MISSING:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def safe_int(value: Any) -> int | None:
    parsed = safe_float(value)
    return int(parsed) if parsed is not None else None


def first_nonempty(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return ""


def nested(mapping: dict[str, Any], *keys: str) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict):
            return ""
        cur = cur.get(key, "")
    return cur


def summary_mean(summary: dict[str, Any], metric: str) -> float | None:
    value = summary.get(metric)
    if isinstance(value, dict):
        return safe_float(value.get("mean"))
    return safe_float(value)


def collect_regime_means(payload: dict[str, Any], metric: str) -> list[float]:
    values: list[float] = []
    test_results = payload.get("test_results", {})
    if not isinstance(test_results, dict):
        return values
    for result in test_results.values():
        if not isinstance(result, dict):
            continue
        summary = result.get("summary", {})
        if not isinstance(summary, dict):
            continue
        value = summary_mean(summary, metric)
        if value is not None:
            values.append(value)
    return values


def aggregate_regime_metric(payload: dict[str, Any], metric: str) -> float | str:
    values = collect_regime_means(payload, metric)
    return mean(values) if values else ""


def extract_timestamp(path: Path, payload: dict[str, Any], run_info: dict[str, Any]) -> str:
    ts = first_nonempty(payload.get("timestamp"), run_info.get("timestamp"))
    if ts:
        return str(ts)
    matches = re.findall(r"20\d{6}_\d{6}(?:_\d{6})?", str(path))
    return matches[-1] if matches else ""


def extract_from_text(text: str, name: str, as_float: bool = False) -> int | float | str:
    match = re.search(rf"(?:^|[_/\-]){re.escape(name)}([0-9]+(?:\.[0-9]+)?)(?:$|[_/\-])", text)
    if not match:
        return ""
    value = float(match.group(1))
    return value if as_float else int(value)


def extract_train_regime(text: str) -> str:
    match = re.search(
        r"(?:^|[_/\-])train([A-Za-z0-9]+(?:_[A-Za-z0-9]+)*?)(?=(?:_cross|_C\d|_k\d|_T\d|_F\d|_seed\d|[_/\-]|$))",
        text,
    )
    if match:
        return match.group(1)
    for regime in sorted(REGIME_NAMES | {"MIX12_EQ"}, key=len, reverse=True):
        if re.search(rf"(?:^|[_/\-]){re.escape(regime)}(?:[_/\-]|$)", text):
            return regime
    return ""


def model_from_text(text: str) -> str:
    lower = text.lower()
    candidates = [
        "maskA_factorized_ac",
        "dual_branch_factorized_ac_gate_regularized",
        "dual_branch_factorized_ac_gate_balanced",
        "dual_branch_factorized_ac_auxrisk",
        "dual_branch_capacity_only",
        "dual_branch_residual_risk",
        "dual_branch_factorized_ac",
        "factorized_ac",
        "basic_ppo",
        "attn_context",
        "baseline",
    ]
    for candidate in candidates:
        if candidate.lower() in lower:
            return candidate
    return ""


def infer_underlying_parent_for_reward(raw: str) -> str:
    if "dual_branch" in raw:
        return "Dual Branch"
    if "basic_ppo" in raw or "factorized_ac" in raw:
        return "PPO/Actor-Critic"
    if "baseline" in raw or "attn_context" in raw:
        return "DQN"
    return "Reward shaping"


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def classify_direction(payload: dict[str, Any], config: dict[str, Any], path: Path) -> str:
    text = f"{payload.get('scenario','')} {payload.get('model_mode','')} {path}".lower()
    reward_mode = str(first_nonempty(payload.get("reward_mode"), nested(config, "reward", "reward_mode")) or "").lower()
    enable_dual = truthy(first_nonempty(
        payload.get("enable_dual_critic"),
        nested(config, "dual_critic", "enable_dual_critic"),
    ))
    # The dual-critic script carries an aux_risk config block by default, even
    # for gate-balanced runs that do not train the aux-risk head. Treat aux risk
    # as enabled only when the run identity explicitly says auxrisk.
    aux_enabled = "auxrisk" in text

    if enable_dual and aux_enabled:
        return "Dual Critic + Aux Risk"
    if enable_dual:
        return "Dual Critic"
    if reward_mode and reward_mode not in {"original", "none"}:
        return "Reward shaping / money reward"
    if "rewardmoney" in text or "reward_mode" in payload and reward_mode == "money":
        return "Reward shaping / money reward"
    if "maska" in text:
        return "MaskA / Masked Factorized AC"
    if "gate_balanced" in text:
        return "Gate-balanced Dual Branch AC"
    if "residual_risk" in text:
        return "Residual Risk Dual"
    if "capacity_only" in text:
        return "Capacity-only Dual"
    if "dual_branch_factorized_ac" in text:
        return "Original Dual Branch AC"
    if "basic_ppo" in text:
        return "Basic PPO"
    if "factorized_ac" in text:
        return "Factorized AC"
    if "baseline" in text or "attn_context" in text:
        return "DQN baseline"
    return "DQN baseline"


def parent_direction(direction: str, payload: dict[str, Any], path: Path) -> str:
    if direction == "DQN baseline":
        return "DQN"
    if direction in {"Basic PPO", "Factorized AC"}:
        return "PPO/Actor-Critic"
    if direction == "MaskA / Masked Factorized AC":
        return "Factorized AC"
    if direction in {
        "Original Dual Branch AC", "Gate-balanced Dual Branch AC",
        "Residual Risk Dual", "Capacity-only Dual", "Dual Critic",
        "Dual Critic + Aux Risk",
    }:
        return "Dual Branch"
    if direction == "Reward shaping / money reward":
        return infer_underlying_parent_for_reward(f"{payload.get('scenario','')} {payload.get('model_mode','')} {path}".lower())
    return ""


def comparison_reference(direction: str) -> str:
    mapping = {
        "Factorized AC": "DQN baseline",
        "Basic PPO": "Factorized AC / DQN baseline",
        "Original Dual Branch AC": "Factorized AC",
        "Gate-balanced Dual Branch AC": "Original Dual Branch AC",
        "Residual Risk Dual": "Original Dual Branch AC",
        "Capacity-only Dual": "Original Dual Branch AC",
        "MaskA / Masked Factorized AC": "Factorized AC",
        "Dual Critic": "Gate-balanced Dual Branch AC",
        "Dual Critic + Aux Risk": "Gate-balanced Dual Branch AC",
        "Reward shaping / money reward": "same model under original reward",
        "DQN baseline": "none",
    }
    return mapping.get(direction, "none")


def experiment_tag(payload: dict[str, Any], config: dict[str, Any], path: Path) -> str:
    text = f"{payload.get('scenario','')} {payload.get('model_mode','')} {path}".lower()
    gate_cfg = config.get("gate", {}) if isinstance(config.get("gate"), dict) else {}
    if "gate_sweep" in text or "gate_regularized" in text:
        return "gate_sweep"
    if gate_cfg:
        defaults = {"gate_temperature": 2.0, "gate_min": 0.1, "gate_max": 0.9, "gate_target": 0.7, "gate_reg_coef": 0.01}
        changed = any(safe_float(gate_cfg.get(k)) not in {None, v} for k, v in defaults.items())
        if changed and "dual_branch" in text:
            return "gate_sweep"
    if "rewardmoney" in text or str(first_nonempty(payload.get("reward_mode"), nested(config, "reward", "reward_mode"))).lower() not in {"", "original"}:
        return "reward_shaping"
    if "maska" in text:
        return "maskA"
    if "dualcritic" in text:
        return "dual_critic"
    return ""


def primary_metric(direction: str, payload: dict[str, Any], config: dict[str, Any]) -> str:
    reward_mode = str(first_nonempty(payload.get("reward_mode"), nested(config, "reward", "reward_mode")) or "").lower()
    if direction == "Reward shaping / money reward" or reward_mode in {"money", "money_normalized", "hybrid_money"}:
        return "mean_eval_money"
    return "mean_value_accept_ratio"


def run_status(payload: dict[str, Any]) -> str:
    if not payload:
        return "failed"
    aggregate = payload.get("aggregate")
    tests = payload.get("test_results")
    if isinstance(aggregate, dict) and isinstance(tests, dict) and tests:
        return "completed"
    if isinstance(aggregate, dict):
        return "aggregate_only"
    return "partial"


def stable_group_key(row: dict[str, Any]) -> str:
    parts = [
        row.get("model_mode"), row.get("direction"), row.get("seed"), row.get("C"), row.get("k"),
        row.get("F"), row.get("T"), row.get("train_regime"), row.get("reward_mode"),
        row.get("maskA_mode"), row.get("maskA_soft_penalty"), row.get("gate_min"),
        row.get("gate_max"), row.get("gate_temperature"), row.get("gate_target"),
        row.get("gate_reg_coef"), row.get("enable_dual_critic"), row.get("dual_critic_coef"),
        row.get("aux_risk_coef"), row.get("aux_risk_window"),
    ]
    text = "|".join("" if p is None else str(p) for p in parts)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{digest}:{text}"


def extract_row(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    run_dir = path.parent
    run_config = read_json(run_dir / "run_config.json")
    run_info = read_json(run_dir / "run_info.json")
    config = run_config or payload.get("config", {}) or run_info.get("config", {})
    if not isinstance(config, dict):
        config = {}
    text = str(path)
    aggregate = payload.get("aggregate", {}) if isinstance(payload.get("aggregate"), dict) else {}
    direction = classify_direction(payload, config, path)
    model_mode = first_nonempty(payload.get("model_mode"), config.get("model_mode"), model_from_text(text))
    scenario = first_nonempty(payload.get("scenario"), path.parent.parent.name)
    reward_mode = first_nonempty(
        payload.get("reward_mode"), config.get("reward_mode"), nested(config, "reward", "reward_mode"), "original"
    )
    gate_min_cfg = first_nonempty(nested(config, "gate", "gate_min"), payload.get("gate_min"))
    gate_max_cfg = first_nonempty(nested(config, "gate", "gate_max"), payload.get("gate_max"))
    gate_min_eval = aggregate_regime_metric(payload, "gate_min")
    gate_max_eval = aggregate_regime_metric(payload, "gate_max")

    gate_means = collect_regime_means(payload, "gate_mean")
    lower = safe_float(gate_min_cfg)
    upper = safe_float(gate_max_cfg)
    gate_near_lower = ""
    gate_near_upper = ""
    if gate_means and lower is not None:
        gate_near_lower = sum(1 for v in gate_means if v <= lower + 0.02) / len(gate_means)
    if gate_means and upper is not None:
        gate_near_upper = sum(1 for v in gate_means if v >= upper - 0.02) / len(gate_means)

    row = {
        "timestamp": extract_timestamp(path, payload, run_info),
        "result_path": rel(path),
        "source_type": SOURCE_TYPE,
        "run_status": run_status(payload),
        "is_rerun": False,
        "run_group_key": "",
        "scenario": scenario,
        "model_mode": model_mode,
        "direction": direction,
        "parent_direction": parent_direction(direction, payload, path),
        "experiment_tag": experiment_tag(payload, config, path),
        "primary_metric_used": primary_metric(direction, payload, config),
        "comparison_reference": comparison_reference(direction),
        "seed": first_nonempty(payload.get("seed"), config.get("seed"), extract_from_text(text, "seed")),
        "C": first_nonempty(nested(config, "env", "C"), extract_from_text(text, "C", as_float=True)),
        "k": first_nonempty(nested(config, "env", "k"), extract_from_text(text, "k")),
        "F": first_nonempty(nested(config, "env", "F"), extract_from_text(text, "F")),
        "T": first_nonempty(nested(config, "env", "T"), extract_from_text(text, "T")),
        "train_regime": first_nonempty(payload.get("train_regime"), nested(config, "data", "train_regime"), extract_train_regime(text)),
        "reward_mode": reward_mode,
        "maskA_mode": first_nonempty(payload.get("maskA_mode"), nested(config, "maskA", "mode")),
        "maskA_soft_penalty": first_nonempty(payload.get("maskA_soft_penalty"), nested(config, "maskA", "soft_penalty")),
        "mask_settle_capacity": first_nonempty(payload.get("maskA_mask_settle_capacity"), nested(config, "maskA", "mask_settle_capacity")),
        "mask_flush_empty_wallet": first_nonempty(payload.get("maskA_mask_flush_empty_wallet"), nested(config, "maskA", "mask_flush_empty_wallet")),
        "mask_same_wallet_conflict": first_nonempty(payload.get("maskA_mask_same_wallet_conflict"), nested(config, "maskA", "mask_same_wallet_conflict")),
        "gate_min": gate_min_cfg,
        "gate_max": gate_max_cfg,
        "gate_temperature": first_nonempty(nested(config, "gate", "gate_temperature"), payload.get("gate_temperature")),
        "gate_target": first_nonempty(nested(config, "gate", "gate_target"), payload.get("gate_target")),
        "gate_reg_coef": first_nonempty(nested(config, "gate", "gate_reg_coef"), payload.get("gate_reg_coef")),
        "enable_dual_critic": first_nonempty(payload.get("enable_dual_critic"), nested(config, "dual_critic", "enable_dual_critic")),
        "dual_critic_coef": first_nonempty(payload.get("dual_critic_coef"), nested(config, "dual_critic", "dual_critic_coef")),
        "aux_risk_coef": first_nonempty(payload.get("aux_risk_coef"), nested(config, "aux_risk", "aux_risk_coef")),
        "aux_risk_window": first_nonempty(payload.get("aux_risk_window"), nested(config, "aux_risk", "aux_risk_window")),
        "mean_value_accept_ratio": aggregate.get("mean_value_accept_ratio", ""),
        "worst_regime_value_accept_ratio": aggregate.get("worst_regime_value_accept_ratio", ""),
        "std_value_accept_ratio_across_regimes": aggregate.get("std_value_accept_ratio_across_regimes", ""),
        "mean_drops": aggregate.get("mean_drops", ""),
        "mean_flushes": aggregate.get("mean_flushes", ""),
        "mean_eval_money": aggregate.get("mean_eval_money", ""),
        "gate_mean": aggregate_regime_metric(payload, "gate_mean"),
        "gate_std": aggregate_regime_metric(payload, "gate_std"),
        "gate_min_eval": gate_min_eval,
        "gate_max_eval": gate_max_eval,
        "gate_near_lower_rate": gate_near_lower,
        "gate_near_upper_rate": gate_near_upper,
        "value_disagreement": aggregate_regime_metric(payload, "value_disagreement"),
        "value_capacity_mean": aggregate_regime_metric(payload, "value_capacity_mean"),
        "value_risk_mean": aggregate_regime_metric(payload, "value_risk_mean"),
    }
    row["run_group_key"] = stable_group_key(row)
    return {key: clean(row.get(key, "")) for key in TIMELINE_COLUMNS}


def discover_runs() -> list[Path]:
    paths = []
    for root in [IDEA4_RESULTS, IDEA3_RESULTS]:
        if root.exists():
            paths.extend(root.glob("**/cross_regime_results.json"))
    return sorted(paths)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def metric_mean(rows: list[dict[str, Any]], col: str) -> float | str:
    vals = [safe_float(r.get(col)) for r in rows]
    vals = [v for v in vals if v is not None]
    return mean(vals) if vals else ""


def metric_std(rows: list[dict[str, Any]], col: str) -> float | str:
    vals = [safe_float(r.get(col)) for r in rows]
    vals = [v for v in vals if v is not None]
    return pstdev(vals) if len(vals) > 1 else ""


def latest_per_group(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row["run_group_key"]
        if key not in latest or str(row.get("timestamp", "")) > str(latest[key].get("timestamp", "")):
            latest[key] = row
    return list(latest.values())


def best_row(rows: list[dict[str, Any]], metric: str = "mean_value_accept_ratio") -> dict[str, Any] | None:
    valid = [(safe_float(r.get(metric)), r) for r in rows]
    valid = [(v, r) for v, r in valid if v is not None]
    if not valid:
        return None
    return max(valid, key=lambda x: x[0])[1]


def group_rows(rows: list[dict[str, Any]], columns: list[str]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(c, "") for c in columns)].append(row)
    return groups


def build_comparison_by_setting(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for key, group in group_rows(rows, COMPARISON_GROUP_COLUMNS).items():
        item = dict(zip(COMPARISON_GROUP_COLUMNS, key))
        for col in METRIC_COLUMNS:
            item[col] = metric_mean(group, col)
        item["run_count"] = len(group)
        item["comparison_reference"] = sorted({str(r.get("comparison_reference", "")) for r in group if r.get("comparison_reference")})[0] if group else ""
        item["source_paths"] = " | ".join(r["result_path"] for r in group)
        out.append(item)
    return sorted(out, key=lambda r: tuple(str(r.get(c, "")) for c in COMPARISON_GROUP_COLUMNS))


def build_best_seed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cols = ["C", "k", "F", "T", "train_regime", "seed"]
    out = []
    for key, group in group_rows(rows, cols).items():
        best = best_row(group)
        if not best:
            continue
        item = dict(zip(cols, key))
        for col in ["direction", "model_mode", "scenario", "result_path"] + METRIC_COLUMNS:
            item[f"best_{col}" if col in {"direction", "model_mode", "scenario", "result_path"} else col] = best.get(col, "")
        out.append(item)
    return sorted(out, key=lambda r: tuple(str(r.get(c, "")) for c in cols))


def build_best_aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cols = ["C", "k", "F", "T", "train_regime"]
    out = []
    for setting_key, setting_rows in group_rows(rows, cols).items():
        direction_groups = group_rows(setting_rows, ["direction", "model_mode"])
        candidates = []
        for (direction, model_mode), group in direction_groups.items():
            candidates.append({
                "direction": direction,
                "model_mode": model_mode,
                "mean_value_accept_ratio": metric_mean(group, "mean_value_accept_ratio"),
                "worst_regime_value_accept_ratio": metric_mean(group, "worst_regime_value_accept_ratio"),
                "mean_eval_money": metric_mean(group, "mean_eval_money"),
                "seed_count": len({str(r.get("seed")) for r in group if r.get("seed") != ""}),
                "run_count": len(group),
                "source_paths": " | ".join(r["result_path"] for r in group),
            })
        valid = [(safe_float(c["mean_value_accept_ratio"]), c) for c in candidates]
        valid = [(v, c) for v, c in valid if v is not None]
        if not valid:
            continue
        best = max(valid, key=lambda x: x[0])[1]
        item = dict(zip(cols, setting_key))
        item.update({f"best_{k}": v for k, v in best.items()})
        out.append(item)
    return sorted(out, key=lambda r: tuple(str(r.get(c, "")) for c in cols))


def evidence_strength_for(rows: list[dict[str, Any]], reference_rows: list[dict[str, Any]] | None = None) -> str:
    seeds = {str(r.get("seed")) for r in rows if r.get("seed") not in {"", None}}
    rerun_groups = group_rows(rows, ["run_group_key"])
    repeated = any(len(g) > 1 for g in rerun_groups.values())
    if len(seeds) >= 3 and (repeated or len(rows) >= 3):
        return "strong"
    if len(seeds) >= 2 or len(rows) >= 2:
        return "tentative"
    return "speculative"


def build_direction_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest_rows = latest_per_group(rows)
    out = []
    all_by_direction = group_rows(rows, ["direction"])
    latest_by_direction = group_rows(latest_rows, ["direction"])
    for (direction,), group in all_by_direction.items():
        latest_group = latest_by_direction.get((direction,), [])
        best = best_row(group) or {}
        item = {
            "direction": direction,
            "parent_direction": best.get("parent_direction", ""),
            "comparison_reference": comparison_reference(direction),
            "evidence_strength": evidence_strength_for(group),
            "all_run_count": len(group),
            "dedup_latest_run_count": len(latest_group),
            "seed_count_all": len({str(r.get("seed")) for r in group if r.get("seed") != ""}),
            "mean_value_accept_ratio_all": metric_mean(group, "mean_value_accept_ratio"),
            "std_value_accept_ratio_all": metric_std(group, "mean_value_accept_ratio"),
            "mean_value_accept_ratio_dedup_latest": metric_mean(latest_group, "mean_value_accept_ratio"),
            "worst_regime_value_accept_ratio_dedup_latest": metric_mean(latest_group, "worst_regime_value_accept_ratio"),
            "mean_drops_dedup_latest": metric_mean(latest_group, "mean_drops"),
            "mean_flushes_dedup_latest": metric_mean(latest_group, "mean_flushes"),
            "mean_eval_money_dedup_latest": metric_mean(latest_group, "mean_eval_money"),
            "best_observed_mean_value_accept_ratio": best.get("mean_value_accept_ratio", ""),
            "best_observed_worst_regime_value_accept_ratio": best.get("worst_regime_value_accept_ratio", ""),
            "best_observed_mean_eval_money": best.get("mean_eval_money", ""),
            "best_observed_scenario": best.get("scenario", ""),
            "best_observed_result_path": best.get("result_path", ""),
            "representative_settings": "; ".join(sorted({
                f"C{r.get('C')}_k{r.get('k')}_F{r.get('F')}_T{r.get('T')}_seed{r.get('seed')}"
                for r in latest_group[:20]
            })),
        }
        out.append(item)
    return sorted(out, key=lambda r: str(r["direction"]))


def build_rerun_index(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for (key,), group in group_rows(rows, ["run_group_key"]).items():
        if len(group) <= 1:
            continue
        latest = max(group, key=lambda r: str(r.get("timestamp", "")))
        best = best_row(group) or latest
        out.append({
            "run_group_key": key,
            "rerun_count": len(group),
            "direction": latest.get("direction", ""),
            "model_mode": latest.get("model_mode", ""),
            "seed": latest.get("seed", ""),
            "C": latest.get("C", ""),
            "k": latest.get("k", ""),
            "F": latest.get("F", ""),
            "T": latest.get("T", ""),
            "train_regime": latest.get("train_regime", ""),
            "latest_timestamp": latest.get("timestamp", ""),
            "latest_run": latest.get("result_path", ""),
            "best_mean_value_accept_ratio": best.get("mean_value_accept_ratio", ""),
            "best_run": best.get("result_path", ""),
            "source_paths": " | ".join(r["result_path"] for r in sorted(group, key=lambda r: str(r.get("timestamp", "")))),
        })
    return sorted(out, key=lambda r: str(r["latest_timestamp"]))


def build_gate_behavior(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        if not any(r.get(c) not in {"", None} for c in ["gate_mean", "gate_std", "value_disagreement"]):
            continue
        out.append({
            "timestamp": r.get("timestamp", ""),
            "scenario": r.get("scenario", ""),
            "seed": r.get("seed", ""),
            "direction": r.get("direction", ""),
            "model_mode": r.get("model_mode", ""),
            "C": r.get("C", ""), "k": r.get("k", ""), "F": r.get("F", ""), "T": r.get("T", ""),
            "gate_min": r.get("gate_min", ""),
            "gate_max": r.get("gate_max", ""),
            "gate_temperature": r.get("gate_temperature", ""),
            "gate_target": r.get("gate_target", ""),
            "gate_reg_coef": r.get("gate_reg_coef", ""),
            "gate_mean": r.get("gate_mean", ""),
            "gate_std": r.get("gate_std", ""),
            "gate_min_eval": r.get("gate_min_eval", ""),
            "gate_max_eval": r.get("gate_max_eval", ""),
            "gate_near_lower_rate": r.get("gate_near_lower_rate", ""),
            "gate_near_upper_rate": r.get("gate_near_upper_rate", ""),
            "value_disagreement": r.get("value_disagreement", ""),
            "value_capacity_mean": r.get("value_capacity_mean", ""),
            "value_risk_mean": r.get("value_risk_mean", ""),
            "result_path": r.get("result_path", ""),
        })
    return sorted(out, key=lambda r: str(r.get("timestamp", "")))


def pct(value: Any) -> str:
    v = safe_float(value)
    return "missing" if v is None else f"{v * 100:.2f}%"


def num(value: Any, digits: int = 3) -> str:
    v = safe_float(value)
    return "missing" if v is None else f"{v:.{digits}f}"


def path_ref(path: str) -> str:
    return f"`{path}`" if path else "`missing`"


def best_for_direction(rows: list[dict[str, Any]], direction: str) -> dict[str, Any]:
    group = [r for r in rows if r.get("direction") == direction]
    return best_row(group) or {}


def direction_hypothesis(direction: str) -> tuple[str, str, str]:
    mapping = {
        "DQN baseline": ("建立可比基线", "先确认旧 DQN 在标准 cross-regime 上能到什么水平。", "src/idea3/results/ 或 dqn_baseline 结果"),
        "Basic PPO": ("测试普通 PPO 是否已经足够强", "如果 flat joint PPO 已经很强，复杂结构就必须证明自己有额外价值。", "kwallet_basic_ppo_fair_benchmark.py"),
        "Factorized AC": ("拆动作空间，缓解 k 增大后的输出爆炸", "把 settle/flush 分头建模，降低策略输出规模。", "run_factorized_ac_benchmark.py"),
        "Original Dual Branch AC": ("把容量约束和未来风险分开建模", "双分支可能比单一表示更能处理钱包压力。", "run_dual_branch_ac_benchmark.py"),
        "Gate-balanced Dual Branch AC": ("防止 gate 偏向单分支", "给 gate 加目标/边界，希望容量和风险分支都被使用。", "run_dual_branch_ac_benchmark.py"),
        "Residual Risk Dual": ("用小残差修正容量分支", "风险只做有限修正，避免整套双分支不稳定。", "run_dual_branch_ac_benchmark.py"),
        "Capacity-only Dual": ("验证风险分支是否真的必要", "只保留容量分支，看是否已经解释大部分收益。", "run_dual_branch_ac_benchmark.py"),
        "MaskA / Masked Factorized AC": ("用硬/软动作 mask 注入可行性先验", "希望减少明显不可行动作，但不改变环境语义。", "kwallet_maskA_factorized_ac.py"),
        "Dual Critic": ("分别训练容量/风险价值头", "希望 critic 更懂两个分支的职责。", "run_dual_branch_ac_dual_critic.py"),
        "Dual Critic + Aux Risk": ("用辅助风险任务逼出风险表征", "让风险分支预测未来 drop 压力。", "run_dual_branch_ac_dual_critic.py"),
        "Reward shaping / money reward": ("直接优化钱的目标", "看真实金额收益是否比原始 reward 更合适。", "basic/factorized/dual reward_mode=money"),
    }
    return mapping.get(direction, ("未归类假设", "需要人工复核。", "missing"))


def compare_to_reference(rows: list[dict[str, Any]], direction: str) -> tuple[str, str]:
    latest = latest_per_group(rows)
    group = [r for r in latest if r.get("direction") == direction]
    ref_name = comparison_reference(direction)
    if ref_name == "none":
        return "基线方向，不做胜负判断。", "tentative"
    ref_directions = set(ref_name.split(" / "))
    refs = [r for r in latest if r.get("direction") in ref_directions]
    if direction == "Reward shaping / money reward":
        refs = [
            r for r in latest
            if str(r.get("reward_mode", "")).lower() in {"", "original"}
            and r.get("model_mode") in {g.get("model_mode") for g in group}
        ]
    if not group or not refs:
        return f"缺少直接可比的 `{ref_name}`，结论尚不确定。", "speculative"

    setting_cols = ["C", "k", "F", "T", "train_regime", "seed"]
    diffs: list[float] = []
    for row in group:
        candidates = [
            ref for ref in refs
            if all(str(ref.get(c, "")) == str(row.get(c, "")) for c in setting_cols)
        ]
        if direction == "Reward shaping / money reward":
            candidates = [ref for ref in candidates if ref.get("model_mode") == row.get("model_mode")]
        if not candidates:
            continue
        row_metric = safe_float(row.get("mean_value_accept_ratio"))
        ref_vals = [safe_float(ref.get("mean_value_accept_ratio")) for ref in candidates]
        ref_vals = [v for v in ref_vals if v is not None]
        if row_metric is not None and ref_vals:
            diffs.append(row_metric - mean(ref_vals))

    if not diffs:
        return f"缺少同 C/k/F/T/train/seed 的 `{ref_name}`，只能说方向可运行，胜负尚不确定。", "speculative"
    diff = mean(diffs)
    noise = max(pstdev(diffs) if len(diffs) > 1 else 0.0, 0.005)
    match_note = f"基于 {len(diffs)} 个同设置/同 seed 匹配"
    if abs(diff) < noise:
        return f"{match_note}，相对 `{ref_name}` 的平均差异约 {diff * 100:.2f} 个百分点，小于或接近波动，弱证据，尚不确定。", "tentative"
    if diff > 0:
        return f"{match_note}，相对 `{ref_name}` 平均高约 {diff * 100:.2f} 个百分点；若匹配数少，仍按弱证据处理。", "tentative"
    return f"{match_note}，相对 `{ref_name}` 平均低约 {abs(diff) * 100:.2f} 个百分点，当前更像退化。", "tentative"


def recommendation(direction: str, rows: list[dict[str, Any]]) -> str:
    best = best_for_direction(rows, direction)
    mean_val = safe_float(metric_mean([r for r in rows if r.get("direction") == direction], "mean_value_accept_ratio"))
    if direction in {"MaskA / Masked Factorized AC", "Dual Critic", "Dual Critic + Aux Risk", "Reward shaping / money reward"}:
        return "停止作为主线；只保留小规模变体复查。" if mean_val is None or mean_val < 0.85 else "修改后再试，不能直接当主线。"
    if direction in {"Basic PPO", "Gate-balanced Dual Branch AC", "Factorized AC"}:
        return "继续作为重点比较对象。" if best else "保留，但需要补结果。"
    if direction in {"Residual Risk Dual", "Capacity-only Dual"}:
        return "作为消融保留，不建议扩大成主线。"
    return "作为参照保留。"


def write_direction_summary(rows: list[dict[str, Any]], direction_rows: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    lines.append("# 实验方向总结\n")
    lines.append("本报告只使用 Idea3/Idea4 的原始 `cross_regime_results.json` 作为主证据；聚合表只作为辅助。所有 rerun 保留，解释时同时参考去重后的 latest-per-setting。\n")

    ordered = sorted(rows, key=lambda r: str(r.get("timestamp", "")))
    if ordered:
        lines.append("## 1. 按时间线看实验演化\n")
        lines.append(f"- 最早结果：{ordered[0].get('timestamp')}，来源 {path_ref(ordered[0].get('result_path',''))}。")
        lines.append(f"- 最新结果：{ordered[-1].get('timestamp')}，来源 {path_ref(ordered[-1].get('result_path',''))}。")
        lines.append("- 大致演化：先有 DQN/attention 旧基线，然后进入 Factorized AC 与 Basic PPO 对比，再扩展到 Dual Branch、Gate-balanced、MaskA、money reward、Dual Critic 和 aux-risk。")
        lines.append("- 详细逐 run 记录见 `experiment_timeline.csv`。\n")

    lines.append("## 2. 按研究方向总结\n")
    for item in direction_rows:
        direction = item["direction"]
        hypo, intuition, script = direction_hypothesis(direction)
        best_path = item.get("best_observed_result_path", "")
        compare_text, compare_strength = compare_to_reference(rows, direction)
        evidence = item.get("evidence_strength", compare_strength)
        lines.append(f"### {direction}")
        lines.append(f"- 假设：{hypo}")
        lines.append(f"- 直觉：{intuition}")
        lines.append(f"- 代码/脚本：`{script}`")
        lines.append(f"- 设置范围：{item.get('representative_settings') or 'missing'}")
        lines.append(f"- 最好结果：mean={pct(item.get('best_observed_mean_value_accept_ratio'))}，worst={pct(item.get('best_observed_worst_regime_value_accept_ratio'))}，来源 {path_ref(best_path)}")
        lines.append(f"- 对比基线：`{item.get('comparison_reference')}`。{compare_text}")
        lines.append(f"- 证据强度：`{evidence}`。小差异不自动算胜利；跨 seed/setting 不一致时按弱证据处理。")
        lines.append(f"- 建议：{recommendation(direction, rows)}")
        lines.append(f"- 一句话结论：{compare_text}\n")

    lines.append("## 3. 失败或负向方向\n")
    for direction in [
        "MaskA / Masked Factorized AC",
        "Dual Critic",
        "Dual Critic + Aux Risk",
        "Reward shaping / money reward",
    ]:
        group = [r for r in rows if r.get("direction") == direction]
        if not group:
            continue
        best = best_row(group) or {}
        compare_text, _ = compare_to_reference(rows, direction)
        lines.append(f"### {direction}")
        lines.append(f"- 试了什么：{len(group)} 个 completed run，代表来源 {path_ref(best.get('result_path',''))}。")
        lines.append(f"- 观察结果：最好 mean={pct(best.get('mean_value_accept_ratio'))}，平均表现见 `direction_mean_summary.csv`。{compare_text}")
        if direction == "MaskA / Masked Factorized AC":
            reason = "硬/软 mask 可能限制了探索，且可行性先验未必等价于长期收益。"
        elif direction == "Dual Critic":
            reason = "coef=0.1 可能把 critic 训练目标拉复杂，但没有让 actor 获得稳定收益。"
        elif direction == "Dual Critic + Aux Risk":
            reason = "aux-risk 目标可能和原 reward 的主优化目标不完全一致，增加了训练干扰。"
        else:
            reason = "直接 money reward 的尺度和稀疏性可能破坏了原 reward 下已学到的稳定策略。"
        lines.append(f"- 可能原因：{reason}")
        lines.append("- 是否值得小改：可以小规模试一个更温和版本，但不建议作为当前主线。\n")

    lines.append("## 4. 当前最强模型\n")
    for metric, label in [
        ("mean_value_accept_ratio", "平均 value acceptance"),
        ("worst_regime_value_accept_ratio", "最差 regime value acceptance"),
        ("mean_eval_money", "平均 eval money"),
    ]:
        best = best_row(rows, metric)
        if best:
            lines.append(f"- 按 {label}：`{best.get('direction')}` 最好，值={num(best.get(metric), 4)}，来源 {path_ref(best.get('result_path',''))}。")
    stress = [r for r in rows if safe_int(r.get("k")) and safe_int(r.get("k")) >= 24 or safe_float(r.get("C")) == 800.0]
    best_stress = best_row(stress)
    if best_stress:
        lines.append(f"- 大动作/压力设置：`{best_stress.get('direction')}` 当前最好，mean={pct(best_stress.get('mean_value_accept_ratio'))}，来源 {path_ref(best_stress.get('result_path',''))}。")
    lines.append("- Basic PPO vs Gate-balanced Dual：逐设置赢家见 `best_model_by_setting_seed_level.csv` 和 `best_model_by_setting_aggregate.csv`；若差异接近 seed 波动，不视为决定性胜利。\n")

    lines.append("## 5. Gate 行为分析\n")
    gate_rows = [r for r in rows if r.get("gate_mean") not in {"", None}]
    if gate_rows:
        near_upper = metric_mean(gate_rows, "gate_near_upper_rate")
        gate_std = metric_mean(gate_rows, "gate_std")
        disagreement = metric_mean(gate_rows, "value_disagreement")
        example = gate_rows[0]
        lines.append(f"- gate 诊断见 `gate_behavior_summary.csv`，代表来源 {path_ref(example.get('result_path',''))}。")
        lines.append(f"- 平均 gate_std={num(gate_std, 6)}，near_upper_rate={num(near_upper, 3)}，value_disagreement={num(disagreement, 4)}。")
        lines.append("- 目前很多 dual run 的 gate_mean 接近上边界，state-dependent 使用不明显；这是 gate-collapse 的直接证据之一。")
        lines.append("- value disagreement 有记录的主要在 dual critic 系列，但它没有自动转化为更好策略，因此“分支专门化有效”仍是弱证据。\n")
    else:
        lines.append("- 没有可用 gate 诊断，无法判断 gate 是否 collapse。\n")

    lines.append("## 6. 最终研究建议\n")
    lines.append("- 当前主故事：Factorized/PPO 系列解决动作空间扩展问题；Dual Branch 的动机合理，但 gate collapse 让“风险分支是否真的工作”仍未完全证明。")
    lines.append("- 最强基线：Basic PPO 和 Factorized AC 都必须保留；具体设置下以 `best_model_by_setting_aggregate.csv` 为准。")
    lines.append("- 最强当前模型：按原始结果通常是低 k 设置下的 Basic PPO/Factorized/Dual 系列；压力设置需要单独看表，不混在一起下结论。")
    lines.append("- 应停止：MaskA 硬 masking、money reward 主线、dual critic coef=0.1 主线。")
    lines.append("- 应继续：gate-collapse 诊断、温和 gate variance/entropy 约束、同设置多 seed 的 Basic PPO vs Gate-balanced 对比。")
    lines.append("- 最多 3 个下一步实验：1) gate variance/anti-collapse 小系数；2) dual critic coef 更小如 0.01；3) 同 C/k/F/T 下补齐 Basic PPO、Factorized、Gate-balanced 的 3 seed 对比。\n")

    lines.append("# 当前最可信的结论\n")
    lines.append("## A. 强支持结论\n")
    strong = [d for d in direction_rows if d.get("evidence_strength") == "strong"]
    if strong:
        for d in strong:
            compare_text, _ = compare_to_reference(rows, d["direction"])
            lines.append(
                f"- `{d['direction']}` 的证据量较足，但不等于方向一定有效；当前判断是：{compare_text} "
                f"具体数值见 `direction_mean_summary.csv`，最好来源 {path_ref(d.get('best_observed_result_path',''))}。"
            )
    else:
        lines.append("- 目前没有足够多方向同时满足多 seed 和稳定重复趋势，强结论应保持克制。")
    lines.append("\n## B. 暂定假设\n")
    lines.append("- Gate-balanced 的结构动机仍值得保留，但 gate 接近边界说明风险分支未必被真正使用。")
    lines.append("- Basic PPO 在部分设置很强，可能是当前最硬的工程基线；但跨压力设置不能直接外推。")
    lines.append("\n## C. 推测性想法\n")
    lines.append("- learnable masking、gate variance regularization、更小 dual critic coef 都还没有形成充分证据，只能作为下一步小实验。\n")

    (OUT_DIR / "direction_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_project_story(rows: list[dict[str, Any]]) -> None:
    phase_data = [
        ("阶段 1：DQN 基线与可扩展性问题", "先用旧 DQN/attention 路线建立参照。随着 k 增大，flat 动作空间会迅速变大，DQN 的可扩展性成为问题。"),
        ("阶段 2：Factorized AC 改进", "把动作拆成 settle/flush 两个头，核心动机是降低输出规模。结果显示它在若干 k 设置下能保持竞争力，因此成为新主线之一。"),
        ("阶段 3：Basic PPO 对比", "Basic PPO 用更直接的 actor-critic 训练方式检验：是不是不需要复杂 factorization/dual branch 也能跑好。它在部分低 k 设置很强，所以必须作为强基线。"),
        ("阶段 4：Dual Branch 探索", "Dual Branch 试图区分容量约束和未来风险。原始版本证明这个想法可以跑通，但不等于证明两个分支真的分工。"),
        ("阶段 5：Gate-balanced 突破与问题", "Gate-balanced 试图约束 gate，避免单分支垄断。结果让 dual branch 进入可比较范围，但 gate 诊断也暴露了接近边界的问题。"),
        ("阶段 6：MaskA 与硬 masking 失败", "MaskA 把可行性先验硬塞进动作 logits。当前结果不理想，可能因为它减少探索，且短期可行不等于长期收益。"),
        ("阶段 7：Dual critic 与 aux-risk 实验", "Dual critic/aux-risk 想让风险分支学到更明确的信号。当前 coef=0.1 方向没有压过 gate-balanced baseline，属于负结果。"),
        ("阶段 8：当前 gate-collapse 调查", "gate_mean 接近上边界、gate_std 很小，说明 gate 很可能没有按状态动态切换。下一步应先解决 collapse，再谈分支专门化。"),
        ("阶段 9：当前最强方向与下一步假设", "当前最可信做法是保留 Basic PPO/Factorized/Gate-balanced 三条可比线，停止大规模 MaskA/money/dual-critic 主线，只做小而可诊断的变体。"),
    ]
    lines = ["# 项目研究故事\n", "这不是逐日流水账，而是按研究阶段记录：为什么做、试了什么、学到了什么、为什么进入下一阶段。\n"]
    for title, body in phase_data:
        lines.append(f"## {title}")
        lines.append(body)
        if "DQN" in title:
            src = best_for_direction(rows, "DQN baseline").get("result_path", "")
        elif "Factorized" in title:
            src = best_for_direction(rows, "Factorized AC").get("result_path", "")
        elif "Basic PPO" in title:
            src = best_for_direction(rows, "Basic PPO").get("result_path", "")
        elif "Gate-balanced" in title or "gate-collapse" in title:
            src = best_for_direction(rows, "Gate-balanced Dual Branch AC").get("result_path", "")
        elif "MaskA" in title:
            src = best_for_direction(rows, "MaskA / Masked Factorized AC").get("result_path", "")
        elif "Dual critic" in title:
            src = best_for_direction(rows, "Dual Critic + Aux Risk").get("result_path", "") or best_for_direction(rows, "Dual Critic").get("result_path", "")
        else:
            src = ""
        lines.append(f"证据入口：{path_ref(src) if src else '`experiment_timeline.csv`'}。\n")
    (OUT_DIR / "project_story.md").write_text("\n".join(lines), encoding="utf-8")


def write_unfinished(rows: list[dict[str, Any]]) -> None:
    items = [
        ("gate sweep", "想找到不 collapse 的 gate 设置。", "已有 gate-balanced/gate-regularized 痕迹，但系统 sweep 证据不足。", "当前 gate 多接近边界，先暂停扩大。", "partial", "值得，用小网格和 gate variance 指标复查。", "tentative"),
        ("learnable masking", "希望比硬 MaskA 更温和地注入可行性先验。", "当前主要是 MaskA hard/soft，learnable mask 没有充分完成。", "MaskA 负结果后优先级下降。", "paused", "可晚点重访，但必须避免硬剪探索。", "speculative"),
        ("smaller dual critic coef", "coef=0.1 可能太强，想测试更小辅助损失。", "已有 dual critic coef=0.1。", "0.1 没有证明收益，先不扩大。", "paused", "值得小试 0.01，但只做诊断实验。", "tentative"),
        ("reward shaping variants", "直接优化 money 或 hybrid money。", "已有 rewardMONEY/tau10 等结果。", "当前 value acceptance 明显退化，money 尺度可能不稳。", "mostly stopped", "除非重做尺度归一化，否则不建议主线继续。", "tentative"),
        ("gate variance regularization", "直接惩罚 gate collapse，让 gate 随状态变化。", "目前更多是 gate target/balance，不是明确 variance objective。", "尚未形成完整实验。", "not started", "值得作为下一步三实验之一。", "speculative"),
    ]
    lines = ["# 未完成/暂停方向\n", "这些方向不等于完全失败；它们只是证据不足、暂时暂停，或需要更小的诊断实验。\n"]
    for name, motivation, tested, stopped, status, revisit, evidence in items:
        lines.append(f"## {name}")
        lines.append(f"- evidence_strength：`{evidence}`")
        lines.append(f"- 原始动机：{motivation}")
        lines.append(f"- 已部分测试：{tested}")
        lines.append(f"- 为什么停下：{stopped}")
        zh_status = {
            "partial": "部分完成",
            "paused": "暂停",
            "mostly stopped": "基本停止",
            "not started": "尚未开始",
        }.get(status, status)
        lines.append(f"- 当前状态：{zh_status}")
        lines.append(f"- 是否值得以后重访：{revisit}")
        lines.append("- 证据入口：`experiment_timeline.csv`、`direction_summary.md`。\n")
    (OUT_DIR / "unfinished_directions.md").write_text("\n".join(lines), encoding="utf-8")


def write_research_questions(rows: list[dict[str, Any]]) -> None:
    questions = [
        ("Factorized AC 是否解决了动作空间扩展问题？", "partially supported", "Factorized AC 与 DQN/Basic PPO 的 k scaling 结果", "压力设置和 seed 覆盖仍不均衡", "补齐 C/k/F/T 相同的 3 seed 对比。"),
        ("Basic PPO 是否已经是最强基线？", "partially supported", "Basic PPO 在若干低 k run 中表现很强", "不同 k/C/F 下未必一致", "按设置输出 winner 表后补缺 seed。"),
        ("Dual Branch 的风险分支是否真的有用？", "unclear", "Gate-balanced run 可运行且有较好结果", "gate 接近边界，分支专门化证据弱", "做 gate variance/anti-collapse 诊断实验。"),
        ("MaskA 硬 masking 是否有帮助？", "contradicted", "MaskA 当前最好结果弱于对应 Factorized/Basic PPO 压力设置", "只测了少量 hard/soft 形式", "除非改成 learnable/soft prior，否则停止主线。"),
        ("Dual Critic coef=0.1 是否提升 Gate-balanced？", "contradicted", "dual critic/aux-risk run 没有压过 gate-balanced baseline", "coef 可能过大，且只代表一种损失权重", "只小试 coef=0.01，不再大规模跑 0.1。"),
        ("Money reward 是否应作为主目标？", "contradicted", "rewardMONEY run 的 value acceptance 明显崩掉", "money objective 的尺度和评价目标冲突仍未完全拆开", "若重试，只做 normalized/hybrid 小实验。"),
    ]
    lines = ["# 研究问题清单\n", "这个文件把问题和证据分开，避免把直觉写成结论。\n"]
    for q, status, support, uncertainty, next_exp in questions:
        lines.append(f"## {q}")
        lines.append(f"- 当前证据状态：`{status}`")
        lines.append(f"- 最强支持实验：{support}；见 `experiment_timeline.csv` 和 `direction_summary.md`。")
        lines.append(f"- 未解决不确定性：{uncertainty}")
        lines.append(f"- 推荐下一步实验：{next_exp}\n")
    (OUT_DIR / "research_questions.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = discover_runs()
    rows = [extract_row(path) for path in paths]
    rows.sort(key=lambda r: str(r.get("timestamp", "")))

    group_counts = defaultdict(int)
    for row in rows:
        group_counts[row["run_group_key"]] += 1
    for row in rows:
        row["is_rerun"] = group_counts[row["run_group_key"]] > 1

    comparison_rows = build_comparison_by_setting(rows)
    best_seed_rows = build_best_seed(rows)
    best_aggregate_rows = build_best_aggregate(rows)
    direction_rows = build_direction_summary(rows)
    rerun_rows = build_rerun_index(rows)
    gate_rows = build_gate_behavior(rows)

    write_csv(OUT_DIR / "experiment_timeline.csv", rows, TIMELINE_COLUMNS)
    write_csv(
        OUT_DIR / "comparison_by_setting.csv",
        comparison_rows,
        COMPARISON_GROUP_COLUMNS + METRIC_COLUMNS + ["run_count", "comparison_reference", "source_paths"],
    )
    write_csv(
        OUT_DIR / "best_model_by_setting_seed_level.csv",
        best_seed_rows,
        ["C", "k", "F", "T", "train_regime", "seed", "best_direction", "best_model_mode",
         "best_scenario", "best_result_path"] + METRIC_COLUMNS,
    )
    write_csv(
        OUT_DIR / "best_model_by_setting_aggregate.csv",
        best_aggregate_rows,
        ["C", "k", "F", "T", "train_regime", "best_direction", "best_model_mode",
         "best_mean_value_accept_ratio", "best_worst_regime_value_accept_ratio",
         "best_mean_eval_money", "best_seed_count", "best_run_count", "best_source_paths"],
    )
    write_csv(
        OUT_DIR / "direction_mean_summary.csv",
        direction_rows,
        ["direction", "parent_direction", "comparison_reference", "evidence_strength",
         "all_run_count", "dedup_latest_run_count", "seed_count_all",
         "mean_value_accept_ratio_all", "std_value_accept_ratio_all",
         "mean_value_accept_ratio_dedup_latest", "worst_regime_value_accept_ratio_dedup_latest",
         "mean_drops_dedup_latest", "mean_flushes_dedup_latest", "mean_eval_money_dedup_latest",
         "best_observed_mean_value_accept_ratio", "best_observed_worst_regime_value_accept_ratio",
         "best_observed_mean_eval_money", "best_observed_scenario", "best_observed_result_path",
         "representative_settings"],
    )
    write_csv(
        OUT_DIR / "rerun_index.csv",
        rerun_rows,
        ["run_group_key", "rerun_count", "direction", "model_mode", "seed", "C", "k", "F", "T",
         "train_regime", "latest_timestamp", "latest_run", "best_mean_value_accept_ratio",
         "best_run", "source_paths"],
    )
    write_csv(
        OUT_DIR / "gate_behavior_summary.csv",
        gate_rows,
        ["timestamp", "scenario", "seed", "direction", "model_mode", "C", "k", "F", "T",
         "gate_min", "gate_max", "gate_temperature", "gate_target", "gate_reg_coef",
         "gate_mean", "gate_std", "gate_min_eval", "gate_max_eval",
         "gate_near_lower_rate", "gate_near_upper_rate",
         "value_disagreement", "value_capacity_mean", "value_risk_mean", "result_path"],
    )

    write_direction_summary(rows, direction_rows)
    write_project_story(rows)
    write_unfinished(rows)
    write_research_questions(rows)

    print(f"Discovered raw cross_regime_results.json: {len(paths)}")
    print(f"Timeline rows written: {len(rows)}")
    print(f"Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
