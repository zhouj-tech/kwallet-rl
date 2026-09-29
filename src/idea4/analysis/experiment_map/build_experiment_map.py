from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


OUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUT_DIR.parents[3]
OUTPUT_CSV = OUT_DIR / "experiment_map.csv"

OUTPUT_COLUMNS = [
    "source_type",
    "source_path",
    "timestamp",
    "model_raw",
    "model_clean",
    "variant_clean",
    "reward_mode",
    "C",
    "k",
    "F",
    "T",
    "train_regime",
    "seed",
    "n_regimes",
    "has_cross_regime_metrics",
    "notes",
]

DUPLICATE_KEY = [
    "model_clean",
    "variant_clean",
    "reward_mode",
    "C",
    "k",
    "F",
    "T",
    "seed",
    "train_regime",
]

RUN_GROUP_KEY = DUPLICATE_KEY

REGIME_NAMES = {
    "US",
    "TLS",
    "LNS",
    "TLNS",
    "TPLS",
    "PLS",
    "UB",
    "TLB",
    "LNB",
    "TLNB",
    "TPLB",
    "PLB",
}

SECONDARY_MARKERS = [
    "final_comparison_tables",
    "model_average_across_seeds",
    "four_model_average_across_seeds",
    "k3_k6_k12",
    "money_ci",
    "summary_money",
    "/comparison/",
    "main_factorized_ac_story",
    "expected_vs_found",
    "excluded_results",
    "difficulty_ranking",
    "regime_level_wide",
    "seed_comparison_wide",
    "summary_by_model",
    "efficiency",
    "timeline",
]

AGGREGATE_MARKERS = [
    "aggregate",
    "aggregated",
    "summary",
    "final_comparison_tables",
    "comparison",
    "summary_money",
    "main_factorized_ac_story",
]


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def normalize_empty(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    return value


def to_int(value: Any) -> int | None:
    value = normalize_empty(value)
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def to_float(value: Any) -> float | None:
    value = normalize_empty(value)
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def scalar(value: Any) -> Any:
    if isinstance(value, pd.Series):
        values = [normalize_empty(v) for v in value.tolist()]
        return next((v for v in values if v is not None), None)
    return normalize_empty(value)


def first_available(mapping: dict[str, Any], names: list[str]) -> Any:
    lowered = {str(k).lower(): k for k in mapping.keys()}
    for name in names:
        key = lowered.get(name.lower())
        if key is not None:
            value = scalar(mapping[key])
            if value is not None:
                return value
    return None


def extract_number(text: str, name: str, as_float: bool = False) -> int | float | None:
    match = re.search(rf"(?:^|[_/\-]){re.escape(name)}([0-9]+(?:\.[0-9]+)?)(?:$|[_/\-])", text)
    if not match:
        return None
    return float(match.group(1)) if as_float else int(float(match.group(1)))


def extract_timestamp(path_text: str) -> str | None:
    matches = re.findall(r"\b20\d{6}_\d{6}(?:_\d{6})?\b", path_text)
    return matches[-1] if matches else None


def extract_train_regime(text: str) -> str | None:
    match = re.search(r"(?:^|[_/\-])train([A-Za-z0-9]+(?:_[A-Za-z0-9]+)*?)(?=(?:_cross|_C\d|_k\d|_T\d|_F\d|_seed\d|[_/\-]|$))", text)
    if match:
        return match.group(1)
    match = re.search(r"(?:^|[_/\-])(MIX12_EQ|MIXED_EQ|US|TLS|LNS|TLNS|TPLS|PLS|UB|TLB|LNB|TLNB|TPLB|PLB)(?:[_/\-]|$)", text)
    return match.group(1) if match else None


def infer_model_raw(text: str, data_value: Any = None) -> str:
    data_value = normalize_empty(data_value)
    if data_value is not None:
        return str(data_value)

    lower = text.lower()
    candidates = [
        "maskA_factorized_ac",
        "maskB_factorized_ac",
        "dual_branch_factorized_ac_gate_regularized",
        "dual_branch_factorized_ac_gate_balanced",
        "dual_branch_capacity_only",
        "dual_branch_residual_risk",
        "dual_branch_factorized_ac",
        "factorized_ac",
        "basic_ppo",
        "dqn_baseline",
        "baseline",
        "attn_context",
    ]
    for candidate in candidates:
        if candidate.lower() in lower:
            return candidate
    if "dqn" in lower:
        return "dqn"
    return "unknown"


def clean_model(model_raw: Any, text: str = "") -> str:
    raw = f"{model_raw or ''} {text}".lower()
    if "maska_factorized_ac" in raw or "maska" in raw:
        return "MaskA Factorized AC"
    if "maskb_factorized_ac" in raw or "maskb" in raw:
        return "MaskB Factorized AC"
    if "gate_regularized" in raw:
        return "Gate-regularized Dual AC"
    if "gate_balanced" in raw:
        return "Gate-balanced Dual AC"
    if "capacity_only" in raw:
        return "Capacity-only Dual AC"
    if "residual_risk" in raw:
        return "Residual-risk Dual AC"
    if "dual_branch_factorized_ac" in raw or "dual_ac" in raw:
        return "Dual-branch AC"
    if "basic_ppo" in raw:
        return "Basic PPO"
    if "factorized_ac" in raw:
        return "Factorized AC"
    if "baseline" in raw or "dqn" in raw or "attn_context" in raw or "ideaextra/results" in raw:
        return "DQN baseline"
    return "unknown"


def clean_variant(text: str) -> str:
    lower = text.lower()
    variants: list[str] = []
    checks = [
        ("maska", "maskA"),
        ("maskb", "maskB"),
        ("gate_balanced", "gate_balanced"),
        ("gate_regularized", "gate_regularized"),
        ("capacity_only", "capacity_only"),
        ("residual_risk", "residual_risk"),
    ]
    for marker, label in checks:
        if marker in lower and label not in variants:
            variants.append(label)
    if "hard" in lower and "hard" not in variants:
        variants.append("hard")
    if "soft" in lower and "soft" not in variants:
        variants.append("soft")
    if "penalty5" in lower and "penalty5" not in variants:
        variants.append("penalty5")
    return "+".join(variants) if variants else "base"


def infer_reward_mode(text: str, mapping: dict[str, Any] | None = None) -> str:
    lower = text.lower()
    if "rewardmoney" in lower or "reward_money" in lower or "money_ci" in lower:
        return "money"
    if mapping:
        for name in ["reward_mode", "training_objective", "objective_label", "reward_formula"]:
            value = first_available(mapping, [name])
            if value is not None and "money" in str(value).lower():
                return "money"
    if "reward" in lower and "money" not in lower:
        return "unknown"
    return "original"


def path_metadata(path: Path) -> dict[str, Any]:
    text = rel(path)
    return {
        "timestamp": extract_timestamp(text),
        "C": extract_number(text, "C", as_float=True),
        "k": extract_number(text, "k"),
        "F": extract_number(text, "F"),
        "T": extract_number(text, "T"),
        "seed": extract_number(text, "seed"),
        "train_regime": extract_train_regime(text),
    }


def count_json_regimes(payload: dict[str, Any]) -> int | None:
    test_results = payload.get("test_results")
    if isinstance(test_results, dict):
        return len(test_results)
    if isinstance(test_results, list):
        regimes = {
            row.get("test_regime")
            for row in test_results
            if isinstance(row, dict) and row.get("test_regime")
        }
        return len(regimes) if regimes else len(test_results)

    aggregate = payload.get("aggregate")
    if isinstance(aggregate, dict):
        for key in ["n_regimes", "num_regimes", "num_test_regimes"]:
            if key in aggregate:
                return to_int(aggregate[key])
    return None


def has_json_cross_regime_metrics(payload: dict[str, Any]) -> bool:
    if isinstance(payload.get("test_results"), (dict, list)):
        return True
    aggregate = payload.get("aggregate")
    if isinstance(aggregate, dict):
        keys = {str(k).lower() for k in aggregate}
        return bool(
            keys
            & {
                "mean_value_accept_ratio",
                "worst_regime_value_accept_ratio",
                "std_value_accept_ratio_across_regimes",
                "n_regimes",
            }
        )
    return False


def parse_json_run(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    env = config.get("env") if isinstance(config.get("env"), dict) else {}
    data = config.get("data") if isinstance(config.get("data"), dict) else {}
    meta = path_metadata(path)
    path_text = rel(path)
    scenario = payload.get("scenario") or path.parent.parent.name
    text = f"{path_text} {scenario}"

    model_raw = infer_model_raw(text, payload.get("model_mode") or config.get("model_mode"))
    row = {
        "source_type": "json_run",
        "source_path": path_text,
        "timestamp": payload.get("timestamp") or meta["timestamp"],
        "model_raw": model_raw,
        "model_clean": clean_model(model_raw, text),
        "variant_clean": clean_variant(text),
        "reward_mode": infer_reward_mode(text, payload),
        "C": to_float(env.get("C")) or meta["C"],
        "k": to_int(env.get("k")) or meta["k"],
        "F": to_int(env.get("F")) or meta["F"],
        "T": to_int(env.get("T")) or meta["T"],
        "train_regime": payload.get("train_regime") or data.get("train_regime") or meta["train_regime"],
        "seed": to_int(payload.get("seed")) or to_int(config.get("seed")) or meta["seed"],
        "n_regimes": count_json_regimes(payload),
        "has_cross_regime_metrics": has_json_cross_regime_metrics(payload),
        "notes": "raw-like json_run; scenario=%s" % scenario,
    }
    if row["reward_mode"] == "money":
        row["notes"] += "; money reward detected"
    return normalize_row(row)


def include_csv(path: Path) -> bool:
    text = rel(path).lower()
    if text.endswith("experiment_map.csv"):
        return False
    if "train_logs.csv" in text or "eval_per_episode.csv" in text or "eval_summary.csv" in text:
        return False
    return any(marker in text for marker in AGGREGATE_MARKERS)


def classify_csv(path: Path) -> tuple[str, str]:
    text = rel(path).lower()
    if any(marker in text for marker in SECONDARY_MARKERS):
        return "secondary_summary_csv", "secondary summary"
    if "aggregate" in text or "aggregated" in text:
        return "aggregate_csv", "aggregate-like"
    return "secondary_summary_csv", "secondary summary"


def source_path_list(value: Any) -> str:
    value = normalize_empty(value)
    if value is None:
        return ""
    parts = [part.strip() for part in str(value).split(";") if part.strip()]
    return " ".join(parts)


def row_text(path: Path, row: dict[str, Any]) -> str:
    pieces = [rel(path)]
    for name in ["source_file", "source_path", "scenario", "model_mode", "model_family", "model"]:
        value = row.get(name)
        if value is not None:
            pieces.append(str(value))
    return " ".join(pieces)


def metadata_from_csv_row(path: Path, row: dict[str, Any]) -> dict[str, Any]:
    text = row_text(path, row)
    meta = path_metadata(path)
    source_files = source_path_list(first_available(row, ["source_file", "source_path"]))
    if source_files:
        source_meta = path_metadata(Path(source_files.split()[0]))
        for key, value in source_meta.items():
            if meta.get(key) is None and value is not None:
                meta[key] = value
        text = f"{text} {source_files}"

    model_value = first_available(row, ["model_mode", "model_raw", "model", "model_family", "algorithm"])
    model_raw = infer_model_raw(text, model_value)
    return {
        "timestamp": first_available(row, ["timestamp", "run_timestamp"]) or extract_timestamp(text) or meta["timestamp"],
        "model_raw": model_raw,
        "model_clean": clean_model(model_raw, text),
        "variant_clean": clean_variant(text),
        "reward_mode": infer_reward_mode(text, row),
        "C": to_float(first_available(row, ["C", "capacity"])) or meta["C"],
        "k": to_int(first_available(row, ["k"])) or meta["k"],
        "F": to_int(first_available(row, ["F"])) or meta["F"],
        "T": to_int(first_available(row, ["T", "horizon"])) or meta["T"],
        "train_regime": first_available(row, ["train_regime", "regime", "train"]) or meta["train_regime"],
        "seed": to_int(first_available(row, ["seed", "run_seed"])) or meta["seed"],
    }


def has_regime_columns(columns: list[str]) -> bool:
    lowered = {column.lower() for column in columns}
    return bool(lowered & {"test_regime", "regime", "num_test_regimes", "n_regimes"})


def detect_n_regimes(group: pd.DataFrame) -> int | None:
    for column in ["test_regime", "regime"]:
        if column in group.columns:
            values = {str(v) for v in group[column].dropna().tolist() if str(v)}
            regime_values = values & REGIME_NAMES
            return len(regime_values or values) if values else None
    for column in ["n_regimes", "num_regimes", "num_test_regimes"]:
        if column in group.columns:
            values = [to_int(v) for v in group[column].tolist()]
            values = [v for v in values if v is not None]
            if values:
                return max(values)
    return None


def normalize_key_value(value: Any) -> Any:
    value = normalize_empty(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {column: row.get(column) for column in OUTPUT_COLUMNS}
    for name in ["C"]:
        normalized[name] = to_float(normalized[name])
    for name in ["k", "F", "T", "seed", "n_regimes"]:
        normalized[name] = to_int(normalized[name])
    normalized["has_cross_regime_metrics"] = bool(normalized["has_cross_regime_metrics"])
    for name in ["source_type", "source_path", "timestamp", "model_raw", "model_clean", "variant_clean", "reward_mode", "train_regime", "notes"]:
        value = normalize_empty(normalized[name])
        normalized[name] = value if value is not None else ""
    return normalized


def build_csv_rows(path: Path) -> list[dict[str, Any]]:
    source_type, source_note = classify_csv(path)
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return [
            normalize_row(
                {
                    "source_type": source_type,
                    "source_path": rel(path),
                    "model_clean": "unknown",
                    "variant_clean": "base",
                    "reward_mode": "unknown",
                    "has_cross_regime_metrics": False,
                    "notes": f"{source_note}; failed to read csv: {exc}",
                }
            )
        ]

    if df.empty:
        return [
            normalize_row(
                {
                    "source_type": source_type,
                    "source_path": rel(path),
                    "model_clean": "unknown",
                    "variant_clean": "base",
                    "reward_mode": "unknown",
                    "has_cross_regime_metrics": False,
                    "notes": f"{source_note}; empty csv",
                }
            )
        ]

    records = df.to_dict("records")
    meta_rows = [metadata_from_csv_row(path, record) for record in records]
    meta_df = pd.DataFrame(meta_rows)
    source_path = rel(path)
    has_cross = has_regime_columns(list(df.columns)) or any("cross_regime" in col.lower() for col in df.columns)

    # Summary files often have one row per setting/model rather than one row per raw run.
    group_key = [key for key in RUN_GROUP_KEY if key in meta_df.columns]
    grouped = meta_df.assign(_row_id=range(len(meta_df))).groupby(group_key, dropna=False, sort=False) if group_key else [((), meta_df)]

    out_rows: list[dict[str, Any]] = []
    for _, group in grouped:
        raw_indexes = group["_row_id"].tolist()
        source_group = df.iloc[raw_indexes]
        first = group.iloc[0].to_dict()
        collapsed_rows = len(source_group)
        source_files = []
        if "source_file" in source_group.columns:
            for value in source_group["source_file"].dropna().astype(str).tolist():
                source_files.extend([part.strip() for part in value.split(";") if part.strip()])
        source_files = sorted(set(source_files))

        notes = [
            source_note,
            f"collapsed_rows={collapsed_rows}",
            f"columns={','.join(df.columns)}",
        ]
        if source_type == "secondary_summary_csv":
            notes.append("reference only; do not use for raw metric aggregation")
        if source_files:
            notes.append(f"source_files={len(source_files)}")
        if first.get("reward_mode") == "money":
            notes.append("money reward detected")

        out_rows.append(
            normalize_row(
                {
                    "source_type": source_type,
                    "source_path": source_path,
                    "timestamp": first.get("timestamp"),
                    "model_raw": first.get("model_raw"),
                    "model_clean": first.get("model_clean"),
                    "variant_clean": first.get("variant_clean"),
                    "reward_mode": first.get("reward_mode"),
                    "C": first.get("C"),
                    "k": first.get("k"),
                    "F": first.get("F"),
                    "T": first.get("T"),
                    "train_regime": first.get("train_regime"),
                    "seed": first.get("seed"),
                    "n_regimes": detect_n_regimes(source_group),
                    "has_cross_regime_metrics": bool(has_cross),
                    "notes": "; ".join(notes),
                }
            )
        )
    return out_rows


def discover_json_runs() -> list[Path]:
    return sorted(REPO_ROOT.glob("**/cross_regime_results.json"))


def discover_csvs() -> list[Path]:
    return sorted(path for path in REPO_ROOT.glob("**/*.csv") if include_csv(path))


def print_section(title: str, body: Any) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if isinstance(body, pd.DataFrame):
        if body.empty:
            print("(none)")
        else:
            print(body.to_string(index=False))
    else:
        print(body)


def report(df: pd.DataFrame) -> None:
    rawish = df[df["source_type"].isin(["json_run", "aggregate_csv"])].copy()

    model_counts = rawish["model_clean"].replace("", "unknown").value_counts(dropna=False).rename_axis("model_clean").reset_index(name="count")
    print_section("1. number of detected runs by model_clean", model_counts)

    setting_counts = (
        rawish.groupby(["C", "k", "F", "T"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["C", "k", "F", "T"], na_position="last")
    )
    print_section("2. number of runs by C,k,F,T", setting_counts)

    seed_groups = (
        rawish.groupby(["model_clean", "variant_clean", "reward_mode", "C", "k", "F", "T", "train_regime"], dropna=False)["seed"]
        .agg(lambda values: sorted({int(v) for v in values.dropna().tolist()}))
        .reset_index(name="seeds")
    )
    seed_groups["n_seeds"] = seed_groups["seeds"].map(len)
    multi_seed = seed_groups[seed_groups["n_seeds"] > 1].sort_values(["model_clean", "C", "k", "F", "T"])
    one_seed = seed_groups[seed_groups["n_seeds"] == 1].sort_values(["model_clean", "C", "k", "F", "T"])
    print_section("3. settings with multiple seeds", multi_seed)
    print_section("4. settings with only one seed", one_seed)

    dupes = rawish[rawish.duplicated(DUPLICATE_KEY, keep=False)].copy()
    if not dupes.empty:
        dupes = (
            dupes.groupby(DUPLICATE_KEY, dropna=False)["source_path"]
            .agg(lambda values: " | ".join(sorted(set(values))))
            .reset_index(name="source_paths")
            .sort_values(DUPLICATE_KEY)
        )
    print_section("5. possible duplicate runs", dupes)


def main() -> None:
    rows: list[dict[str, Any]] = []

    for path in discover_json_runs():
        try:
            rows.append(parse_json_run(path))
        except Exception as exc:
            rows.append(
                normalize_row(
                    {
                        "source_type": "json_run",
                        "source_path": rel(path),
                        "model_clean": "unknown",
                        "variant_clean": "base",
                        "reward_mode": "unknown",
                        "has_cross_regime_metrics": False,
                        "notes": f"raw-like json_run; failed to parse json: {exc}",
                    }
                )
            )

    for path in discover_csvs():
        rows.extend(build_csv_rows(path))

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    df = df.sort_values(["source_type", "model_clean", "C", "k", "F", "T", "seed", "source_path"], na_position="last")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Wrote {len(df)} rows to {rel(OUTPUT_CSV)}")
    print(f"JSON runs: {(df['source_type'] == 'json_run').sum()}")
    print(f"Aggregate CSV rows: {(df['source_type'] == 'aggregate_csv').sum()}")
    print(f"Secondary summary CSV rows: {(df['source_type'] == 'secondary_summary_csv').sum()}")
    report(df)


if __name__ == "__main__":
    main()
