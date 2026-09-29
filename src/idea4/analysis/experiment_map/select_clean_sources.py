from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


OUT_DIR = Path(__file__).resolve().parent
INPUT_CSV = OUT_DIR / "experiment_map.csv"
OUTPUT_CSV = OUT_DIR / "clean_source_selection.csv"

DUPLICATE_KEY_COLUMNS = [
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

KEY_METADATA_COLUMNS = ["C", "k", "F", "T", "seed"]

MAIN_ORIGINAL_MODELS = {
    "DQN baseline",
    "Basic PPO",
    "Factorized AC",
    "Dual-branch AC",
}

DUAL_ABLATION_MODELS = {
    "Gate-balanced Dual AC",
    "Gate-regularized Dual AC",
    "Capacity-only Dual AC",
    "Residual-risk Dual AC",
}


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


def normalize_key_value(value: Any, numeric: bool = False) -> str:
    value = normalize_empty(value)
    if value is None:
        return ""
    if numeric:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return str(value)
        if parsed.is_integer():
            return str(int(parsed))
        return ("%f" % parsed).rstrip("0").rstrip(".")
    return str(value)


def duplicate_key(row: pd.Series) -> str:
    parts = []
    for column in DUPLICATE_KEY_COLUMNS:
        parts.append(f"{column}={normalize_key_value(row.get(column), numeric=column in {'C', 'k', 'F', 'T', 'seed'})}")
    return "|".join(parts)


def has_missing_key_metadata(row: pd.Series) -> bool:
    for column in KEY_METADATA_COLUMNS:
        if normalize_empty(row.get(column)) is None:
            return True
    return False


def result_group(row: pd.Series) -> str:
    model_clean = str(row.get("model_clean") or "")
    variant_clean = str(row.get("variant_clean") or "")
    reward_mode = str(row.get("reward_mode") or "")
    variant_lower = variant_clean.lower()

    if reward_mode == "money":
        return "money_reward"
    if "maska" in variant_lower or "maskb" in variant_lower or model_clean in {"MaskA Factorized AC", "MaskB Factorized AC"}:
        return "mask_ablation"
    if model_clean in DUAL_ABLATION_MODELS or any(
        marker in variant_lower
        for marker in ["gate_balanced", "gate_regularized", "capacity_only", "residual_risk"]
    ):
        return "dual_ablation"
    if reward_mode == "original" and model_clean in MAIN_ORIGINAL_MODELS:
        return "main_or_stress_original"
    if reward_mode == "original":
        return "original_reward"
    return "unknown"


def timestamp_sort_value(row: pd.Series) -> pd.Timestamp:
    value = normalize_empty(row.get("timestamp"))
    parsed = pd.to_datetime(value, errors="coerce") if value is not None else pd.NaT
    if pd.notna(parsed):
        return parsed

    source_path = str(row.get("source_path") or "")
    matches = re.findall(r"\b(20\d{6}_\d{6}(?:_\d{6})?)\b", source_path)
    if not matches:
        return pd.Timestamp.min
    token = matches[-1]
    fmt = "%Y%m%d_%H%M%S_%f" if token.count("_") == 2 else "%Y%m%d_%H%M%S"
    parsed = pd.to_datetime(token, format=fmt, errors="coerce")
    return parsed if pd.notna(parsed) else pd.Timestamp.min


def initial_priority(row: pd.Series) -> int:
    source_type = str(row.get("source_type") or "")
    if "archive/" in str(row.get("source_path") or ""):
        return 90
    if source_type == "secondary_summary_csv":
        return 80
    if has_missing_key_metadata(row):
        return 70
    if source_type == "json_run":
        return 10
    if source_type == "aggregate_csv":
        return 20
    return 60


def apply_base_exclusions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["use_for_final"] = True
    df["exclusion_reason"] = ""
    df["result_group"] = df.apply(result_group, axis=1)
    df["duplicate_key"] = df.apply(duplicate_key, axis=1)
    df["selection_priority"] = df.apply(initial_priority, axis=1)
    df["_timestamp_sort"] = df.apply(timestamp_sort_value, axis=1)

    archive_mask = df["source_path"].fillna("").astype(str).str.contains("archive/", regex=False)
    secondary_mask = df["source_type"].eq("secondary_summary_csv")
    missing_mask = df.apply(has_missing_key_metadata, axis=1)

    # Specific exclusion labels are assigned in the order requested. Secondary
    # summaries keep their own reason even when they also lack run metadata.
    df.loc[archive_mask, ["use_for_final", "exclusion_reason"]] = [False, "archive source"]
    df.loc[secondary_mask, ["use_for_final", "exclusion_reason"]] = [False, "secondary summary table"]
    needs_missing_reason = missing_mask & df["exclusion_reason"].eq("")
    df.loc[needs_missing_reason, ["use_for_final", "exclusion_reason"]] = [False, "missing key metadata"]
    return df


def select_preferred_sources(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eligible_mask = (
        df["use_for_final"].eq(True)
        & df["source_type"].isin(["json_run", "aggregate_csv"])
        & df["exclusion_reason"].eq("")
    )

    for _, indexes in df[eligible_mask].groupby("duplicate_key", dropna=False).groups.items():
        group = df.loc[list(indexes)]
        json_group = group[group["source_type"].eq("json_run")]
        aggregate_group = group[group["source_type"].eq("aggregate_csv")]

        if not json_group.empty:
            newest_json_index = json_group.sort_values(
                ["_timestamp_sort", "source_path"],
                ascending=[False, False],
            ).index[0]
            older_json_indexes = [idx for idx in json_group.index if idx != newest_json_index]
            if older_json_indexes:
                df.loc[older_json_indexes, ["use_for_final", "exclusion_reason"]] = [
                    False,
                    "older duplicate json run",
                ]
                df.loc[older_json_indexes, "selection_priority"] = 30
            if not aggregate_group.empty:
                df.loc[aggregate_group.index, ["use_for_final", "exclusion_reason"]] = [
                    False,
                    "duplicate aggregate; json_run preferred",
                ]
                df.loc[aggregate_group.index, "selection_priority"] = 40
            df.loc[newest_json_index, "selection_priority"] = 10
            continue

        if len(aggregate_group) > 1:
            keep_index = aggregate_group.sort_values(
                ["_timestamp_sort", "source_path"],
                ascending=[False, False],
            ).index[0]
            drop_indexes = [idx for idx in aggregate_group.index if idx != keep_index]
            df.loc[drop_indexes, ["use_for_final", "exclusion_reason"]] = [
                False,
                "duplicate aggregate",
            ]
            df.loc[drop_indexes, "selection_priority"] = 50

    return df


def print_section(title: str, body: pd.DataFrame | pd.Series | str) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if isinstance(body, pd.Series):
        if body.empty:
            print("(none)")
        else:
            print(body.to_string())
    elif isinstance(body, pd.DataFrame):
        if body.empty:
            print("(none)")
        else:
            print(body.to_string(index=False))
    else:
        print(body)


def report(df: pd.DataFrame) -> None:
    kept = df[df["use_for_final"].eq(True)].copy()
    excluded_reasons = df.loc[~df["use_for_final"], "exclusion_reason"].replace("", "unspecified").value_counts()

    print(f"1. number of rows kept for final aggregation: {len(kept)}")
    print_section("2. number excluded by reason", excluded_reasons)
    print_section("3. kept rows by model_clean", kept["model_clean"].replace("", "unknown").value_counts())
    print_section("4. kept rows by result_group", kept["result_group"].replace("", "unknown").value_counts())

    by_setting = (
        kept.groupby(["C", "k", "F", "T"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["C", "k", "F", "T"], na_position="last")
    )
    print_section("5. kept rows by C,k,F,T", by_setting)

    duplicate_counts = (
        df.groupby("duplicate_key", dropna=False)
        .size()
        .reset_index(name="original_row_count")
    )
    duplicate_counts = duplicate_counts[duplicate_counts["original_row_count"] > 1].sort_values(
        ["original_row_count", "duplicate_key"],
        ascending=[False, True],
    )
    print_section("6. duplicate keys where more than one row was originally present", duplicate_counts)


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    df = apply_base_exclusions(df)
    df = select_preferred_sources(df)

    output_columns = list(pd.read_csv(INPUT_CSV, nrows=0).columns) + [
        "use_for_final",
        "result_group",
        "exclusion_reason",
        "duplicate_key",
        "selection_priority",
    ]
    df = df[output_columns + ["_timestamp_sort"]]
    df = df.sort_values(["use_for_final", "selection_priority", "model_clean", "C", "k", "F", "T", "seed", "source_path"], ascending=[False, True, True, True, True, True, True, True, True])
    df[output_columns].to_csv(OUTPUT_CSV, index=False)

    print(f"Wrote {len(df)} rows to {OUTPUT_CSV.relative_to(OUT_DIR.parents[3])}")
    report(df[output_columns])


if __name__ == "__main__":
    main()
