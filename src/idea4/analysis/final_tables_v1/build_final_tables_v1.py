from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import t


OUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUT_DIR.parents[3]
SELECTION_DIR = REPO_ROOT / "src/idea4/analysis/experiment_map"
SELECTION_V2_CSV = SELECTION_DIR / "clean_source_selection_v2.csv"
SELECTION_CSV = SELECTION_DIR / "clean_source_selection.csv"

MAIN_OUT = OUT_DIR / "main_k_scaling_summary.csv"
STRESS_OUT = OUT_DIR / "stress_complexity_summary.csv"
ABLATION_OUT = OUT_DIR / "ablation_summary.csv"
WORKBOOK_OUT = OUT_DIR / "final_summary_tables_v1.xlsx"
SEED_LEVEL_RUNS_OUT = OUT_DIR / "seed_level_runs_used.csv"
CLEAN_SELECTION_USED_OUT = OUT_DIR / "clean_selection_used_for_final_tables.csv"

STRICT_EXCLUDE_MARKERS = [
    "src/ideaextra/results/trainLNB_",
    "src/ideaextra/results/trainLNS_",
    "src/ideaextra/results/trainPLB_",
    "src/ideaextra/results/trainPLS_",
    "src/ideaextra/results/trainTLB_",
    "src/ideaextra/results/trainTLNB_",
    "src/ideaextra/results/trainTLNS_",
    "src/ideaextra/results/trainTLS_",
    "src/ideaextra/results/trainTPLB_",
    "src/ideaextra/results/trainTPLS_",
    "src/ideaextra/results/trainUB_",
    "src/ideaextra/results/trainUS_",
    "src/ideaextra/results_by_seed/",
    "context_attention",
    "trainATTN_CTX",
]

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

GROUP_COLUMNS = [
    "model_clean",
    "variant_clean",
    "C",
    "k",
    "F",
    "T",
    "train_regime",
    "reward_mode",
]

SUMMARY_COLUMNS = [
    "result_table",
    "model_clean",
    "variant_clean",
    "C",
    "k",
    "F",
    "T",
    "train_regime",
    "reward_mode",
    "n_seeds",
    "seeds",
    "mean_value_accept_ratio_mean",
    "mean_value_accept_ratio_ci95",
    "worst_regime_value_accept_ratio_mean",
    "worst_regime_value_accept_ratio_ci95",
    "std_value_accept_ratio_across_regimes_mean",
    "value_accept_ratio_mean",
    "value_accept_ratio_ci95",
    "drops_mean",
    "drops_ci95",
    "flushes_mean",
    "flushes_ci95",
    "drop_rate_mean",
    "drop_rate_ci95",
    "count_accept_ratio_mean",
    "count_accept_ratio_ci95",
    "ci_note",
    "source_paths",
]

RUN_METRICS = [
    "mean_value_accept_ratio",
    "worst_regime_value_accept_ratio",
    "std_value_accept_ratio_across_regimes",
    "value_accept_ratio",
    "drops",
    "flushes",
    "drop_rate",
    "count_accept_ratio",
]

CI_METRICS = [
    "mean_value_accept_ratio",
    "worst_regime_value_accept_ratio",
    "value_accept_ratio",
    "drops",
    "flushes",
    "drop_rate",
    "count_accept_ratio",
]

MAIN_MODELS = ["DQN baseline", "Basic PPO", "Factorized AC", "Dual-branch AC"]
STRESS_MODELS = ["DQN baseline", "Basic PPO", "Factorized AC"]
ABLATION_MODELS = [
    "Gate-balanced Dual AC",
    "Gate-regularized Dual AC",
    "Capacity-only Dual AC",
    "Residual-risk Dual AC",
    "MaskA Factorized AC",
]


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


def to_float(value: Any) -> float | None:
    value = normalize_empty(value)
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(parsed) else parsed


def to_int(value: Any) -> int | None:
    value = normalize_empty(value)
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def repo_path(source_path: str) -> Path:
    path = Path(str(source_path))
    return path if path.is_absolute() else REPO_ROOT / path


def numeric_key(value: Any) -> str:
    parsed = to_float(value)
    if parsed is None:
        return ""
    return str(int(parsed)) if parsed.is_integer() else ("%f" % parsed).rstrip("0").rstrip(".")


def duplicate_key(row: pd.Series) -> str:
    pieces: list[str] = []
    for column in DUPLICATE_KEY_COLUMNS:
        value = row.get(column)
        if column in {"C", "k", "F", "T", "seed"}:
            value = numeric_key(value)
        else:
            value = normalize_empty(value) or ""
        pieces.append(f"{column}={value}")
    return "|".join(pieces)


def timestamp_sort_value(row: pd.Series) -> pd.Timestamp:
    value = normalize_empty(row.get("timestamp"))
    parsed = pd.to_datetime(value, errors="coerce") if value is not None else pd.NaT
    if pd.notna(parsed):
        return parsed
    matches = re.findall(r"\b(20\d{6}_\d{6}(?:_\d{6})?)\b", str(row.get("source_path") or ""))
    if not matches:
        return pd.Timestamp.min
    token = matches[-1]
    fmt = "%Y%m%d_%H%M%S_%f" if token.count("_") == 2 else "%Y%m%d_%H%M%S"
    parsed = pd.to_datetime(token, format=fmt, errors="coerce")
    return parsed if pd.notna(parsed) else pd.Timestamp.min


def selection_priority(row: pd.Series) -> int:
    if row.get("source_type") == "json_run":
        return 0
    if row.get("source_type") == "aggregate_csv":
        return 1
    return 2


def load_selection() -> tuple[pd.DataFrame, Path]:
    input_path = SELECTION_V2_CSV if SELECTION_V2_CSV.exists() else SELECTION_CSV
    if not input_path.exists():
        raise FileNotFoundError(f"Missing clean source selection CSV: {input_path}")
    return pd.read_csv(input_path), input_path


def apply_strict_filters(selection: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = selection.copy()
    if "use_for_final" in df.columns:
        use = df["use_for_final"]
        if use.dtype == object:
            use = use.astype(str).str.lower().isin(["true", "1", "yes"])
        df = df[use.eq(True)].copy()

    missing_key_mask = df[["C", "k", "F", "T", "seed"]].isna().any(axis=1)
    missing_key_rows = df[missing_key_mask].copy()
    df = df[~missing_key_mask].copy()

    df = df[df["source_type"].ne("secondary_summary_csv")].copy()
    for marker in STRICT_EXCLUDE_MARKERS:
        df = df[~df["source_path"].astype(str).str.contains(marker, regex=False)].copy()

    df["_duplicate_key"] = df.apply(duplicate_key, axis=1)
    df["_timestamp_sort"] = df.apply(timestamp_sort_value, axis=1)
    df["_priority"] = df.apply(selection_priority, axis=1)

    duplicate_rows = df[df.duplicated("_duplicate_key", keep=False)].copy()
    kept_indexes: list[int] = []
    for _, group in df.groupby("_duplicate_key", dropna=False, sort=False):
        ordered = group.sort_values(
            ["_priority", "_timestamp_sort", "source_path"],
            ascending=[True, False, False],
        )
        kept_indexes.append(int(ordered.index[0]))

    deduped = df.loc[kept_indexes].copy().sort_values(
        ["model_clean", "C", "k", "F", "T", "seed", "source_path"],
        na_position="last",
    )
    return deduped, missing_key_rows, duplicate_rows


def summary_mean(summary: dict[str, Any], metric: str) -> float | None:
    value = summary.get(metric)
    if isinstance(value, dict):
        return to_float(value.get("mean"))
    return to_float(value)


def extract_regime_metric_rows(test_results: Any) -> list[dict[str, float | None]]:
    rows: list[dict[str, float | None]] = []
    if isinstance(test_results, dict):
        iterable = test_results.values()
    elif isinstance(test_results, list):
        iterable = test_results
    else:
        iterable = []

    for item in iterable:
        if not isinstance(item, dict):
            continue
        summary = item.get("summary") if isinstance(item.get("summary"), dict) else item
        rows.append(
            {
                "value_accept_ratio": summary_mean(summary, "value_accept_ratio"),
                "drops": summary_mean(summary, "drops"),
                "flushes": summary_mean(summary, "flushes"),
                "drop_rate": summary_mean(summary, "drop_rate"),
                "count_accept_ratio": summary_mean(summary, "count_accept_ratio"),
            }
        )
    return rows


def mean_clean(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return float(sum(clean) / len(clean)) if clean else None


def min_clean(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return float(min(clean)) if clean else None


def parse_json_metrics(source_row: pd.Series) -> dict[str, float | None]:
    path = repo_path(str(source_row["source_path"]))
    payload = json.loads(path.read_text(encoding="utf-8"))
    aggregate = payload.get("aggregate") if isinstance(payload.get("aggregate"), dict) else {}
    regime_rows = extract_regime_metric_rows(payload.get("test_results"))

    metrics = {
        "mean_value_accept_ratio": to_float(aggregate.get("mean_value_accept_ratio")),
        "worst_regime_value_accept_ratio": to_float(aggregate.get("worst_regime_value_accept_ratio")),
        "std_value_accept_ratio_across_regimes": to_float(aggregate.get("std_value_accept_ratio_across_regimes")),
        "value_accept_ratio": mean_clean([r["value_accept_ratio"] for r in regime_rows]),
        "drops": mean_clean([r["drops"] for r in regime_rows]),
        "flushes": mean_clean([r["flushes"] for r in regime_rows]),
        "drop_rate": mean_clean([r["drop_rate"] for r in regime_rows]),
        "count_accept_ratio": mean_clean([r["count_accept_ratio"] for r in regime_rows]),
    }

    if metrics["mean_value_accept_ratio"] is None:
        metrics["mean_value_accept_ratio"] = metrics["value_accept_ratio"]
    if metrics["worst_regime_value_accept_ratio"] is None:
        metrics["worst_regime_value_accept_ratio"] = min_clean([r["value_accept_ratio"] for r in regime_rows])
    return metrics


def values_match(series: pd.Series, value: Any, numeric: bool = False) -> pd.Series:
    if numeric:
        return pd.to_numeric(series, errors="coerce").eq(to_float(value))
    return series.astype(str).fillna("").eq(str(value))


def parse_aggregate_metrics(source_row: pd.Series) -> dict[str, float | None]:
    path = repo_path(str(source_row["source_path"]))
    df = pd.read_csv(path)
    for column in ["model_clean", "variant_clean", "reward_mode", "C", "k", "F", "T", "seed", "train_regime"]:
        if column not in df.columns:
            continue
        numeric = column in {"C", "k", "F", "T", "seed"}
        df = df[values_match(df[column], source_row[column], numeric=numeric)]

    if df.empty:
        # Most aggregate CSVs carry model_mode instead of model_clean; fall back
        # to the metadata columns that are consistently present.
        df = pd.read_csv(path)
        for column in ["C", "k", "F", "T", "seed", "train_regime"]:
            if column in df.columns:
                df = df[values_match(df[column], source_row[column], numeric=column in {"C", "k", "F", "T", "seed"})]

    def repeated_or_mean(metric: str) -> float | None:
        if metric not in df.columns:
            return None
        values = pd.to_numeric(df[metric], errors="coerce").dropna()
        if values.empty:
            return None
        return float(values.iloc[0]) if values.nunique(dropna=True) == 1 else float(values.mean())

    metrics = {
        "mean_value_accept_ratio": repeated_or_mean("mean_value_accept_ratio"),
        "worst_regime_value_accept_ratio": repeated_or_mean("worst_regime_value_accept_ratio"),
        "std_value_accept_ratio_across_regimes": repeated_or_mean("std_value_accept_ratio_across_regimes"),
        "value_accept_ratio": repeated_or_mean("value_accept_ratio"),
        "drops": repeated_or_mean("drops"),
        "flushes": repeated_or_mean("flushes"),
        "drop_rate": repeated_or_mean("drop_rate"),
        "count_accept_ratio": repeated_or_mean("count_accept_ratio"),
    }
    if metrics["mean_value_accept_ratio"] is None:
        metrics["mean_value_accept_ratio"] = metrics["value_accept_ratio"]
    if metrics["worst_regime_value_accept_ratio"] is None and "value_accept_ratio" in df.columns:
        values = pd.to_numeric(df["value_accept_ratio"], errors="coerce").dropna()
        metrics["worst_regime_value_accept_ratio"] = float(values.min()) if not values.empty else None
    return metrics


def build_seed_level_rows(source_rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for _, source_row in source_rows.iterrows():
        if source_row["source_type"] == "json_run":
            metrics = parse_json_metrics(source_row)
        elif source_row["source_type"] == "aggregate_csv":
            metrics = parse_aggregate_metrics(source_row)
        else:
            continue

        record = {column: source_row.get(column) for column in source_rows.columns}
        record.update(metrics)
        records.append(record)
    return pd.DataFrame(records)


def ci95(values: pd.Series) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    n = len(clean)
    if n < 2:
        return None
    sample_sd = clean.std(ddof=1)
    if pd.isna(sample_sd):
        return None
    return float(t.ppf(0.975, n - 1) * sample_sd / math.sqrt(n))


def metric_mean(values: pd.Series) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.mean()) if len(clean) else None


def summarize(seed_rows: pd.DataFrame, result_table: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_values, group in seed_rows.groupby(GROUP_COLUMNS, dropna=False, sort=True):
        group_dict = dict(zip(GROUP_COLUMNS, group_values))
        seeds = sorted({int(v) for v in pd.to_numeric(group["seed"], errors="coerce").dropna().tolist()})
        output = {
            "result_table": result_table,
            **group_dict,
            "n_seeds": len(seeds),
            "seeds": ";".join(str(seed) for seed in seeds),
            "ci_note": (
                f"seed-level CI across {len(seeds)} seeds"
                if len(seeds) >= 2
                else "CI not available; single seed only"
            ),
            "source_paths": " | ".join(sorted(set(group["source_path"].astype(str).tolist()))),
        }
        for metric in RUN_METRICS:
            output[f"{metric}_mean"] = metric_mean(group[metric]) if metric in group.columns else None
            if metric in CI_METRICS:
                output[f"{metric}_ci95"] = ci95(group[metric]) if metric in group.columns else None
        rows.append(output)

    result = pd.DataFrame(rows)
    for column in SUMMARY_COLUMNS:
        if column not in result.columns:
            result[column] = None
    return result[SUMMARY_COLUMNS].sort_values(["C", "k", "F", "T", "model_clean", "variant_clean"], na_position="last")


def table_filters(seed_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    original = seed_rows["reward_mode"].eq("original")
    main = seed_rows[
        original
        & seed_rows["C"].eq(1200)
        & seed_rows["F"].eq(3)
        & seed_rows["T"].eq(1000)
        & seed_rows["k"].isin([3, 6, 12])
        & seed_rows["model_clean"].isin(MAIN_MODELS)
    ].copy()
    stress = seed_rows[
        original
        & seed_rows["T"].eq(1000)
        & seed_rows["model_clean"].isin(STRESS_MODELS)
        & (
            (seed_rows["C"].eq(1200) & seed_rows["k"].eq(24) & seed_rows["F"].eq(3))
            | (seed_rows["C"].eq(800) & seed_rows["k"].eq(12) & seed_rows["F"].eq(3))
            | (seed_rows["C"].eq(800) & seed_rows["k"].eq(24) & seed_rows["F"].eq(3))
        )
    ].copy()
    ablation = seed_rows[
        original
        & seed_rows["model_clean"].isin(ABLATION_MODELS)
    ].copy()
    return main, stress, ablation


def readme_sheet() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "README": [
                "CI is computed across seeds using a t critical 95% confidence interval.",
                "If only one seed exists, CI is NA.",
                "Secondary summary tables are not used as raw sources.",
                "Original reward and money reward are separated; this workbook only uses original reward.",
                f"Generated by {rel(Path(__file__))}.",
            ]
        }
    )


def write_outputs(
    main_summary: pd.DataFrame,
    stress_summary: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    seed_rows: pd.DataFrame,
    selected_rows: pd.DataFrame,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main_summary.to_csv(MAIN_OUT, index=False)
    stress_summary.to_csv(STRESS_OUT, index=False)
    ablation_summary.to_csv(ABLATION_OUT, index=False)
    seed_rows.to_csv(SEED_LEVEL_RUNS_OUT, index=False)
    selected_rows.to_csv(CLEAN_SELECTION_USED_OUT, index=False)

    seed_sheet_columns = [
        "source_type",
        "source_path",
        "model_clean",
        "variant_clean",
        "reward_mode",
        "C",
        "k",
        "F",
        "T",
        "train_regime",
        "seed",
        *RUN_METRICS,
    ]
    with pd.ExcelWriter(WORKBOOK_OUT, engine="openpyxl") as writer:
        readme_sheet().to_excel(writer, sheet_name="README", index=False)
        main_summary.to_excel(writer, sheet_name="Main K Scaling", index=False)
        stress_summary.to_excel(writer, sheet_name="Stress Complexity", index=False)
        ablation_summary.to_excel(writer, sheet_name="Ablation", index=False)
        seed_rows[seed_sheet_columns].to_excel(writer, sheet_name="Seed Level Runs Used", index=False)


def print_section(title: str, body: Any) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if isinstance(body, pd.DataFrame):
        print("(none)" if body.empty else body.to_string(index=False))
    elif isinstance(body, pd.Series):
        print("(none)" if body.empty else body.to_string())
    else:
        print(body)


def settings_summary(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=["C", "k", "F", "T", "n_seed_rows"])
    return (
        rows.groupby(["C", "k", "F", "T"], dropna=False)
        .size()
        .reset_index(name="n_seed_rows")
        .sort_values(["C", "k", "F", "T"], na_position="last")
    )


def report(
    selection_path: Path,
    selected_rows: pd.DataFrame,
    missing_key_rows: pd.DataFrame,
    duplicate_rows: pd.DataFrame,
    main_rows: pd.DataFrame,
    stress_rows: pd.DataFrame,
    ablation_rows: pd.DataFrame,
) -> None:
    print(f"Input selection: {rel(selection_path)}")
    print(f"Rows used from clean selection: {len(selected_rows)}")
    print(f"Wrote: {rel(MAIN_OUT)}")
    print(f"Wrote: {rel(STRESS_OUT)}")
    print(f"Wrote: {rel(ABLATION_OUT)}")
    print(f"Wrote: {rel(SEED_LEVEL_RUNS_OUT)}")
    print(f"Wrote: {rel(CLEAN_SELECTION_USED_OUT)}")
    print(f"Wrote: {rel(WORKBOOK_OUT)}")

    print_section(
        "Rows used by table",
        pd.DataFrame(
            [
                {"table": "main_k_scaling_summary", "seed_level_rows": len(main_rows)},
                {"table": "stress_complexity_summary", "seed_level_rows": len(stress_rows)},
                {"table": "ablation_summary", "seed_level_rows": len(ablation_rows)},
            ]
        ),
    )
    print_section(
        "Models included in each table",
        pd.DataFrame(
            [
                {"table": "main_k_scaling_summary", "models": ", ".join(sorted(main_rows["model_clean"].dropna().unique()))},
                {"table": "stress_complexity_summary", "models": ", ".join(sorted(stress_rows["model_clean"].dropna().unique()))},
                {"table": "ablation_summary", "models": ", ".join(sorted(ablation_rows["model_clean"].dropna().unique()))},
            ]
        ),
    )
    print_section("Settings in main_k_scaling_summary", settings_summary(main_rows))
    print_section("Settings in stress_complexity_summary", settings_summary(stress_rows))
    print_section("Settings in ablation_summary", settings_summary(ablation_rows))
    print_section(
        "Selected rows with missing C/k/F/T/seed",
        missing_key_rows[["source_type", "source_path", "model_clean", "C", "k", "F", "T", "seed"]]
        if not missing_key_rows.empty
        else pd.DataFrame(),
    )

    duplicate_summary = (
        duplicate_rows.groupby("_duplicate_key", dropna=False)
        .size()
        .reset_index(name="rows_before_dedup")
        .sort_values(["rows_before_dedup", "_duplicate_key"], ascending=[False, True])
    )
    print_section("Selected duplicate keys after filtering", duplicate_summary)


def main() -> None:
    selection, selection_path = load_selection()
    selected_rows, missing_key_rows, duplicate_rows = apply_strict_filters(selection)
    seed_rows = build_seed_level_rows(selected_rows)
    main_rows, stress_rows, ablation_rows = table_filters(seed_rows)

    main_summary = summarize(main_rows, "main_k_scaling_summary")
    stress_summary = summarize(stress_rows, "stress_complexity_summary")
    ablation_summary = summarize(ablation_rows, "ablation_summary")

    write_outputs(main_summary, stress_summary, ablation_summary, seed_rows, selected_rows)
    report(selection_path, selected_rows, missing_key_rows, duplicate_rows, main_rows, stress_rows, ablation_rows)


if __name__ == "__main__":
    main()
