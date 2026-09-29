import os
import re
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Target setting
# ============================================================

TARGET_K = 24
TARGET_C = 1200
TARGET_F = 3
TARGET_T = 1000
TARGET_P = 1.0
TARGET_TAU = 10.0


# ============================================================
# 2. Input patterns
# ============================================================

INPUT_PATTERNS = {
    "Basic PPO": (
        "./src/idea4/ac/results/basic_ppo/runs/"
        "basic_ppo_trainMIX12_EQ_C1200_k24_T1000_F3_seed*/"
        "*/cross_regime_results.json"
    ),

    "IFAC": (
        "./src/idea4/ac/results/factorized_ac/runs/"
        "factorized_ac_trainMIX12_EQ_C1200_k24_T1000_F3_seed*/"
        "*/cross_regime_results.json"
    ),

    "SC-FAC": (
        "./src/idea4/ac/results/conditional_factorized_ac/runs/"
        "conditional_factorized_ac_trainMIX12_EQ_C1200_k24_T1000_F3_seed*_condE32_condH256/"
        "*/cross_regime_results.json"
    ),
}


# ============================================================
# 3. Output
# ============================================================

OUTPUT_DIR = (
    "./src/idea4/ac/results/final_comparison_tables/"
    "per_regime_heatmaps_k24_from_json"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 4. Orders
# ============================================================

REGIME_ORDER = [
    "US", "TLS", "LNS", "TLNS", "TPLS", "PLS",
    "UB", "TLB", "LNB", "TLNB", "TPLB", "PLB"
]

MODEL_ORDER = ["Basic PPO", "IFAC", "SC-FAC"]


# ============================================================
# 5. Basic helpers
# ============================================================

def parse_seed(path):
    match = re.search(r"_seed(\d+)", path)
    if not match:
        return None
    return int(match.group(1))


def normalize_regime(x):
    if x is None:
        return None
    return str(x).strip().upper()


def to_float(x):
    if x is None:
        return np.nan

    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return np.nan

    return np.nan


def flatten_with_paths(obj, prefix=""):
    """
    Flatten nested dict into path -> scalar value.

    Example:
    {"summary": {"flushes": {"mean": 10, "std": 2}}}
    becomes:
    {
        "summary.flushes.mean": 10,
        "summary.flushes.std": 2
    }
    """
    flat = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)

            if isinstance(value, dict):
                flat.update(flatten_with_paths(value, path))
            elif isinstance(value, list):
                continue
            else:
                flat[path] = value

    return flat


def is_uncertainty_key(key):
    """
    Reject uncertainty fields.

    Important:
    Do NOT reject substring '.se' blindly, because 'summary.settled.mean'
    contains '.se' at the beginning of '.settled'.
    """
    low = key.lower()
    parts = re.split(r"[._]", low)

    bad_tokens = {
        "std",
        "se",
        "ci",
        "ci95",
        "stderr",
        "standarderror",
        "standard_error",
    }

    return any(part in bad_tokens for part in parts)


def find_metric(flat, include_terms, exclude_terms=None, prefer_terms=None):
    """
    Search flattened path-value dictionary by key-name patterns.

    Rules:
    1. Reject std/se/ci fields.
    2. Strongly prefer .mean fields.
    3. Prefer exact metric names.
    """

    if exclude_terms is None:
        exclude_terms = []

    if prefer_terms is None:
        prefer_terms = []

    candidates = []

    for key, value in flat.items():
        low_key = key.lower()

        if is_uncertainty_key(low_key):
            continue

        if not any(term in low_key for term in include_terms):
            continue

        if any(term in low_key for term in exclude_terms):
            continue

        val = to_float(value)

        if np.isnan(val):
            continue

        score = 0

        # Strongly prefer mean fields.
        if low_key.endswith(".mean"):
            score += 10000
        elif ".mean" in low_key:
            score += 9000

        # Prefer summary fields over config/meta fields.
        if low_key.startswith("summary."):
            score += 500

        # Prefer exact/simple metric names.
        for i, term in enumerate(prefer_terms):
            if term in low_key:
                score += 200 - i

        # Avoid configuration fields.
        bad_contexts = [
            "config",
            "objective",
            "formula",
            "reward_mode",
            "training_objective",
            "settled_scale",
            "tau_scaled",
            "money_tau",
            "money_p",
        ]

        if any(bad in low_key for bad in bad_contexts):
            score -= 1000

        # Prefer shorter paths slightly.
        score -= low_key.count(".")

        candidates.append((score, key, val))

    if not candidates:
        return np.nan, None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][2], candidates[0][1]


# ============================================================
# 6. JSON extraction
# ============================================================

def extract_regime_records_from_test_results(obj):
    """
    Extract per-regime records from cross_regime_results.json.

    Supports:
    1. test_results = {"US": {...}, "TLS": {...}}
    2. test_results = [{"test_regime": "US", ...}, ...]
    3. nested dict/list structures
    """

    records = []

    def walk(x, inherited_regime=None):
        if isinstance(x, list):
            for item in x:
                walk(item, inherited_regime)
            return

        if not isinstance(x, dict):
            return

        # Case 1: dict directly keyed by regime names.
        direct_regime_keys = [
            key for key in x.keys()
            if normalize_regime(key) in REGIME_ORDER
        ]

        if direct_regime_keys:
            for key in direct_regime_keys:
                regime = normalize_regime(key)
                value = x[key]

                if isinstance(value, dict):
                    rec = dict(value)
                    rec["test_regime"] = regime
                    records.append(rec)

                elif isinstance(value, (int, float)):
                    records.append({
                        "test_regime": regime,
                        "money": value
                    })
            return

        # Case 2: current dict is already one regime result.
        possible_regime = (
            x.get("test_regime")
            or x.get("regime")
            or x.get("eval_regime")
            or x.get("name")
            or x.get("scenario")
            or inherited_regime
        )

        flat = flatten_with_paths(x)

        metric_terms = [
            "value_accept",
            "val_acc",
            "drops",
            "flushes",
            "settled",
            "accepted",
            "requested",
            "money",
            "drop_rate",
            "count_accept",
        ]

        has_metric = any(
            any(term in key.lower() for term in metric_terms)
            for key in flat.keys()
        )

        if possible_regime is not None and has_metric:
            rec = dict(x)
            rec["test_regime"] = normalize_regime(possible_regime)
            records.append(rec)
            return

        # Case 3: recursive search.
        for key, value in x.items():
            next_regime = inherited_regime

            if normalize_regime(key) in REGIME_ORDER:
                next_regime = normalize_regime(key)

            if isinstance(value, (dict, list)):
                walk(value, next_regime)

    if isinstance(obj, dict) and "test_results" in obj:
        walk(obj["test_results"])
    else:
        walk(obj)

    return records


def read_one_json(path, model_name):
    seed = parse_seed(path)

    with open(path, "r") as f:
        obj = json.load(f)

    records = extract_regime_records_from_test_results(obj)

    if not records:
        print("\nWARNING: no regime records extracted from:")
        print(path)

        if isinstance(obj, dict):
            print("Top-level keys:", list(obj.keys())[:30])

            if "test_results" in obj:
                tr = obj["test_results"]
                print("test_results type:", type(tr))

                if isinstance(tr, dict):
                    print("test_results keys:", list(tr.keys())[:30])

                    for k, v in tr.items():
                        print("sample test_results key:", k)
                        print("sample value type:", type(v))
                        if isinstance(v, dict):
                            print("sample value keys:", list(v.keys())[:30])
                        break

                elif isinstance(tr, list):
                    print("test_results length:", len(tr))
                    if len(tr) > 0:
                        print("first item type:", type(tr[0]))
                        if isinstance(tr[0], dict):
                            print("first item keys:", list(tr[0].keys())[:30])

        return []

    rows = []

    for rec in records:
        flat = flatten_with_paths(rec)

        regime = normalize_regime(
            rec.get("test_regime")
            or rec.get("regime")
            or rec.get("eval_regime")
            or rec.get("name")
            or rec.get("scenario")
        )

        if regime not in REGIME_ORDER:
            continue

        value_accept_ratio, value_accept_key = find_metric(
            flat,
            include_terms=["value_accept_ratio", "value_accept", "val_acc"],
            exclude_terms=["count"],
            prefer_terms=[
                "value_accept_ratio",
                "mean_value_accept_ratio",
                "val_acc"
            ]
        )

        drops, drops_key = find_metric(
            flat,
            include_terms=["drops", "drop_count", "oversize_drops"],
            exclude_terms=["rate", "ratio"],
            prefer_terms=["drops", "oversize_drops", "mean_drops"]
        )

        flushes, flushes_key = find_metric(
            flat,
            include_terms=["flushes", "num_flushes", "flush_count"],
            exclude_terms=["cost", "rate", "ratio"],
            prefer_terms=["flushes", "mean_flushes"]
        )

        drop_rate, drop_rate_key = find_metric(
            flat,
            include_terms=["drop_rate"],
            exclude_terms=[],
            prefer_terms=["drop_rate"]
        )

        count_accept_ratio, count_accept_key = find_metric(
            flat,
            include_terms=["count_accept_ratio", "count_accept"],
            exclude_terms=[],
            prefer_terms=["count_accept_ratio"]
        )

        total_requested_value, requested_key = find_metric(
            flat,
            include_terms=[
                "total_requested_value",
                "requested_value",
                "total_value",
                "requested"
            ],
            exclude_terms=[
                "ratio",
                "rate",
                "count",
                "std",
                "ci",
            ],
            prefer_terms=[
                "total_requested_value",
                "requested_value",
                "total_value"
            ]
        )

        accepted_value, accepted_key = find_metric(
            flat,
            include_terms=[
                "accepted_value",
                "settled_value",
                "total_settled_value",
                "settled"
            ],
            exclude_terms=[
                "ratio",
                "rate",
                "count",
                "num",
                "drop",
                "flush",
                "scale",
            ],
            prefer_terms=[
                "accepted_value",
                "settled_value",
                "total_settled_value",
                "settled"
            ]
        )

        # Reconstruct accepted value if only ratio and total requested value exist.
        if np.isnan(accepted_value):
            if not np.isnan(value_accept_ratio) and not np.isnan(total_requested_value):
                accepted_value = value_accept_ratio * total_requested_value
                accepted_key = "reconstructed_from_value_accept_ratio_x_total_requested_value"

        explicit_money, money_key = find_metric(
            flat,
            include_terms=[
                "eval_money",
                "money",
                "reward_money",
                "mean_money"
            ],
            exclude_terms=[
                "money_p",
                "money_tau",
                "method",
                "mode",
                "formula",
                "objective",
                "flush_cost",
            ],
            prefer_terms=[
                "eval_money",
                "money",
                "reward_money",
                "mean_money"
            ]
        )

        # For paper consistency, compute money from mean accepted value and mean flushes.
        # If these are unavailable, fall back to explicit eval_money.mean.
        if not np.isnan(accepted_value) and not np.isnan(flushes):
            money = TARGET_P * accepted_value - TARGET_TAU * flushes
            money_source = "computed_from_mean_accepted_value_minus_tau_mean_flushes"
        elif not np.isnan(explicit_money):
            money = explicit_money
            money_source = f"explicit:{money_key}"
        else:
            money = np.nan
            money_source = "missing"

        rows.append({
            "model": model_name,
            "seed": seed,
            "k": TARGET_K,
            "C": TARGET_C,
            "F": TARGET_F,
            "T": TARGET_T,
            "test_regime": regime,
            "value_accept_ratio": value_accept_ratio,
            "drops": drops,
            "flushes": flushes,
            "drop_rate": drop_rate,
            "count_accept_ratio": count_accept_ratio,
            "total_requested_value": total_requested_value,
            "accepted_value": accepted_value,
            "p": TARGET_P,
            "tau": TARGET_TAU,
            "money": money,
            "money_source": money_source,
            "debug_value_accept_key": value_accept_key,
            "debug_drops_key": drops_key,
            "debug_flushes_key": flushes_key,
            "debug_requested_key": requested_key,
            "debug_accepted_key": accepted_key,
            "debug_money_key": money_key,
            "source_file": path,
        })

    return rows


# ============================================================
# 7. Plotting
# ============================================================

def save_heatmap(matrix, title, output_path, value_format=".0f"):
    fig_width = max(6, 1.7 * len(matrix.columns))
    fig_height = max(5, 0.42 * len(matrix.index))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    data = matrix.values.astype(float)
    im = ax.imshow(data, aspect="auto")

    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))

    ax.set_xticklabels(matrix.columns, rotation=25, ha="right")
    ax.set_yticklabels(matrix.index)

    ax.set_title(title)
    ax.set_ylabel("Evaluation regime")

    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            value = data[i, j]
            if not np.isnan(value):
                ax.text(
                    j,
                    i,
                    format(value, value_format),
                    ha="center",
                    va="center",
                    fontsize=8
                )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# ============================================================
# 8. Main
# ============================================================

def main():
    all_rows = []

    print("\n=== Searching input files ===")

    for model_name, pattern in INPUT_PATTERNS.items():
        paths = sorted(glob.glob(pattern, recursive=True))

        print(f"\n{model_name}")
        print("Pattern:", pattern)
        print("Found files:", len(paths))

        for path in paths:
            seed = parse_seed(path)
            print(f"  seed={seed}  {path}")

            rows = read_one_json(path, model_name)
            all_rows.extend(rows)

    if not all_rows:
        raise ValueError("No rows extracted. Check JSON structure or input patterns.")

    df_all = pd.DataFrame(all_rows)

    all_raw_path = os.path.join(
        OUTPUT_DIR,
        f"k{TARGET_K}_C{TARGET_C}_all_extracted_rows_before_drop_tau{int(TARGET_TAU)}.csv"
    )
    df_all.to_csv(all_raw_path, index=False)

    print("\n=== All extracted rows before dropping missing money ===")
    print("Shape:", df_all.shape)
    print(df_all.head(10).to_string(index=False))
    print("Saved:")
    print(all_raw_path)

    # Check whether wrong std fields are still selected.
    debug_cols = [
        "debug_value_accept_key",
        "debug_drops_key",
        "debug_flushes_key",
        "debug_requested_key",
        "debug_accepted_key",
        "debug_money_key",
    ]

    print("\n=== Debug selected metric keys ===")
    existing_debug_cols = [c for c in debug_cols if c in df_all.columns]
    print(df_all[existing_debug_cols].head(12).to_string(index=False))

    std_key_mask = pd.Series(False, index=df_all.index)

    for col in existing_debug_cols:
        std_key_mask = std_key_mask | df_all[col].astype(str).str.contains(
            r"\.std|_std|\.ci|_ci|ci95|stderr|standard_error",
            case=False,
            regex=True,
            na=False
        )

    if std_key_mask.any():
        bad_key_path = os.path.join(
            OUTPUT_DIR,
            f"debug_bad_std_keys_k{TARGET_K}_C{TARGET_C}.csv"
        )
        df_all[std_key_mask].to_csv(bad_key_path, index=False)

        raise ValueError(
            "Some selected metric keys are still std/ci/se fields. "
            f"Saved bad-key debug file:\n{bad_key_path}"
        )

    missing_money = df_all[df_all["money"].isna()].copy()

    if not missing_money.empty:
        debug_missing_path = os.path.join(
            OUTPUT_DIR,
            f"debug_missing_money_rows_k{TARGET_K}_C{TARGET_C}.csv"
        )
        missing_money.to_csv(debug_missing_path, index=False)

        print(f"\nRows with missing money: {len(missing_money)}")
        print("Saved missing-money debug file:")
        print(debug_missing_path)

        print("\nSample missing-money rows:")
        print(missing_money.head(10).to_string(index=False))

    df = df_all.dropna(subset=["money"]).copy()

    print("\nRows before dropping missing money:", len(df_all))
    print("Rows after dropping missing money:", len(df))

    if df.empty:
        raise ValueError(
            "All extracted rows have missing money. "
            "Open debug_missing_money_rows_k24_C1200.csv and inspect metric names."
        )

    raw_path = os.path.join(
        OUTPUT_DIR,
        f"k{TARGET_K}_C{TARGET_C}_per_regime_rows_from_json_tau{int(TARGET_TAU)}.csv"
    )
    df.to_csv(raw_path, index=False)

    print("\n=== Extracted usable rows ===")
    print("Shape:", df.shape)
    print(df.head(10).to_string(index=False))

    print("\n=== Seed coverage ===")
    coverage = (
        df.groupby("model")["seed"]
        .nunique()
        .reindex(MODEL_ORDER)
    )
    print(coverage.to_string())

    print("\n=== Regime count by model ===")
    regime_count = (
        df.groupby("model")["test_regime"]
        .nunique()
        .reindex(MODEL_ORDER)
    )
    print(regime_count.to_string())

    expected_rows = len(MODEL_ORDER) * 10 * len(REGIME_ORDER)
    print("\nExpected rows if 3 models × 10 seeds × 12 regimes:", expected_rows)
    print("Actual usable rows:", len(df))

    # ------------------------------------------------------------
    # Aggregate over seeds
    # ------------------------------------------------------------

    summary = (
        df.groupby(["test_regime", "model"], as_index=False)
        .agg(
            mean_money=("money", "mean"),
            std_money=("money", "std"),
            n_seeds=("seed", "nunique"),
            mean_valacc=("value_accept_ratio", "mean"),
            mean_drops=("drops", "mean"),
            mean_flushes=("flushes", "mean"),
            mean_accepted_value=("accepted_value", "mean"),
        )
    )

    summary["se_money"] = summary["std_money"] / np.sqrt(summary["n_seeds"])
    summary["ci95_money"] = 1.96 * summary["se_money"]

    summary_path = os.path.join(
        OUTPUT_DIR,
        f"k{TARGET_K}_C{TARGET_C}_per_regime_summary_tau{int(TARGET_TAU)}.csv"
    )
    summary.to_csv(summary_path, index=False)

    # ------------------------------------------------------------
    # Money matrix
    # ------------------------------------------------------------

    money_matrix = summary.pivot(
        index="test_regime",
        columns="model",
        values="mean_money"
    )

    money_matrix = money_matrix.reindex(
        index=REGIME_ORDER,
        columns=MODEL_ORDER
    )

    money_matrix_path = os.path.join(
        OUTPUT_DIR,
        f"k{TARGET_K}_C{TARGET_C}_money_matrix_tau{int(TARGET_TAU)}.csv"
    )
    money_matrix.to_csv(money_matrix_path)

    save_heatmap(
        money_matrix,
        f"Money by Regime and Model, k={TARGET_K}, C={TARGET_C}, tau={int(TARGET_TAU)}",
        os.path.join(
            OUTPUT_DIR,
            f"k{TARGET_K}_C{TARGET_C}_heatmap_money_by_regime_tau{int(TARGET_TAU)}.png"
        ),
        value_format=".0f"
    )

    # ------------------------------------------------------------
    # Delta matrix
    # ------------------------------------------------------------

    delta_matrix = pd.DataFrame(index=money_matrix.index)

    delta_matrix["SC-FAC - IFAC"] = (
        money_matrix["SC-FAC"] - money_matrix["IFAC"]
    )

    delta_matrix["IFAC - Basic PPO"] = (
        money_matrix["IFAC"] - money_matrix["Basic PPO"]
    )

    delta_matrix["SC-FAC - Basic PPO"] = (
        money_matrix["SC-FAC"] - money_matrix["Basic PPO"]
    )

    delta_matrix_path = os.path.join(
        OUTPUT_DIR,
        f"k{TARGET_K}_C{TARGET_C}_delta_matrix_tau{int(TARGET_TAU)}.csv"
    )
    delta_matrix.to_csv(delta_matrix_path)

    save_heatmap(
        delta_matrix,
        f"Money Gain by Regime, k={TARGET_K}, C={TARGET_C}, tau={int(TARGET_TAU)}",
        os.path.join(
            OUTPUT_DIR,
            f"k{TARGET_K}_C{TARGET_C}_heatmap_money_gain_by_regime_tau{int(TARGET_TAU)}.png"
        ),
        value_format=".0f"
    )

    # ------------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------------

    interpretation = money_matrix.copy()
    interpretation["SC-FAC - IFAC"] = delta_matrix["SC-FAC - IFAC"]
    interpretation["SC-FAC - Basic PPO"] = delta_matrix["SC-FAC - Basic PPO"]
    interpretation["SC-FAC beats IFAC"] = interpretation["SC-FAC - IFAC"] > 0
    interpretation["SC-FAC beats Basic PPO"] = interpretation["SC-FAC - Basic PPO"] > 0

    interpretation_path = os.path.join(
        OUTPUT_DIR,
        f"k{TARGET_K}_C{TARGET_C}_interpretation_tau{int(TARGET_TAU)}.csv"
    )
    interpretation.to_csv(interpretation_path)

    # ------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------

    print("\n=== Money matrix ===")
    print(money_matrix.round(2).to_string())

    print("\n=== Delta matrix ===")
    print(delta_matrix.round(2).to_string())

    print("\n=== Top 3 regimes where SC-FAC improves most over IFAC ===")
    print(
        delta_matrix["SC-FAC - IFAC"]
        .sort_values(ascending=False)
        .head(3)
        .round(2)
        .to_string()
    )

    print("\n=== Bottom 3 regimes where SC-FAC is weakest relative to IFAC ===")
    print(
        delta_matrix["SC-FAC - IFAC"]
        .sort_values(ascending=True)
        .head(3)
        .round(2)
        .to_string()
    )

    print("\nSaved files:")
    print(raw_path)
    print(summary_path)
    print(money_matrix_path)
    print(delta_matrix_path)
    print(interpretation_path)
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()