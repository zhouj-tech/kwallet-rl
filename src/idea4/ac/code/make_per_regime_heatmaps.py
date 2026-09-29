import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Input / Output
# ============================================================

INPUT_CSV = "./src/idea4/ac/results/final_comparison_tables/money_ci_regime_seed_long.csv"

OUTPUT_DIR = "./src/idea4/ac/results/final_comparison_tables/per_regime_heatmaps"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. Fixed setting for paper figure
# ============================================================

TARGET_K = 12
TARGET_C = 1200
TARGET_P = 1.0
TARGET_TAU = 10.0


# ============================================================
# 3. Regime and model order
# ============================================================

REGIME_ORDER = [
    "US", "TLS", "LNS", "TLNS", "TPLS", "PLS",
    "UB", "TLB", "LNB", "TLNB", "TPLB", "PLB"
]

MODEL_ORDER = ["Basic PPO", "IFAC", "SC-FAC"]


# ============================================================
# 4. Helper functions
# ============================================================

def clean_colnames(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def normalize_model_name(raw_name):
    """
    Convert raw model labels in csv into paper names.

    Expected examples:
    - basic_ppo -> Basic PPO
    - factorized_ac -> IFAC
    - conditional_factorized_ac -> SC-FAC
    - dual_branch_factorized_ac -> SC-FAC
    """

    name = str(raw_name).strip().lower()

    # Basic PPO / flat joint actor-critic baseline
    if name in ["basic_ppo", "ja_ppo", "basic ppo", "ja-ppo"]:
        return "Basic PPO"

    # SC-FAC / conditional / dual-branch factorized model
    if (
        "conditional" in name
        or "cond" in name
        or "sc_fac" in name
        or "sc-fac" in name
        or "dual_branch_factorized_ac" in name
    ):
        return "SC-FAC"

    # Independent factorized actor-critic
    # Important: put this after conditional check.
    if "factorized" in name:
        return "IFAC"

    return None


def save_heatmap(matrix, title, output_path, value_format=".0f"):
    """
    Simple matplotlib heatmap.
    Rows = regimes.
    Columns = models or delta column.
    """
    fig_width = max(6, 1.6 * len(matrix.columns))
    fig_height = max(5, 0.42 * len(matrix.index))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    data = matrix.values.astype(float)
    im = ax.imshow(data, aspect="auto")

    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))

    ax.set_xticklabels(matrix.columns, rotation=25, ha="right")
    ax.set_yticklabels(matrix.index)

    ax.set_title(title)
    ax.set_xlabel("")
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


def print_available_values(df):
    print("\n=== Available model_label values ===")
    print(sorted(df["model_label"].astype(str).unique()))

    print("\n=== Available k values ===")
    print(sorted(df["k"].dropna().unique()))

    print("\n=== Available C values ===")
    print(sorted(df["C"].dropna().unique()))

    print("\n=== Available tau values ===")
    print(sorted(df["tau"].dropna().unique()))


# ============================================================
# 5. Main logic
# ============================================================

def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input file does not exist:\n{INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    df = clean_colnames(df)

    print("\nLoaded file:")
    print(INPUT_CSV)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    required_cols = [
        "k", "model_label", "seed", "C", "test_regime",
        "p", "tau", "money"
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: "
            + str(missing)
            + "\nCurrent columns are: "
            + str(list(df.columns))
        )

    print_available_values(df)

    # ------------------------------------------------------------
    # Filter target setting
    # ------------------------------------------------------------
    df = df[
        (df["k"] == TARGET_K)
        & (np.isclose(df["C"].astype(float), TARGET_C))
        & (np.isclose(df["p"].astype(float), TARGET_P))
        & (np.isclose(df["tau"].astype(float), TARGET_TAU))
    ].copy()

    print("\nAfter k/C/p/tau filtering:")
    print("Shape:", df.shape)

    if df.empty:
        raise ValueError(
            "No rows remain after filtering. "
            "Check TARGET_K, TARGET_C, TARGET_P, TARGET_TAU."
        )

    # ------------------------------------------------------------
    # Normalize model names
    # ------------------------------------------------------------
    df["model_clean"] = df["model_label"].apply(normalize_model_name)

    print("\nModel mapping preview:")
    print(
        df[["model_label", "model_clean"]]
        .drop_duplicates()
        .sort_values(["model_clean", "model_label"])
        .to_string(index=False)
    )

    df = df[df["model_clean"].isin(MODEL_ORDER)].copy()

    print("\nAfter model filtering:")
    print("Shape:", df.shape)

    if df.empty:
        raise ValueError(
            "No rows remain after model filtering. "
            "Check normalize_model_name()."
        )

    # ------------------------------------------------------------
    # Keep only known 12 regimes
    # ------------------------------------------------------------
    df = df[df["test_regime"].isin(REGIME_ORDER)].copy()

    print("\nAfter regime filtering:")
    print("Shape:", df.shape)

    if df.empty:
        raise ValueError(
            "No rows remain after regime filtering. "
            "Check REGIME_ORDER or test_regime names."
        )

    # ------------------------------------------------------------
    # Aggregate over seeds
    # ------------------------------------------------------------
    summary = (
        df.groupby(["test_regime", "model_clean"], as_index=False)
        .agg(
            mean_money=("money", "mean"),
            std_money=("money", "std"),
            n_seeds=("seed", "nunique")
        )
    )

    summary["se_money"] = summary["std_money"] / np.sqrt(summary["n_seeds"])
    summary["ci95_money"] = 1.96 * summary["se_money"]

    summary_path = os.path.join(
        OUTPUT_DIR,
        f"per_regime_summary_k{TARGET_K}_C{TARGET_C}_tau{int(TARGET_TAU)}.csv"
    )
    summary.to_csv(summary_path, index=False)

    # ------------------------------------------------------------
    # Heatmap 1: absolute money
    # ------------------------------------------------------------
    money_matrix = summary.pivot(
        index="test_regime",
        columns="model_clean",
        values="mean_money"
    )

    money_matrix = money_matrix.reindex(
        index=REGIME_ORDER,
        columns=MODEL_ORDER
    )

    money_matrix_path = os.path.join(
        OUTPUT_DIR,
        f"heatmap1_money_matrix_k{TARGET_K}_C{TARGET_C}_tau{int(TARGET_TAU)}.csv"
    )
    money_matrix.to_csv(money_matrix_path)

    save_heatmap(
        money_matrix,
        f"Money by Regime and Model, k={TARGET_K}, C={TARGET_C}, tau={int(TARGET_TAU)}",
        os.path.join(
            OUTPUT_DIR,
            f"heatmap1_money_by_regime_k{TARGET_K}_C{TARGET_C}_tau{int(TARGET_TAU)}.png"
        ),
        value_format=".0f"
    )

    # ------------------------------------------------------------
    # Heatmap 2: SC-FAC - IFAC
    # ------------------------------------------------------------
    if "SC-FAC" not in money_matrix.columns or "IFAC" not in money_matrix.columns:
        raise ValueError(
            "Cannot compute SC-FAC - IFAC because one column is missing.\n"
            f"Current columns: {list(money_matrix.columns)}"
        )

    delta_scfac_ifac = money_matrix["SC-FAC"] - money_matrix["IFAC"]
    delta_basic_ifac = money_matrix["IFAC"] - money_matrix["Basic PPO"]
    delta_scfac_basic = money_matrix["SC-FAC"] - money_matrix["Basic PPO"]

    delta_matrix = pd.DataFrame({
        "SC-FAC - IFAC": delta_scfac_ifac,
        "IFAC - Basic PPO": delta_basic_ifac,
        "SC-FAC - Basic PPO": delta_scfac_basic,
    })

    delta_matrix_path = os.path.join(
        OUTPUT_DIR,
        f"heatmap2_delta_matrix_k{TARGET_K}_C{TARGET_C}_tau{int(TARGET_TAU)}.csv"
    )
    delta_matrix.to_csv(delta_matrix_path)

    save_heatmap(
        delta_matrix,
        f"Money Gain by Regime, k={TARGET_K}, C={TARGET_C}, tau={int(TARGET_TAU)}",
        os.path.join(
            OUTPUT_DIR,
            f"heatmap2_money_gain_by_regime_k{TARGET_K}_C{TARGET_C}_tau{int(TARGET_TAU)}.png"
        ),
        value_format=".0f"
    )

    # ------------------------------------------------------------
    # Compact interpretation table
    # ------------------------------------------------------------
    interpretation = money_matrix.copy()
    interpretation["SC-FAC - IFAC"] = delta_scfac_ifac
    interpretation["SC-FAC beats IFAC"] = delta_scfac_ifac > 0

    interpretation_path = os.path.join(
        OUTPUT_DIR,
        f"per_regime_interpretation_k{TARGET_K}_C{TARGET_C}_tau{int(TARGET_TAU)}.csv"
    )
    interpretation.to_csv(interpretation_path)

    # ------------------------------------------------------------
    # Print outputs
    # ------------------------------------------------------------
    print("\n=== Money matrix ===")
    print(money_matrix.round(2).to_string())

    print("\n=== Delta matrix ===")
    print(delta_matrix.round(2).to_string())

    print("\n=== Top 3 regimes where SC-FAC improves most over IFAC ===")
    print(delta_scfac_ifac.sort_values(ascending=False).head(3).round(2).to_string())

    print("\n=== Bottom 3 regimes where SC-FAC is weakest relative to IFAC ===")
    print(delta_scfac_ifac.sort_values(ascending=True).head(3).round(2).to_string())

    print("\nSaved files:")
    print(summary_path)
    print(money_matrix_path)
    print(delta_matrix_path)
    print(interpretation_path)
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()