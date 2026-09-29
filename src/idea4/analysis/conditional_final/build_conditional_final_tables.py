from pathlib import Path
import pandas as pd
import numpy as np

SUMMARY_PATH = Path("src/idea4/ac/results/conditional_E32_H256_k24_10seed_status/k24_10seed_summary_basic_factorized_condE32H256.csv")
PAIRED_PATH = Path("src/idea4/ac/results/conditional_E32_H256_k24_10seed_status/k24_10seed_paired_diff_condE32H256.csv")

OUT_DIR = Path("src/idea4/analysis/conditional_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)

summary = pd.read_csv(SUMMARY_PATH)
paired = pd.read_csv(PAIRED_PATH)

model_name_map = {
    "basic_ppo": "Basic PPO",
    "factorized_ac": "Factorized AC",
    "conditional_factorized_ac": "Conditional Factorized AC",
}

summary["model_display"] = summary["model"].map(model_name_map).fillna(summary["model"])

def fmt_mean_ci(mean, ci, percent=False):
    if pd.isna(mean):
        return ""
    if percent:
        return f"{100 * mean:.2f} ± {100 * ci:.2f}%"
    return f"{mean:.2f} ± {ci:.2f}"

def fmt_reduction(x):
    if pd.isna(x):
        return ""
    return f"{100 * x:.0f}%"

rows = []
for _, r in summary.iterrows():
    rows.append({
        "C": int(r["C"]),
        "k": int(r["k"]),
        "Model": r["model_display"],
        "Policy Output": int(r["policy_output_size"]),
        "Output Reduction": fmt_reduction(r["output_reduction_vs_basic"]),
        "Mean ValAcc": fmt_mean_ci(r["mean_valacc_mean"], r["mean_valacc_ci95"], percent=True),
        "Worst ValAcc": fmt_mean_ci(r["worst_valacc_mean"], r["worst_valacc_ci95"], percent=True),
        "Drops": fmt_mean_ci(r["mean_drops_mean"], r["mean_drops_ci95"]),
        "Flushes": fmt_mean_ci(r["mean_flushes_mean"], r["mean_flushes_ci95"]),
        "Money": fmt_mean_ci(r["mean_money_mean"], r["mean_money_ci95"]),
        "Seeds": r["seeds"],
    })

paper_table = pd.DataFrame(rows)

# Sort in paper-friendly order.
model_order = {
    "Basic PPO": 0,
    "Factorized AC": 1,
    "Conditional Factorized AC": 2,
}
paper_table["model_order"] = paper_table["Model"].map(model_order)
paper_table = paper_table.sort_values(["C", "model_order"]).drop(columns=["model_order"])

paper_table.to_csv(OUT_DIR / "table_k24_stress_comparison_conditional_E32H256.csv", index=False)

# Markdown table
md_table = paper_table.drop(columns=["Seeds"]).to_markdown(index=False)
(OUT_DIR / "table_k24_stress_comparison_conditional_E32H256.md").write_text(md_table, encoding="utf-8")

# Paired difference paper table
paired_keep = paired[
    (
        (paired["comparison"] == "conditional_minus_basic")
        & (paired["metric"].isin(["mean_valacc", "worst_valacc", "mean_drops", "mean_money"]))
    )
    |
    (
        (paired["comparison"] == "conditional_minus_factorized")
        & (paired["metric"].isin(["mean_valacc", "worst_valacc", "mean_drops", "mean_money"]))
    )
].copy()

metric_map = {
    "mean_valacc": "Mean ValAcc",
    "worst_valacc": "Worst ValAcc",
    "mean_drops": "Drops",
    "mean_money": "Money",
}

comparison_map = {
    "conditional_minus_basic": "Conditional - Basic PPO",
    "conditional_minus_factorized": "Conditional - Factorized AC",
}

def fmt_diff(row):
    metric = row["metric"]
    mean = row["diff_mean"]
    ci = row["diff_ci95"]
    if metric in ["mean_valacc", "worst_valacc"]:
        return f"{100 * mean:+.2f} ± {100 * ci:.2f} pp"
    if metric == "mean_drops":
        return f"{mean:+.2f} ± {ci:.2f}"
    if metric == "mean_money":
        return f"{mean:+.2f} ± {ci:.2f}"
    return f"{mean:+.4f} ± {ci:.4f}"

paired_rows = []
for _, r in paired_keep.iterrows():
    paired_rows.append({
        "C": int(r["C"]),
        "k": int(r["k"]),
        "Comparison": comparison_map.get(r["comparison"], r["comparison"]),
        "Metric": metric_map.get(r["metric"], r["metric"]),
        "Difference ± 95% CI": fmt_diff(r),
        "CI Low": r["diff_ci95_low"],
        "CI High": r["diff_ci95_high"],
        "Interpretation": r["interpretation"],
    })

paired_table = pd.DataFrame(paired_rows)
paired_table.to_csv(OUT_DIR / "table_k24_paired_difference_conditional_E32H256.csv", index=False)
(OUT_DIR / "table_k24_paired_difference_conditional_E32H256.md").write_text(
    paired_table.drop(columns=["CI Low", "CI High"]).to_markdown(index=False),
    encoding="utf-8",
)

writeup = r"""
# Conditional Factorized AC Final Write-up

## Main Model Configuration

The final Conditional Factorized Actor-Critic uses the following fixed hyperparameters:

- settle action embedding dimension: 32
- conditional flush hidden size: 256
- training reward: original environment reward
- validation metric: value acceptance ratio
- training episodes: 1000
- evaluation episodes: 200
- seeds: 123, 323, 532, 777, 999, 2027, 3407, 4501, 6101, 8888

This setting is selected based on the 5-seed top-2 parameter confirmation experiment. Although E=32,H=128 and E=32,H=256 each won two of the four tested settings by mean value acceptance ratio, E=32,H=256 achieved better global average mean value acceptance ratio, better worst-regime value acceptance ratio, higher money score, and better overall rank. Therefore, E=32,H=256 is used as the main Conditional Factorized AC configuration, while E=32,H=128 is retained as a lightweight ablation variant.

## Method: Conditional Factorized Actor-Critic

The flat Basic PPO policy directly models the joint action space over settlement and flushing decisions. For k wallets, this requires a policy output size of (k+1)^2. This becomes expensive when k grows.

The independent Factorized AC reduces the policy output size by splitting the action into two independent categorical decisions:

π(a_s, a_f | s) = π_s(a_s | s) π_f(a_f | s)

where a_s is the settlement action and a_f is the flushing action. This reduces the output size from (k+1)^2 to 2(k+1). However, the independence assumption can be too restrictive because the best flushing decision may depend on the selected settlement decision.

The proposed Conditional Factorized AC keeps the same linear output structure but restores part of the joint-action dependency:

π(a_s, a_f | s) = π_s(a_s | s) π_f(a_f | s, a_s)

The actor first predicts the settlement distribution from the state. Then the selected settlement action is embedded and concatenated with the state representation to condition the flushing distribution. In this way, the flushing head can adapt to the settlement decision while the policy output size remains 2(k+1). For k=24, this gives 50 policy outputs compared with 625 outputs for Basic PPO, corresponding to a 92% reduction.

## Ablation: Why Conditional Factorization Helps

The comparison between independent Factorized AC and Conditional Factorized AC tests whether the dependency between settlement and flushing decisions matters. Independent factorization assumes that the two decisions are conditionally independent given the state. Conditional factorization relaxes this assumption by allowing the flushing policy to observe the selected settlement action.

The 10-seed k=24 stress results show that this conditional dependency is useful. At C=1200,k=24, Conditional Factorized AC significantly improves over independent Factorized AC in mean value acceptance ratio, worst-regime value acceptance ratio, and drops. This suggests that the conditional structure recovers useful joint-action information without returning to the full quadratic action output.

## Result Interpretation

Under the harsher C=800,k=24 setting, Conditional Factorized AC significantly outperforms Basic PPO. It improves mean value acceptance ratio, worst-regime value acceptance ratio, drops, and money score, while using only 50 policy outputs instead of 625. This indicates that conditional factorization is especially effective under high-pressure large-k settings.

Under the C=1200,k=24 setting, Conditional Factorized AC is statistically comparable to Basic PPO in value acceptance and drops, while achieving a significantly higher money score. It also significantly outperforms independent Factorized AC in mean value acceptance ratio, worst-regime value acceptance ratio, and drops. This supports the main claim that conditional factorization can close the performance gap between independent factorization and flat joint-action PPO while preserving the scalability advantage.

Overall, Conditional Factorized AC provides a stronger trade-off than the previous factorized policy: it preserves the 92% output reduction at k=24 while improving the ability to model the dependency between settlement and flushing decisions.
"""

(OUT_DIR / "conditional_E32H256_method_ablation_results_writeup.md").write_text(writeup.strip() + "\n", encoding="utf-8")

print("Saved final outputs:")
print(OUT_DIR / "table_k24_stress_comparison_conditional_E32H256.csv")
print(OUT_DIR / "table_k24_stress_comparison_conditional_E32H256.md")
print(OUT_DIR / "table_k24_paired_difference_conditional_E32H256.csv")
print(OUT_DIR / "table_k24_paired_difference_conditional_E32H256.md")
print(OUT_DIR / "conditional_E32H256_method_ablation_results_writeup.md")
