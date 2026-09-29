#!/usr/bin/env python3
"""Post-hoc money metrics and confidence intervals for final comparisons.

This script reads existing evaluation JSON files and writes derived analysis
tables only. It does not rerun training or modify raw experiment outputs.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
FINAL_DIR = PROJECT_ROOT / "src" / "idea4" / "ac" / "results" / "final_comparison_tables"

P_VALUES = [1.0]
TAU_VALUES = [0.0, 1.0, 5.0, 10.0, 20.0]

MODELS = ["dqn_baseline", "basic_ppo", "factorized_ac", "dual_branch_factorized_ac"]
MODEL_ORDER = {model: idx for idx, model in enumerate(MODELS)}
K_VALUES = [3, 6, 12]
SEEDS = [123, 323]
EXPECTED_REGIMES = ["US", "TLS", "LNS", "TLNS", "TPLS", "PLS", "UB", "TLB", "LNB", "TLNB", "TPLB", "PLB"]

RAW_SOURCES = {
    (3, "dqn_baseline", 123): PROJECT_ROOT
    / "src/idea3/results/old_fair_benchmark_results/runs/baseline_trainMIX12_EQ_C1200_k3_T1000_F3_seed123/20260504_015024_611040/cross_regime_results.json",
    (3, "basic_ppo", 123): PROJECT_ROOT
    / "src/idea4/ac/results/basic_ppo/runs/basic_ppo_trainMIX12_EQ_C1200_k3_T1000_F3_seed123/20260507_212549_286620/cross_regime_results.json",
    (3, "basic_ppo", 323): PROJECT_ROOT
    / "src/idea4/ac/results/basic_ppo/runs/basic_ppo_trainMIX12_EQ_C1200_k3_T1000_F3_seed323/20260507_213733_993592/cross_regime_results.json",
    (3, "factorized_ac", 123): PROJECT_ROOT
    / "src/idea4/ac/results/factorized_ac/runs/factorized_ac_trainMIX12_EQ_C1200_k3_T1000_F3_seed123/20260505_212833_781649/cross_regime_results.json",
    (3, "factorized_ac", 323): PROJECT_ROOT
    / "src/idea4/ac/results/factorized_ac/runs/factorized_ac_trainMIX12_EQ_C1200_k3_T1000_F3_seed323/20260505_214601_304092/cross_regime_results.json",
    (3, "dual_branch_factorized_ac", 123): PROJECT_ROOT
    / "src/idea4/ac/results/dual_branch_ac/runs/dual_branch_factorized_ac_trainMIX12_EQ_C1200_k3_T1000_F3_seed123/20260505_155416_193647/cross_regime_results.json",
    (3, "dual_branch_factorized_ac", 323): PROJECT_ROOT
    / "src/idea4/ac/results/dual_branch_ac/runs/dual_branch_factorized_ac_trainMIX12_EQ_C1200_k3_T1000_F3_seed323/20260505_165828_143067/cross_regime_results.json",
    (6, "dqn_baseline", 123): PROJECT_ROOT
    / "src/ideaextra/results/trainMIX12_EQ_crossUS_TLS_LNS_TLNS_TPLS_PLS_UB_TLB_LNB_TLNB_TPLB_PLB_C1200_k6_T1000_F3/20260507_205843/cross_regime_results.json",
    (6, "dqn_baseline", 323): PROJECT_ROOT
    / "src/ideaextra/results/trainMIX12_EQ_crossUS_TLS_LNS_TLNS_TPLS_PLS_UB_TLB_LNB_TLNB_TPLB_PLB_C1200_k6_T1000_F3/20260507_212725/cross_regime_results.json",
    (6, "basic_ppo", 123): PROJECT_ROOT
    / "src/idea4/ac/results/basic_ppo/runs/basic_ppo_trainMIX12_EQ_C1200_k6_T1000_F3_seed123/20260507_214911_366876/cross_regime_results.json",
    (6, "basic_ppo", 323): PROJECT_ROOT
    / "src/idea4/ac/results/basic_ppo/runs/basic_ppo_trainMIX12_EQ_C1200_k6_T1000_F3_seed323/20260507_220040_799666/cross_regime_results.json",
    (6, "factorized_ac", 123): PROJECT_ROOT
    / "src/idea4/ac/results/factorized_ac/runs/factorized_ac_trainMIX12_EQ_C1200_k6_T1000_F3_seed123/20260507_215312_488583/cross_regime_results.json",
    (6, "factorized_ac", 323): PROJECT_ROOT
    / "src/idea4/ac/results/factorized_ac/runs/factorized_ac_trainMIX12_EQ_C1200_k6_T1000_F3_seed323/20260507_221122_605391/cross_regime_results.json",
    (6, "dual_branch_factorized_ac", 123): PROJECT_ROOT
    / "src/idea4/ac/results/dual_branch_ac/runs/dual_branch_factorized_ac_trainMIX12_EQ_C1200_k6_T1000_F3_seed123/20260505_214117_653382/cross_regime_results.json",
    (6, "dual_branch_factorized_ac", 323): PROJECT_ROOT
    / "src/idea4/ac/results/dual_branch_ac/runs/dual_branch_factorized_ac_trainMIX12_EQ_C1200_k6_T1000_F3_seed323/20260505_221616_688729/cross_regime_results.json",
    (12, "dqn_baseline", 123): PROJECT_ROOT
    / "src/ideaextra/results/trainMIX12_EQ_crossUS_TLS_LNS_TLNS_TPLS_PLS_UB_TLB_LNB_TLNB_TPLB_PLB_C1200_k12_T1000_F3/20260506_121117/cross_regime_results.json",
    (12, "dqn_baseline", 323): PROJECT_ROOT
    / "src/ideaextra/results/trainMIX12_EQ_crossUS_TLS_LNS_TLNS_TPLS_PLS_UB_TLB_LNB_TLNB_TPLB_PLB_C1200_k12_T1000_F3/20260506_141848/cross_regime_results.json",
    (12, "basic_ppo", 123): PROJECT_ROOT
    / "src/idea4/ac/results/basic_ppo/runs/basic_ppo_trainMIX12_EQ_C1200_k12_T1000_F3_seed123/20260507_221205_537294/cross_regime_results.json",
    (12, "basic_ppo", 323): PROJECT_ROOT
    / "src/idea4/ac/results/basic_ppo/runs/basic_ppo_trainMIX12_EQ_C1200_k12_T1000_F3_seed323/20260507_222411_366919/cross_regime_results.json",
    (12, "factorized_ac", 123): PROJECT_ROOT
    / "src/idea4/ac/results/factorized_ac/runs/factorized_ac_trainMIX12_EQ_C1200_k12_T1000_F3_seed123/20260506_144445_439009/cross_regime_results.json",
    (12, "factorized_ac", 323): PROJECT_ROOT
    / "src/idea4/ac/results/factorized_ac/runs/factorized_ac_trainMIX12_EQ_C1200_k12_T1000_F3_seed323/20260506_150113_691196/cross_regime_results.json",
    (12, "dual_branch_factorized_ac", 123): PROJECT_ROOT
    / "src/idea4/ac/results/dual_branch_ac/runs/dual_branch_factorized_ac_trainMIX12_EQ_C1200_k12_T1000_F3_seed123/20260506_151752_310450/cross_regime_results.json",
    (12, "dual_branch_factorized_ac", 323): PROJECT_ROOT
    / "src/idea4/ac/results/dual_branch_ac/runs/dual_branch_factorized_ac_trainMIX12_EQ_C1200_k12_T1000_F3_seed323/20260506_155155_525191/cross_regime_results.json",
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def metric_mean(summary: dict[str, Any], key: str) -> float:
    value = summary.get(key)
    if isinstance(value, dict):
        return float(value.get("mean", math.nan))
    if value is None:
        return math.nan
    return float(value)


def t_critical(n: int) -> tuple[float, str | None]:
    if n < 2:
        return math.nan, "n<2; CI set to NaN"
    try:
        from scipy import stats  # type: ignore

        return float(stats.t.ppf(0.975, n - 1)), None
    except Exception:
        if n == 12:
            return 2.201, "scipy unavailable; using fixed t=2.201 for n=12"
        if n == 24:
            return 2.069, "scipy unavailable; using fixed t=2.069 for n=24"
        return 1.96, f"scipy unavailable; using normal fallback t=1.96 for n={n}"


def ci_stats(values: pd.Series) -> tuple[float, float, float, int, str | None]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    n = int(len(clean))
    if n == 0:
        return math.nan, math.nan, math.nan, 0, "n=0; CI set to NaN"
    mean = float(clean.mean())
    if n < 2:
        return mean, math.nan, math.nan, n, "n<2; CI set to NaN"
    crit, warning = t_critical(n)
    std = float(clean.std(ddof=1))
    margin = crit * std / math.sqrt(n)
    return mean, mean - margin, mean + margin, n, warning


def parse_raw_rows() -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for (k, model_label, seed), path in RAW_SOURCES.items():
        if not path.exists():
            warnings.append(f"missing source: {rel(path)}")
            continue
        data = json.loads(path.read_text())
        config = data.get("config") if isinstance(data.get("config"), dict) else {}
        env = config.get("env") if isinstance(config.get("env"), dict) else {}
        test_results = data.get("test_results") if isinstance(data.get("test_results"), dict) else {}
        if set(test_results) != set(EXPECTED_REGIMES):
            warnings.append(f"non-12-regime source skipped: {rel(path)}")
            continue
        for regime in EXPECTED_REGIMES:
            summary = test_results[regime].get("summary", {})
            settled = metric_mean(summary, "settled")
            total_requested = metric_mean(summary, "total_requested_value")
            value_accept_ratio = metric_mean(summary, "value_accept_ratio")
            flushes = metric_mean(summary, "flushes")
            if math.isnan(settled):
                warnings.append(f"missing settled for {model_label} k={k} seed={seed} regime={regime}; using fallback if available")
            if math.isnan(total_requested):
                warnings.append(f"missing total_requested_value for {model_label} k={k} seed={seed} regime={regime}")
            rows.append(
                {
                    "k": k,
                    "model_label": model_label,
                    "seed": seed,
                    "C": float(env.get("C", 1200.0)),
                    "F": int(env.get("F", 3)),
                    "T": int(env.get("T", 1000)),
                    "test_regime": regime,
                    "value_accept_ratio": value_accept_ratio,
                    "drops": metric_mean(summary, "drops"),
                    "flushes": flushes,
                    "drop_rate": metric_mean(summary, "drop_rate"),
                    "count_accept_ratio": metric_mean(summary, "count_accept_ratio"),
                    "settled": settled,
                    "total_requested_value": total_requested,
                    "source_file": rel(path),
                }
            )
    return pd.DataFrame(rows), warnings


def add_money_rows(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, str, list[str]]:
    rows: list[dict[str, Any]] = []
    methods: set[str] = set()
    warnings: list[str] = []
    for _, row in raw_df.iterrows():
        for p in P_VALUES:
            for tau in TAU_VALUES:
                if not math.isnan(float(row["settled"])):
                    accepted_value = float(row["settled"])
                    method = "true_money_via_settled"
                elif not math.isnan(float(row["total_requested_value"])):
                    accepted_value = float(row["value_accept_ratio"]) * float(row["total_requested_value"])
                    method = "estimated_money_from_total_value"
                    warnings.append(f"estimated money used for {row['model_label']} k={row['k']} seed={row['seed']} {row['test_regime']}")
                else:
                    accepted_value = float(row["value_accept_ratio"])
                    method = "normalized_money_proxy"
                    warnings.append(f"proxy money used for {row['model_label']} k={row['k']} seed={row['seed']} {row['test_regime']}")
                flush_cost = tau * float(row["flushes"])
                methods.add(method)
                out = row.to_dict()
                out.update(
                    {
                        "p": float(p),
                        "tau": float(tau),
                        "accepted_value": accepted_value,
                        "flush_cost": flush_cost,
                        "money": float(p) * accepted_value - flush_cost,
                        "money_method": method,
                    }
                )
                rows.append(out)
    method_summary = next(iter(methods)) if len(methods) == 1 else "mixed_methods"
    return pd.DataFrame(rows), method_summary, warnings


def coverage_note(seeds: list[int]) -> str:
    seeds = sorted(set(int(seed) for seed in seeds))
    if seeds == [123, 323]:
        return "2 seeds"
    if seeds == [123]:
        return "seed123 only"
    if seeds == [323]:
        return "seed323 only"
    if not seeds:
        return "missing"
    return "seed" + ",".join(str(seed) for seed in seeds) + " only"


def build_summary(long_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    group_cols = ["k", "model_label", "C", "F", "T", "p", "tau"]
    for keys, group in long_df.groupby(group_cols, dropna=False):
        val_mean, val_low, val_high, val_n, val_warn = ci_stats(group["value_accept_ratio"])
        drops_mean, drops_low, drops_high, drops_n, drops_warn = ci_stats(group["drops"])
        flush_mean, flush_low, flush_high, flush_n, flush_warn = ci_stats(group["flushes"])
        money_mean, money_low, money_high, money_n, money_warn = ci_stats(group["money"])
        for warn in [val_warn, drops_warn, flush_warn, money_warn]:
            if warn:
                warnings.append(warn)
        method_values = sorted(set(str(x) for x in group["money_method"].dropna()))
        rows.append(
            {
                "k": keys[0],
                "model_label": keys[1],
                "C": keys[2],
                "F": keys[3],
                "T": keys[4],
                "p": keys[5],
                "tau": keys[6],
                "num_seeds": int(group["seed"].nunique()),
                "num_regime_seed_rows": int(len(group)),
                "mean_value_accept_ratio_percent": val_mean * 100,
                "value_accept_ratio_ci_low_percent": val_low * 100 if not math.isnan(val_low) else math.nan,
                "value_accept_ratio_ci_high_percent": val_high * 100 if not math.isnan(val_high) else math.nan,
                "value_accept_ratio_n": val_n,
                "worst_regime_value_accept_ratio_percent": float(group["value_accept_ratio"].min()) * 100,
                "mean_drops": drops_mean,
                "drops_ci_low": drops_low,
                "drops_ci_high": drops_high,
                "drops_n": drops_n,
                "mean_flushes": flush_mean,
                "flushes_ci_low": flush_low,
                "flushes_ci_high": flush_high,
                "flushes_n": flush_n,
                "mean_money": money_mean,
                "money_ci_low": money_low,
                "money_ci_high": money_high,
                "money_n": money_n,
                "money_method": method_values[0] if len(method_values) == 1 else "mixed_methods",
                "coverage_note": coverage_note(list(group["seed"])),
            }
        )
    summary = pd.DataFrame(rows)
    summary["model_order"] = summary["model_label"].map(MODEL_ORDER)
    summary = summary.sort_values(["tau", "k", "model_order"], kind="stable").drop(columns=["model_order"])
    return summary.reset_index(drop=True), sorted(set(warnings))


def build_sensitivity(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for (_k, _p, _tau), group in summary.groupby(["k", "p", "tau"], dropna=False):
        ranked = group.sort_values("mean_money", ascending=False, kind="stable").copy()
        ranked["rank_by_money"] = range(1, len(ranked) + 1)
        rows.append(ranked)
    out = pd.concat(rows, ignore_index=True)
    return out[
        [
            "k",
            "p",
            "tau",
            "rank_by_money",
            "model_label",
            "mean_money",
            "money_ci_low",
            "money_ci_high",
            "mean_value_accept_ratio_percent",
            "mean_flushes",
            "mean_drops",
            "coverage_note",
        ]
    ].sort_values(["k", "tau", "rank_by_money"], kind="stable")


def build_winners(sensitivity: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (k, p, tau), group in sensitivity.groupby(["k", "p", "tau"], dropna=False):
        ranked = group.sort_values("rank_by_money", kind="stable").reset_index(drop=True)
        best = ranked.iloc[0]
        second = ranked.iloc[1] if len(ranked) > 1 else None
        rows.append(
            {
                "k": k,
                "p": p,
                "tau": tau,
                "best_model": best["model_label"],
                "best_mean_money": best["mean_money"],
                "best_money_ci_low": best["money_ci_low"],
                "best_money_ci_high": best["money_ci_high"],
                "best_mean_value_accept_ratio_percent": best["mean_value_accept_ratio_percent"],
                "best_mean_flushes": best["mean_flushes"],
                "best_mean_drops": best["mean_drops"],
                "second_best_model": second["model_label"] if second is not None else "",
                "money_gap_vs_second_best": best["mean_money"] - second["mean_money"] if second is not None else math.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["k", "tau"], kind="stable")


def ci_text(low: float, high: float, digits: int = 2) -> str:
    if math.isnan(float(low)) or math.isnan(float(high)):
        return ""
    return f"[{low:.{digits}f}, {high:.{digits}f}]"


def build_meeting_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        rows.append(
            {
                "k": int(row["k"]),
                "Model 模型": row["model_label"],
                "Seeds 种子数": int(row["num_seeds"]),
                "Mean ValAcc 平均价值接受率": f"{row['mean_value_accept_ratio_percent']:.2f}%",
                "95% CI ValAcc 价值接受率置信区间": ci_text(row["value_accept_ratio_ci_low_percent"], row["value_accept_ratio_ci_high_percent"]),
                "Worst ValAcc 最差regime价值接受率": f"{row['worst_regime_value_accept_ratio_percent']:.2f}%",
                "Drops 平均丢弃数": f"{row['mean_drops']:.2f}",
                "95% CI Drops 丢弃数置信区间": ci_text(row["drops_ci_low"], row["drops_ci_high"]),
                "Flushes 平均flush次数": f"{row['mean_flushes']:.2f}",
                "95% CI Flushes flush次数置信区间": ci_text(row["flushes_ci_low"], row["flushes_ci_high"]),
                "Money 平均净收益": f"{row['mean_money']:.2f}",
                "95% CI Money 净收益置信区间": ci_text(row["money_ci_low"], row["money_ci_high"]),
                "p settle利润系数": row["p"],
                "tau flush成本": row["tau"],
                "Note 备注": row["coverage_note"],
            }
        )
    meeting = pd.DataFrame(rows)
    meeting["model_order"] = meeting["Model 模型"].map(MODEL_ORDER)
    meeting = meeting.sort_values(["tau flush成本", "k", "model_order"], kind="stable").drop(columns=["model_order"])
    return meeting.reset_index(drop=True)


def build_availability(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for k in K_VALUES:
        for model in MODELS:
            group = raw_df[(raw_df["k"] == k) & (raw_df["model_label"] == model)]
            seeds = sorted(set(int(seed) for seed in group["seed"])) if not group.empty else []
            rows.append(
                {
                    "model_label": model,
                    "k": k,
                    "num_seeds": len(seeds),
                    "num_regime_seed_rows": int(len(group)),
                    "coverage_note": coverage_note(seeds),
                }
            )
    return pd.DataFrame(rows)


def write_readme(method: str, warnings: list[str]) -> None:
    warning_text = "\n".join(f"- {warning}" for warning in sorted(set(warnings))) or "- None"
    text = f"""# Money And CI Post-Hoc Summary

This table adds money and confidence intervals as post-hoc evaluation metrics. The trained policies are unchanged. For each model and k setting, we aggregate over available seed-regime evaluation rows. Money is computed as settlement profit minus flush cost: money = p x accepted value - tau x flush count. The confidence intervals are t-based 95% intervals over regime-seed rows. This allows us to test whether the model with the highest value acceptance also produces the highest economic return after accounting for flush cost.

本表是在已有训练结果基础上进行的后处理评估，没有重新训练模型。对于每个 model 和 k 设置，我们把 seed-regime 级别的测试结果作为样本，计算平均值和 95% 置信区间。Money 指标定义为 settle 收益减去 flush 成本：money = p x accepted value - tau x flush count。这样可以检验最高 value acceptance 的模型是否也能带来最高经济收益。

## Method

- money_method: `{method}`
- p controls settlement profit.
- tau controls flush cost.
- The raw evaluation field `settled` is treated as accepted transaction value.
- CI sample unit is one regime-seed evaluation row.
- This is not retraining and does not modify reward, environment, or policies.

## Warnings

{warning_text}
"""
    (FINAL_DIR / "README_money_ci_summary.md").write_text(text)


def main() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    raw_df, parse_warnings = parse_raw_rows()
    money_long, method, money_warnings = add_money_rows(raw_df)
    summary, ci_warnings = build_summary(money_long)
    sensitivity = build_sensitivity(summary)
    winners = build_winners(sensitivity)
    meeting = build_meeting_table(summary)
    meeting_tau5 = meeting[(meeting["p settle利润系数"] == 1.0) & (meeting["tau flush成本"] == 5.0)].copy()
    availability = build_availability(raw_df)

    all_warnings = parse_warnings + money_warnings + ci_warnings
    if availability[(availability["model_label"] == "dqn_baseline") & (availability["k"] == 3)]["num_seeds"].iloc[0] < 2:
        all_warnings.append("incomplete k=3 DQN seed coverage")
    if method == "true_money_via_settled":
        all_warnings.append("money_method = true_money_via_settled")

    money_long.to_csv(FINAL_DIR / "money_ci_regime_seed_long.csv", index=False)
    summary.to_csv(FINAL_DIR / "money_ci_model_k_summary.csv", index=False)
    sensitivity.to_csv(FINAL_DIR / "money_sensitivity_by_tau.csv", index=False)
    meeting.to_csv(FINAL_DIR / "meeting_money_ci_table.csv", index=False)
    meeting_tau5.to_csv(FINAL_DIR / "meeting_money_ci_table_tau5.csv", index=False)
    winners.to_csv(FINAL_DIR / "money_winner_by_tau.csv", index=False)
    write_readme(method, all_warnings)

    print(f"money_method = {method}")
    print("\nAVAILABILITY CHECK")
    print(availability.to_string(index=False))

    print("\nFULL meeting_money_ci_table.csv")
    print(meeting.to_string(index=False))

    print("\nFULL meeting_money_ci_table_tau5.csv")
    print(meeting_tau5.to_string(index=False))

    print("\nFULL money_sensitivity_by_tau.csv")
    print(sensitivity.to_string(index=False))

    print("\nFULL money_winner_by_tau.csv")
    print(winners.to_string(index=False))

    print("\nWARNINGS")
    print("\n".join(f"- {warning}" for warning in sorted(set(all_warnings))) if all_warnings else "- None")

    print("\nGenerated files:")
    for name in [
        "money_ci_regime_seed_long.csv",
        "money_ci_model_k_summary.csv",
        "money_sensitivity_by_tau.csv",
        "meeting_money_ci_table.csv",
        "meeting_money_ci_table_tau5.csv",
        "money_winner_by_tau.csv",
        "README_money_ci_summary.md",
    ]:
        print(f"- {rel(FINAL_DIR / name)}")


if __name__ == "__main__":
    main()
