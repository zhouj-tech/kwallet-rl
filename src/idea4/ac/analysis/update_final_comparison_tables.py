#!/usr/bin/env python3
"""Regenerate final comparison tables from formal core result runs.

This script is intentionally read-only with respect to raw experiment outputs.
It only rewrites derived tables under src/idea4/ac/results/final_comparison_tables.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
FINAL_DIR = PROJECT_ROOT / "src" / "idea4" / "ac" / "results" / "final_comparison_tables"
EXPECTED_REGIMES = ["US", "TLS", "LNS", "TLNS", "TPLS", "PLS", "UB", "TLB", "LNB", "TLNB", "TPLB", "PLB"]

CORE_MODELS = ["dqn_baseline", "factorized_ac", "dual_branch_factorized_ac"]
FOUR_MODEL_MODELS = ["dqn_baseline", "basic_ppo", "factorized_ac", "dual_branch_factorized_ac"]
MODEL_ORDER = {model: idx for idx, model in enumerate(FOUR_MODEL_MODELS)}
CORE_SEEDS = [123, 323]
CORE_K_VALUES = [3, 6, 12]

CORE_SOURCES = {
    (3, "dqn_baseline", 123): PROJECT_ROOT
    / "src/idea3/results/old_fair_benchmark_results/runs/baseline_trainMIX12_EQ_C1200_k3_T1000_F3_seed123/20260504_015024_611040/cross_regime_results.json",
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
    (12, "factorized_ac", 123): PROJECT_ROOT
    / "src/idea4/ac/results/factorized_ac/runs/factorized_ac_trainMIX12_EQ_C1200_k12_T1000_F3_seed123/20260506_144445_439009/cross_regime_results.json",
    (12, "factorized_ac", 323): PROJECT_ROOT
    / "src/idea4/ac/results/factorized_ac/runs/factorized_ac_trainMIX12_EQ_C1200_k12_T1000_F3_seed323/20260506_150113_691196/cross_regime_results.json",
    (12, "dual_branch_factorized_ac", 123): PROJECT_ROOT
    / "src/idea4/ac/results/dual_branch_ac/runs/dual_branch_factorized_ac_trainMIX12_EQ_C1200_k12_T1000_F3_seed123/20260506_151752_310450/cross_regime_results.json",
    (12, "dual_branch_factorized_ac", 323): PROJECT_ROOT
    / "src/idea4/ac/results/dual_branch_ac/runs/dual_branch_factorized_ac_trainMIX12_EQ_C1200_k12_T1000_F3_seed323/20260506_155155_525191/cross_regime_results.json",
}

OPTIONAL_K6_MODES = {
    "dual_branch_factorized_ac_gate_balanced",
    "dual_branch_factorized_ac_gate_regularized",
    "dual_branch_factorized_ac_auxrisk",
}

SUMMARY_COLS = [
    "model_label",
    "train_regime",
    "seed",
    "C",
    "k",
    "F",
    "T",
    "mean_value_accept_ratio",
    "mean_value_accept_ratio_percent",
    "worst_regime_value_accept_ratio",
    "worst_regime_value_accept_ratio_percent",
    "std_value_accept_ratio_across_regimes",
    "std_value_accept_ratio_percent",
    "mean_drops",
    "mean_flushes",
    "num_regimes",
    "completeness",
]

AVG_COLS = [
    "model_label",
    "C",
    "k",
    "F",
    "T",
    "avg_mean_value_accept_ratio",
    "avg_mean_value_accept_ratio_percent",
    "avg_worst_regime_value_accept_ratio",
    "avg_worst_regime_value_accept_ratio_percent",
    "avg_std_value_accept_ratio_across_regimes",
    "avg_drops",
    "avg_flushes",
    "num_seeds",
]


@dataclass
class ParsedResult:
    model_label: str
    seed: int | None
    C: float | None
    k: int | None
    F: int | None
    T: int | None
    train_regime: str
    episodes: int | None
    eval_episodes: int | None
    scenario: str
    source_path: Path
    accepted: bool
    rejection_reason: str
    rows: list[dict[str, Any]]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def stat_mean(summary: dict[str, Any], key: str) -> float:
    value = summary.get(key)
    if isinstance(value, dict):
        return float(value.get("mean", math.nan))
    if value is None:
        return math.nan
    return float(value)


def config_value(config: dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    value = config.get(section, {})
    if isinstance(value, dict):
        return value.get(key, default)
    return default


def parse_from_scenario(scenario: str, key: str) -> int | None:
    pattern = {
        "C": r"_C(\d+(?:\.\d+)?)",
        "k": r"_k(\d+)",
        "F": r"_F(\d+)",
        "T": r"_T(\d+)",
        "seed": r"_seed(\d+)",
    }[key]
    match = re.search(pattern, scenario)
    if not match:
        return None
    return int(float(match.group(1)))


def parse_train_regime_from_text(text: str) -> str | None:
    match = re.search(r"train([A-Z0-9_]+?)(?:_cross|_C)", text)
    if match:
        return match.group(1)
    return None


def classify_model(data: dict[str, Any], path: Path) -> str:
    scenario = str(data.get("scenario") or "")
    config = data.get("config") if isinstance(data.get("config"), dict) else {}
    model_mode = str(data.get("model_mode") or config.get("model_mode") or "")

    if model_mode == "basic_ppo" or scenario.startswith("basic_ppo_train"):
        return "basic_ppo"
    if model_mode == "baseline" or scenario.startswith("baseline_train"):
        return "dqn_baseline"
    if model_mode == "factorized_ac" or scenario.startswith("factorized_ac_train"):
        return "factorized_ac"
    if model_mode in OPTIONAL_K6_MODES or scenario.startswith(tuple(f"{m}_train" for m in OPTIONAL_K6_MODES)):
        return model_mode if model_mode else scenario.split("_train", 1)[0]
    if model_mode == "dual_branch_factorized_ac" or scenario.startswith("dual_branch_factorized_ac_train"):
        return "dual_branch_factorized_ac"
    if "ideaextra/results" in rel(path) and scenario.startswith("trainMIX12_EQ_cross"):
        return "dqn_baseline"
    return model_mode or "unknown"


def extract_test_results(data: dict[str, Any], path: Path) -> tuple[dict[str, Any], str]:
    """Return a 12-regime result block and diagnostic reason if unavailable."""
    candidates = ["test_results", "cross_regime_results", "results", "regime_results"]
    for key in candidates:
        block = data.get(key)
        if isinstance(block, dict) and set(EXPECTED_REGIMES).issubset(block.keys()):
            return block, ""

    present = {
        key: (type(data.get(key)).__name__, len(data.get(key)) if hasattr(data.get(key), "__len__") else None)
        for key in candidates
        if key in data
    }
    message = (
        f"could not find 12-regime results; top_level_keys={list(data.keys())}; "
        f"candidate_blocks={present}; source={rel(path)}"
    )
    print(f"PARSE WARNING: {message}")
    return {}, message


def parse_result(path: Path, expected_label: str | None = None, expected_k: int | None = None) -> ParsedResult:
    if not path.exists():
        return ParsedResult(
            model_label=expected_label or "unknown",
            seed=None,
            C=None,
            k=None,
            F=None,
            T=None,
            train_regime="",
            episodes=None,
            eval_episodes=None,
            scenario="",
            source_path=path,
            accepted=False,
            rejection_reason="missing file",
            rows=[],
        )

    data = json.loads(path.read_text())
    config = data.get("config") if isinstance(data.get("config"), dict) else {}
    env = config.get("env") if isinstance(config.get("env"), dict) else {}
    train = config.get("train") if isinstance(config.get("train"), dict) else {}
    eval_cfg = config.get("eval") if isinstance(config.get("eval"), dict) else {}
    data_cfg = config.get("data") if isinstance(config.get("data"), dict) else {}

    scenario = str(data.get("scenario") or path.parent.parent.name)
    model_label = classify_model(data, path)
    seed = data.get("seed") or config.get("seed") or parse_from_scenario(scenario, "seed")
    C = env.get("C") or parse_from_scenario(scenario, "C")
    k = env.get("k") or parse_from_scenario(scenario, "k")
    F = env.get("F") or parse_from_scenario(scenario, "F")
    T = env.get("T") or parse_from_scenario(scenario, "T")
    path_text = f"{scenario} {rel(path)}"
    train_regime = str(
        data.get("train_regime")
        or data_cfg.get("train_regime")
        or parse_train_regime_from_text(path_text)
        or "MIX12_EQ"
    )
    episodes = train.get("episodes")
    eval_episodes = eval_cfg.get("num_episodes")
    save_mode = str(config.get("save_mode") or "")
    debug_mode = bool(config.get("debug_mode", False))
    test_results, parse_warning = extract_test_results(data, path)

    reasons: list[str] = []
    if expected_label and model_label != expected_label:
        reasons.append(f"model mismatch expected {expected_label}, got {model_label}")
    if model_label == "factorized_ac" and str(data.get("model_mode") or config.get("model_mode") or "") not in ("factorized_ac", ""):
        reasons.append("plain factorized_ac strict match failed")
    if "20260507_205800" in str(path):
        reasons.append("explicitly rejected earlier DQN folder without cross_regime_results")
    if debug_mode:
        reasons.append("debug_mode true")
    if save_mode and save_mode != "full":
        reasons.append(f"save_mode {save_mode}")
    if episodes is None or int(episodes) < 1000:
        reasons.append("episodes < 1000 or missing")
    if eval_episodes is None or int(eval_episodes) < 100:
        reasons.append("eval_episodes too small or missing")
    if (float(C) if C is not None else None) != 1200.0 or int(F or -1) != 3 or int(T or -1) != 1000:
        reasons.append("not target C1200 F3 T1000")
    if expected_k is not None and int(k or -1) != expected_k:
        reasons.append(f"not expected k={expected_k}")
    elif expected_k is None and int(k or -1) not in {6, 12}:
        reasons.append("not target k in {6,12}")
    if train_regime != "MIX12_EQ":
        reasons.append(f"train_regime {train_regime}")
    if set(test_results) != set(EXPECTED_REGIMES):
        reasons.append(parse_warning or "missing or non-12 static regimes")
    lowered = str(path).lower()
    if any(token in lowered for token in ["mock", "debug", "smoke", "diagnostic"]):
        reasons.append("mock/debug/smoke/diagnostic path")

    rows: list[dict[str, Any]] = []
    if not reasons:
        for regime in EXPECTED_REGIMES:
            summary = test_results[regime].get("summary", {})
            rows.append(
                {
                    "model_label": model_label,
                    "family": "DQN" if model_label == "dqn_baseline" else ("Basic PPO" if model_label == "basic_ppo" else "AC"),
                    "train_regime": train_regime,
                    "seed": int(seed),
                    "C": float(C),
                    "k": int(k),
                    "F": int(F),
                    "T": int(T),
                    "test_regime": regime,
                    "value_accept_ratio": stat_mean(summary, "value_accept_ratio"),
                    "drops": stat_mean(summary, "drops"),
                    "flushes": stat_mean(summary, "flushes"),
                    "drop_rate": stat_mean(summary, "drop_rate"),
                    "count_accept_ratio": stat_mean(summary, "count_accept_ratio"),
                    "scenario": scenario,
                    "source_file": rel(path),
                }
            )

    return ParsedResult(
        model_label=model_label,
        seed=int(seed) if seed is not None else None,
        C=float(C) if C is not None else None,
        k=int(k) if k is not None else None,
        F=int(F) if F is not None else None,
        T=int(T) if T is not None else None,
        train_regime=train_regime,
        episodes=int(episodes) if episodes is not None else None,
        eval_episodes=int(eval_episodes) if eval_episodes is not None else None,
        scenario=scenario,
        source_path=path,
        accepted=not reasons,
        rejection_reason="; ".join(reasons),
        rows=rows,
    )


def discover_optional_k6() -> list[Path]:
    paths: list[Path] = []
    root = PROJECT_ROOT / "src" / "idea4" / "ac" / "results" / "dual_branch_ac" / "runs"
    for mode in OPTIONAL_K6_MODES:
        paths.extend(root.glob(f"{mode}_trainMIX12_EQ_C1200_k6_T1000_F3_seed*/**/cross_regime_results.json"))
    return sorted(paths)


def discover_basic_ppo() -> list[Path]:
    root = PROJECT_ROOT / "src" / "idea4" / "ac" / "results" / "basic_ppo" / "runs"
    paths: list[Path] = []
    for k in CORE_K_VALUES:
        for seed in CORE_SEEDS:
            paths.extend(root.glob(f"basic_ppo_trainMIX12_EQ_C1200_k{k}_T1000_F3_seed{seed}/**/cross_regime_results.json"))
    return sorted(paths)


def load_base_long() -> pd.DataFrame:
    path = FINAL_DIR / "all_regime_rows_long.csv"
    if not path.exists():
        return pd.DataFrame(columns=["model_label", "family", "train_regime", "seed", "C", "k", "F", "T", "test_regime", "value_accept_ratio", "drops", "flushes", "drop_rate", "count_accept_ratio", "scenario", "source_file"])
    return pd.read_csv(path)


def summarize_seed(long_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["model_label", "train_regime", "seed", "C", "k", "F", "T"]
    for keys, group in long_df.groupby(group_cols, dropna=False):
        values = group["value_accept_ratio"].astype(float)
        drops = group["drops"].astype(float)
        flushes = group["flushes"].astype(float)
        num_regimes = int(group["test_regime"].nunique())
        mean_var = float(values.mean())
        worst_var = float(values.min())
        std_var = float(values.std(ddof=0))
        rows.append(
            dict(
                zip(group_cols, keys),
                mean_value_accept_ratio=mean_var,
                mean_value_accept_ratio_percent=mean_var * 100,
                worst_regime_value_accept_ratio=worst_var,
                worst_regime_value_accept_ratio_percent=worst_var * 100,
                std_value_accept_ratio_across_regimes=std_var,
                std_value_accept_ratio_percent=std_var * 100,
                mean_drops=float(drops.mean()),
                mean_flushes=float(flushes.mean()),
                num_regimes=num_regimes,
                completeness="complete" if num_regimes == 12 else "incomplete",
            )
        )
    df = pd.DataFrame(rows, columns=SUMMARY_COLS)
    return df.sort_values(["k", "model_label", "seed"], kind="stable").reset_index(drop=True)


def summarize_average(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["model_label", "C", "k", "F", "T"]
    for keys, group in seed_df.groupby(group_cols, dropna=False):
        mean_var = float(group["mean_value_accept_ratio"].mean())
        worst_var = float(group["worst_regime_value_accept_ratio"].mean())
        rows.append(
            dict(
                zip(group_cols, keys),
                avg_mean_value_accept_ratio=mean_var,
                avg_mean_value_accept_ratio_percent=mean_var * 100,
                avg_worst_regime_value_accept_ratio=worst_var,
                avg_worst_regime_value_accept_ratio_percent=worst_var * 100,
                avg_std_value_accept_ratio_across_regimes=float(group["std_value_accept_ratio_across_regimes"].mean()),
                avg_drops=float(group["mean_drops"].mean()),
                avg_flushes=float(group["mean_flushes"].mean()),
                num_seeds=int(group["seed"].nunique()),
            )
        )
    df = pd.DataFrame(rows, columns=AVG_COLS)
    return df.sort_values(["k", "model_label"], kind="stable").reset_index(drop=True)


def build_regime_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame(columns=["k", "test_regime"])

    data = long_df.copy()
    data["model_seed"] = data.apply(lambda r: f"{r['model_label']}_seed{int(r['seed'])}", axis=1)
    pieces: list[pd.DataFrame] = []
    for metric in ["value_accept_ratio", "drops", "flushes"]:
        pivot = data.pivot_table(
            index=["k", "test_regime"],
            columns="model_seed",
            values=metric,
            aggfunc="mean",
        )
        pivot.columns = [f"{col}_{metric}" for col in pivot.columns]
        pieces.append(pivot)

    wide = pd.concat(pieces, axis=1).reset_index()
    return wide.sort_values(["k", "test_regime"], kind="stable").reset_index(drop=True)


def build_difficulty(long_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = long_df.groupby(["k", "test_regime"], dropna=False)
    for (k, regime), group in grouped:
        avg_var = float(group["value_accept_ratio"].astype(float).mean())
        rows.append(
            {
                "k": k,
                "test_regime": regime,
                "avg_value_accept_ratio": avg_var,
                "avg_drops": float(group["drops"].astype(float).mean()),
                "avg_flushes": float(group["flushes"].astype(float).mean()),
                "num_model_seed_rows": int(len(group)),
                "simple_interpretation": "harder" if avg_var < 0.95 else "easy_or_near_ceiling",
            }
        )
    return pd.DataFrame(rows).sort_values(["k", "avg_value_accept_ratio"], kind="stable").reset_index(drop=True)


def build_k12_model_summary(seed_df: pd.DataFrame) -> pd.DataFrame:
    k12 = seed_df[
        (seed_df["k"] == 12)
        & (seed_df["model_label"].isin(CORE_MODELS))
        & (seed_df["seed"].isin(CORE_SEEDS))
    ].copy()
    if k12.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "seed",
                "k",
                "num_regimes",
                "mean_value_accept_ratio",
                "mean_value_accept_ratio_percent",
                "worst_regime_value_accept_ratio",
                "worst_regime_value_accept_ratio_percent",
                "std_value_accept_ratio",
                "std_value_accept_ratio_percent",
                "mean_drops",
                "mean_flushes",
            ]
        )
    out = k12.rename(
        columns={
            "model_label": "model",
            "std_value_accept_ratio_across_regimes": "std_value_accept_ratio",
        }
    )
    return out[
        [
            "model",
            "seed",
            "k",
            "num_regimes",
            "mean_value_accept_ratio",
            "mean_value_accept_ratio_percent",
            "worst_regime_value_accept_ratio",
            "worst_regime_value_accept_ratio_percent",
            "std_value_accept_ratio",
            "std_value_accept_ratio_percent",
            "mean_drops",
            "mean_flushes",
        ]
    ].sort_values(["model", "seed"], kind="stable")


def build_core_18_check(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    core = seed_df[seed_df["model_label"].isin(CORE_MODELS)].copy()
    for k in CORE_K_VALUES:
        for model in CORE_MODELS:
            for seed in CORE_SEEDS:
                match = core[
                    (core["k"].astype(int) == k)
                    & (core["model_label"] == model)
                    & (core["seed"].astype(int) == seed)
                    & (core["num_regimes"].astype(int) == 12)
                ]
                rows.append(
                    {
                        "k": k,
                        "model_label": model,
                        "seed": seed,
                        "found": not match.empty,
                        "status": "complete" if not match.empty else "missing",
                        "source_note": "formal/current table row" if not match.empty else "",
                    }
                )
    return pd.DataFrame(rows)


def build_four_model_summary(seed_df: pd.DataFrame) -> pd.DataFrame:
    out = seed_df[
        (seed_df["k"].isin(CORE_K_VALUES))
        & (seed_df["model_label"].isin(FOUR_MODEL_MODELS))
        & (seed_df["seed"].isin(CORE_SEEDS))
    ].copy()
    out["model_order"] = out["model_label"].map(MODEL_ORDER)
    return out.sort_values(["k", "model_order", "seed"], kind="stable").drop(columns=["model_order"]).reset_index(drop=True)


def build_four_model_average(four_summary: pd.DataFrame) -> pd.DataFrame:
    if four_summary.empty:
        return pd.DataFrame(columns=AVG_COLS)
    avg = summarize_average(four_summary)
    avg = avg[avg["model_label"].isin(FOUR_MODEL_MODELS)].copy()
    avg["model_order"] = avg["model_label"].map(MODEL_ORDER)
    return avg.sort_values(["k", "model_order"], kind="stable").drop(columns=["model_order"]).reset_index(drop=True)


def build_ppo_factorization_comparison(four_avg: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pairs = [
        ("basic_ppo", "factorized_ac"),
        ("basic_ppo", "dual_branch_factorized_ac"),
        ("factorized_ac", "dual_branch_factorized_ac"),
    ]
    metric_map = {
        "mean_value_accept_ratio": "avg_mean_value_accept_ratio",
        "worst_regime_value_accept_ratio": "avg_worst_regime_value_accept_ratio",
        "mean_drops": "avg_drops",
        "mean_flushes": "avg_flushes",
    }
    for k in CORE_K_VALUES:
        k_df = four_avg[four_avg["k"].astype(int) == k].set_index("model_label")
        for model_a, model_b in pairs:
            row: dict[str, Any] = {"k": k, "model_a": model_a, "model_b": model_b, "delta_direction": f"{model_b}_minus_{model_a}"}
            if model_a not in k_df.index or model_b not in k_df.index:
                row.update({f"delta_{metric}": math.nan for metric in metric_map})
                row["comparison_note"] = "missing source model"
            else:
                for metric, col in metric_map.items():
                    row[f"delta_{metric}"] = float(k_df.loc[model_b, col] - k_df.loc[model_a, col])
                row["comparison_note"] = "complete"
            rows.append(row)
    return pd.DataFrame(rows)


def build_meeting_four_model_comparison(four_avg: pd.DataFrame, four_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    avg_lookup = four_avg.set_index(["k", "model_label"]) if not four_avg.empty else pd.DataFrame()
    summary_lookup = four_summary.groupby(["k", "model_label"])["seed"].apply(lambda s: sorted(int(x) for x in s)).to_dict()
    for k in CORE_K_VALUES:
        for model in FOUR_MODEL_MODELS:
            key = (k, model)
            seeds = summary_lookup.get(key, [])
            if len(seeds) == 2:
                coverage_note = "2 seeds"
            elif seeds == [123]:
                coverage_note = "seed123 only"
            elif seeds:
                coverage_note = "seed" + ",".join(str(seed) for seed in seeds) + " only"
            else:
                coverage_note = "missing"

            if not four_avg.empty and key in avg_lookup.index:
                avg_row = avg_lookup.loc[key]
                rows.append(
                    {
                        "k": k,
                        "model_label": model,
                        "num_seeds": int(avg_row["num_seeds"]),
                        "avg_mean_value_accept_ratio_percent": float(avg_row["avg_mean_value_accept_ratio_percent"]),
                        "avg_worst_regime_value_accept_ratio_percent": float(avg_row["avg_worst_regime_value_accept_ratio_percent"]),
                        "avg_drops": float(avg_row["avg_drops"]),
                        "avg_flushes": float(avg_row["avg_flushes"]),
                        "coverage_note": coverage_note,
                    }
                )
            else:
                rows.append(
                    {
                        "k": k,
                        "model_label": model,
                        "num_seeds": 0,
                        "avg_mean_value_accept_ratio_percent": math.nan,
                        "avg_worst_regime_value_accept_ratio_percent": math.nan,
                        "avg_drops": math.nan,
                        "avg_flushes": math.nan,
                        "coverage_note": coverage_note,
                    }
                )
    return pd.DataFrame(rows)


def build_model_seed_check(seed_df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for k in CORE_K_VALUES:
        for model in models:
            for seed in CORE_SEEDS:
                found = not seed_df[
                    (seed_df["k"].astype(int) == k)
                    & (seed_df["model_label"] == model)
                    & (seed_df["seed"].astype(int) == seed)
                    & (seed_df["num_regimes"].astype(int) == 12)
                ].empty
                rows.append({"k": k, "model_label": model, "seed": seed, "found": found, "status": "complete" if found else "missing"})
    return pd.DataFrame(rows)


def write_markdown(
    seed_df: pd.DataFrame,
    avg_df: pd.DataFrame,
    missing: list[tuple[int, str, int]],
    sources: list[Path],
    core_check: pd.DataFrame,
    four_avg: pd.DataFrame,
    ppo_comparison: pd.DataFrame,
    meeting: pd.DataFrame,
) -> None:
    k6_seed = seed_df[seed_df["k"] == 6]
    k6_avg = avg_df[avg_df["k"] == 6]
    k12_seed = seed_df[seed_df["k"] == 12]
    note_missing = "None" if not missing else ", ".join(f"k{k} {m} seed {s}" for k, m, s in missing)
    source_lines = "\n".join(f"- `{rel(p)}`" for p in sources)
    readme = f"""# Final Comparison Tables Summary

These derived tables were regenerated from existing result files only. No training was rerun.

## k=6 Refill Note

k=6 DQN baseline and plain factorized_ac were refilled after the earlier summary generation. The updated k=6 comparison now supports a fairer DQN vs factorized AC vs dual-branch AC comparison.

## Sources Read

{source_lines}

## Core Formal Coverage

- Required k=6/k=12 groups: dqn_baseline seeds 123/323, factorized_ac seeds 123/323, dual_branch_factorized_ac seeds 123/323.
- Missing core groups: {note_missing}.

## Updated k=6 Model-Seed Summary

```text
{k6_seed.to_string(index=False)}
```

## Updated k=6 Average Across Seeds

```text
{k6_avg.to_string(index=False)}
```

## Updated k=12 Model-Seed Summary

```text
{k12_seed.to_string(index=False)}
```

## 18-Combination Core Check

```text
{core_check.to_string(index=False)}
```

## Four-Model Comparison

Basic PPO was added as an additional baseline. This separates three effects:

1. DQN baseline -> Basic PPO: moving from value-based DQN to policy-gradient PPO.
2. Basic PPO -> Factorized AC: action factorization.
3. Factorized AC -> Dual-Branch AC: dual state-reasoning branches.

```text
{meeting.to_string(index=False)}
```

## PPO Factorization Comparison

```text
{ppo_comparison.to_string(index=False)}
```
"""
    (FINAL_DIR / "README_summary.md").write_text(readme)

    interpretation = f"""# Final Interpretation

The final comparison tables now include formal k=6 and k=12 DQN baseline, plain factorized_ac, and dual_branch_factorized_ac runs where available. Basic PPO has also been added as a policy-gradient baseline.

Use `meeting_four_model_comparison.csv` for a compact meeting view of DQN baseline, Basic PPO, Factorized AC, and Dual-Branch AC. Interpret the comparison cautiously: the tables separate the DQN-to-PPO shift, the Basic PPO-to-Factorized AC action-factorization effect, and the Factorized AC-to-Dual-Branch effect. Missing core formal groups: {note_missing}.
"""
    (FINAL_DIR / "final_interpretation.md").write_text(interpretation)


def main() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    candidates: list[tuple[int | None, str | None, Path]] = [
        (k, label, path) for (k, label, _seed), path in CORE_SOURCES.items()
    ]
    candidates.extend((6, None, path) for path in discover_optional_k6())
    candidates.extend((parse_from_scenario(str(path), "k"), "basic_ppo", path) for path in discover_basic_ppo())

    parsed = [parse_result(path, expected_label=expected, expected_k=expected_k) for expected_k, expected, path in candidates]
    validation_rows = [
        {
            "model_label": item.model_label,
            "seed": item.seed,
            "k": item.k,
            "C": item.C,
            "F": item.F,
            "T": item.T,
            "episodes": item.episodes,
            "eval_episodes": item.eval_episodes,
            "source_path": rel(item.source_path),
            "status": "accepted" if item.accepted else "rejected",
            "rejection_reason": item.rejection_reason,
        }
        for item in parsed
    ]
    validation_df = pd.DataFrame(validation_rows)

    print("\nPRE-WRITE VALIDATION TABLE")
    print(validation_df.to_string(index=False))

    basic_validation_df = validation_df[validation_df["model_label"].eq("basic_ppo")]
    print("\nBASIC PPO CANDIDATE VALIDATION")
    print(basic_validation_df.to_string(index=False) if not basic_validation_df.empty else "No basic_ppo candidates found.")

    accepted = [item for item in parsed if item.accepted]
    source_files = [item.source_path for item in accepted]
    missing_core: list[tuple[int, str, int]] = []
    for (k, model, seed), _path in CORE_SOURCES.items():
        if not any(item.accepted and item.k == k and item.model_label == model and item.seed == seed for item in accepted):
            print(f"MISSING CORE K={k} FORMAL RUN: model={model}, seed={seed}")
            missing_core.append((k, model, seed))

    base = load_base_long()
    if not base.empty:
        base["model_label"] = base["model_label"].replace({"dual_branch_ac": "dual_branch_factorized_ac"})
        mask_remove = (
            (
                (base["k"].astype(int).isin(CORE_K_VALUES))
                & (base["model_label"].isin(set(CORE_MODELS) | OPTIONAL_K6_MODES))
            )
            | (
                (base["k"].astype(int).isin(CORE_K_VALUES))
                & (base["model_label"].eq("basic_ppo"))
            )
        )
        mask_remove = mask_remove & base["seed"].astype(int).isin([123, 323, 532, 999])
        base = base.loc[~mask_remove].copy()

    new_rows = [row for item in accepted for row in item.rows]
    new_df = pd.DataFrame(new_rows, columns=base.columns if not base.empty else None)
    long_df = pd.concat([base, new_df], ignore_index=True)
    long_df = long_df.sort_values(["k", "model_label", "seed", "test_regime"], kind="stable").reset_index(drop=True)

    seed_df = summarize_seed(long_df)
    avg_df = summarize_average(seed_df)
    k6_models = [
        "dqn_baseline",
        "factorized_ac",
        "dual_branch_factorized_ac",
        "dual_branch_factorized_ac_gate_balanced",
        "dual_branch_factorized_ac_gate_regularized",
        "dual_branch_factorized_ac_auxrisk",
    ]
    k6_df = seed_df[(seed_df["k"] == 6) & (seed_df["model_label"].isin(k6_models))].copy()
    k12_model_summary = build_k12_model_summary(seed_df)
    core_check = build_core_18_check(seed_df)
    four_summary = build_four_model_summary(seed_df)
    four_avg = build_four_model_average(four_summary)
    ppo_comparison = build_ppo_factorization_comparison(four_avg)
    meeting = build_meeting_four_model_comparison(four_avg, four_summary)
    basic_ppo_check = build_model_seed_check(seed_df, ["basic_ppo"])
    four_model_check = build_model_seed_check(seed_df, FOUR_MODEL_MODELS)
    regime_wide = build_regime_wide(long_df)
    difficulty = build_difficulty(long_df)

    long_df.to_csv(FINAL_DIR / "all_regime_rows_long.csv", index=False)
    seed_df.to_csv(FINAL_DIR / "model_seed_summary.csv", index=False)
    avg_df.to_csv(FINAL_DIR / "model_average_across_seeds.csv", index=False)
    k6_df.to_csv(FINAL_DIR / "k6_dqn_factorized_dual.csv", index=False)
    k12_model_summary.to_csv(FINAL_DIR / "k12_model_summary.csv", index=False)
    four_summary.to_csv(FINAL_DIR / "k3_k6_k12_four_model_summary.csv", index=False)
    four_avg.to_csv(FINAL_DIR / "four_model_average_across_seeds.csv", index=False)
    ppo_comparison.to_csv(FINAL_DIR / "ppo_factorization_comparison.csv", index=False)
    meeting.to_csv(FINAL_DIR / "meeting_four_model_comparison.csv", index=False)
    regime_wide.to_csv(FINAL_DIR / "regime_level_wide.csv", index=False)
    difficulty.to_csv(FINAL_DIR / "difficulty_ranking.csv", index=False)
    write_markdown(seed_df, avg_df, missing_core, source_files, core_check, four_avg, ppo_comparison, meeting)

    added_counts = new_df.groupby(["model_label", "seed"]).size().reset_index(name="rows_added") if not new_df.empty else pd.DataFrame()
    k6_seed_print_cols = [
        "model_label",
        "seed",
        "k",
        "mean_value_accept_ratio",
        "worst_regime_value_accept_ratio",
        "std_value_accept_ratio_across_regimes",
        "mean_drops",
        "mean_flushes",
    ]
    k6_avg_print_cols = [
        "model_label",
        "k",
        "avg_mean_value_accept_ratio",
        "avg_worst_regime_value_accept_ratio",
        "avg_std_value_accept_ratio_across_regimes",
        "avg_drops",
        "avg_flushes",
    ]
    k6_seed_slice = seed_df[seed_df["k"] == 6][k6_seed_print_cols].sort_values(["model_label", "seed"], kind="stable")
    k6_avg_slice = avg_df[avg_df["k"] == 6][k6_avg_print_cols].sort_values(["model_label"], kind="stable")
    k12_seed_slice = seed_df[seed_df["k"] == 12][k6_seed_print_cols].sort_values(["model_label", "seed"], kind="stable")
    rejected_k12 = validation_df[(validation_df["k"] == 12) & (validation_df["status"] == "rejected")]
    four_print_cols = [
        "model_label",
        "seed",
        "k",
        "mean_value_accept_ratio",
        "worst_regime_value_accept_ratio",
        "std_value_accept_ratio_across_regimes",
        "mean_drops",
        "mean_flushes",
    ]

    print("\nSOURCE FILES READ")
    for path in source_files:
        print(f"- {rel(path)}")

    print("\nROWS ADDED BY MODEL/SEED")
    print(added_counts.to_string(index=False) if not added_counts.empty else "No rows added.")

    print("\nMISSING MODEL/SEED COMBINATIONS")
    print("None" if not missing_core else "\n".join(f"- k{k} {model} seed {seed}" for k, model, seed in missing_core))

    print("\nFULL k6_dqn_factorized_dual.csv")
    print(k6_df.to_string(index=False))

    print("\nK=6 SLICE OF model_seed_summary.csv")
    print(k6_seed_slice.to_string(index=False))

    print("\nK=6 SLICE OF model_average_across_seeds.csv")
    print(k6_avg_slice.to_string(index=False))

    print("\nK=12 SLICE OF model_seed_summary.csv")
    print(k12_seed_slice.to_string(index=False))

    print("\nFULL k12_model_summary.csv")
    print(k12_model_summary.to_string(index=False))

    print("\n18-COMBINATION CORE CHECK")
    print(core_check.to_string(index=False))

    print("\nREJECTED k=12 CANDIDATES")
    print(rejected_k12.to_string(index=False) if not rejected_k12.empty else "None")

    print("\nBASIC PPO AVAILABILITY CHECK")
    print(basic_ppo_check.to_string(index=False))

    print("\nFOUR-MODEL COMBINATION CHECK")
    print(four_model_check.to_string(index=False))

    for k_value in CORE_K_VALUES:
        k_slice = four_summary[four_summary["k"].astype(int).eq(k_value)][four_print_cols]
        print(f"\nK={k_value} FOUR-MODEL SUMMARY")
        print(k_slice.to_string(index=False))

    print("\nFULL four_model_average_across_seeds.csv")
    print(four_avg.to_string(index=False))

    print("\nFULL ppo_factorization_comparison.csv")
    print(ppo_comparison.to_string(index=False))

    print("\nFULL meeting_four_model_comparison.csv")
    print(meeting.to_string(index=False))

    print("\nGenerated files:")
    for name in [
        "all_regime_rows_long.csv",
        "model_seed_summary.csv",
        "model_average_across_seeds.csv",
        "k6_dqn_factorized_dual.csv",
        "k12_model_summary.csv",
        "k3_k6_k12_four_model_summary.csv",
        "four_model_average_across_seeds.csv",
        "ppo_factorization_comparison.csv",
        "meeting_four_model_comparison.csv",
        "regime_level_wide.csv",
        "difficulty_ranking.csv",
        "README_summary.md",
        "final_interpretation.md",
    ]:
        print(f"- {rel(FINAL_DIR / name)}")


if __name__ == "__main__":
    main()
