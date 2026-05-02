#3/27 idea2 4regimes（保证结构差异的前提下近似难度）
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
第一阶段：构造并分析四类 static regimes（精简版）
================================================
目标：
1. 用“一个统一 generator + 四类 regime 配置”生成 transaction 序列
2. 计算关键结构描述符（descriptor）
3. 输出简洁、可比、可复现的统计结果
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ============================================================
# 1) 全局常量：后续调试最常改这里
# ============================================================
GLOBAL = {
    "episode_length": 2000,          # 每个 episode 的交易步数
    "episodes_per_regime": 30,       # 每类 regime 生成多少个 episode 来估计 descriptor
    "base_seed": 20260327,           # 随机种子基准
    "high_load_threshold_ratio": 1.20,  # 用于“高负载连续段”统计的阈值比例
    "output_dir": "phase1_outputs_compact",
}

# 参考尺度：方便理解参数，不强制等于最终均值
MU_STAR = 50.0


# ============================================================
# 2) 四类 regime 参数：后续调试最常改这里
#    当前版本已经加入了更稳妥的 burst 调整
# ============================================================
REGIME_CONFIGS: Dict[str, Dict] = {
    "SL": {
        "name": "SL",
        "time_structure": "smooth",
        "size_structure": "light_tail",
        "description": "smooth + light-tail；通过提高整体负载来抬高难度，但不引入burst和重尾。",

        "base_mean": 1.08 * MU_STAR,
        "base_std": 0.28 * MU_STAR,
        "max_tx": 1.75 * MU_STAR,

        "burst_start_prob": 0.0,
        "burst_length_mean": 1.0,
        "burst_uplift": 0.0,

        "large_prob_normal": 0.0,
        "large_prob_burst": 0.0,
        "large_mean": None,
        "large_std": None,
    },

    "SH": {
        "name": "SH",
        "time_structure": "smooth",
        "size_structure": "heavy_tail",
        "description": "smooth + heavy-tail；保持平稳，但加入少量大单。",

        "base_mean": 0.90 * MU_STAR,
        "base_std": 0.20 * MU_STAR,
        "max_tx": 2.80 * MU_STAR,

        "burst_start_prob": 0.0,
        "burst_length_mean": 1.0,
        "burst_uplift": 0.0,

        "large_prob_normal": 0.10,
        "large_prob_burst": 0.10,
        "large_mean": 2.25 * MU_STAR,
        "large_std": 0.24 * MU_STAR,
    },

    "BL": {
        "name": "BL",
        "time_structure": "bursty",
        "size_structure": "light_tail",
        "description": "bursty + light-tail；保留时间簇拥，但压低burst强度。",

        "base_mean": 0.92 * MU_STAR,
        "base_std": 0.22 * MU_STAR,
        "max_tx": 1.90 * MU_STAR,

        "burst_start_prob": 0.05,
        "burst_length_mean": 6.0,
        "burst_uplift": 0.22,

        "large_prob_normal": 0.0,
        "large_prob_burst": 0.0,
        "large_mean": None,
        "large_std": None,
    },

    "BH": {
        "name": "BH",
        "time_structure": "bursty",
        "size_structure": "heavy_tail",
        "description": "bursty + heavy-tail；双重结构保留，但进一步压轻 burst 期强度。",

        "base_mean": 0.86 * MU_STAR,
        "base_std": 0.20 * MU_STAR,
        "max_tx": 2.60 * MU_STAR,

        "burst_start_prob": 0.045,
        "burst_length_mean": 6.0,
        "burst_uplift": 0.18,

        "large_prob_normal": 0.04,
        "large_prob_burst": 0.10,
        "large_mean": 2.10 * MU_STAR,
        "large_std": 0.20 * MU_STAR,
    },
}


# ============================================================
# 3) 基础工具函数
# ============================================================
def ensure_output_dir(path_str: str) -> Path:
    path = Path(path_str)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b != 0 else 0.0


# ============================================================
# 4) 采样函数
# ============================================================
def sample_truncated_normal(mean: float, std: float, max_tx: float, rng: np.random.Generator) -> float:
    """轻尾分布：normal + 裁剪。"""
    x = rng.normal(mean, std)
    x = max(1.0, min(float(x), float(max_tx)))
    return x


def sample_heavy_tail(
    base_mean: float,
    base_std: float,
    large_prob: float,
    large_mean: float,
    large_std: float,
    max_tx: float,
    rng: np.random.Generator,
) -> float:
    """重尾分布：mixture，大部分小额 + 少量大额。"""
    if rng.random() < large_prob:
        x = rng.normal(large_mean, large_std)
    else:
        x = rng.normal(base_mean, base_std)

    x = max(1.0, min(float(x), float(max_tx)))
    return x


def sample_burst_length(mean_len: float, rng: np.random.Generator) -> int:
    """burst 持续长度：用几何分布近似。"""
    if mean_len <= 1:
        return 1
    p = 1.0 / mean_len
    return int(rng.geometric(p))


# ============================================================
# 5) 统一 generator
# ============================================================
def generate_episode(config: Dict, T: int, seed: int | None = None) -> Tuple[np.ndarray, List[str]]:
    """
    输出：
        tx_sizes: 长度为 T 的交易大小序列
        state_labels: 每一步所属状态，取值 normal / burst
    """
    rng = np.random.default_rng(seed)

    tx_sizes: List[float] = []
    state_labels: List[str] = []

    remaining_burst = 0

    for _ in range(T):
        # ---- 先判定当前属于 normal 还是 burst ----
        if config["time_structure"] == "smooth":
            state = "normal"
        else:
            if remaining_burst > 0:
                state = "burst"
                remaining_burst -= 1
            else:
                if rng.random() < config["burst_start_prob"]:
                    state = "burst"
                    remaining_burst = sample_burst_length(config["burst_length_mean"], rng) - 1
                else:
                    state = "normal"

        # ---- 根据状态决定当前均值 ----
        current_mean = config["base_mean"]
        current_std = config["base_std"]

        if state == "burst":
            current_mean = current_mean * (1.0 + config["burst_uplift"])

        # ---- 根据 tail 类型采样 ----
        if config["size_structure"] == "light_tail":
            x = sample_truncated_normal(
                mean=current_mean,
                std=current_std,
                max_tx=config["max_tx"],
                rng=rng,
            )
        else:
            large_prob = config["large_prob_burst"] if state == "burst" else config["large_prob_normal"]
            x = sample_heavy_tail(
                base_mean=current_mean,
                base_std=current_std,
                large_prob=large_prob,
                large_mean=config["large_mean"],
                large_std=config["large_std"],
                max_tx=config["max_tx"],
                rng=rng,
            )

        tx_sizes.append(x)
        state_labels.append(state)

    return np.array(tx_sizes, dtype=float), state_labels


# ============================================================
# 6) descriptor 计算
# ============================================================
def compute_lag1_autocorr(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    x0 = x[:-1]
    x1 = x[1:]
    std0 = np.std(x0)
    std1 = np.std(x1)
    if std0 < 1e-12 or std1 < 1e-12:
        return 0.0
    return float(np.corrcoef(x0, x1)[0, 1])


def compute_avg_burst_length_by_threshold(x: np.ndarray, threshold: float) -> float:
    """
    辅助指标：
    用“高于阈值的连续 high-load 段长度”的平均值来近似高压连续段。
    注意：这不是生成机制中的真 burst，只是负载层面的辅助视角。
    """
    lengths = []
    current = 0

    for value in x:
        if value > threshold:
            current += 1
        else:
            if current > 0:
                lengths.append(current)
                current = 0

    if current > 0:
        lengths.append(current)

    return float(np.mean(lengths)) if lengths else 0.0


def compute_avg_burst_length_from_states(state_labels: List[str]) -> float:
    """
    主时间结构指标：
    直接根据 generator 生成的 state_labels 计算 burst 段平均长度。
    这比固定阈值法更适合区分 smooth / bursty。
    """
    lengths = []
    current = 0

    for s in state_labels:
        if s == "burst":
            current += 1
        else:
            if current > 0:
                lengths.append(current)
                current = 0

    if current > 0:
        lengths.append(current)

    return float(np.mean(lengths)) if lengths else 0.0


def compute_episode_descriptors(tx_sizes: np.ndarray, high_load_threshold: float, state_labels: List[str]) -> Dict:
    mean_size = float(np.mean(tx_sizes))
    std_size = float(np.std(tx_sizes))
    cv = safe_div(std_size, mean_size)

    p95 = float(np.percentile(tx_sizes, 95))
    p99 = float(np.percentile(tx_sizes, 99))
    p95_over_mean = safe_div(p95, mean_size)
    p99_over_mean = safe_div(p99, mean_size)

    lag1 = compute_lag1_autocorr(tx_sizes)

    # 基于“生成状态”的主时间结构指标
    state_avg_burst_length = compute_avg_burst_length_from_states(state_labels)
    state_burst_fraction = safe_div(sum(1 for s in state_labels if s == "burst"), len(state_labels))

    # 基于“大小阈值”的辅助高压指标
    avg_highload_run_length = compute_avg_burst_length_by_threshold(tx_sizes, high_load_threshold)
    high_load_fraction = safe_div(int(np.sum(tx_sizes > high_load_threshold)), len(tx_sizes))

    return {
        "mean_size": mean_size,
        "std_size": std_size,
        "cv": cv,
        "p95_over_mean": p95_over_mean,
        "p99_over_mean": p99_over_mean,
        "lag1_autocorr": lag1,
        "state_avg_burst_length": state_avg_burst_length,
        "state_burst_fraction": state_burst_fraction,
        "avg_highload_run_length": avg_highload_run_length,
        "high_load_fraction": high_load_fraction,
    }


def aggregate_descriptor_dicts(dict_list: List[Dict]) -> Dict:
    if not dict_list:
        return {}

    keys = dict_list[0].keys()
    out = {}
    for key in keys:
        vals = [d[key] for d in dict_list]
        out[key] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals))
    return out


# ============================================================
# 7) 终端输出：保留关键对比，不做过度展示
# ============================================================
def print_key_config_table() -> None:
    print("\n[Regime 参数总览]")
    print("-" * 108)
    print(
        f"{'Name':<6}{'Time':<10}{'Size':<12}"
        f"{'base_mean':>12}{'base_std':>11}{'max_tx':>10}"
        f"{'burst_p':>10}{'burst_len':>11}{'uplift':>10}"
        f"{'Lp_N':>9}{'Lp_B':>9}"
    )
    print("-" * 108)

    for name, cfg in REGIME_CONFIGS.items():
        print(
            f"{name:<6}{cfg['time_structure']:<10}{cfg['size_structure']:<12}"
            f"{cfg['base_mean']:>12.2f}{cfg['base_std']:>11.2f}{cfg['max_tx']:>10.2f}"
            f"{cfg['burst_start_prob']:>10.3f}{cfg['burst_length_mean']:>11.2f}{cfg['burst_uplift']:>10.2f}"
            f"{cfg['large_prob_normal']:>9.2f}{cfg['large_prob_burst']:>9.2f}"
        )


def print_descriptor_summary(rows: List[Dict]) -> None:
    print("\n[关键 descriptor 汇总]")
    print("-" * 132)
    print(
        f"{'Name':<6}"
        f"{'mean':>10}{'CV':>10}"
        f"{'p95/mean':>12}{'p99/mean':>12}"
        f"{'lag1':>10}"
        f"{'stateBurstLen':>15}{'stateBurstFrac':>16}"
        f"{'highLoadLen':>13}{'highLoadFrac':>14}"
    )
    print("-" * 132)

    for row in rows:
        print(
            f"{row['name']:<6}"
            f"{row['mean_size']:>10.3f}{row['cv']:>10.3f}"
            f"{row['p95_over_mean']:>12.3f}{row['p99_over_mean']:>12.3f}"
            f"{row['lag1_autocorr']:>10.3f}"
            f"{row['state_avg_burst_length']:>15.3f}{row['state_burst_fraction']:>16.3f}"
            f"{row['avg_highload_run_length']:>13.3f}{row['high_load_fraction']:>14.3f}"
        )


def print_quick_reading_guide() -> None:
    print("\n[结果解读要点]")
    print("1. SH / BH 的 p95/mean、p99/mean 应明显高于 SL / BL（尾部结构）。")
    print("2. BL / BH 的 state_avg_burst_length、state_burst_fraction 应明显高于 SL / SH（时间结构）。")
    print("3. mean_size 不应飘太散，否则你比较的是总压力，不只是结构。")
    print("4. avg_highload_run_length 只是辅助指标；真正主时间结构请优先看 state_* 指标。")


# ============================================================
# 8) 主流程
# ============================================================
def main() -> None:
    output_dir = ensure_output_dir(GLOBAL["output_dir"])
    high_load_threshold = GLOBAL["high_load_threshold_ratio"] * MU_STAR

    # 保存配置，方便复现
    save_json(GLOBAL, output_dir / "global_config.json")
    save_json(REGIME_CONFIGS, output_dir / "regime_configs.json")

    summary_rows: List[Dict] = []

    for idx, (name, cfg) in enumerate(REGIME_CONFIGS.items()):
        episode_descriptor_list = []

        for ep in range(GLOBAL["episodes_per_regime"]):
            seed = GLOBAL["base_seed"] + 10000 * idx + ep

            tx_sizes, state_labels = generate_episode(
                config=cfg,
                T=GLOBAL["episode_length"],
                seed=seed,
            )

            desc = compute_episode_descriptors(
                tx_sizes=tx_sizes,
                high_load_threshold=high_load_threshold,
                state_labels=state_labels,
            )
            episode_descriptor_list.append(desc)

        agg = aggregate_descriptor_dicts(episode_descriptor_list)

        row = {
            "name": name,
            "time_structure": cfg["time_structure"],
            "size_structure": cfg["size_structure"],
            "description": cfg["description"],
            **agg,
        }
        summary_rows.append(row)

    # 只保存必要、清晰的统计信息
    save_csv(summary_rows, output_dir / "regime_descriptor_summary.csv")

    compact_json = {
        "global": GLOBAL,
        "mu_star": MU_STAR,
        "high_load_threshold": high_load_threshold,
        "regimes": summary_rows,
    }
    save_json(compact_json, output_dir / "regime_descriptor_summary.json")

    # 终端只打印关键表和简短解释
    print_key_config_table()
    print_descriptor_summary(summary_rows)
    print_quick_reading_guide()

    print(f"\n结果已保存到：{output_dir.resolve()}")
    print("生成文件：")
    print("  - regime_configs.json")
    print("  - regime_descriptor_summary.csv")
    print("  - regime_descriptor_summary.json")


if __name__ == "__main__":
    main()
