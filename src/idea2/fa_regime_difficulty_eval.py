#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FA baseline：直接按四类 regime 生成数据并做 difficulty screening
==============================================================

说明：
1. 直接导入 phase1_regime_generator_compact.py 中的 REGIME_CONFIGS / GLOBAL / generate_episode
2. 不依赖 tx_pool.npy
3. 只输出当前阶段真正需要的 summary / per-episode 统计

运行：
    python fa_regime_difficulty_eval.py
"""

from __future__ import annotations

import csv
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from idea1.regime_generator2 import REGIME_CONFIGS, GLOBAL, generate_episode


# ============================================================
# 1) 统一配置
# ============================================================
CONFIG = {
    "seed": 123,
    "env": {
        "C": 3000.0,   # 总容量
        "k": 3,        # 钱包数
        "T": 1000,     # 交易额上界（仅做一致性检查）
        "F": 1,        # flush 冻结长度
    },
    "eval": {
        "regime_names": ["SL", "SH", "BL", "BH"],
        "num_episodes_per_regime": 100,
        "max_steps": GLOBAL["episode_length"],
        "seed_stride_per_regime": 10000,
        "seed_stride_per_episode": 1,
    },
    "output": {
        "save_results": True,
        "results_dir": "fa_regime_eval_outputs",
        "summary_json": "fa_regime_summary.json",
        "summary_csv": "fa_regime_summary.csv",
        "per_episode_csv": "fa_regime_per_episode.csv",
    }
}


# ============================================================
# 2) FA baseline：修正版
# ============================================================
class KWalletFA:
    """
    Flush-All baseline（与当前 FWF 脚本尽量保持一致的度量口径）

    规则：
    - 当前若所有钱包都在冷却，则系统离线，本步 drop
    - 若有可用钱包，尝试在任一可用钱包中放入（first-fit）
    - 若没有任何可用钱包能容纳当前交易，则 FLUSH ALL，并丢弃当前交易
    - 若 tx > 单钱包容量，则直接 oversize drop
    """

    def __init__(self, C: float, k: int, T: int, F: int, max_steps: int, seed: int = 123):
        self.C = float(C)
        self.k = int(k)
        self.size = self.C / self.k
        self.T = int(T)
        self.F = int(F)
        self.max_steps = int(max_steps)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.reset_state()

    def reset_state(self):
        self.t = 0
        self.wallets = np.full(self.k, self.size, dtype=float)
        self.freeze_until = np.full(self.k, -1, dtype=int)

        self.total_accepted = 0.0
        self.total_offered = 0.0
        self.num_flushes = 0
        self.num_drops = 0
        self.oversize_drops = 0
        self.insufficient_drops = 0
        self.wait_steps = 0

    def _usable(self, i: int) -> bool:
        # 与 FWF 脚本口径一致：t > freeze_until 才视为可用
        return self.t > self.freeze_until[i]

    def _flush_all(self):
        # 立即把余额恢复满，但在 freeze_until 之前不可用
        for i in range(self.k):
            self.wallets[i] = self.size
            self.freeze_until[i] = self.t + self.F
        self.num_flushes += 1

    def reset(self, tx_stream: Optional[List[float]] = None):
        self.reset_state()

        if tx_stream is None:
            self.tx_stream = [int(self.rng.integers(1, self.T + 1)) for _ in range(self.max_steps)]
        else:
            if len(tx_stream) < self.max_steps:
                raise ValueError(f"tx_stream length < max_steps: {len(tx_stream)} < {self.max_steps}")
            self.tx_stream = list(tx_stream[:self.max_steps])

        self.current_tx = float(self.tx_stream[self.t])

    def step(self):
        tx = float(self.current_tx)
        self.total_offered += tx

        accepted = 0.0
        placed = False
        flushes_this_step = 0

        # 1) oversize：任何钱包都不可能放下
        if tx > self.size:
            self.num_drops += 1
            self.oversize_drops += 1

        else:
            usable_wallets = [i for i in range(self.k) if self._usable(i)]

            # 2) 若所有钱包都在冷却，则系统离线
            if not usable_wallets:
                self.num_drops += 1
                self.insufficient_drops += 1
                self.wait_steps += 1

            else:
                # 3) 在当前所有可用钱包中找 first-fit
                for i in usable_wallets:
                    if self.wallets[i] >= tx:
                        self.wallets[i] -= tx
                        accepted = tx
                        placed = True
                        self.total_accepted += accepted
                        break

                # 4) 若所有可用钱包都放不下，则 flush all，并丢弃当前 tx
                if not placed:
                    self._flush_all()
                    flushes_this_step += 1
                    self.num_drops += 1
                    self.insufficient_drops += 1

        # 推进时间
        self.t += 1
        if self.t < self.max_steps:
            self.current_tx = float(self.tx_stream[self.t])

        return accepted, placed, flushes_this_step

    def get_metrics(self) -> Dict[str, float]:
        val_acc = self.total_accepted / self.total_offered if self.total_offered > 0 else 0.0
        drop_rate = self.num_drops / self.max_steps if self.max_steps > 0 else 0.0
        difficulty = 0.6 * (1.0 - val_acc) + 0.4 * drop_rate

        return {
            "settled": self.total_accepted,
            "offered": self.total_offered,
            "drops": self.num_drops,
            "oversize_drops": self.oversize_drops,
            "insufficient_drops": self.insufficient_drops,
            "flushes": self.num_flushes,
            "utilization": self.total_accepted / (self.C * self.max_steps),
            "avg_tx_value": self.total_accepted / max(1, self.max_steps - self.num_drops),
            "drop_rate": drop_rate,
            "value_acceptance": val_acc,
            "difficulty": difficulty,
            "offline_ratio": self.wait_steps / max(1, self.max_steps),
        }


# ============================================================
# 3) 工具函数
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


def aggregate_metric_dicts(dict_list: List[Dict]) -> Dict:
    """
    只聚合数值型字段，跳过 regime / episode_index / seed 等标识字段。
    """
    out = {}
    if not dict_list:
        return out

    exclude_keys = {"regime", "episode_index", "seed"}
    sample = dict_list[0]
    metric_keys = []

    for key, value in sample.items():
        if key in exclude_keys:
            continue
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            metric_keys.append(key)

    for key in metric_keys:
        values = [float(d[key]) for d in dict_list]
        out[key] = float(np.mean(values))
        out[f"{key}_std"] = float(np.std(values))
        out[f"{key}_min"] = float(np.min(values))
        out[f"{key}_max"] = float(np.max(values))
    return out


def compute_regime_data_fingerprint(regime_name: str, tx_episodes: List[np.ndarray]) -> str:
    arr = np.stack(tx_episodes, axis=0)
    digest = hashlib.md5(arr.tobytes()).hexdigest()
    return f"{regime_name}:{digest}"


# ============================================================
# 4) 核心：按 regime 现生成数据并评估
# ============================================================
def evaluate_one_regime(regime_name: str, config: Dict) -> Dict:
    if regime_name not in REGIME_CONFIGS:
        raise ValueError(f"未知 regime: {regime_name}")

    regime_cfg = REGIME_CONFIGS[regime_name]
    num_eps = config["eval"]["num_episodes_per_regime"]
    max_steps = config["eval"]["max_steps"]

    env = KWalletFA(
        C=config["env"]["C"],
        k=config["env"]["k"],
        T=config["env"]["T"],
        F=config["env"]["F"],
        max_steps=max_steps,
        seed=config["seed"],
    )

    per_episode_results = []
    tx_episodes = []

    regime_index = config["eval"]["regime_names"].index(regime_name)

    for ep in range(num_eps):
        seed = (
            config["seed"]
            + regime_index * config["eval"]["seed_stride_per_regime"]
            + ep * config["eval"]["seed_stride_per_episode"]
        )

        tx_sizes, state_labels = generate_episode(
            config=regime_cfg,
            T=max_steps,
            seed=seed,
        )

        tx_episodes.append(tx_sizes.copy())

        max_tx_observed = float(np.max(tx_sizes))
        if max_tx_observed > config["env"]["T"]:
            print(f"⚠️  警告：{regime_name} 第 {ep} 个 episode 出现 tx={max_tx_observed:.2f}，超过 env.T={config['env']['T']}")

        env.reset(tx_stream=tx_sizes)
        for _ in range(max_steps):
            env.step()

        metrics = env.get_metrics()
        metrics["regime"] = regime_name
        metrics["episode_index"] = ep
        metrics["seed"] = seed
        metrics["state_burst_fraction"] = float(sum(1 for s in state_labels if s == "burst") / len(state_labels))
        per_episode_results.append(metrics)

    fingerprint = compute_regime_data_fingerprint(regime_name, tx_episodes)
    summary = aggregate_metric_dicts(per_episode_results)

    summary_row = {
        "regime": regime_name,
        "time_structure": regime_cfg["time_structure"],
        "size_structure": regime_cfg["size_structure"],
        "description": regime_cfg["description"],
        "fingerprint": fingerprint,
        **summary,
    }

    return {
        "regime_name": regime_name,
        "summary_row": summary_row,
        "per_episode_results": per_episode_results,
    }


def run_all_regimes(config: Dict) -> Dict:
    all_summary_rows = []
    all_per_episode_rows = []

    print("\n[FA difficulty screening]")
    print("-" * 72)
    print(f"评估 regimes: {config['eval']['regime_names']}")
    print(f"每类 episode 数: {config['eval']['num_episodes_per_regime']}")
    print(f"每个 episode 步数: {config['eval']['max_steps']}")
    print("-" * 72)

    for regime_name in config["eval"]["regime_names"]:
        result = evaluate_one_regime(regime_name, config)
        all_summary_rows.append(result["summary_row"])
        all_per_episode_rows.extend(result["per_episode_results"])
        print(f"完成 {regime_name}")

    return {
        "config": config,
        "summary_rows": all_summary_rows,
        "per_episode_rows": all_per_episode_rows,
    }


# ============================================================
# 5) 输出
# ============================================================
def print_summary_table(summary_rows: List[Dict]) -> None:
    print("\n[FA regime summary]")
    print("-" * 130)
    print(
        f"{'Regime':<8}"
        f"{'Settled':>12}"
        f"{'Drops':>10}"
        f"{'Flushes':>10}"
        f"{'ValAcc':>10}"
        f"{'DropRate':>11}"
        f"{'Difficulty':>12}"
        f"{'BurstFrac':>11}"
    )
    print("-" * 130)

    for row in summary_rows:
        print(
            f"{row['regime']:<8}"
            f"{row['settled']:>12.2f}"
            f"{row['drops']:>10.2f}"
            f"{row['flushes']:>10.2f}"
            f"{row['value_acceptance']:>10.4f}"
            f"{row['drop_rate']:>11.4f}"
            f"{row['difficulty']:>12.4f}"
            f"{row['state_burst_fraction']:>11.4f}"
        )

    print("-" * 130)
    print("Difficulty = 0.6 * (1 - ValAcc) + 0.4 * DropRate")


def save_results(results: Dict, config: Dict) -> None:
    out_dir = ensure_output_dir(config["output"]["results_dir"])

    save_json(
        {
            "config": config,
            "summary_rows": results["summary_rows"],
        },
        out_dir / config["output"]["summary_json"],
    )

    save_csv(results["summary_rows"], out_dir / config["output"]["summary_csv"])
    save_csv(results["per_episode_rows"], out_dir / config["output"]["per_episode_csv"])

    print(f"\n结果已保存到：{out_dir.resolve()}")
    print(f"  - {config['output']['summary_json']}")
    print(f"  - {config['output']['summary_csv']}")
    print(f"  - {config['output']['per_episode_csv']}")


# ============================================================
# 6) 主程序
# ============================================================
if __name__ == "__main__":
    results = run_all_regimes(CONFIG)
    print_summary_table(results["summary_rows"])

    if CONFIG["output"]["save_results"]:
        save_results(results, CONFIG)
