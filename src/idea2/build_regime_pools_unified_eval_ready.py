#4/1
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
统一生成 k-wallet 项目用的 regime pools（含独立 eval pools）
=========================================================
用途：
1. 生成四个 static train pools（SL / SH / BL / BH）
2. 可选生成 mixed full/generalist train pool
3. 可选生成 mixed equal/generalist train pool（总条数固定，适合公平比较）
4. 生成一个或多个 switching train pools
5. 额外生成一套 seed 完全独立的 static eval pools 和 switching eval pools

你以后最常改的地方：
- GENERATOR_CANDIDATES：generator 文件名/位置
- POOL_CONFIG：训练池 / 评估池规模
- SWITCHING_SPECS：切换顺序和比例
"""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ============================================================
# 1) 自动寻找 generator 文件
#    以后如果你的 generator 文件换了位置，只改这里
# ============================================================
THIS_FILE = Path(__file__).resolve()
CUR_DIR = THIS_FILE.parent
SRC_DIR = CUR_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

GENERATOR_CANDIDATES = [
    CUR_DIR / "regime_generator2.py",      # 例如和本脚本都在 src/idea3
    SRC_DIR / "regime_generator2.py",      # 例如 generator 在 src 根目录
    SRC_DIR / "idea2" / "regime_generator2.py",  # 例如 generator 在 src/idea2
]

GENERATOR_FILE = None
for candidate in GENERATOR_CANDIDATES:
    if candidate.exists():
        GENERATOR_FILE = candidate
        break

if GENERATOR_FILE is None:
    raise FileNotFoundError(
        "找不到 generator 文件。\n"
        "请检查 GENERATOR_CANDIDATES 配置。\n"
        f"已检查路径: {[str(p) for p in GENERATOR_CANDIDATES]}"
    )

spec = importlib.util.spec_from_file_location("regime_generator2", GENERATOR_FILE)
if spec is None or spec.loader is None:
    raise ImportError(f"无法为 generator 文件创建 import spec: {GENERATOR_FILE}")

gen_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen_mod)

try:
    REGIME_CONFIGS = gen_mod.REGIME_CONFIGS
    generate_episode = gen_mod.generate_episode
except AttributeError as e:
    raise AttributeError(
        "导入成功，但 generator 文件里缺少 REGIME_CONFIGS 或 generate_episode。\n"
        f"当前 generator 文件: {GENERATOR_FILE}\n原始报错: {e}"
    )

GENERATOR_TAG = GENERATOR_FILE.stem


# ============================================================
# 2) 总配置：以后最常改这里
# ============================================================
POOL_CONFIG = {
    # 输出目录
    "output_dir": "data/pools",

    # -------------------------
    # train pools
    # -------------------------
    "static_num_episodes": 1000,
    "static_episode_length": 1000,

    # mixed pools
    "save_mixed_full_pool": True,
    "save_mixed_equal_pool": True,
    "mixed_equal_total_episodes": 1200,   # 公平比较版 mixed，总条数固定

    # switching train pools
    "switching_num_episodes": 1000,
    "switching_total_length": 1000,

    # -------------------------
    # 独立 eval pools
    # -------------------------
    "save_eval_pools": True,
    "eval_num_episodes": 200,
    "eval_episode_length": 1000,

    # -------------------------
    # 种子设置
    # train 和 eval 必须彻底分开
    # -------------------------
    "base_seed": 20260331,
    "eval_seed_offset": 1_000_000,
}


# ============================================================
# 3) switching specs：以后想加新切换模板，就在这里加
#    比例不要求和为 1，代码里会自动归一化
# ============================================================
SWITCHING_SPECS = [
    {
        "name": "SL_BH_SH_20_50_30",
        "sequence": ["SL", "BH", "SH"],
        "weights": {"SL": 0.2, "BH": 0.5, "SH": 0.3},
        "num_episodes": 1000,
        "total_length": 1000,
        "seed_offset": 90000,
    },
]


# ============================================================
# 4) regime 固定 seed 偏移：不要用 hash()，可复现性更稳
# ============================================================
REGIME_SEED_OFFSET = {
    "SL": 100,
    "SH": 200,
    "BL": 300,
    "BH": 400,
}


# ============================================================
# 5) 基础工具函数
# ============================================================
def ensure_dir(path_str: str) -> Path:
    path = PROJECT_ROOT / path_str
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)



def save_pool_npy(pool: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, pool)



def pretty_print_preview(pool: np.ndarray, title: str) -> None:
    """简单打印，方便肉眼检查 pool 大致正常"""
    print(f"\n=== {title} ===")
    print("shape =", pool.shape)
    print("第1条 episode 前10笔 =", pool[0][:10].tolist())
    if pool.shape[1] >= 110:
        print("第95~105笔 =", pool[0][95:105].tolist())
    if pool.shape[1] >= 210:
        print("第195~205笔 =", pool[0][195:205].tolist())
    print("最后10笔 =", pool[0][-10:].tolist())


# ============================================================
# 6) 长度分配函数：把比例转成具体步数
# ============================================================
def allocate_segment_lengths(total_length: int, regime_weights: Dict[str, float]) -> Dict[str, int]:
    regime_names = list(regime_weights.keys())
    weights = np.array([regime_weights[r] for r in regime_names], dtype=float)

    if np.any(weights < 0):
        raise ValueError("regime_weights 里不能有负数")
    if weights.sum() <= 0:
        raise ValueError("regime_weights 总和必须大于 0")

    weights = weights / weights.sum()
    raw_lengths = weights * total_length
    floor_lengths = np.floor(raw_lengths).astype(int)

    remainder = total_length - floor_lengths.sum()
    fractional_parts = raw_lengths - floor_lengths
    order = np.argsort(-fractional_parts)

    for i in range(remainder):
        floor_lengths[order[i]] += 1

    return {regime_names[i]: int(floor_lengths[i]) for i in range(len(regime_names))}


# ============================================================
# 7) 生成 static pool
# ============================================================
def generate_static_pool(
    config: Dict,
    num_episodes: int,
    episode_length: int,
    base_seed: int,
) -> np.ndarray:
    episodes = []

    for ep in range(num_episodes):
        seed = base_seed + ep
        tx_sizes, _ = generate_episode(config=config, T=episode_length, seed=seed)
        episodes.append(tx_sizes)

    return np.array(episodes, dtype=float)


# ============================================================
# 8) 生成 mixed full pool（四类全部拼起来）
# ============================================================
def generate_mixed_full_pool(
    regime_names: List[str],
    regime_configs: Dict[str, Dict],
    num_episodes_per_regime: int,
    episode_length: int,
    base_seed: int,
) -> np.ndarray:
    all_rows = []

    for regime_name in regime_names:
        regime_seed_base = base_seed + REGIME_SEED_OFFSET[regime_name]
        pool = generate_static_pool(
            config=regime_configs[regime_name],
            num_episodes=num_episodes_per_regime,
            episode_length=episode_length,
            base_seed=regime_seed_base,
        )
        all_rows.append(pool)

    mixed_pool = np.concatenate(all_rows, axis=0)
    return mixed_pool


# ============================================================
# 9) 生成 mixed equal pool（总条数固定，适合公平比较）
# ============================================================
def generate_mixed_equal_pool(
    regime_names: List[str],
    regime_configs: Dict[str, Dict],
    total_episodes: int,
    episode_length: int,
    base_seed: int,
) -> np.ndarray:
    num_regimes = len(regime_names)
    if total_episodes % num_regimes != 0:
        raise ValueError(
            f"total_episodes={total_episodes} 不能被 regime 数量 {num_regimes} 整除"
        )

    episodes_per_regime = total_episodes // num_regimes
    all_rows = []

    for regime_name in regime_names:
        regime_seed_base = base_seed + REGIME_SEED_OFFSET[regime_name]
        pool = generate_static_pool(
            config=regime_configs[regime_name],
            num_episodes=episodes_per_regime,
            episode_length=episode_length,
            base_seed=regime_seed_base,
        )
        all_rows.append(pool)

    mixed_pool = np.concatenate(all_rows, axis=0)

    # 打乱顺序，让 mixed 更自然
    rng = np.random.default_rng(base_seed + 999)
    rng.shuffle(mixed_pool, axis=0)

    return mixed_pool


# ============================================================
# 10) 生成 switching pool（顺序固定，比例可调）
# ============================================================
def generate_switching_pool_by_ratio(
    regime_sequence: List[str],
    regime_weights: Dict[str, float],
    regime_configs: Dict[str, Dict],
    total_length: int,
    num_episodes: int,
    base_seed: int,
) -> Tuple[np.ndarray, Dict[str, int]]:
    filtered_weights = {r: regime_weights[r] for r in regime_sequence}
    allocated_lengths = allocate_segment_lengths(total_length, filtered_weights)
    segment_lengths = [allocated_lengths[r] for r in regime_sequence]

    episodes = []

    for ep in range(num_episodes):
        full_episode = []

        for seg_idx, regime_name in enumerate(regime_sequence):
            cfg = regime_configs[regime_name]
            seg_len = segment_lengths[seg_idx]

            # 每个 segment 的 seed 生成规则
            seed = base_seed + ep * 1000 + REGIME_SEED_OFFSET[regime_name] + seg_idx

            tx_sizes, _ = generate_episode(config=cfg, T=seg_len, seed=seed)
            full_episode.extend(tx_sizes.tolist())

        episodes.append(full_episode)

    return np.array(episodes, dtype=float), allocated_lengths


# ============================================================
# 11) 文件名工具：统一命名
# ============================================================
def build_static_pool_filename(regime_name: str, num_episodes: int, episode_length: int) -> str:
    return f"{regime_name}_static_pool_E{num_episodes}_T{episode_length}.npy"


def build_static_meta_filename(regime_name: str, num_episodes: int, episode_length: int) -> str:
    return f"{regime_name}_static_pool_E{num_episodes}_T{episode_length}_meta.json"


def build_static_eval_pool_filename(regime_name: str, num_episodes: int, episode_length: int) -> str:
    return f"{regime_name}_static_eval_pool_E{num_episodes}_T{episode_length}.npy"


def build_static_eval_meta_filename(regime_name: str, num_episodes: int, episode_length: int) -> str:
    return f"{regime_name}_static_eval_pool_E{num_episodes}_T{episode_length}_meta.json"


def build_mixed_full_pool_filename(regime_names: List[str], num_total_episodes: int, episode_length: int) -> str:
    tag = "".join(regime_names)
    return f"MIXED_{tag}_pool_E{num_total_episodes}_T{episode_length}.npy"


def build_mixed_full_meta_filename(regime_names: List[str], num_total_episodes: int, episode_length: int) -> str:
    tag = "".join(regime_names)
    return f"MIXED_{tag}_pool_E{num_total_episodes}_T{episode_length}_meta.json"


def build_mixed_equal_pool_filename(regime_names: List[str], total_episodes: int, episode_length: int) -> str:
    tag = "".join(regime_names)
    return f"MIXED_EQ_{tag}_pool_E{total_episodes}_T{episode_length}.npy"


def build_mixed_equal_meta_filename(regime_names: List[str], total_episodes: int, episode_length: int) -> str:
    tag = "".join(regime_names)
    return f"MIXED_EQ_{tag}_pool_E{total_episodes}_T{episode_length}_meta.json"


def build_switch_pool_filename(spec_name: str, num_episodes: int, total_length: int) -> str:
    return f"{spec_name}_switch_pool_E{num_episodes}_T{total_length}.npy"


def build_switch_meta_filename(spec_name: str, num_episodes: int, total_length: int) -> str:
    return f"{spec_name}_switch_pool_E{num_episodes}_T{total_length}_meta.json"


def build_switch_eval_pool_filename(spec_name: str, num_episodes: int, total_length: int) -> str:
    return f"{spec_name}_switch_eval_pool_E{num_episodes}_T{total_length}.npy"


def build_switch_eval_meta_filename(spec_name: str, num_episodes: int, total_length: int) -> str:
    return f"{spec_name}_switch_eval_pool_E{num_episodes}_T{total_length}_meta.json"


# ============================================================
# 12) 主函数：统一生成 train + eval pools
# ============================================================
def build_and_save_all_pools() -> None:
    output_dir = ensure_dir(POOL_CONFIG["output_dir"])

    static_num_episodes = POOL_CONFIG["static_num_episodes"]
    static_episode_length = POOL_CONFIG["static_episode_length"]
    switching_num_episodes = POOL_CONFIG["switching_num_episodes"]
    switching_total_length = POOL_CONFIG["switching_total_length"]

    eval_num_episodes = POOL_CONFIG["eval_num_episodes"]
    eval_episode_length = POOL_CONFIG["eval_episode_length"]

    base_seed = POOL_CONFIG["base_seed"]
    eval_seed_offset = POOL_CONFIG["eval_seed_offset"]
    eval_base_seed = base_seed + eval_seed_offset

    static_regime_names = list(REGIME_CONFIGS.keys())

    # --------------------------------------------------------
    # A1. 生成四个 static train pools
    # --------------------------------------------------------
    print("\n" + "=" * 72)
    print("开始生成 static train pools")
    print("=" * 72)

    for regime_name in static_regime_names:
        cfg = REGIME_CONFIGS[regime_name]
        regime_seed_base = base_seed + REGIME_SEED_OFFSET[regime_name]

        pool = generate_static_pool(
            config=cfg,
            num_episodes=static_num_episodes,
            episode_length=static_episode_length,
            base_seed=regime_seed_base,
        )

        pool_path = output_dir / build_static_pool_filename(regime_name, static_num_episodes, static_episode_length)
        meta_path = output_dir / build_static_meta_filename(regime_name, static_num_episodes, static_episode_length)

        save_pool_npy(pool, pool_path)
        save_json(
            {
                "type": "static_train_pool",
                "generator_file": str(GENERATOR_FILE),
                "generator_tag": GENERATOR_TAG,
                "regime": regime_name,
                "num_episodes": static_num_episodes,
                "episode_length": static_episode_length,
                "base_seed": regime_seed_base,
                "config": cfg,
            },
            meta_path,
        )

        print(f"[已保存] {pool_path}")
        pretty_print_preview(pool, f"static train pool - {regime_name}")

    # --------------------------------------------------------
    # A2. 生成四个独立的 static eval pools（seed 与 train 不重叠）
    # --------------------------------------------------------
    if POOL_CONFIG.get("save_eval_pools", False):
        print("\n" + "=" * 72)
        print("开始生成独立的 static eval pools")
        print("=" * 72)

        for regime_name in static_regime_names:
            cfg = REGIME_CONFIGS[regime_name]
            regime_seed_base = eval_base_seed + REGIME_SEED_OFFSET[regime_name]

            eval_pool = generate_static_pool(
                config=cfg,
                num_episodes=eval_num_episodes,
                episode_length=eval_episode_length,
                base_seed=regime_seed_base,
            )

            eval_pool_path = output_dir / build_static_eval_pool_filename(regime_name, eval_num_episodes, eval_episode_length)
            eval_meta_path = output_dir / build_static_eval_meta_filename(regime_name, eval_num_episodes, eval_episode_length)

            save_pool_npy(eval_pool, eval_pool_path)
            save_json(
                {
                    "type": "static_eval_pool",
                    "generator_file": str(GENERATOR_FILE),
                    "generator_tag": GENERATOR_TAG,
                    "regime": regime_name,
                    "num_episodes": eval_num_episodes,
                    "episode_length": eval_episode_length,
                    "base_seed": regime_seed_base,
                    "config": cfg,
                    "note": "独立评估池，seed 与 train pools 不重叠",
                },
                eval_meta_path,
            )

            print(f"[已保存] {eval_pool_path}")
            pretty_print_preview(eval_pool, f"static eval pool - {regime_name}")

    # --------------------------------------------------------
    # B1. 可选：生成 mixed full pool（四类全部拼起来）
    # --------------------------------------------------------
    if POOL_CONFIG.get("save_mixed_full_pool", False):
        print("\n" + "=" * 72)
        print("开始生成 mixed full train pool")
        print("=" * 72)

        mixed_full_pool = generate_mixed_full_pool(
            regime_names=static_regime_names,
            regime_configs=REGIME_CONFIGS,
            num_episodes_per_regime=static_num_episodes,
            episode_length=static_episode_length,
            base_seed=base_seed + 50000,
        )

        total_eps = mixed_full_pool.shape[0]
        mixed_full_pool_path = output_dir / build_mixed_full_pool_filename(static_regime_names, total_eps, static_episode_length)
        mixed_full_meta_path = output_dir / build_mixed_full_meta_filename(static_regime_names, total_eps, static_episode_length)

        save_pool_npy(mixed_full_pool, mixed_full_pool_path)
        save_json(
            {
                "type": "mixed_full_train_pool",
                "generator_file": str(GENERATOR_FILE),
                "generator_tag": GENERATOR_TAG,
                "regime_names": static_regime_names,
                "num_episodes_per_regime": static_num_episodes,
                "total_episodes": total_eps,
                "episode_length": static_episode_length,
                "base_seed": base_seed + 50000,
            },
            mixed_full_meta_path,
        )

        print(f"[已保存] {mixed_full_pool_path}")
        pretty_print_preview(mixed_full_pool, "mixed full train pool")

    # --------------------------------------------------------
    # B2. 可选：生成 mixed equal pool（总条数固定，适合公平比较）
    # --------------------------------------------------------
    if POOL_CONFIG.get("save_mixed_equal_pool", False):
        print("\n" + "=" * 72)
        print("开始生成 mixed equal train pool")
        print("=" * 72)

        mixed_equal_total_episodes = POOL_CONFIG["mixed_equal_total_episodes"]

        mixed_equal_pool = generate_mixed_equal_pool(
            regime_names=static_regime_names,
            regime_configs=REGIME_CONFIGS,
            total_episodes=mixed_equal_total_episodes,
            episode_length=static_episode_length,
            base_seed=base_seed + 60000,
        )

        mixed_equal_pool_path = output_dir / build_mixed_equal_pool_filename(static_regime_names, mixed_equal_total_episodes, static_episode_length)
        mixed_equal_meta_path = output_dir / build_mixed_equal_meta_filename(static_regime_names, mixed_equal_total_episodes, static_episode_length)

        save_pool_npy(mixed_equal_pool, mixed_equal_pool_path)
        save_json(
            {
                "type": "mixed_equal_train_pool",
                "generator_file": str(GENERATOR_FILE),
                "generator_tag": GENERATOR_TAG,
                "regime_names": static_regime_names,
                "total_episodes": mixed_equal_total_episodes,
                "episode_length": static_episode_length,
                "base_seed": base_seed + 60000,
            },
            mixed_equal_meta_path,
        )

        print(f"[已保存] {mixed_equal_pool_path}")
        pretty_print_preview(mixed_equal_pool, "mixed equal train pool")

    # --------------------------------------------------------
    # C1. 生成 switching train pools
    # --------------------------------------------------------
    print("\n" + "=" * 72)
    print("开始生成 switching train pools")
    print("=" * 72)

    for spec in SWITCHING_SPECS:
        spec_name = spec["name"]
        regime_sequence = spec["sequence"]
        regime_weights = spec["weights"]
        num_episodes = spec.get("num_episodes", switching_num_episodes)
        total_length = spec.get("total_length", switching_total_length)
        seed_offset = spec.get("seed_offset", 90000)

        pool, allocated_lengths = generate_switching_pool_by_ratio(
            regime_sequence=regime_sequence,
            regime_weights=regime_weights,
            regime_configs=REGIME_CONFIGS,
            total_length=total_length,
            num_episodes=num_episodes,
            base_seed=base_seed + seed_offset,
        )

        pool_path = output_dir / build_switch_pool_filename(spec_name, num_episodes, total_length)
        meta_path = output_dir / build_switch_meta_filename(spec_name, num_episodes, total_length)

        save_pool_npy(pool, pool_path)
        save_json(
            {
                "type": "switching_train_pool",
                "generator_file": str(GENERATOR_FILE),
                "generator_tag": GENERATOR_TAG,
                "sequence": regime_sequence,
                "weights": regime_weights,
                "allocated_lengths": allocated_lengths,
                "num_episodes": num_episodes,
                "total_length": total_length,
                "base_seed": base_seed + seed_offset,
            },
            meta_path,
        )

        print(f"[已保存] {pool_path}")
        print("allocated_lengths =", allocated_lengths)
        pretty_print_preview(pool, f"switching train pool - {spec_name}")

        # ----------------------------------------------------
        # C2. 生成独立的 switching eval pool（seed 与 train 不重叠）
        # ----------------------------------------------------
        if POOL_CONFIG.get("save_eval_pools", False):
            eval_switch_seed = eval_base_seed + seed_offset

            eval_switch_pool, eval_allocated_lengths = generate_switching_pool_by_ratio(
                regime_sequence=regime_sequence,
                regime_weights=regime_weights,
                regime_configs=REGIME_CONFIGS,
                total_length=eval_episode_length,
                num_episodes=eval_num_episodes,
                base_seed=eval_switch_seed,
            )

            eval_switch_path = output_dir / build_switch_eval_pool_filename(spec_name, eval_num_episodes, eval_episode_length)
            eval_switch_meta_path = output_dir / build_switch_eval_meta_filename(spec_name, eval_num_episodes, eval_episode_length)

            save_pool_npy(eval_switch_pool, eval_switch_path)
            save_json(
                {
                    "type": "switching_eval_pool",
                    "generator_file": str(GENERATOR_FILE),
                    "generator_tag": GENERATOR_TAG,
                    "sequence": regime_sequence,
                    "weights": regime_weights,
                    "allocated_lengths": eval_allocated_lengths,
                    "num_episodes": eval_num_episodes,
                    "total_length": eval_episode_length,
                    "base_seed": eval_switch_seed,
                    "note": "独立 switching 评估池，seed 与 train pools 不重叠",
                },
                eval_switch_meta_path,
            )

            print(f"[已保存] {eval_switch_path}")
            print("eval allocated_lengths =", eval_allocated_lengths)
            pretty_print_preview(eval_switch_pool, f"switching eval pool - {spec_name}")

    print("\n全部 pool 生成完成。")
    print(f"当前 generator: {GENERATOR_FILE}")
    print(f"输出目录：{output_dir.resolve()}")


# ============================================================
# 13) 程序入口
# ============================================================
if __name__ == "__main__":
    build_and_save_all_pools()
