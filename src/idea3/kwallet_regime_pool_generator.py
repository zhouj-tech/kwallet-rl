#4/3 新数据生成器（train+test+validation）
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# =========================================================
# 路径
# =========================================================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent
DATA_POOL_DIR = PROJECT_ROOT / "data" / "pools"
DATA_REPORT_DIR = PROJECT_ROOT / "data" / "reports"


# =========================================================
# Regime 定义
# 这些参数不是“唯一正确值”，而是一套可复现实验用的结构化设定。
# =========================================================
@dataclass(frozen=True)
class RegimeSpec:
    name: str
    base_mean: float
    base_std: float
    max_tx: float
    burst_start_prob: float
    burst_length_mean: float
    burst_uplift: float
    large_prob_normal: float
    large_prob_burst: float


REGIME_SPECS: Dict[str, RegimeSpec] = {
    "SL": RegimeSpec(
        name="SL",
        base_mean=54.0,
        base_std=14.0,
        max_tx=87.5,
        burst_start_prob=0.0,
        burst_length_mean=0.0,
        burst_uplift=0.0,
        large_prob_normal=0.00,
        large_prob_burst=0.00,
    ),
    "SH": RegimeSpec(
        name="SH",
        base_mean=45.0,
        base_std=10.0,
        max_tx=140.0,
        burst_start_prob=0.0,
        burst_length_mean=0.0,
        burst_uplift=0.0,
        large_prob_normal=0.10,
        large_prob_burst=0.10,
    ),
    "BL": RegimeSpec(
        name="BL",
        base_mean=46.0,
        base_std=11.0,
        max_tx=95.0,
        burst_start_prob=0.05,
        burst_length_mean=6.0,
        burst_uplift=0.22,
        large_prob_normal=0.02,
        large_prob_burst=0.05,
    ),
    "BH": RegimeSpec(
        name="BH",
        base_mean=42.0,
        base_std=12.0,
        max_tx=150.0,
        burst_start_prob=0.05,
        burst_length_mean=6.0,
        burst_uplift=0.24,
        large_prob_normal=0.08,
        large_prob_burst=0.20,
    ),
}


# 你当前常用的 hidden switching 结构
SWITCHING_SPECS = {
    "SL_BH_SH_20_50_30": [("SL", 0.20), ("BH", 0.50), ("SH", 0.30)],
}


DEFAULT_CONFIG = {
    # 全局
    "base_seed": 532,
    "train_seed_offset": 0,
    "val_seed_offset": 2_000_000,
    "eval_seed_offset": 1_000_000,
    "save_reports": True,

    # episode 长度
    "episode_length": 1000,

    # 是否生成哪些文件
    "save_static_master_pools": True,
    "save_static_eval_pools": True,
    "save_mixed_full_master_pool": True,
    "save_mixed_equal_master_pool": True,
    "save_mixed_equal_val_pool": True,
    "save_switching_master_pools": True,
    "save_switching_eval_pools": True,

    # 规模：以后主要改这些数，不需要改文件名
    "static_master_episodes": 5000,
    "static_eval_episodes": 200,
    "mixed_full_master_episodes": 5000,
    "mixed_equal_master_episodes": 5000,
    "mixed_equal_val_episodes": 300,
    "switching_master_episodes": 5000,
    "switching_eval_episodes": 200,
}


# =========================================================
# 工具函数
# =========================================================
def ensure_dirs() -> None:
    DATA_POOL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_REPORT_DIR.mkdir(parents=True, exist_ok=True)


def build_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def summarize_pool(pool: np.ndarray) -> Dict[str, float]:
    flat = pool.reshape(-1).astype(np.float64)
    return {
        "episodes": int(pool.shape[0]),
        "steps": int(pool.shape[1]),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "p50": float(np.percentile(flat, 50)),
        "p90": float(np.percentile(flat, 90)),
        "p95": float(np.percentile(flat, 95)),
        "p99": float(np.percentile(flat, 99)),
        "max": float(np.max(flat)),
    }


def save_pool(array: np.ndarray, file_name: str, meta: Dict) -> Path:
    ensure_dirs()
    pool_path = DATA_POOL_DIR / file_name
    np.save(pool_path, array.astype(np.int32))

    if DEFAULT_CONFIG["save_reports"]:
        report_path = DATA_REPORT_DIR / f"{pool_path.stem}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[saved] {pool_path}")
    return pool_path


# =========================================================
# 单 regime 生成逻辑
# =========================================================
def _sample_one_tx(spec: RegimeSpec, rng: np.random.Generator, in_burst: bool) -> int:
    mean = spec.base_mean * (1.0 + spec.burst_uplift if in_burst else 1.0)
    std = spec.base_std * (1.0 + 0.50 * spec.burst_uplift if in_burst else 1.0)

    large_prob = spec.large_prob_burst if in_burst else spec.large_prob_normal
    if rng.random() < large_prob:
        tx = rng.uniform(max(spec.max_tx * 0.55, mean), spec.max_tx)
    else:
        tx = rng.normal(loc=mean, scale=std)

    tx = max(1.0, min(float(tx), spec.max_tx))
    return int(round(tx))


def generate_regime_episode(
    spec: RegimeSpec,
    episode_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    out = np.zeros(episode_length, dtype=np.int32)
    burst_remaining = 0

    for t in range(episode_length):
        if burst_remaining > 0:
            in_burst = True
            burst_remaining -= 1
        else:
            if spec.burst_start_prob > 0 and rng.random() < spec.burst_start_prob:
                # 至少持续 1 步
                burst_remaining = max(1, int(rng.poisson(spec.burst_length_mean)))
                in_burst = True
                burst_remaining -= 1
            else:
                in_burst = False

        out[t] = _sample_one_tx(spec, rng, in_burst)

    return out


def generate_regime_pool(
    regime_name: str,
    num_episodes: int,
    episode_length: int,
    base_seed: int,
) -> np.ndarray:
    spec = REGIME_SPECS[regime_name]
    pool = np.zeros((num_episodes, episode_length), dtype=np.int32)
    for ep in range(num_episodes):
        rng = build_rng(base_seed + ep)
        pool[ep] = generate_regime_episode(spec, episode_length, rng)
    return pool


# =========================================================
# mixed pools
# =========================================================
def generate_mixed_full_master_pool(
    regime_order: List[str],
    num_episodes: int,
    episode_length: int,
    base_seed: int,
) -> np.ndarray:
    # mixed_full: 每个 episode 先随机抽一个 regime，然后整条 episode 都来自该 regime
    rng = build_rng(base_seed)
    chosen = rng.choice(regime_order, size=num_episodes, replace=True)
    pool = np.zeros((num_episodes, episode_length), dtype=np.int32)
    for ep, regime_name in enumerate(chosen):
        ep_rng = build_rng(base_seed + ep + 10_000)
        pool[ep] = generate_regime_episode(REGIME_SPECS[str(regime_name)], episode_length, ep_rng)
    return pool


def generate_mixed_equal_pool(
    regime_order: List[str],
    total_episodes: int,
    episode_length: int,
    base_seed: int,
) -> np.ndarray:
    # mixed_equal: 尽量平均地从各 regime 取 episode，再整体打乱顺序
    pieces = []
    n_regimes = len(regime_order)
    base_each = total_episodes // n_regimes
    remainder = total_episodes % n_regimes

    cursor_seed = base_seed
    for i, regime_name in enumerate(regime_order):
        count = base_each + (1 if i < remainder else 0)
        part = generate_regime_pool(regime_name, count, episode_length, cursor_seed)
        pieces.append(part)
        cursor_seed += 100_000

    pool = np.concatenate(pieces, axis=0)
    shuffle_rng = build_rng(base_seed + 999_999)
    shuffle_rng.shuffle(pool, axis=0)
    return pool


# =========================================================
# switching pools
# =========================================================
def _segment_lengths(episode_length: int, spec: List[Tuple[str, float]]) -> List[int]:
    raw = [episode_length * ratio for _, ratio in spec]
    lengths = [int(x) for x in raw]
    gap = episode_length - sum(lengths)
    # 把舍入误差补到最后一段
    lengths[-1] += gap
    return lengths


def generate_switching_episode(
    switching_spec: List[Tuple[str, float]],
    episode_length: int,
    rng_seed: int,
) -> np.ndarray:
    lengths = _segment_lengths(episode_length, switching_spec)
    out_parts = []
    cursor = rng_seed
    for (regime_name, _), seg_len in zip(switching_spec, lengths):
        seg_rng = build_rng(cursor)
        out_parts.append(generate_regime_episode(REGIME_SPECS[regime_name], seg_len, seg_rng))
        cursor += 1_000
    return np.concatenate(out_parts, axis=0)


def generate_switching_pool(
    switching_name: str,
    num_episodes: int,
    episode_length: int,
    base_seed: int,
) -> np.ndarray:
    spec = SWITCHING_SPECS[switching_name]
    pool = np.zeros((num_episodes, episode_length), dtype=np.int32)
    for ep in range(num_episodes):
        pool[ep] = generate_switching_episode(spec, episode_length, base_seed + ep)
    return pool


# =========================================================
# 文件名：用稳定名字，不把 episode 数写进 train/val 文件名里。
# 这样以后换 1000 / 3000 时，agent 配置不用反复改文件名。
# =========================================================
def static_master_name(regime_name: str, T: int) -> str:
    return f"{regime_name}_static_master_T{T}.npy"


def static_eval_name(regime_name: str, T: int) -> str:
    return f"{regime_name}_static_eval_T{T}.npy"


def mixed_full_master_name(regime_order: List[str], T: int) -> str:
    joined = "".join(regime_order)
    return f"MIXED_FULL_{joined}_master_T{T}.npy"


def mixed_equal_master_name(regime_order: List[str], T: int) -> str:
    joined = "".join(regime_order)
    return f"MIXED_EQ_{joined}_master_T{T}.npy"


def mixed_equal_val_name(regime_order: List[str], T: int) -> str:
    joined = "".join(regime_order)
    return f"MIXED_EQ_{joined}_val_T{T}.npy"


def switching_master_name(switching_name: str, T: int) -> str:
    return f"{switching_name}_switch_master_T{T}.npy"


def switching_eval_name(switching_name: str, T: int) -> str:
    return f"{switching_name}_switch_eval_T{T}.npy"


# =========================================================
# 命令行覆盖：以后要从 1000 改到 3000，不需要进文件里改。
# 示例：
# python kwallet_regime_pool_generator.py --mixed-eq-master-episodes 6000 --mixed-eq-val-episodes 500
# =========================================================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="K-Wallet regime pool generator")
    parser.add_argument("--base-seed", type=int, default=DEFAULT_CONFIG["base_seed"])
    parser.add_argument("--episode-length", type=int, default=DEFAULT_CONFIG["episode_length"])

    parser.add_argument("--static-master-episodes", type=int, default=DEFAULT_CONFIG["static_master_episodes"])
    parser.add_argument("--static-eval-episodes", type=int, default=DEFAULT_CONFIG["static_eval_episodes"])
    parser.add_argument("--mixed-full-master-episodes", type=int, default=DEFAULT_CONFIG["mixed_full_master_episodes"])
    parser.add_argument("--mixed-eq-master-episodes", type=int, default=DEFAULT_CONFIG["mixed_equal_master_episodes"])
    parser.add_argument("--mixed-eq-val-episodes", type=int, default=DEFAULT_CONFIG["mixed_equal_val_episodes"])
    parser.add_argument("--switching-master-episodes", type=int, default=DEFAULT_CONFIG["switching_master_episodes"])
    parser.add_argument("--switching-eval-episodes", type=int, default=DEFAULT_CONFIG["switching_eval_episodes"])

    return parser


# =========================================================
# 主流程
# =========================================================
def main() -> None:
    args = build_arg_parser().parse_args()
    ensure_dirs()

    regime_order = ["SL", "SH", "BL", "BH"]
    T = int(args.episode_length)
    base_seed = int(args.base_seed)

    master_manifest: Dict[str, Dict] = {
        "config": {
            "base_seed": base_seed,
            "train_seed_offset": DEFAULT_CONFIG["train_seed_offset"],
            "val_seed_offset": DEFAULT_CONFIG["val_seed_offset"],
            "eval_seed_offset": DEFAULT_CONFIG["eval_seed_offset"],
            "episode_length": T,
            "static_master_episodes": int(args.static_master_episodes),
            "static_eval_episodes": int(args.static_eval_episodes),
            "mixed_full_master_episodes": int(args.mixed_full_master_episodes),
            "mixed_equal_master_episodes": int(args.mixed_eq_master_episodes),
            "mixed_equal_val_episodes": int(args.mixed_eq_val_episodes),
            "switching_master_episodes": int(args.switching_master_episodes),
            "switching_eval_episodes": int(args.switching_eval_episodes),
        },
        "regimes": {name: asdict(spec) for name, spec in REGIME_SPECS.items()},
        "files": {},
    }

    # A. static master pools
    if DEFAULT_CONFIG["save_static_master_pools"]:
        print("\n=== A. static master pools ===")
        for idx, regime_name in enumerate(regime_order):
            seed = base_seed + DEFAULT_CONFIG["train_seed_offset"] + idx * 100_000
            pool = generate_regime_pool(regime_name, int(args.static_master_episodes), T, seed)
            fname = static_master_name(regime_name, T)
            meta = {
                "type": "static_master",
                "regime": regime_name,
                "seed": seed,
                "summary": summarize_pool(pool),
            }
            save_pool(pool, fname, meta)
            master_manifest["files"][fname] = meta

    # A2. static eval pools
    if DEFAULT_CONFIG["save_static_eval_pools"]:
        print("\n=== A2. static eval pools ===")
        for idx, regime_name in enumerate(regime_order):
            seed = base_seed + DEFAULT_CONFIG["eval_seed_offset"] + idx * 100_000
            pool = generate_regime_pool(regime_name, int(args.static_eval_episodes), T, seed)
            fname = static_eval_name(regime_name, T)
            meta = {
                "type": "static_eval",
                "regime": regime_name,
                "seed": seed,
                "summary": summarize_pool(pool),
            }
            save_pool(pool, fname, meta)
            master_manifest["files"][fname] = meta

    # B1. mixed full master pool
    if DEFAULT_CONFIG["save_mixed_full_master_pool"]:
        print("\n=== B1. mixed full master pool ===")
        seed = base_seed + DEFAULT_CONFIG["train_seed_offset"] + 500_000
        pool = generate_mixed_full_master_pool(regime_order, int(args.mixed_full_master_episodes), T, seed)
        fname = mixed_full_master_name(regime_order, T)
        meta = {
            "type": "mixed_full_master",
            "regime_order": regime_order,
            "seed": seed,
            "summary": summarize_pool(pool),
        }
        save_pool(pool, fname, meta)
        master_manifest["files"][fname] = meta

    # B2. mixed equal master pool
    if DEFAULT_CONFIG["save_mixed_equal_master_pool"]:
        print("\n=== B2. mixed equal master pool ===")
        seed = base_seed + DEFAULT_CONFIG["train_seed_offset"] + 700_000
        pool = generate_mixed_equal_pool(regime_order, int(args.mixed_eq_master_episodes), T, seed)
        fname = mixed_equal_master_name(regime_order, T)
        meta = {
            "type": "mixed_equal_master",
            "regime_order": regime_order,
            "seed": seed,
            "summary": summarize_pool(pool),
        }
        save_pool(pool, fname, meta)
        master_manifest["files"][fname] = meta

    # B3. dedicated mixed equal val pool
    if DEFAULT_CONFIG["save_mixed_equal_val_pool"]:
        print("\n=== B3. mixed equal val pool ===")
        seed = base_seed + DEFAULT_CONFIG["val_seed_offset"] + 700_000
        pool = generate_mixed_equal_pool(regime_order, int(args.mixed_eq_val_episodes), T, seed)
        fname = mixed_equal_val_name(regime_order, T)
        meta = {
            "type": "mixed_equal_val",
            "regime_order": regime_order,
            "seed": seed,
            "summary": summarize_pool(pool),
        }
        save_pool(pool, fname, meta)
        master_manifest["files"][fname] = meta

    # C1. switching master pools
    if DEFAULT_CONFIG["save_switching_master_pools"]:
        print("\n=== C1. switching master pools ===")
        for idx, switching_name in enumerate(SWITCHING_SPECS.keys()):
            seed = base_seed + DEFAULT_CONFIG["train_seed_offset"] + 900_000 + idx * 100_000
            pool = generate_switching_pool(switching_name, int(args.switching_master_episodes), T, seed)
            fname = switching_master_name(switching_name, T)
            meta = {
                "type": "switching_master",
                "switching_spec": SWITCHING_SPECS[switching_name],
                "seed": seed,
                "summary": summarize_pool(pool),
            }
            save_pool(pool, fname, meta)
            master_manifest["files"][fname] = meta

    # C2. switching eval pools
    if DEFAULT_CONFIG["save_switching_eval_pools"]:
        print("\n=== C2. switching eval pools ===")
        for idx, switching_name in enumerate(SWITCHING_SPECS.keys()):
            seed = base_seed + DEFAULT_CONFIG["eval_seed_offset"] + 900_000 + idx * 100_000
            pool = generate_switching_pool(switching_name, int(args.switching_eval_episodes), T, seed)
            fname = switching_eval_name(switching_name, T)
            meta = {
                "type": "switching_eval",
                "switching_spec": SWITCHING_SPECS[switching_name],
                "seed": seed,
                "summary": summarize_pool(pool),
            }
            save_pool(pool, fname, meta)
            master_manifest["files"][fname] = meta

    manifest_path = DATA_REPORT_DIR / "pool_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(master_manifest, f, ensure_ascii=False, indent=2)

    print("\n=== Done ===")
    print(f"pools dir   : {DATA_POOL_DIR}")
    print(f"reports dir : {DATA_REPORT_DIR}")
    print(f"manifest    : {manifest_path}")
    print("\n建议你在 agent 里用这些固定文件名：")
    print(f"  train_pool_file = {mixed_equal_master_name(regime_order, T)}")
    print(f"  val_pool_file   = {mixed_equal_val_name(regime_order, T)}")
    print("  test_pool_files = {")
    for regime_name in regime_order:
        print(f"    '{regime_name}': '{static_eval_name(regime_name, T)}',")
    for switching_name in SWITCHING_SPECS.keys():
        print(f"    'SW': '{switching_eval_name(switching_name, T)}',")
    print("  }")


if __name__ == "__main__":
    main()
