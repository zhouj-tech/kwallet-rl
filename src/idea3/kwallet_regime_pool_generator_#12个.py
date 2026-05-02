
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
DATA_POOL_DIR = PROJECT_ROOT / "data" / "pools" / "ideaextra"


# =========================================================
# 设计说明
# ---------------------------------------------------------
# 1) 12 个 regime：
#       US, TLS, LNS, TLNS, TPLS, PLS,
#       UB, TLB, LNB, TLNB, TPLB, PLB
# 2) 所有 regime 的最终目标均值都尽量校准到 50
#    注意：bursty 组不是先校准 raw family 再乘 burst，
#    而是对 “raw + burst + clipping + rounding” 的完整机制校准
# 3) train / eval / val 用统一 seed builder 隔离
# 4) 每个 pool 都额外输出一个 *_summary.json，方便解释
# =========================================================


# =========================================================
# Regime 定义
# =========================================================
@dataclass(frozen=True)
class RegimeSpec:
    key: str
    label: str
    dist_family: str
    bursty: bool
    note: str

    target_mean: float = 50.0
    max_tx: float = 200.0
    min_tx: float = 1.0

    # uniform
    uniform_half_width: float = 30.0

    # truncated light (truncated normal)
    trunc_light_std: float = 15.0
    trunc_light_low: float = 1.0
    trunc_light_high: float = 120.0

    # lognormal / truncated lognormal
    log_sigma: float = 0.5
    trunc_lognormal_max: float = 120.0

    # power law / truncated power law
    power_alpha: float = 2.5
    power_xmin: float = 5.0
    trunc_powerlaw_max: float = 120.0

    # burst settings
    burst_start_prob: float = 0.0
    burst_length_mean: float = 0.0
    burst_size_multiplier: float = 1.0


# =========================================================
# 12 个 regime：短 key + 可读 label
# =========================================================
REGIME_SPECS: Dict[str, RegimeSpec] = {
    "US": RegimeSpec(
        key="US",
        label="U-S",
        dist_family="uniform",
        bursty=False,
        note="bounded baseline; smooth uniform transactions",
        target_mean=50.0,
        max_tx=100.0,
        uniform_half_width=30.0,
    ),
    "TLS": RegimeSpec(
        key="TLS",
        label="TL-S",
        dist_family="trunc_light",
        bursty=False,
        note="light-tail smooth regime; truncated normal around target mean",
        target_mean=50.0,
        max_tx=120.0,
        trunc_light_std=15.0,
        trunc_light_low=1.0,
        trunc_light_high=120.0,
    ),
    "LNS": RegimeSpec(
        key="LNS",
        label="LN-S",
        dist_family="lognormal",
        bursty=False,
        note="smooth lognormal; realistic right-skew heavy tail",
        target_mean=50.0,
        max_tx=180.0,
        log_sigma=0.65,
    ),
    "TLNS": RegimeSpec(
        key="TLNS",
        label="TLN-S",
        dist_family="trunc_lognormal",
        bursty=False,
        note="smooth truncated lognormal; controlled heavy tail",
        target_mean=50.0,
        max_tx=120.0,
        log_sigma=0.55,
        trunc_lognormal_max=120.0,
    ),
    "TPLS": RegimeSpec(
        key="TPLS",
        label="TPL-S",
        dist_family="trunc_powerlaw",
        bursty=False,
        note="smooth truncated power law; strong tail with cap",
        target_mean=50.0,
        max_tx=120.0,
        power_alpha=2.3,
        power_xmin=8.0,
        trunc_powerlaw_max=120.0,
    ),
    "PLS": RegimeSpec(
        key="PLS",
        label="PL-S",
        dist_family="powerlaw",
        bursty=False,
        note="smooth power law; extreme heavy-tail stress test",
        target_mean=50.0,
        max_tx=250.0,
        power_alpha=2.2,
        power_xmin=8.0,
    ),

    # B 组统一 burst 参数，避免和 tail 一起变化
    "UB": RegimeSpec(
        key="UB",
        label="U-B",
        dist_family="uniform",
        bursty=True,
        note="bursty uniform; isolates time-structure effect without heavy tail",
        target_mean=50.0,
        max_tx=100.0,
        uniform_half_width=30.0,
        burst_start_prob=0.035,
        burst_length_mean=6.0,
        burst_size_multiplier=1.4,
    ),
    "TLB": RegimeSpec(
        key="TLB",
        label="TL-B",
        dist_family="trunc_light",
        bursty=True,
        note="bursty truncated light-tail; mild-value complexity plus burst",
        target_mean=50.0,
        max_tx=120.0,
        trunc_light_std=15.0,
        trunc_light_low=1.0,
        trunc_light_high=120.0,
        burst_start_prob=0.035,
        burst_length_mean=6.0,
        burst_size_multiplier=1.4,
    ),
    "LNB": RegimeSpec(
        key="LNB",
        label="LN-B",
        dist_family="lognormal",
        bursty=True,
        note="bursty lognormal; realistic complex regime with skew and burst",
        target_mean=50.0,
        max_tx=180.0,
        log_sigma=0.70,
        burst_start_prob=0.035,
        burst_length_mean=6.0,
        burst_size_multiplier=1.4,
    ),
    "TLNB": RegimeSpec(
        key="TLNB",
        label="TLN-B",
        dist_family="trunc_lognormal",
        bursty=True,
        note="bursty truncated lognormal; stable realistic heavy-tail burst regime",
        target_mean=50.0,
        max_tx=120.0,
        log_sigma=0.58,
        trunc_lognormal_max=120.0,
        burst_start_prob=0.035,
        burst_length_mean=6.0,
        burst_size_multiplier=1.4,
    ),
    "TPLB": RegimeSpec(
        key="TPLB",
        label="TPL-B",
        dist_family="trunc_powerlaw",
        bursty=True,
        note="bursty truncated power law; high-pressure heavy tail with cap",
        target_mean=50.0,
        max_tx=120.0,
        power_alpha=2.2,
        power_xmin=8.0,
        trunc_powerlaw_max=120.0,
        burst_start_prob=0.035,
        burst_length_mean=6.0,
        burst_size_multiplier=1.4,
    ),
    "PLB": RegimeSpec(
        key="PLB",
        label="PL-B",
        dist_family="powerlaw",
        bursty=True,
        note="bursty power law; worst-case stress regime",
        target_mean=50.0,
        max_tx=250.0,
        power_alpha=2.1,
        power_xmin=8.0,
        burst_start_prob=0.035,
        burst_length_mean=6.0,
        burst_size_multiplier=1.4,
    ),
}

REGIME_ORDER = [
    "US", "TLS", "LNS", "TLNS", "TPLS", "PLS",
    "UB", "TLB", "LNB", "TLNB", "TPLB", "PLB",
]

SWITCHING_SPECS = {
    "US_PLB_LNS_20_50_30": [("US", 0.20), ("PLB", 0.50), ("LNS", 0.30)],
    "TLS_TPLB_TLNS_30_40_30": [("TLS", 0.30), ("TPLB", 0.40), ("TLNS", 0.30)],
}


# =========================================================
# 全局配置
# =========================================================
DEFAULT_CONFIG = {
    "base_seed": 532,

    # split 级别大区间：彻底隔离
    "train_seed_offset": 0,
    "eval_seed_offset": 100_000_000,
    "val_seed_offset": 200_000_000,

    "save_reports": True,
    "episode_length": 1000,

    "save_static_master_pools": True,
    "save_static_eval_pools": True,
    "save_mixed_full_master_pool": True,
    "save_mixed_equal_master_pool": True,
    "save_mixed_equal_val_pool": True,
    "save_switching_master_pools": True,
    "save_switching_eval_pools": True,

    "static_master_episodes": 5000,
    "static_eval_episodes": 200,
    "mixed_full_master_episodes": 5000,
    "mixed_equal_master_episodes": 5000,
    "mixed_equal_val_episodes": 300,
    "switching_master_episodes": 5000,
    "switching_eval_episodes": 200,

    # 完整机制校准用样本量
    "calibration_sample_size": 200000,
}


# =========================================================
# 基础工具
# =========================================================
def ensure_dirs() -> None:
    DATA_POOL_DIR.mkdir(parents=True, exist_ok=True)


def build_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def build_pool_seed(
    *,
    base_seed: int,
    split: str,       # "train" / "eval" / "val"
    pool_type: str,   # "static" / "mixed_full" / "mixed_equal" / "switching"
    item_idx: int = 0,
) -> int:
    split_offset_map = {
        "train": DEFAULT_CONFIG["train_seed_offset"],
        "eval": DEFAULT_CONFIG["eval_seed_offset"],
        "val": DEFAULT_CONFIG["val_seed_offset"],
    }

    pool_type_offset_map = {
        "static": 0,
        "mixed_full": 10_000_000,
        "mixed_equal": 20_000_000,
        "switching": 30_000_000,
    }

    item_stride = 1_000_000

    return (
        base_seed
        + split_offset_map[split]
        + pool_type_offset_map[pool_type]
        + item_idx * item_stride
    )


def summarize_pool(pool: np.ndarray) -> Dict[str, float]:
    flat = pool.reshape(-1).astype(np.float64)
    mean = float(np.mean(flat))
    std = float(np.std(flat))
    p95 = float(np.percentile(flat, 95))
    p99 = float(np.percentile(flat, 99))
    mx = float(np.max(flat))
    return {
        "episodes": int(pool.shape[0]),
        "steps_per_episode": int(pool.shape[1]),
        "mean": mean,
        "std": std,
        "cv": float(std / mean) if mean > 0 else 0.0,
        "min": float(np.min(flat)),
        "p50": float(np.percentile(flat, 50)),
        "p90": float(np.percentile(flat, 90)),
        "p95": p95,
        "p99": p99,
        "p95_over_mean": float(p95 / mean) if mean > 0 else 0.0,
        "p99_over_mean": float(p99 / mean) if mean > 0 else 0.0,
        "max": mx,
        "max_over_mean": float(mx / mean) if mean > 0 else 0.0,
    }


def summarize_true_burst_mask(burst_mask_pool: np.ndarray) -> Dict[str, float]:
    flat = burst_mask_pool.reshape(-1)
    total_steps = int(flat.shape[0])
    steps_in_burst = int(np.sum(flat))

    lengths = []
    run = 0
    for flag in flat:
        if flag:
            run += 1
        else:
            if run > 0:
                lengths.append(run)
                run = 0
    if run > 0:
        lengths.append(run)

    return {
        "burst_fraction": float(steps_in_burst / total_steps) if total_steps > 0 else 0.0,
        "avg_burst_length": float(np.mean(lengths)) if lengths else 0.0,
        "max_burst_length": int(np.max(lengths)) if lengths else 0,
        "num_bursts": int(len(lengths)),
        "steps_in_burst": steps_in_burst,
        "steps_total": total_steps,
    }


def save_pool_and_summary(array: np.ndarray, file_name: str, summary: Dict) -> Path:
    ensure_dirs()
    pool_path = DATA_POOL_DIR / file_name
    np.save(pool_path, array.astype(np.int32))

    summary_path = DATA_POOL_DIR / f"{pool_path.stem}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[saved] {pool_path}")
    print(f"[saved] {summary_path}")
    return pool_path


# =========================================================
# 各分布族采样
# =========================================================
def _sample_raw_uniform(spec: RegimeSpec, rng: np.random.Generator, size: int) -> np.ndarray:
    low = max(spec.min_tx, spec.target_mean - spec.uniform_half_width)
    high = min(spec.max_tx, spec.target_mean + spec.uniform_half_width)
    return rng.uniform(low, high, size=size)


def _sample_raw_trunc_light(spec: RegimeSpec, rng: np.random.Generator, size: int) -> np.ndarray:
    out = []
    needed = size
    while needed > 0:
        cand = rng.normal(loc=spec.target_mean, scale=spec.trunc_light_std, size=max(needed * 2, 1000))
        cand = cand[(cand >= spec.trunc_light_low) & (cand <= spec.trunc_light_high)]
        if cand.size > 0:
            take = min(needed, cand.size)
            out.append(cand[:take])
            needed -= take
    return np.concatenate(out, axis=0)


def _sample_raw_lognormal(spec: RegimeSpec, rng: np.random.Generator, size: int) -> np.ndarray:
    return rng.lognormal(mean=0.0, sigma=spec.log_sigma, size=size)


def _sample_raw_trunc_lognormal(spec: RegimeSpec, rng: np.random.Generator, size: int) -> np.ndarray:
    out = []
    needed = size
    while needed > 0:
        cand = _sample_raw_lognormal(spec, rng, max(needed * 3, 3000))
        cand = cand[cand <= spec.trunc_lognormal_max]
        if cand.size > 0:
            take = min(needed, cand.size)
            out.append(cand[:take])
            needed -= take
    return np.concatenate(out, axis=0)


def _sample_raw_powerlaw(spec: RegimeSpec, rng: np.random.Generator, size: int) -> np.ndarray:
    y = rng.pareto(spec.power_alpha, size=size)
    x = spec.power_xmin * (1.0 + y)
    return x


def _sample_raw_trunc_powerlaw(spec: RegimeSpec, rng: np.random.Generator, size: int) -> np.ndarray:
    out = []
    needed = size
    while needed > 0:
        cand = _sample_raw_powerlaw(spec, rng, max(needed * 3, 3000))
        cand = cand[cand <= spec.trunc_powerlaw_max]
        if cand.size > 0:
            take = min(needed, cand.size)
            out.append(cand[:take])
            needed -= take
    return np.concatenate(out, axis=0)


def sample_raw_by_family(spec: RegimeSpec, rng: np.random.Generator, size: int) -> np.ndarray:
    family = spec.dist_family
    if family == "uniform":
        return _sample_raw_uniform(spec, rng, size)
    elif family == "trunc_light":
        return _sample_raw_trunc_light(spec, rng, size)
    elif family == "lognormal":
        return _sample_raw_lognormal(spec, rng, size)
    elif family == "trunc_lognormal":
        return _sample_raw_trunc_lognormal(spec, rng, size)
    elif family == "powerlaw":
        return _sample_raw_powerlaw(spec, rng, size)
    elif family == "trunc_powerlaw":
        return _sample_raw_trunc_powerlaw(spec, rng, size)
    else:
        raise ValueError(f"Unknown dist_family: {family}")


# =========================================================
# 关键：完整机制校准
# =========================================================
_CALIBRATION_CACHE: Dict[str, float] = {}


def _sample_one_tx_with_scale(
    spec: RegimeSpec,
    rng: np.random.Generator,
    in_burst: bool,
    base_scale: float,
) -> int:
    raw = float(sample_raw_by_family(spec, rng, 1)[0])

    tx = raw * base_scale
    if in_burst:
        tx *= spec.burst_size_multiplier

    tx = max(spec.min_tx, min(float(tx), spec.max_tx))
    return int(round(tx))


def _simulate_episode_with_scale_and_mask(
    spec: RegimeSpec,
    episode_length: int,
    rng: np.random.Generator,
    base_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    txs = np.zeros(episode_length, dtype=np.int32)
    burst_mask = np.zeros(episode_length, dtype=bool)

    burst_remaining = 0

    for t in range(episode_length):
        if burst_remaining > 0:
            in_burst = True
            burst_remaining -= 1
        else:
            if spec.bursty and spec.burst_start_prob > 0 and rng.random() < spec.burst_start_prob:
                burst_remaining = max(1, int(rng.poisson(spec.burst_length_mean)))
                in_burst = True
                burst_remaining -= 1
            else:
                in_burst = False

        burst_mask[t] = in_burst
        txs[t] = _sample_one_tx_with_scale(
            spec=spec,
            rng=rng,
            in_burst=in_burst,
            base_scale=base_scale,
        )

    return txs, burst_mask


def get_effective_mean_scale(
    spec: RegimeSpec,
    calibration_sample_size: int,
    seed: int = 1234567,
    n_iter: int = 3,
) -> float:
    """
    不只对 raw family 校准，
    而是对“raw family + burst + clipping + rounding”的完整机制校准。
    """
    cache_key = f"{spec.key}|{calibration_sample_size}|effective"
    if cache_key in _CALIBRATION_CACHE:
        return _CALIBRATION_CACHE[cache_key]

    # 先按 raw family 粗校准
    rng0 = build_rng(seed)
    raw = sample_raw_by_family(spec, rng0, calibration_sample_size)
    raw_mean = float(np.mean(raw))
    scale = spec.target_mean / raw_mean if raw_mean > 0 else 1.0

    # 再对完整机制做多轮修正
    steps_per_episode = min(1000, calibration_sample_size)
    num_episodes = max(20, calibration_sample_size // steps_per_episode)

    for i in range(n_iter):
        sim_vals = []

        for ep in range(num_episodes):
            rng = build_rng(seed + 10_000 * (i + 1) + ep)
            ep_vals, _ = _simulate_episode_with_scale_and_mask(
                spec=spec,
                episode_length=steps_per_episode,
                rng=rng,
                base_scale=scale,
            )
            sim_vals.append(ep_vals)

        sim = np.concatenate(sim_vals).astype(np.float64)
        realized_mean = float(np.mean(sim))
        if realized_mean > 0:
            scale *= spec.target_mean / realized_mean

    _CALIBRATION_CACHE[cache_key] = scale
    return scale


# =========================================================
# episode / pool 生成
# =========================================================
def generate_regime_episode(
    spec: RegimeSpec,
    episode_length: int,
    rng: np.random.Generator,
    calibration_sample_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    scale = get_effective_mean_scale(
        spec=spec,
        calibration_sample_size=calibration_sample_size,
    )
    return _simulate_episode_with_scale_and_mask(
        spec=spec,
        episode_length=episode_length,
        rng=rng,
        base_scale=scale,
    )


def generate_regime_pool(
    regime_name: str,
    num_episodes: int,
    episode_length: int,
    base_seed: int,
    calibration_sample_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    spec = REGIME_SPECS[regime_name]
    pool = np.zeros((num_episodes, episode_length), dtype=np.int32)
    burst_mask_pool = np.zeros((num_episodes, episode_length), dtype=bool)

    for ep in range(num_episodes):
        rng = build_rng(base_seed + ep)
        txs, burst_mask = generate_regime_episode(
            spec=spec,
            episode_length=episode_length,
            rng=rng,
            calibration_sample_size=calibration_sample_size,
        )
        pool[ep] = txs
        burst_mask_pool[ep] = burst_mask

    return pool, burst_mask_pool


def generate_mixed_full_master_pool(
    regime_order: List[str],
    num_episodes: int,
    episode_length: int,
    base_seed: int,
    calibration_sample_size: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    rng = build_rng(base_seed)
    chosen = rng.choice(regime_order, size=num_episodes, replace=True)

    pool = np.zeros((num_episodes, episode_length), dtype=np.int32)
    burst_mask_pool = np.zeros((num_episodes, episode_length), dtype=bool)
    counts = {r: 0 for r in regime_order}

    for ep, regime_name in enumerate(chosen):
        regime_name = str(regime_name)
        counts[regime_name] += 1
        ep_rng = build_rng(base_seed + 10_000 + ep)
        txs, burst_mask = generate_regime_episode(
            REGIME_SPECS[regime_name],
            episode_length,
            ep_rng,
            calibration_sample_size,
        )
        pool[ep] = txs
        burst_mask_pool[ep] = burst_mask

    return pool, burst_mask_pool, counts


def generate_mixed_equal_pool(
    regime_order: List[str],
    total_episodes: int,
    episode_length: int,
    base_seed: int,
    calibration_sample_size: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    pieces = []
    mask_pieces = []
    counts = {}

    n_regimes = len(regime_order)
    base_each = total_episodes // n_regimes
    remainder = total_episodes % n_regimes

    cursor_seed = base_seed
    for i, regime_name in enumerate(regime_order):
        count = base_each + (1 if i < remainder else 0)
        counts[regime_name] = count
        part, part_mask = generate_regime_pool(
            regime_name,
            count,
            episode_length,
            cursor_seed,
            calibration_sample_size,
        )
        pieces.append(part)
        mask_pieces.append(part_mask)
        cursor_seed += 1_000_000

    pool = np.concatenate(pieces, axis=0)
    burst_mask_pool = np.concatenate(mask_pieces, axis=0)

    shuffle_rng = build_rng(base_seed + 999_999)
    perm = shuffle_rng.permutation(pool.shape[0])
    pool = pool[perm]
    burst_mask_pool = burst_mask_pool[perm]

    return pool, burst_mask_pool, counts


def _segment_lengths(episode_length: int, spec: List[Tuple[str, float]]) -> List[int]:
    raw = [episode_length * ratio for _, ratio in spec]
    lengths = [int(x) for x in raw]
    gap = episode_length - sum(lengths)
    lengths[-1] += gap
    return lengths


def generate_switching_episode(
    switching_spec: List[Tuple[str, float]],
    episode_length: int,
    rng_seed: int,
    calibration_sample_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    lengths = _segment_lengths(episode_length, switching_spec)

    out_parts = []
    mask_parts = []
    cursor = rng_seed

    for (regime_name, _), seg_len in zip(switching_spec, lengths):
        seg_rng = build_rng(cursor)
        txs, mask = generate_regime_episode(
            REGIME_SPECS[regime_name],
            seg_len,
            seg_rng,
            calibration_sample_size,
        )
        out_parts.append(txs)
        mask_parts.append(mask)
        cursor += 10_000

    return np.concatenate(out_parts, axis=0), np.concatenate(mask_parts, axis=0)


def generate_switching_pool(
    switching_name: str,
    num_episodes: int,
    episode_length: int,
    base_seed: int,
    calibration_sample_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    spec = SWITCHING_SPECS[switching_name]
    pool = np.zeros((num_episodes, episode_length), dtype=np.int32)
    burst_mask_pool = np.zeros((num_episodes, episode_length), dtype=bool)

    for ep in range(num_episodes):
        txs, mask = generate_switching_episode(
            spec,
            episode_length,
            base_seed + ep,
            calibration_sample_size,
        )
        pool[ep] = txs
        burst_mask_pool[ep] = mask

    return pool, burst_mask_pool


# =========================================================
# 文件名
# =========================================================
def static_master_name(regime_name: str, T: int) -> str:
    return f"{regime_name}_static_master_T{T}.npy"


def static_eval_name(regime_name: str, T: int) -> str:
    return f"{regime_name}_static_eval_T{T}.npy"


def mixed_full_master_name(regime_order: List[str], T: int) -> str:
    joined = "_".join(regime_order)
    return f"MIX12_FULL_{joined}_master_T{T}.npy"


def mixed_equal_master_name(regime_order: List[str], T: int) -> str:
    joined = "_".join(regime_order)
    return f"MIX12_EQ_{joined}_master_T{T}.npy"


def mixed_equal_val_name(regime_order: List[str], T: int) -> str:
    joined = "_".join(regime_order)
    return f"MIX12_EQ_{joined}_val_T{T}.npy"


def switching_master_name(switching_name: str, T: int) -> str:
    return f"{switching_name}_switch_master_T{T}.npy"


def switching_eval_name(switching_name: str, T: int) -> str:
    return f"{switching_name}_switch_eval_T{T}.npy"


# =========================================================
# summary 构造
# =========================================================
def build_common_settings_summary() -> Dict:
    return {
        "base_target_mean_goal": 50.0,
        "bursty_groups_share_same_burst_parameters": {
            "burst_start_prob": 0.035,
            "burst_length_mean": 6.0,
            "burst_size_multiplier": 1.4,
        },
        "calibration_sample_size": DEFAULT_CONFIG["calibration_sample_size"],
        "note": (
            "The generator calibrates the full mechanism rather than only raw family samples. "
            "That means the final realized mean is calibrated after raw-family sampling, "
            "burst multiplication, clipping to [min_tx, max_tx], and integer rounding."
        ),
    }


def build_static_summary(
    *,
    file_name: str,
    pool_type: str,
    seed: int,
    num_episodes: int,
    episode_length: int,
    regime_name: str,
    pool: np.ndarray,
    burst_mask_pool: np.ndarray,
) -> Dict:
    spec = REGIME_SPECS[regime_name]
    if spec.bursty:
        burst_sentence = (
            f"Bursty = True with burst_start_prob = {spec.burst_start_prob}, "
            f"burst_length_mean = {spec.burst_length_mean}, "
            f"burst_size_multiplier = {spec.burst_size_multiplier}."
        )
    else:
        burst_sentence = "Bursty = False; transactions are generated without explicit burst segments."

    return {
        "file_name": file_name,
        "pool_type": pool_type,
        "output_dir": str(DATA_POOL_DIR.resolve()),
        "seed": seed,
        "num_episodes": int(num_episodes),
        "episode_length": int(episode_length),
        "generation_summary": (
            f"This pool was generated from regime {spec.key} ({spec.label}). "
            f"Distribution family = {spec.dist_family}. "
            f"Target mean = {spec.target_mean}. "
            f"{burst_sentence} "
            f"Note: {spec.note}"
        ),
        "realized_stats": summarize_pool(pool),
        "true_burst_stats": summarize_true_burst_mask(burst_mask_pool),
        "common_settings": build_common_settings_summary(),
        "regime": {
            "key": spec.key,
            "label": spec.label,
            "spec": asdict(spec),
        },
    }


def build_mixed_summary(
    *,
    file_name: str,
    pool_type: str,
    seed: int,
    num_episodes: int,
    episode_length: int,
    regime_order: List[str],
    counts: Dict[str, int],
    pool: np.ndarray,
    burst_mask_pool: np.ndarray,
    note: str,
) -> Dict:
    return {
        "file_name": file_name,
        "pool_type": pool_type,
        "output_dir": str(DATA_POOL_DIR.resolve()),
        "seed": seed,
        "num_episodes": int(num_episodes),
        "episode_length": int(episode_length),
        "generation_summary": note,
        "realized_stats": summarize_pool(pool),
        "true_burst_stats": summarize_true_burst_mask(burst_mask_pool),
        "common_settings": build_common_settings_summary(),
        "regime_order": regime_order,
        "regime_counts": counts,
    }


def build_switching_summary(
    *,
    file_name: str,
    pool_type: str,
    seed: int,
    num_episodes: int,
    episode_length: int,
    switching_name: str,
    pool: np.ndarray,
    burst_mask_pool: np.ndarray,
) -> Dict:
    return {
        "file_name": file_name,
        "pool_type": pool_type,
        "output_dir": str(DATA_POOL_DIR.resolve()),
        "seed": seed,
        "num_episodes": int(num_episodes),
        "episode_length": int(episode_length),
        "generation_summary": (
            f"This is a switching pool generated from switching spec {switching_name}. "
            f"The sequence/weights are {SWITCHING_SPECS[switching_name]}."
        ),
        "realized_stats": summarize_pool(pool),
        "true_burst_stats": summarize_true_burst_mask(burst_mask_pool),
        "common_settings": build_common_settings_summary(),
        "switching_spec": SWITCHING_SPECS[switching_name],
    }


# =========================================================
# 命令行参数
# =========================================================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="K-Wallet 12-regime pool generator (ideaextra version)")
    parser.add_argument("--base-seed", type=int, default=DEFAULT_CONFIG["base_seed"])
    parser.add_argument("--episode-length", type=int, default=DEFAULT_CONFIG["episode_length"])
    parser.add_argument("--static-master-episodes", type=int, default=DEFAULT_CONFIG["static_master_episodes"])
    parser.add_argument("--static-eval-episodes", type=int, default=DEFAULT_CONFIG["static_eval_episodes"])
    parser.add_argument("--mixed-full-master-episodes", type=int, default=DEFAULT_CONFIG["mixed_full_master_episodes"])
    parser.add_argument("--mixed-eq-master-episodes", type=int, default=DEFAULT_CONFIG["mixed_equal_master_episodes"])
    parser.add_argument("--mixed-eq-val-episodes", type=int, default=DEFAULT_CONFIG["mixed_equal_val_episodes"])
    parser.add_argument("--switching-master-episodes", type=int, default=DEFAULT_CONFIG["switching_master_episodes"])
    parser.add_argument("--switching-eval-episodes", type=int, default=DEFAULT_CONFIG["switching_eval_episodes"])
    parser.add_argument("--calibration-sample-size", type=int, default=DEFAULT_CONFIG["calibration_sample_size"])
    parser.add_argument(
        "--regimes",
        nargs="*",
        default=None,
        help="Optional subset of regimes. Example: --regimes US LNS PLB"
    )
    return parser


# =========================================================
# 主程序
# =========================================================
def main() -> None:
    args = build_arg_parser().parse_args()
    ensure_dirs()

    if args.regimes is None or len(args.regimes) == 0:
        regime_order = list(REGIME_ORDER)
    else:
        unknown = [r for r in args.regimes if r not in REGIME_SPECS]
        if unknown:
            raise ValueError(f"Unknown regimes in --regimes: {unknown}")
        regime_order = list(args.regimes)

    T = int(args.episode_length)
    base_seed = int(args.base_seed)
    calibration_sample_size = int(args.calibration_sample_size)

    # 让 summary 中显示当前运行参数
    DEFAULT_CONFIG["calibration_sample_size"] = calibration_sample_size

    manifest: Dict[str, Dict] = {
        "generator_name": THIS_FILE.name,
        "output_dir": str(DATA_POOL_DIR.resolve()),
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
            "calibration_sample_size": calibration_sample_size,
            "regime_order": regime_order,
        },
        "regimes": {name: asdict(spec) for name, spec in REGIME_SPECS.items()},
        "switching_specs": SWITCHING_SPECS,
        "files": {},
    }

    # -----------------------------------------------------
    # A1. static master pools
    # -----------------------------------------------------
    if DEFAULT_CONFIG["save_static_master_pools"]:
        print("\n=== A1. static master pools ===")
        for idx, regime_name in enumerate(regime_order):
            seed = build_pool_seed(
                base_seed=base_seed,
                split="train",
                pool_type="static",
                item_idx=idx,
            )

            pool, burst_mask_pool = generate_regime_pool(
                regime_name,
                int(args.static_master_episodes),
                T,
                seed,
                calibration_sample_size,
            )

            fname = static_master_name(regime_name, T)
            summary = build_static_summary(
                file_name=fname,
                pool_type="static_master",
                seed=seed,
                num_episodes=int(args.static_master_episodes),
                episode_length=T,
                regime_name=regime_name,
                pool=pool,
                burst_mask_pool=burst_mask_pool,
            )
            save_pool_and_summary(pool, fname, summary)
            manifest["files"][fname] = summary

    # -----------------------------------------------------
    # A2. static eval pools
    # -----------------------------------------------------
    if DEFAULT_CONFIG["save_static_eval_pools"]:
        print("\n=== A2. static eval pools ===")
        for idx, regime_name in enumerate(regime_order):
            seed = build_pool_seed(
                base_seed=base_seed,
                split="eval",
                pool_type="static",
                item_idx=idx,
            )

            pool, burst_mask_pool = generate_regime_pool(
                regime_name,
                int(args.static_eval_episodes),
                T,
                seed,
                calibration_sample_size,
            )

            fname = static_eval_name(regime_name, T)
            summary = build_static_summary(
                file_name=fname,
                pool_type="static_eval",
                seed=seed,
                num_episodes=int(args.static_eval_episodes),
                episode_length=T,
                regime_name=regime_name,
                pool=pool,
                burst_mask_pool=burst_mask_pool,
            )
            save_pool_and_summary(pool, fname, summary)
            manifest["files"][fname] = summary

    # -----------------------------------------------------
    # B1. mixed full master pool
    # -----------------------------------------------------
    if DEFAULT_CONFIG["save_mixed_full_master_pool"]:
        print("\n=== B1. mixed full master pool ===")
        seed = build_pool_seed(
            base_seed=base_seed,
            split="train",
            pool_type="mixed_full",
            item_idx=0,
        )

        pool, burst_mask_pool, counts = generate_mixed_full_master_pool(
            regime_order,
            int(args.mixed_full_master_episodes),
            T,
            seed,
            calibration_sample_size,
        )

        fname = mixed_full_master_name(regime_order, T)
        summary = build_mixed_summary(
            file_name=fname,
            pool_type="mixed_full_master",
            seed=seed,
            num_episodes=int(args.mixed_full_master_episodes),
            episode_length=T,
            regime_order=regime_order,
            counts=counts,
            pool=pool,
            burst_mask_pool=burst_mask_pool,
            note=(
                "This mixed-full pool was created by randomly choosing a regime for each episode "
                "from the selected regime_order with replacement, then generating that episode "
                "under the chosen regime."
            ),
        )
        save_pool_and_summary(pool, fname, summary)
        manifest["files"][fname] = summary

    # -----------------------------------------------------
    # B2. mixed equal master pool
    # -----------------------------------------------------
    if DEFAULT_CONFIG["save_mixed_equal_master_pool"]:
        print("\n=== B2. mixed equal master pool ===")
        seed = build_pool_seed(
            base_seed=base_seed,
            split="train",
            pool_type="mixed_equal",
            item_idx=0,
        )

        pool, burst_mask_pool, counts = generate_mixed_equal_pool(
            regime_order,
            int(args.mixed_eq_master_episodes),
            T,
            seed,
            calibration_sample_size,
        )

        fname = mixed_equal_master_name(regime_order, T)
        summary = build_mixed_summary(
            file_name=fname,
            pool_type="mixed_equal_master",
            seed=seed,
            num_episodes=int(args.mixed_eq_master_episodes),
            episode_length=T,
            regime_order=regime_order,
            counts=counts,
            pool=pool,
            burst_mask_pool=burst_mask_pool,
            note=(
                "This mixed-equal master pool allocates nearly equal numbers of episodes to each "
                "selected regime, concatenates them, and then shuffles the episode order."
            ),
        )
        save_pool_and_summary(pool, fname, summary)
        manifest["files"][fname] = summary

    # -----------------------------------------------------
    # B3. mixed equal val pool
    # -----------------------------------------------------
    if DEFAULT_CONFIG["save_mixed_equal_val_pool"]:
        print("\n=== B3. mixed equal val pool ===")
        seed = build_pool_seed(
            base_seed=base_seed,
            split="val",
            pool_type="mixed_equal",
            item_idx=0,
        )

        pool, burst_mask_pool, counts = generate_mixed_equal_pool(
            regime_order,
            int(args.mixed_eq_val_episodes),
            T,
            seed,
            calibration_sample_size,
        )

        fname = mixed_equal_val_name(regime_order, T)
        summary = build_mixed_summary(
            file_name=fname,
            pool_type="mixed_equal_val",
            seed=seed,
            num_episodes=int(args.mixed_eq_val_episodes),
            episode_length=T,
            regime_order=regime_order,
            counts=counts,
            pool=pool,
            burst_mask_pool=burst_mask_pool,
            note=(
                "This mixed-equal validation pool uses the same equal-allocation construction as "
                "the mixed-equal master pool, but with a separate validation seed range."
            ),
        )
        save_pool_and_summary(pool, fname, summary)
        manifest["files"][fname] = summary

    # -----------------------------------------------------
    # C1. switching master pools
    # -----------------------------------------------------
    if DEFAULT_CONFIG["save_switching_master_pools"]:
        print("\n=== C1. switching master pools ===")
        for idx, switching_name in enumerate(SWITCHING_SPECS.keys()):
            seed = build_pool_seed(
                base_seed=base_seed,
                split="train",
                pool_type="switching",
                item_idx=idx,
            )

            pool, burst_mask_pool = generate_switching_pool(
                switching_name,
                int(args.switching_master_episodes),
                T,
                seed,
                calibration_sample_size,
            )

            fname = switching_master_name(switching_name, T)
            summary = build_switching_summary(
                file_name=fname,
                pool_type="switching_master",
                seed=seed,
                num_episodes=int(args.switching_master_episodes),
                episode_length=T,
                switching_name=switching_name,
                pool=pool,
                burst_mask_pool=burst_mask_pool,
            )
            save_pool_and_summary(pool, fname, summary)
            manifest["files"][fname] = summary

    # -----------------------------------------------------
    # C2. switching eval pools
    # -----------------------------------------------------
    if DEFAULT_CONFIG["save_switching_eval_pools"]:
        print("\n=== C2. switching eval pools ===")
        for idx, switching_name in enumerate(SWITCHING_SPECS.keys()):
            seed = build_pool_seed(
                base_seed=base_seed,
                split="eval",
                pool_type="switching",
                item_idx=idx,
            )

            pool, burst_mask_pool = generate_switching_pool(
                switching_name,
                int(args.switching_eval_episodes),
                T,
                seed,
                calibration_sample_size,
            )

            fname = switching_eval_name(switching_name, T)
            summary = build_switching_summary(
                file_name=fname,
                pool_type="switching_eval",
                seed=seed,
                num_episodes=int(args.switching_eval_episodes),
                episode_length=T,
                switching_name=switching_name,
                pool=pool,
                burst_mask_pool=burst_mask_pool,
            )
            save_pool_and_summary(pool, fname, summary)
            manifest["files"][fname] = summary

    # 总 manifest
    manifest_path = DATA_POOL_DIR / "ideaextra_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n=== Done ===")
    print(f"pools dir : {DATA_POOL_DIR}")
    print(f"manifest  : {manifest_path}")

    print("\n建议你在 agent 里优先先用：")
    print(f"  train_pool_file = {mixed_equal_master_name(regime_order, T)}")
    print(f"  val_pool_file   = {mixed_equal_val_name(regime_order, T)}")
    print("  static_eval_files = {")
    for regime_name in regime_order:
        print(f"    '{regime_name}': '{static_eval_name(regime_name, T)}',")
    print("  }")


if __name__ == "__main__":
    main()