#4/9
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
DATA_IDEAEXTRA_DIR = PROJECT_ROOT / "data" / "pools" 


# =========================================================
# 设计说明
# ---------------------------------------------------------
# 1) 12 个 regime：6 个 smooth + 6 个 bursty
# 2) 所有 regime 的 target mean 统一设为 50
# 3) 内部 regime key 用短名：US / TLS / LNS / TLNS / TPLS / PLS / UB / ...
# 4) 所有输出都保存到 data/pools/ideaextra
# 5) 每个 .npy pool 都附带一个同名 *_summary.json，包含：
#    - 生成方式说明
#    - regime 参数
#    - seed
#    - realized summary stats
#    - true burst stats（基于真实生成状态，不是 quantile proxy）
# 6) train / val / eval seeds 分离
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

    uniform_half_width: float = 30.0

    trunc_light_std: float = 15.0
    trunc_light_low: float = 1.0
    trunc_light_high: float = 120.0

    log_sigma: float = 0.5
    trunc_lognormal_max: float = 120.0

    power_alpha: float = 2.5
    power_xmin: float = 5.0
    trunc_powerlaw_max: float = 120.0

    burst_start_prob: float = 0.0
    burst_length_mean: float = 0.0
    burst_size_multiplier: float = 1.0


# 说明：
# - B 组先统一 burst 参数，避免 “tail 越重，burst 也越强” 这种混杂设计。
# - 你以后如果真想做第二版，再系统改变 burst 参数即可。
BURST_START_PROB = 0.035
BURST_LENGTH_MEAN = 6.0
BURST_SIZE_MULTIPLIER = 1.40

REGIME_ORDER: List[str] = [
    "US", "TLS", "LNS", "TLNS", "TPLS", "PLS",
    "UB", "TLB", "LNB", "TLNB", "TPLB", "PLB",
]

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
    "UB": RegimeSpec(
        key="UB",
        label="U-B",
        dist_family="uniform",
        bursty=True,
        note="bursty uniform; isolates time-structure effect without heavy tail",
        target_mean=50.0,
        max_tx=100.0,
        uniform_half_width=30.0,
        burst_start_prob=BURST_START_PROB,
        burst_length_mean=BURST_LENGTH_MEAN,
        burst_size_multiplier=BURST_SIZE_MULTIPLIER,
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
        burst_start_prob=BURST_START_PROB,
        burst_length_mean=BURST_LENGTH_MEAN,
        burst_size_multiplier=BURST_SIZE_MULTIPLIER,
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
        burst_start_prob=BURST_START_PROB,
        burst_length_mean=BURST_LENGTH_MEAN,
        burst_size_multiplier=BURST_SIZE_MULTIPLIER,
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
        burst_start_prob=BURST_START_PROB,
        burst_length_mean=BURST_LENGTH_MEAN,
        burst_size_multiplier=BURST_SIZE_MULTIPLIER,
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
        burst_start_prob=BURST_START_PROB,
        burst_length_mean=BURST_LENGTH_MEAN,
        burst_size_multiplier=BURST_SIZE_MULTIPLIER,
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
        burst_start_prob=BURST_START_PROB,
        burst_length_mean=BURST_LENGTH_MEAN,
        burst_size_multiplier=BURST_SIZE_MULTIPLIER,
    ),
}

# Switching pool 用短 key 命名，文件更干净
SWITCHING_SPECS: Dict[str, List[Tuple[str, float]]] = {
    "US_PLB_LNS_20_50_30": [("US", 0.20), ("PLB", 0.50), ("LNS", 0.30)],
    "TLS_TPLB_TLNS_30_40_30": [("TLS", 0.30), ("TPLB", 0.40), ("TLNS", 0.30)],
}

DEFAULT_CONFIG = {
    "base_seed": 532,
    "train_seed_offset": 0,
    "val_seed_offset": 2_000_000,
    "eval_seed_offset": 1_000_000,
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
    "calibration_sample_size": 200000,
}


# =========================================================
# 基础工具
# =========================================================
def ensure_dirs() -> None:
    DATA_IDEAEXTRA_DIR.mkdir(parents=True, exist_ok=True)


def build_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def summarize_pool(pool: np.ndarray) -> Dict[str, float]:
    flat = pool.reshape(-1).astype(np.float64)
    mean = float(np.mean(flat))
    std = float(np.std(flat))
    p95 = float(np.percentile(flat, 95))
    p99 = float(np.percentile(flat, 99))
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
        "max": float(np.max(flat)),
        "max_over_mean": float(np.max(flat) / mean) if mean > 0 else 0.0,
    }


def true_burst_stats(mask: np.ndarray) -> Dict[str, float]:
    flat = mask.reshape(-1).astype(np.int8)
    total_len = int(flat.shape[0])
    total_on = int(flat.sum())

    lengths: List[int] = []
    run = 0
    for x in flat:
        if x == 1:
            run += 1
        else:
            if run > 0:
                lengths.append(run)
                run = 0
    if run > 0:
        lengths.append(run)

    return {
        "burst_fraction": float(total_on / total_len) if total_len > 0 else 0.0,
        "avg_burst_length": float(np.mean(lengths)) if lengths else 0.0,
        "max_burst_length": int(np.max(lengths)) if lengths else 0,
        "num_bursts": int(len(lengths)),
        "steps_in_burst": total_on,
        "steps_total": total_len,
    }


def build_generation_explanation(kind: str, *, regime_key: str | None = None, switching_name: str | None = None) -> str:
    if kind.startswith("static") and regime_key is not None:
        spec = REGIME_SPECS[regime_key]
        base = (
            f"This pool was generated from regime {spec.key} ({spec.label}). "
            f"Distribution family = {spec.dist_family}. "
            f"Target mean = {spec.target_mean}. "
        )
        if spec.bursty:
            burst_part = (
                f"Bursty = True with burst_start_prob = {spec.burst_start_prob}, "
                f"burst_length_mean = {spec.burst_length_mean}, "
                f"burst_size_multiplier = {spec.burst_size_multiplier}. "
            )
        else:
            burst_part = "Bursty = False; transactions are generated without explicit burst segments. "
        return base + burst_part + f"Note: {spec.note}"

    if kind.startswith("mixed_full"):
        return (
            "This pool was generated by sampling regime labels with replacement from all 12 regimes, "
            "then drawing one full episode from the selected regime for each episode. "
            "The resulting composition is random rather than perfectly balanced."
        )

    if kind.startswith("mixed_equal"):
        return (
            "This pool was generated by allocating episodes across all 12 regimes as evenly as possible, "
            "then shuffling the pooled episodes. This is the fairer mixed pool for generalist training/evaluation."
        )

    if kind.startswith("switching") and switching_name is not None:
        spec = SWITCHING_SPECS[switching_name]
        parts = ", ".join([f"{name}:{ratio:.2f}" for name, ratio in spec])
        return (
            f"This switching pool was generated by concatenating regime segments within each episode. "
            f"Segment ratios = {parts}. Each segment uses its own regime-specific generator."
        )

    return "Pool generated by the k-wallet ideaextra regime generator."


def save_pool_with_summary(array: np.ndarray, file_name: str, summary: Dict) -> Path:
    ensure_dirs()
    pool_path = DATA_IDEAEXTRA_DIR / file_name
    np.save(pool_path, array.astype(np.int32))

    summary_path = DATA_IDEAEXTRA_DIR / f"{pool_path.stem}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[saved] {pool_path}")
    print(f"[saved] {summary_path}")
    return pool_path


# =========================================================
# 原始分布采样
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
    raw_mu = 0.0
    return rng.lognormal(mean=raw_mu, sigma=spec.log_sigma, size=size)


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
    if family == "trunc_light":
        return _sample_raw_trunc_light(spec, rng, size)
    if family == "lognormal":
        return _sample_raw_lognormal(spec, rng, size)
    if family == "trunc_lognormal":
        return _sample_raw_trunc_lognormal(spec, rng, size)
    if family == "powerlaw":
        return _sample_raw_powerlaw(spec, rng, size)
    if family == "trunc_powerlaw":
        return _sample_raw_trunc_powerlaw(spec, rng, size)
    raise ValueError(f"Unknown dist_family: {family}")


_CALIBRATION_CACHE: Dict[str, float] = {}


def get_mean_scale(spec: RegimeSpec, calibration_sample_size: int, seed: int = 1234567) -> float:
    cache_key = f"{spec.key}|{calibration_sample_size}"
    if cache_key in _CALIBRATION_CACHE:
        return _CALIBRATION_CACHE[cache_key]

    rng = build_rng(seed)
    raw = sample_raw_by_family(spec, rng, calibration_sample_size)
    raw_mean = float(np.mean(raw))
    scale = spec.target_mean / raw_mean if raw_mean > 0 else 1.0
    _CALIBRATION_CACHE[cache_key] = scale
    return scale


def _sample_one_tx(
    spec: RegimeSpec,
    rng: np.random.Generator,
    in_burst: bool,
    calibration_sample_size: int,
) -> int:
    raw = float(sample_raw_by_family(spec, rng, 1)[0])
    scale = get_mean_scale(spec, calibration_sample_size=calibration_sample_size)

    tx = raw * scale
    if in_burst:
        tx *= spec.burst_size_multiplier

    tx = max(spec.min_tx, min(float(tx), spec.max_tx))
    return int(round(tx))


# =========================================================
# Episode / Pool 生成
# =========================================================
def generate_regime_episode(
    spec: RegimeSpec,
    episode_length: int,
    rng: np.random.Generator,
    calibration_sample_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    out = np.zeros(episode_length, dtype=np.int32)
    burst_mask = np.zeros(episode_length, dtype=np.int8)
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

        out[t] = _sample_one_tx(
            spec=spec,
            rng=rng,
            in_burst=in_burst,
            calibration_sample_size=calibration_sample_size,
        )
        burst_mask[t] = 1 if in_burst else 0

    return out, burst_mask


def generate_regime_pool(
    regime_key: str,
    num_episodes: int,
    episode_length: int,
    base_seed: int,
    calibration_sample_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    spec = REGIME_SPECS[regime_key]
    pool = np.zeros((num_episodes, episode_length), dtype=np.int32)
    burst_mask = np.zeros((num_episodes, episode_length), dtype=np.int8)
    for ep in range(num_episodes):
        rng = build_rng(base_seed + ep)
        ep_values, ep_mask = generate_regime_episode(spec, episode_length, rng, calibration_sample_size)
        pool[ep] = ep_values
        burst_mask[ep] = ep_mask
    return pool, burst_mask


def generate_mixed_full_master_pool(
    regime_order: List[str],
    num_episodes: int,
    episode_length: int,
    base_seed: int,
    calibration_sample_size: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    rng = build_rng(base_seed)
    chosen = rng.choice(regime_order, size=num_episodes, replace=True).tolist()
    pool = np.zeros((num_episodes, episode_length), dtype=np.int32)
    burst_mask = np.zeros((num_episodes, episode_length), dtype=np.int8)

    for ep, regime_key in enumerate(chosen):
        ep_rng = build_rng(base_seed + ep + 10_000)
        ep_values, ep_mask = generate_regime_episode(
            REGIME_SPECS[regime_key],
            episode_length,
            ep_rng,
            calibration_sample_size,
        )
        pool[ep] = ep_values
        burst_mask[ep] = ep_mask

    return pool, burst_mask, chosen


def generate_mixed_equal_pool(
    regime_order: List[str],
    total_episodes: int,
    episode_length: int,
    base_seed: int,
    calibration_sample_size: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    pieces = []
    mask_pieces = []
    counts: Dict[str, int] = {}
    n_regimes = len(regime_order)
    base_each = total_episodes // n_regimes
    remainder = total_episodes % n_regimes

    cursor_seed = base_seed
    for i, regime_key in enumerate(regime_order):
        count = base_each + (1 if i < remainder else 0)
        counts[regime_key] = count
        part, part_mask = generate_regime_pool(
            regime_key,
            count,
            episode_length,
            cursor_seed,
            calibration_sample_size,
        )
        pieces.append(part)
        mask_pieces.append(part_mask)
        cursor_seed += 100_000

    pool = np.concatenate(pieces, axis=0)
    burst_mask = np.concatenate(mask_pieces, axis=0)

    shuffle_rng = build_rng(base_seed + 999_999)
    perm = shuffle_rng.permutation(pool.shape[0])
    pool = pool[perm]
    burst_mask = burst_mask[perm]
    return pool, burst_mask, counts


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

    for (regime_key, _), seg_len in zip(switching_spec, lengths):
        seg_rng = build_rng(cursor)
        part, part_mask = generate_regime_episode(
            REGIME_SPECS[regime_key],
            seg_len,
            seg_rng,
            calibration_sample_size,
        )
        out_parts.append(part)
        mask_parts.append(part_mask)
        cursor += 1_000

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
    burst_mask = np.zeros((num_episodes, episode_length), dtype=np.int8)
    for ep in range(num_episodes):
        ep_values, ep_mask = generate_switching_episode(spec, episode_length, base_seed + ep, calibration_sample_size)
        pool[ep] = ep_values
        burst_mask[ep] = ep_mask
    return pool, burst_mask


# =========================================================
# 文件命名
# =========================================================
def static_master_name(regime_key: str, T: int) -> str:
    return f"{regime_key}_static_master_T{T}.npy"


def static_eval_name(regime_key: str, T: int) -> str:
    return f"{regime_key}_static_eval_T{T}.npy"


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
# CLI
# =========================================================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="K-Wallet ideaextra regime generator")
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
        help="Optional subset of regime keys. Example: --regimes US LNS PLB",
    )
    return parser


# =========================================================
# Summary 组装
# =========================================================
def build_pool_summary(
    *,
    file_name: str,
    pool_type: str,
    seed: int,
    episode_length: int,
    num_episodes: int,
    pool: np.ndarray,
    burst_mask: np.ndarray,
    regime_key: str | None = None,
    regime_order: List[str] | None = None,
    mixed_counts: Dict[str, int] | None = None,
    chosen_regimes_head: List[str] | None = None,
    switching_name: str | None = None,
) -> Dict:
    summary = {
        "file_name": file_name,
        "pool_type": pool_type,
        "output_dir": str(DATA_IDEAEXTRA_DIR),
        "seed": seed,
        "num_episodes": num_episodes,
        "episode_length": episode_length,
        "generation_summary": build_generation_explanation(
            pool_type,
            regime_key=regime_key,
            switching_name=switching_name,
        ),
        "realized_stats": summarize_pool(pool),
        "true_burst_stats": true_burst_stats(burst_mask),
        "common_settings": {
            "base_target_mean_goal": 50.0,
            "bursty_groups_share_same_burst_parameters": {
                "burst_start_prob": BURST_START_PROB,
                "burst_length_mean": BURST_LENGTH_MEAN,
                "burst_size_multiplier": BURST_SIZE_MULTIPLIER,
            },
            "calibration_sample_size": DEFAULT_CONFIG["calibration_sample_size"],
            "note": (
                "raw family samples are first calibrated toward the target mean, then optionally multiplied in burst state, "
                "and finally clipped to [min_tx, max_tx] and rounded to int."
            ),
        },
    }

    if regime_key is not None:
        spec = REGIME_SPECS[regime_key]
        summary["regime"] = {
            "key": spec.key,
            "label": spec.label,
            "spec": asdict(spec),
        }

    if regime_order is not None:
        summary["regime_order"] = regime_order

    if mixed_counts is not None:
        summary["mixed_counts"] = mixed_counts

    if chosen_regimes_head is not None:
        summary["chosen_regimes_preview_first_30"] = chosen_regimes_head[:30]

    if switching_name is not None:
        summary["switching_name"] = switching_name
        summary["switching_spec"] = SWITCHING_SPECS[switching_name]
        summary["segment_lengths"] = _segment_lengths(episode_length, SWITCHING_SPECS[switching_name])

    return summary


# =========================================================
# 主函数
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

    manifest: Dict[str, Dict] = {
        "generator_name": THIS_FILE.name,
        "output_dir": str(DATA_IDEAEXTRA_DIR),
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
        "regimes": {key: asdict(REGIME_SPECS[key]) for key in REGIME_ORDER},
        "switching_specs": SWITCHING_SPECS,
        "files": {},
    }

    if DEFAULT_CONFIG["save_static_master_pools"]:
        print("\n=== A. static master pools ===")
        for idx, regime_key in enumerate(regime_order):
            seed = base_seed + DEFAULT_CONFIG["train_seed_offset"] + idx * 100_000
            pool, burst_mask = generate_regime_pool(
                regime_key,
                int(args.static_master_episodes),
                T,
                seed,
                calibration_sample_size,
            )
            fname = static_master_name(regime_key, T)
            summary = build_pool_summary(
                file_name=fname,
                pool_type="static_master",
                seed=seed,
                episode_length=T,
                num_episodes=int(args.static_master_episodes),
                pool=pool,
                burst_mask=burst_mask,
                regime_key=regime_key,
            )
            save_pool_with_summary(pool, fname, summary)
            manifest["files"][fname] = summary

    if DEFAULT_CONFIG["save_static_eval_pools"]:
        print("\n=== A2. static eval pools ===")
        for idx, regime_key in enumerate(regime_order):
            seed = base_seed + DEFAULT_CONFIG["eval_seed_offset"] + idx * 100_000
            pool, burst_mask = generate_regime_pool(
                regime_key,
                int(args.static_eval_episodes),
                T,
                seed,
                calibration_sample_size,
            )
            fname = static_eval_name(regime_key, T)
            summary = build_pool_summary(
                file_name=fname,
                pool_type="static_eval",
                seed=seed,
                episode_length=T,
                num_episodes=int(args.static_eval_episodes),
                pool=pool,
                burst_mask=burst_mask,
                regime_key=regime_key,
            )
            save_pool_with_summary(pool, fname, summary)
            manifest["files"][fname] = summary

    if DEFAULT_CONFIG["save_mixed_full_master_pool"]:
        print("\n=== B1. mixed full master pool ===")
        seed = base_seed + DEFAULT_CONFIG["train_seed_offset"] + 500_000
        pool, burst_mask, chosen = generate_mixed_full_master_pool(
            regime_order,
            int(args.mixed_full_master_episodes),
            T,
            seed,
            calibration_sample_size,
        )
        fname = mixed_full_master_name(regime_order, T)
        counts = {key: int(sum(1 for x in chosen if x == key)) for key in regime_order}
        summary = build_pool_summary(
            file_name=fname,
            pool_type="mixed_full_master",
            seed=seed,
            episode_length=T,
            num_episodes=int(args.mixed_full_master_episodes),
            pool=pool,
            burst_mask=burst_mask,
            regime_order=regime_order,
            mixed_counts=counts,
            chosen_regimes_head=chosen,
        )
        save_pool_with_summary(pool, fname, summary)
        manifest["files"][fname] = summary

    if DEFAULT_CONFIG["save_mixed_equal_master_pool"]:
        print("\n=== B2. mixed equal master pool ===")
        seed = base_seed + DEFAULT_CONFIG["train_seed_offset"] + 700_000
        pool, burst_mask, counts = generate_mixed_equal_pool(
            regime_order,
            int(args.mixed_eq_master_episodes),
            T,
            seed,
            calibration_sample_size,
        )
        fname = mixed_equal_master_name(regime_order, T)
        summary = build_pool_summary(
            file_name=fname,
            pool_type="mixed_equal_master",
            seed=seed,
            episode_length=T,
            num_episodes=int(args.mixed_eq_master_episodes),
            pool=pool,
            burst_mask=burst_mask,
            regime_order=regime_order,
            mixed_counts=counts,
        )
        save_pool_with_summary(pool, fname, summary)
        manifest["files"][fname] = summary

    if DEFAULT_CONFIG["save_mixed_equal_val_pool"]:
        print("\n=== B3. mixed equal val pool ===")
        seed = base_seed + DEFAULT_CONFIG["val_seed_offset"] + 700_000
        pool, burst_mask, counts = generate_mixed_equal_pool(
            regime_order,
            int(args.mixed_eq_val_episodes),
            T,
            seed,
            calibration_sample_size,
        )
        fname = mixed_equal_val_name(regime_order, T)
        summary = build_pool_summary(
            file_name=fname,
            pool_type="mixed_equal_val",
            seed=seed,
            episode_length=T,
            num_episodes=int(args.mixed_eq_val_episodes),
            pool=pool,
            burst_mask=burst_mask,
            regime_order=regime_order,
            mixed_counts=counts,
        )
        save_pool_with_summary(pool, fname, summary)
        manifest["files"][fname] = summary

    if DEFAULT_CONFIG["save_switching_master_pools"]:
        print("\n=== C1. switching master pools ===")
        for idx, switching_name in enumerate(SWITCHING_SPECS.keys()):
            seed = base_seed + DEFAULT_CONFIG["train_seed_offset"] + 900_000 + idx * 100_000
            pool, burst_mask = generate_switching_pool(
                switching_name,
                int(args.switching_master_episodes),
                T,
                seed,
                calibration_sample_size,
            )
            fname = switching_master_name(switching_name, T)
            summary = build_pool_summary(
                file_name=fname,
                pool_type="switching_master",
                seed=seed,
                episode_length=T,
                num_episodes=int(args.switching_master_episodes),
                pool=pool,
                burst_mask=burst_mask,
                switching_name=switching_name,
            )
            save_pool_with_summary(pool, fname, summary)
            manifest["files"][fname] = summary

    if DEFAULT_CONFIG["save_switching_eval_pools"]:
        print("\n=== C2. switching eval pools ===")
        for idx, switching_name in enumerate(SWITCHING_SPECS.keys()):
            seed = base_seed + DEFAULT_CONFIG["eval_seed_offset"] + 900_000 + idx * 100_000
            pool, burst_mask = generate_switching_pool(
                switching_name,
                int(args.switching_eval_episodes),
                T,
                seed,
                calibration_sample_size,
            )
            fname = switching_eval_name(switching_name, T)
            summary = build_pool_summary(
                file_name=fname,
                pool_type="switching_eval",
                seed=seed,
                episode_length=T,
                num_episodes=int(args.switching_eval_episodes),
                pool=pool,
                burst_mask=burst_mask,
                switching_name=switching_name,
            )
            save_pool_with_summary(pool, fname, summary)
            manifest["files"][fname] = summary

    manifest_path = DATA_IDEAEXTRA_DIR / "ideaextra_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n=== Done ===")
    print(f"ideaextra dir : {DATA_IDEAEXTRA_DIR}")
    print(f"manifest      : {manifest_path}")
    print("\n建议在 agent 里优先使用：")
    print(f"  train_pool_file = {mixed_equal_master_name(regime_order, T)}")
    print(f"  val_pool_file   = {mixed_equal_val_name(regime_order, T)}")
    print("  static_eval_files = {")
    for regime_key in regime_order:
        print(f"    '{regime_key}': '{static_eval_name(regime_key, T)}',")
    print("  }")


if __name__ == "__main__":
    main()
