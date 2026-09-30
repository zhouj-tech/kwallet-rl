"""Twelve transaction regimes for K-Wallet.

Faithful port of the provenance generator
``src/ideaextra/kwallet_ideaextra_generator.py`` (the DQN-era code whose regime
keys, parameters, burst process and seed namespaces match the old paper's
Table I and Sec. V.B: 12 regimes, target value near 50, 200 static eval
episodes/regime, 5000-episode MIX12_EQ training pool, separated train/val/eval
seed offsets, base seed 532).

IMPORTANT (paper Sec. V.A + task brief 7.2): the target mean is ~50 for the raw
families, but burst multiplication, truncation and rounding move the realised
pool means; these are *capped* heavy tails and must not be used to claim
unbounded heavy-tail asymptotics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

GENERATOR_VERSION = "regimes_v1_port_from_ideaextra"

# Burst parameters shared by all B-group regimes (kept identical to source so
# tail severity and burst strength are not confounded).
BURST_START_PROB = 0.035
BURST_LENGTH_MEAN = 6.0
BURST_SIZE_MULTIPLIER = 1.40

REGIME_ORDER: List[str] = [
    "US", "TLS", "LNS", "TLNS", "TPLS", "PLS",
    "UB", "TLB", "LNB", "TLNB", "TPLB", "PLB",
]


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


REGIME_SPECS: Dict[str, RegimeSpec] = {
    "US": RegimeSpec("US", "U-S", "uniform", False,
                     "bounded baseline; smooth uniform transactions",
                     max_tx=100.0, uniform_half_width=30.0),
    "TLS": RegimeSpec("TLS", "TL-S", "trunc_light", False,
                      "light-tail smooth; truncated normal around target",
                      max_tx=120.0, trunc_light_std=15.0,
                      trunc_light_low=1.0, trunc_light_high=120.0),
    "LNS": RegimeSpec("LNS", "LN-S", "lognormal", False,
                      "smooth lognormal; right-skew heavy tail",
                      max_tx=180.0, log_sigma=0.65),
    "TLNS": RegimeSpec("TLNS", "TLN-S", "trunc_lognormal", False,
                       "smooth truncated lognormal; capped skew",
                       max_tx=120.0, log_sigma=0.55, trunc_lognormal_max=120.0),
    "TPLS": RegimeSpec("TPLS", "TPL-S", "trunc_powerlaw", False,
                       "smooth truncated power law; capped strong tail",
                       max_tx=120.0, power_alpha=2.3, power_xmin=8.0,
                       trunc_powerlaw_max=120.0),
    "PLS": RegimeSpec("PLS", "PL-S", "powerlaw", False,
                      "smooth power law; extreme heavy-tail stress",
                      max_tx=250.0, power_alpha=2.2, power_xmin=8.0),
    "UB": RegimeSpec("UB", "U-B", "uniform", True,
                     "bursty uniform; bursts without heavy tail",
                     max_tx=100.0, uniform_half_width=30.0,
                     burst_start_prob=BURST_START_PROB,
                     burst_length_mean=BURST_LENGTH_MEAN,
                     burst_size_multiplier=BURST_SIZE_MULTIPLIER),
    "TLB": RegimeSpec("TLB", "TL-B", "trunc_light", True,
                      "bursty truncated light-tail",
                      max_tx=120.0, trunc_light_std=15.0,
                      trunc_light_low=1.0, trunc_light_high=120.0,
                      burst_start_prob=BURST_START_PROB,
                      burst_length_mean=BURST_LENGTH_MEAN,
                      burst_size_multiplier=BURST_SIZE_MULTIPLIER),
    "LNB": RegimeSpec("LNB", "LN-B", "lognormal", True,
                      "bursty lognormal; skew + burst",
                      max_tx=180.0, log_sigma=0.70,
                      burst_start_prob=BURST_START_PROB,
                      burst_length_mean=BURST_LENGTH_MEAN,
                      burst_size_multiplier=BURST_SIZE_MULTIPLIER),
    "TLNB": RegimeSpec("TLNB", "TLN-B", "trunc_lognormal", True,
                       "bursty truncated lognormal; capped skew bursts",
                       max_tx=120.0, log_sigma=0.58, trunc_lognormal_max=120.0,
                       burst_start_prob=BURST_START_PROB,
                       burst_length_mean=BURST_LENGTH_MEAN,
                       burst_size_multiplier=BURST_SIZE_MULTIPLIER),
    "TPLB": RegimeSpec("TPLB", "TPL-B", "trunc_powerlaw", True,
                       "bursty truncated power law; capped tail bursts",
                       max_tx=120.0, power_alpha=2.2, power_xmin=8.0,
                       trunc_powerlaw_max=120.0,
                       burst_start_prob=BURST_START_PROB,
                       burst_length_mean=BURST_LENGTH_MEAN,
                       burst_size_multiplier=BURST_SIZE_MULTIPLIER),
    "PLB": RegimeSpec("PLB", "PL-B", "powerlaw", True,
                      "bursty power law; worst-case stress",
                      max_tx=250.0, power_alpha=2.1, power_xmin=8.0,
                      burst_start_prob=BURST_START_PROB,
                      burst_length_mean=BURST_LENGTH_MEAN,
                      burst_size_multiplier=BURST_SIZE_MULTIPLIER),
}

# Default generation config (mirrors source DEFAULT_CONFIG).
DEFAULT_GEN_CONFIG = {
    "base_seed": 532,
    "train_seed_offset": 0,
    "val_seed_offset": 2_000_000,
    "eval_seed_offset": 1_000_000,
    "episode_length": 1000,
    "static_eval_episodes": 200,
    "mixed_equal_master_episodes": 5000,
    "mixed_equal_val_episodes": 300,
    "calibration_sample_size": 200_000,
}


def build_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# ----------------------------------------------------------------------
# raw distribution sampling (exact port of the source functions)
# ----------------------------------------------------------------------
def _sample_raw_uniform(spec, rng, size):
    low = max(spec.min_tx, spec.target_mean - spec.uniform_half_width)
    high = min(spec.max_tx, spec.target_mean + spec.uniform_half_width)
    return rng.uniform(low, high, size=size)


def _sample_raw_trunc_light(spec, rng, size):
    out, needed = [], size
    while needed > 0:
        cand = rng.normal(loc=spec.target_mean, scale=spec.trunc_light_std,
                          size=max(needed * 2, 1000))
        cand = cand[(cand >= spec.trunc_light_low) & (cand <= spec.trunc_light_high)]
        if cand.size > 0:
            take = min(needed, cand.size)
            out.append(cand[:take])
            needed -= take
    return np.concatenate(out, axis=0)


def _sample_raw_lognormal(spec, rng, size):
    return rng.lognormal(mean=0.0, sigma=spec.log_sigma, size=size)


def _sample_raw_trunc_lognormal(spec, rng, size):
    out, needed = [], size
    while needed > 0:
        cand = _sample_raw_lognormal(spec, rng, max(needed * 3, 3000))
        cand = cand[cand <= spec.trunc_lognormal_max]
        if cand.size > 0:
            take = min(needed, cand.size)
            out.append(cand[:take])
            needed -= take
    return np.concatenate(out, axis=0)


def _sample_raw_powerlaw(spec, rng, size):
    y = rng.pareto(spec.power_alpha, size=size)
    return spec.power_xmin * (1.0 + y)


def _sample_raw_trunc_powerlaw(spec, rng, size):
    out, needed = [], size
    while needed > 0:
        cand = _sample_raw_powerlaw(spec, rng, max(needed * 3, 3000))
        cand = cand[cand <= spec.trunc_powerlaw_max]
        if cand.size > 0:
            take = min(needed, cand.size)
            out.append(cand[:take])
            needed -= take
    return np.concatenate(out, axis=0)


def sample_raw_by_family(spec, rng, size):
    family = spec.dist_family
    return {
        "uniform": _sample_raw_uniform,
        "trunc_light": _sample_raw_trunc_light,
        "lognormal": _sample_raw_lognormal,
        "trunc_lognormal": _sample_raw_trunc_lognormal,
        "powerlaw": _sample_raw_powerlaw,
        "trunc_powerlaw": _sample_raw_trunc_powerlaw,
    }[family](spec, rng, size)


_CALIBRATION_CACHE: Dict[str, float] = {}


def get_mean_scale(spec: RegimeSpec, calibration_sample_size: int,
                   seed: int = 1234567) -> float:
    """Per-regime multiplicative scale so the RAW family mean ~ target_mean.

    Computed from the unclipped raw distribution (fixed calibration seed), so
    realised clipped means are not exactly 50 (documented).
    """
    cache_key = f"{spec.key}|{calibration_sample_size}"
    if cache_key in _CALIBRATION_CACHE:
        return _CALIBRATION_CACHE[cache_key]
    rng = build_rng(seed)
    raw = sample_raw_by_family(spec, rng, calibration_sample_size)
    raw_mean = float(np.mean(raw))
    scale = spec.target_mean / raw_mean if raw_mean > 0 else 1.0
    _CALIBRATION_CACHE[cache_key] = scale
    return scale


def _sample_one_tx(spec, rng, in_burst, calibration_sample_size):
    raw = float(sample_raw_by_family(spec, rng, 1)[0])
    scale = get_mean_scale(spec, calibration_sample_size=calibration_sample_size)
    tx = raw * scale
    if in_burst:
        tx *= spec.burst_size_multiplier
    tx = max(spec.min_tx, min(float(tx), spec.max_tx))
    return int(round(tx))


def generate_regime_episode(spec, episode_length, rng, calibration_sample_size
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """One episode of values plus the true burst mask (burst mask never observed
    by the policy)."""
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
        out[t] = _sample_one_tx(spec, rng, in_burst, calibration_sample_size)
        burst_mask[t] = 1 if in_burst else 0
    return out, burst_mask


def generate_regime_pool(regime_key, num_episodes, episode_length, base_seed,
                         calibration_sample_size):
    """Static per-regime pool; episode e uses rng = default_rng(base_seed + e)."""
    spec = REGIME_SPECS[regime_key]
    pool = np.zeros((num_episodes, episode_length), dtype=np.int32)
    masks = np.zeros((num_episodes, episode_length), dtype=np.int8)
    for ep in range(num_episodes):
        rng = build_rng(base_seed + ep)
        v, m = generate_regime_episode(spec, episode_length, rng,
                                       calibration_sample_size)
        pool[ep] = v
        masks[ep] = m
    return pool, masks


def generate_mixed_equal_pool(regime_order, total_episodes, episode_length,
                              base_seed, calibration_sample_size):
    """MIX12_EQ: episodes allocated as evenly as possible across regimes then
    shuffled (matches source). Returns (pool, burst_masks, counts)."""
    pieces, mask_pieces, counts = [], [], {}
    n_regimes = len(regime_order)
    base_each = total_episodes // n_regimes
    remainder = total_episodes % n_regimes
    cursor_seed = base_seed
    for i, regime_key in enumerate(regime_order):
        count = base_each + (1 if i < remainder else 0)
        counts[regime_key] = count
        part, part_mask = generate_regime_pool(
            regime_key, count, episode_length, cursor_seed,
            calibration_sample_size)
        pieces.append(part)
        mask_pieces.append(part_mask)
        cursor_seed += 100_000
    pool = np.concatenate(pieces, axis=0)
    masks = np.concatenate(mask_pieces, axis=0)
    shuffle_rng = build_rng(base_seed + 999_999)
    perm = shuffle_rng.permutation(pool.shape[0])
    return pool[perm], masks[perm], counts
