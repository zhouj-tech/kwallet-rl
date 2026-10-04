"""SCFAC-DESIGN-v3 episode seed rule and historical seed-namespace audit.

Seed rule (MODEL_DESIGN.md, section 8, mandatory):
  seed(split, regime, episode, counter) =
      first 8 bytes (big-endian) of
      SHA256("SCFAC-DESIGN-v3|<split>|<regime>|<zero-based-episode>|20261003|<collision-counter>")

The collision counter starts at 0 and may be incremented ONLY when a verified
seed or episode-fingerprint collision occurs; every increment is logged.

Historical episode seeds (for seed-namespace disjointness audit), reconstructed
from the frozen generator semantics and the frozen pool summaries:
  - MIX12_EQ master (base seed 20000532): per-regime cursor seeds
    20000532 + 100000*i (i = regime index in REGIME_ORDER), episode j = 0..count_i-1,
    seed_i_j = cursor_i + j. (Stored pool order is a shuffle; the SEED SET is what matters.)
  - MIX12_EQ val (base seed 220000532, verified against frozen summary):
    seed_i_j = 220000532 + 100000*i + j, j = 0..24.
  - 12 static eval pools: seed_i_j = 100000532 + 1000000*i + j, j = 0..199.
  - NEW12-v1: seed_i_j = 710000001 + 100000*i + j, j = 0..199.
  - calibration seed 1234567 and both master/val shuffle seeds are included in the
    reserved-seed namespace check.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Set, Tuple

from .frozen_kwallet_ideaextra_generator import REGIME_ORDER

SEED_RULE_PREFIX = "SCFAC-DESIGN-v3"
SEED_RULE_DATE = "20261003"

MASTER_BASE_SEED = 20000532
MASTER_SHUFFLE_SEED = MASTER_BASE_SEED + 999_999          # 30000531
MASTER_PER_REGIME_COUNTS = [417] * 8 + [416] * 4          # 5000 = 8*417 + 4*416
VAL_BASE_SEED = 220_000_532                                # verified against frozen val summary
VAL_PER_REGIME_COUNT = 25
STATIC_EVAL_BASE_SEED = 100_000_532
STATIC_EVAL_PER_REGIME_COUNT = 200
NEW12_BASE_SEED = 710_000_001
NEW12_PER_REGIME_COUNT = 200
CALIBRATION_SEED = 1_234_567


def design_v3_seed(split: str, regime: str, episode: int, collision_counter: int = 0) -> int:
    payload = (
        f"{SEED_RULE_PREFIX}|{split}|{regime}|{episode}|{SEED_RULE_DATE}|{collision_counter}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big")


def historical_seed_namespace() -> Dict[str, object]:
    """Reconstructed historical episode-seed set + reserved constants."""
    seeds: Set[int] = set()
    per_pool: Dict[str, List[Tuple[str, int]]] = {}

    def add(pool: str, split_label: str, base: int, stride: int, counts) -> None:
        entries = []
        for i, regime in enumerate(REGIME_ORDER):
            count = counts[i] if isinstance(counts, list) else counts
            for j in range(count):
                s = base + stride * i + j
                seeds.add(s)
                entries.append((regime, s))
        per_pool[pool] = entries

    add("MIX12_EQ_master", "historical_train", MASTER_BASE_SEED, 100_000, MASTER_PER_REGIME_COUNTS)
    add("MIX12_EQ_val", "historical_val", VAL_BASE_SEED, 100_000, VAL_PER_REGIME_COUNT)
    add("static_eval", "historical_eval", STATIC_EVAL_BASE_SEED, 1_000_000, STATIC_EVAL_PER_REGIME_COUNT)
    add("NEW12_v1", "new12", NEW12_BASE_SEED, 100_000, NEW12_PER_REGIME_COUNT)

    reserved = {CALIBRATION_SEED, MASTER_SHUFFLE_SEED, VAL_BASE_SEED + 999_999}
    return {
        "seed_count": len(seeds),
        "seeds": seeds,
        "reserved_constants": sorted(reserved),
        "per_pool_first_seeds": {
            k: [s for _, s in v[:2]] for k, v in per_pool.items()
        },
    }
