"""Streaming / non-stationary streams with regime change-points (Phase-2).

The stationary pools in ``pools.py`` draw every episode from a single regime.
For the out-of-distribution (OOD) streaming test we splice two regimes within
one T-step episode: steps ``[0, sp)`` behave like regime A and steps ``[sp, T)``
like regime B. The policies are trained ONLY on stationary pools; at test time
they must cope with an abrupt distribution shift they have not been trained on.

Splicing uses independent deterministic episodes (same base_seed) for the two
regimes, so each switching episode is reproducible. We focus on the hard
directions calm -> burst and burst -> calm.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

# Regime labels mirror data/regimes.py REGIME_ORDER.
REGIMES = ["US", "TLS", "LNS", "TLNS", "TPLS", "PLS",
           "UB", "TLB", "LNB", "TLNB", "TPLB", "PLB"]

# (from, to, switch fraction). Calm=US/UB/steady; burst=burst-size regimes.
SWITCH_PAIRS: List[Tuple[str, str, float]] = [
    ("US", "TPLS", 0.5),    # steady -> large burst-spike
    ("US", "TPLB", 0.5),    # steady -> burst large
    ("UB", "TPLB", 0.5),    # unif big -> burst large
    ("TPLS", "US", 0.5),    # burst -> steady (policy must stop over-reserving)
    ("US", "TLS", 0.25),    # early switch
    ("US", "TLS", 0.75),    # late switch
]


def build_switch_streams(eval_pools: Dict[str, np.ndarray],
                         regime_a: str, regime_b: str,
                         switch_frac: float = 0.5,
                         n: int = None) -> np.ndarray:
    """Return (m, T) streams: first ``switch_frac`` of steps from regime A,
    rest from regime B. ``eval_pools[regime]`` is a (n_ep, T) array."""
    pa = np.asarray(eval_pools[regime_a], dtype=np.float64)
    pb = np.asarray(eval_pools[regime_b], dtype=np.float64)
    T = pa.shape[1]
    sp = int(round(switch_frac * T))
    m = min(pa.shape[0], pb.shape[0]) if n is None else min(n, pa.shape[0], pb.shape[0])
    out = np.empty((m, T), dtype=np.float64)
    for i in range(m):
        out[i, :sp] = pa[i, :sp]
        out[i, sp:] = pb[i, sp:]
    return out


def switch_point(regime_a: str, regime_b: str, switch_frac: float, T: int = 1000):
    return int(round(switch_frac * T))
