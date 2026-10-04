"""IA-SC extra features: q(x) (9 state features) and c(x, a_s) (4 flush-condition features).

Feature spec (MODEL_DESIGN.md section 6), all derived from the frozen E0
baseline observation (state width 3k+2) and environment bookkeeping:

State layout of the 74-dim base observation (per frozen_env):
  [0:k]   b_i / w      (balance / wallet_size)
  [k:2k]  0.0 usable / 1.0 frozen
  [2k:3k] remaining freeze time rem / F   (rem = max(0, freeze_until - time))
  [3k]    current_tx / max_transaction
  [3k+1]  time / (max_steps - 1)

Refill-delay semantics (frozen E0): a frozen wallet with observed normalized
remaining r = rem/F refills so that its usable-at-decision delay is
d = F * r + 1 decision steps; a frozen zero timer (r == 0) still means exactly
one step remains. A usable wallet is not scheduled (not counted in any p_l).
With F = 3 the scheduled delays are exactly d in {1, 2, 3}.

q(x) (9 features):
  1. sum_i u_i / k                       usable-wallet fraction
  2. |V| / k                             feasible-wallet fraction (u_i=1 and b_i >= v)
  3. sum_i u_i * b_i / C                 usable balance / C
  4. max_{u_i=1} b_i / w                 largest usable balance / w (0 if none usable)
  5. v / w                               current demand / wallet capacity
  6. 1[v > w]                            oversize indicator
  7. p_1  8. p_2  9. p_3                 fraction of wallets scheduled to refill
                                         in l decision steps (l = 1, 2, 3)

c(x, a_s) (4 features; all zero when a_s == k, i.e. settlement no-op):
  1. u_s
  2. b_s / w
  3. 1[s in V]
  4. 1[s in V] * (b_s - v) / w           hypothetical residual balance if settlement succeeds
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .config import C, K, MAX_TRANSACTION, WALLET_SIZE


def _resolve_w(capacity, wallet_size) -> float:
    """Effective wallet capacity w = C/k. Explicit wallet_size wins; then capacity/k;
    then the frozen config default (C=800). Formulas are unchanged; only the
    capacity source is threaded so C1200 cells use w = 1200/24 = 50."""
    if wallet_size is not None:
        return float(wallet_size)
    if capacity is not None:
        return float(capacity) / K
    return float(WALLET_SIZE)


def _resolve_C(capacity) -> float:
    if capacity is not None:
        return float(capacity)
    return float(C)


def extract_q_features(base_state: np.ndarray, capacity=None, wallet_size=None) -> np.ndarray:
    x = np.asarray(base_state, dtype=np.float64)
    assert x.shape[0] == 3 * K + 2, f"expected base state width {3*K+2}, got {x.shape[0]}"
    w = _resolve_w(capacity, wallet_size)
    bal = x[:K] * w                                # actual balances b_i
    frozen = x[K:2 * K] > 0.5                       # True = frozen
    usable = ~frozen
    rem_norm = x[2 * K:3 * K]                       # rem / F
    v = x[3 * K] * MAX_TRANSACTION                  # current demand (actual value)

    q = np.zeros(9, dtype=np.float64)

    q[0] = usable.sum() / K
    feasible = usable & (bal >= v)
    q[1] = feasible.sum() / K
    q[2] = (usable * bal).sum() / _resolve_C(capacity)
    if usable.any():
        q[3] = bal[usable].max() / w
    else:
        q[3] = 0.0
    q[4] = v / w
    q[5] = 1.0 if v > w else 0.0

    # scheduled refill delays: d = F * rem_norm + 1 for frozen wallets (rem_norm in
    # {0, 1/3, 2/3} -> d in {1, 2, 3}); usable wallets are not scheduled.
    delays = np.rint(3.0 * rem_norm + 1.0).astype(int)
    for i in range(K):
        if frozen[i]:
            d = delays[i]
            if 1 <= d <= 3:
                q[5 + d] += 1.0 / K
    return q


def extract_c_features(base_state: np.ndarray, settle_action: int,
                       capacity=None, wallet_size=None) -> np.ndarray:
    x = np.asarray(base_state, dtype=np.float64)
    assert x.shape[0] == 3 * K + 2
    w = _resolve_w(capacity, wallet_size)
    bal = x[:K] * w
    frozen = x[K:2 * K] > 0.5
    usable = ~frozen
    v = x[3 * K] * MAX_TRANSACTION

    c = np.zeros(4, dtype=np.float64)
    if settle_action == K:          # settlement no-op: all four zero
        return c
    assert 0 <= settle_action < K
    u_s = bool(usable[settle_action])
    in_V = bool(u_s and bal[settle_action] >= v)
    c[0] = 1.0 if u_s else 0.0
    c[1] = bal[settle_action] / w
    c[2] = 1.0 if in_V else 0.0
    c[3] = (bal[settle_action] - v) / w if in_V else 0.0
    return c


def augment_state(base_state: np.ndarray, capacity=None, wallet_size=None) -> np.ndarray:
    """83-dim IA observation = [base 74, q(x) 9]."""
    return np.concatenate([np.asarray(base_state, dtype=np.float32),
                           extract_q_features(base_state, capacity=capacity,
                                              wallet_size=wallet_size).astype(np.float32)]
                          ).astype(np.float32)


def ia_state_size() -> int:
    return 3 * K + 2 + 9


def ia_flush_cond_size() -> int:
    return 128 + 32 + 4  # encoder hidden + settle embedding + c(x,a_s); used only in docs
