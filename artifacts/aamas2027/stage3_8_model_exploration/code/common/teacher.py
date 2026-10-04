"""TP-SC teacher: deterministic BF labels derived from the frozen BF-T0.5 controller.

PROVENANCE:
  BF semantics frozen from /Users/zhouzhou/Desktop/kwallet-aamas-b0/tools/aamas_stage2/adapter.py
  (bf_action, lines ~121-127):
    - settlement: candidates = usable wallets with balance >= tx; choose min by (balance, index);
      no-op (= k) when none feasible.
    - flush: eligible = usable wallets, EXCLUDING the settlement choice, with balance < 0.5 * C/k;
      choose min by (balance, index); no-op when none eligible.
  For the TP-SC teacher prior the flush exclusion uses the LEARNER'S ACTUAL sampled
  settlement action a_s (not BF's own settlement), per MODEL_DESIGN.md section 7:
    q_BF(x, a_s): settlement teacher b(x) = BF feasible minimum-balance settlement with
    original deterministic tie break, no-op when none feasible; flush teacher = lowest-balance
    usable wallet below 0.5*C/k excluding the learner's sampled settlement action a_s, or no-op.

IMPORTANT: this module is TRAINING-ONLY. The deployment / deterministic evaluation
pipeline (common/evaluate.py) must not import it. A runtime call counter is exposed
so tests can assert zero BF invocations during deployment evaluation.
"""
from __future__ import annotations

from typing import List, Tuple

from .config import C, K, TP_FLUSH_THRESHOLD, WALLET_SIZE

# Runtime counters (tests assert these stay at zero during deployment evaluation)
bf_settlement_calls = 0
bf_flush_calls = 0

TEACHER_LABEL_NONE = K  # encoded no-op for both heads


def bf_settlement_label(balances: List[float], frozen_flags: List[bool], current_tx: float) -> int:
    """b(x): feasible (usable AND balance >= tx) min-(balance, index) settlement; else no-op k."""
    global bf_settlement_calls
    bf_settlement_calls += 1
    n = len(balances)
    candidates = [
        i for i in range(n)
        if (not frozen_flags[i]) and balances[i] >= current_tx
    ]
    if not candidates:
        return n  # no-op
    return min(candidates, key=lambda i: (balances[i], i))


def bf_flush_label(balances: List[float], frozen_flags: List[bool],
                   learner_settle_action: int, wallet_size: float = None) -> int:
    """q_BF(x, a_s): lowest-balance usable wallet with balance < 0.5*C/k, excluding the
    learner's sampled settlement action a_s; else no-op k.

    wallet_size = C/k threads the run's capacity (default: frozen config C=800).
    Threshold is strictly 0.5 * C/k at the run capacity."""
    global bf_flush_calls
    bf_flush_calls += 1
    w = WALLET_SIZE if wallet_size is None else float(wallet_size)
    n = len(balances)
    threshold = TP_FLUSH_THRESHOLD * w
    eligible = [
        i for i in range(n)
        if (not frozen_flags[i]) and i != learner_settle_action and balances[i] < threshold
    ]
    if not eligible:
        return n  # no-op
    return min(eligible, key=lambda i: (balances[i], i))


def teacher_labels(env, settle_action: int) -> Tuple[int, int]:
    """(b(x), q_BF(x, a_s)) from the live pre-action environment state.
    The flush threshold uses env.wallet_size = C/k, i.e. the run's actual capacity."""
    balances = [float(env.wallets[i]) for i in range(K)]
    frozen = [not env._usable(i) for i in range(K)]
    b = bf_settlement_label(balances, frozen, float(env.current_tx))
    q = bf_flush_label(balances, frozen, settle_action,
                       wallet_size=float(env.wallet_size))
    return b, q


def reset_counters() -> None:
    global bf_settlement_calls, bf_flush_calls
    bf_settlement_calls = 0
    bf_flush_calls = 0


def lambda_bf(episode: int, lambda0: float = 0.01, decay_episodes: int = 250) -> float:
    """lambda_BF(e) = 0.01 * max(1 - e/250, 0); e is the zero-based episode index."""
    return lambda0 * max(1.0 - episode / decay_episodes, 0.0)
