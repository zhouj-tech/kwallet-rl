"""Vendored K-Wallet E0 environment (baseline mode), frozen semantics.

PROVENANCE (do not edit the semantics of this file by hand):
  Source : /Users/zhouzhou/Desktop/kwallet-rl/src/idea3/context_attention/kwallet_ctx_attn_fair_benchmark.py
  SHA256 : recorded in common/config.py as FROZEN_SOURCE_SHA256
  Symbol : KWalletEnv (class), used with model_mode="baseline".
  Frozen evidence commit of the source worktree: 281e02920a32e9878177593d62c7f2d841ed6078

This vendored copy restricts the original class to the "baseline" observation
mode (state width 3k+2), which is the mode used by the historical SC-FAC
("conditional_factorized_ac", env dual_env_mode -> "baseline") runs.
Attention-context code paths were removed; all environment semantics
(reset, _usable, _get_base_state, _decode_action, step, get_metrics) are kept
byte-for-byte identical to the frozen source. A unit test verifies behavioral
equivalence against the original frozen module when it is importable.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class KWalletEnv:
    """
    Shared fixed-setting K-wallet environment (frozen E0 semantics).

    Leakage rule:
    - At decision step t, current_tx is tx[t].
    - tx_history is appended only after tx[t] has been processed in step().
    """

    def __init__(
        self,
        C: float = 1200.0,
        k: int = 3,
        F: int = 3,
        max_transaction: int = 1000,
        max_steps: int = 1000,
        seed: int = 123,
        model_mode: str = "baseline",
        attention_window_size: int = 50,
        enable_shaping: bool = False,
        alpha_drop: float = 0.02,
        beta_flush: float = 0.01,
    ):
        if model_mode not in {"baseline", "attn_context"}:
            raise ValueError(f"Unsupported model_mode={model_mode}")
        if model_mode != "baseline":
            raise NotImplementedError(
                "This vendored copy implements only model_mode='baseline' "
                "(the mode used by the historical SC-FAC runs)."
            )
        self.C = float(C)
        self.k = int(k)
        self.F = int(F)
        self.max_transaction = int(max_transaction)
        self.max_steps = int(max_steps)
        self.wallet_size = self.C / self.k
        self.num_actions = (self.k + 1) ** 2

        self.model_mode = model_mode
        self.attention_window_size = int(attention_window_size)
        self.tx_history = deque(maxlen=self.attention_window_size)

        self.enable_shaping = bool(enable_shaping)
        self.alpha_drop = float(alpha_drop)
        self.beta_flush = float(beta_flush)
        self.IMBALANCE_PENALTY = 0.02
        self.WASTEFUL_REFRESH_PENALTY = 0.02
        self.WASTEFUL_REFRESH_THRESH = 0.6
        self.INVALID_ACTION_PENALTY = 0.05

        self.rng = np.random.default_rng(seed)
        self._tx_stream: Optional[List[int]] = None

        self.reset()

    @property
    def base_state_size(self) -> int:
        return 3 * self.k + 2

    @property
    def state_size(self) -> int:
        return self.base_state_size

    def reset(self, tx_stream: Optional[np.ndarray | List[int]] = None) -> np.ndarray:
        self.wallets = [self.wallet_size] * self.k
        self.freeze_until = [-1] * self.k
        self.pending_refill = [False] * self.k

        self.total_settled = 0.0
        self.total_accepted = 0.0
        self.num_flushes = 0
        self.drops = 0
        self.oversize_drops = 0
        self.insufficient_drops = 0
        self.time = 0
        self.tx_history.clear()

        if tx_stream is not None:
            self._tx_stream = [int(x) for x in tx_stream]
        else:
            self._tx_stream = [
                int(self.rng.integers(1, self.max_transaction + 1))
                for _ in range(self.max_steps)
            ]

        self.current_tx = self._tx_stream[self.time]
        return self._get_state()

    def _usable(self, i: int) -> bool:
        return self.time > self.freeze_until[i]

    def _get_base_state(self) -> List[float]:
        state = []
        for w in self.wallets:
            state.append(w / self.wallet_size)
        for i in range(self.k):
            state.append(0.0 if self._usable(i) else 1.0)
        for i in range(self.k):
            rem = max(0, self.freeze_until[i] - self.time)
            state.append((rem / self.F) if self.F > 0 else 0.0)
        state.append(self.current_tx / self.max_transaction)
        state.append(self.time / max(1, self.max_steps - 1))
        return state

    def _get_state(self) -> np.ndarray:
        return np.array(self._get_base_state(), dtype=np.float32)

    def _decode_action(self, action_int: int) -> Tuple[int, int]:
        if not (0 <= action_int < self.num_actions):
            raise ValueError(f"Action out of range: {action_int}")
        base = self.k + 1
        settle_choice = action_int // base
        flush_choice = action_int % base
        return settle_choice, flush_choice

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        reward = 0.0
        flushes_this_step = 0
        refresh_targets = []
        tx = self.current_tx
        settle_choice, flush_choice = self._decode_action(action)
        pre_refresh_balances = {i: self.wallets[i] for i in range(self.k)}

        if flush_choice < self.k:
            if self._usable(flush_choice):
                self.pending_refill[flush_choice] = True
                self.wallets[flush_choice] = 0.0
                self.freeze_until[flush_choice] = self.time + self.F - 1
                self.num_flushes += 1
                flushes_this_step += 1
                refresh_targets.append(flush_choice)
            elif self.enable_shaping:
                reward -= self.INVALID_ACTION_PENALTY

        fit_idx = None
        if tx > self.wallet_size:
            self.drops += 1
            self.oversize_drops += 1
            reward -= self.alpha_drop
        elif settle_choice < self.k:
            if (
                self._usable(settle_choice)
                and (settle_choice not in refresh_targets)
                and self.wallets[settle_choice] >= tx
            ):
                self.wallets[settle_choice] -= tx
                self.total_settled += tx
                self.total_accepted += tx
                reward += float(tx) / self.max_transaction
                fit_idx = settle_choice
            else:
                self.drops += 1
                self.insufficient_drops += 1
                reward -= self.alpha_drop
        else:
            self.drops += 1
            self.insufficient_drops += 1
            reward -= self.alpha_drop

        reward -= self.beta_flush * flushes_this_step

        if self.enable_shaping:
            usable_balances = [self.wallets[i] for i in range(self.k) if self._usable(i)]
            if len(usable_balances) >= 2:
                std_norm = float(np.std(np.array(usable_balances)) / self.wallet_size)
                reward -= self.IMBALANCE_PENALTY * std_norm
            for i in refresh_targets:
                if (pre_refresh_balances[i] / self.wallet_size) >= self.WASTEFUL_REFRESH_THRESH:
                    reward -= self.WASTEFUL_REFRESH_PENALTY

        self.tx_history.append(float(tx))
        self.time += 1

        for i in range(self.k):
            if self.pending_refill[i] and self._usable(i):
                self.wallets[i] = self.wallet_size
                self.pending_refill[i] = False

        if self.time < len(self._tx_stream):
            self.current_tx = self._tx_stream[self.time]

        done = self.time >= self.max_steps
        info = {
            "fit_idx": fit_idx,
            "tx": tx,
            "settle_choice": settle_choice,
            "flush_choice": flush_choice,
            "settled_value": float(tx if fit_idx is not None else 0.0),
            "accepted": bool(fit_idx is not None),
            "dropped": bool(fit_idx is None and tx <= self.wallet_size),
            "oversize_dropped": bool(tx > self.wallet_size),
            "flushes_this_step": flushes_this_step,
        }
        return self._get_state(), float(reward), bool(done), info

    def get_metrics(self) -> Dict[str, float]:
        return {
            "settled": self.total_settled,
            "drops": self.drops,
            "oversize_drops": self.oversize_drops,
            "insufficient_drops": self.insufficient_drops,
            "flushes": self.num_flushes,
            "utilization": self.total_accepted / (self.C * self.max_steps),
            "avg_tx_value": self.total_settled / max(1, self.max_steps - self.drops),
            "drop_rate": self.drops / self.max_steps,
        }
