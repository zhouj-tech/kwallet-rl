from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


SETTLE_DISCARD = 0
SETTLE_ACCEPT = 1
POOL_A = 0
POOL_B = 1
SAFE_MASK_EPS = 1e-8


def build_flush_fractions(flush_levels: int, flush_grid: str = "uniform") -> List[float]:
    if str(flush_grid) != "uniform":
        raise ValueError("general_model_v2 currently supports only flush_grid='uniform'.")
    flush_levels = int(flush_levels)
    if flush_levels < 2:
        raise ValueError("flush_levels must be at least 2.")
    return [float(i / (flush_levels - 1)) for i in range(flush_levels)]


@dataclass
class PendingRelease:
    return_time: int
    amount: float


class TwoPoolCollateralEnv:
    """Two-pool general collateral environment.

    This is intentionally simpler than K-Wallet:
    - two symmetric collateral pools A/B,
    - one typed transaction per step,
    - one settle decision,
    - one pool-specific flush action.
    """

    def __init__(
        self,
        C: float = 1000.0,
        F: int = 3,
        max_transaction: float = 1000.0,
        max_steps: int = 1000,
        seed: int = 123,
        money_p: float = 1.0,
        money_tau: float = 100.0,
        drop_penalty: float = 0.0,
        flush_levels: int = 17,
        flush_grid: str = "uniform",
        state_feature_mode: str = "base",
        mask_mode: str = "none",
    ) -> None:
        if C <= 0:
            raise ValueError("C must be positive.")
        if F < 0:
            raise ValueError("F must be non-negative.")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if max_transaction <= 0:
            raise ValueError("max_transaction must be positive.")
        if money_p < 0:
            raise ValueError("money_p must be non-negative.")
        if money_tau < 0:
            raise ValueError("money_tau must be non-negative.")
        if drop_penalty < 0:
            raise ValueError("drop_penalty must be non-negative.")

        self.C = float(C)
        self.F = int(F)
        self.max_transaction = float(max_transaction)
        self.max_steps = int(max_steps)
        self.money_p = float(money_p)
        self.money_tau = float(money_tau)
        self.drop_penalty = float(drop_penalty)
        self.flush_levels = int(flush_levels)
        self.flush_grid = str(flush_grid)
        self.flush_fractions = build_flush_fractions(self.flush_levels, self.flush_grid)
        self.num_flush_choices = 1 + 2 * (self.flush_levels - 1)
        self.action_size = 2 * self.num_flush_choices
        self.num_actions = self.action_size
        self.factorized_policy_output_size = 2 + self.num_flush_choices

        self.state_feature_mode = str(state_feature_mode)
        if self.state_feature_mode not in {"base", "pressure", "pressure_release"}:
            raise ValueError(
                "state_feature_mode must be 'base', 'pressure', or 'pressure_release'."
            )
        self.mask_mode = str(mask_mode)
        if self.mask_mode not in {"none", "safe"}:
            raise ValueError("mask_mode must be 'none' or 'safe'.")

        self.rng = np.random.default_rng(seed)
        self._tx_values: List[float] = []
        self._tx_types: List[int] = []
        self.reset()

    @property
    def base_state_size(self) -> int:
        return 8

    @property
    def state_size(self) -> int:
        if self.state_feature_mode == "base":
            return self.base_state_size
        if self.state_feature_mode == "pressure":
            return self.base_state_size + 6
        return self.base_state_size + 6 + 8

    def decode_action(self, action_id: int) -> Tuple[int, int]:
        action_id = int(action_id)
        if not 0 <= action_id < self.action_size:
            raise ValueError(f"action_id out of range: {action_id}")
        return action_id // self.num_flush_choices, action_id % self.num_flush_choices

    def encode_action(self, settle_choice: int, flush_action: int) -> int:
        if int(settle_choice) not in (SETTLE_DISCARD, SETTLE_ACCEPT):
            raise ValueError(f"settle_choice must be 0 or 1, got {settle_choice}")
        if not 0 <= int(flush_action) < self.num_flush_choices:
            raise ValueError(
                f"flush_action must be in [0, {self.num_flush_choices - 1}], got {flush_action}"
            )
        return int(settle_choice) * self.num_flush_choices + int(flush_action)

    def decode_flush_action(self, flush_action: int) -> Tuple[Optional[int], float]:
        flush_action = int(flush_action)
        if not 0 <= flush_action < self.num_flush_choices:
            raise ValueError(
                f"flush_action must be in [0, {self.num_flush_choices - 1}], got {flush_action}"
            )
        if flush_action == 0:
            return None, 0.0
        if 1 <= flush_action <= self.flush_levels - 1:
            return POOL_A, float(self.flush_fractions[flush_action])
        pool_b_fraction_idx = flush_action - (self.flush_levels - 1)
        return POOL_B, float(self.flush_fractions[pool_b_fraction_idx])

    def flush_action_decoding_summary(self) -> Dict[str, Any]:
        return {
            "no_flush": 0,
            "pool_A_actions": [1, self.flush_levels - 1],
            "pool_B_actions": [self.flush_levels, self.num_flush_choices - 1],
            "pool_A_first_fraction": self.flush_fractions[1],
            "pool_A_full_action": self.flush_levels - 1,
            "pool_B_first_action": self.flush_levels,
            "pool_B_first_fraction": self.flush_fractions[1],
            "pool_B_full_action": self.num_flush_choices - 1,
        }

    def reset(
        self,
        tx_values: Optional[Sequence[float]] = None,
        tx_types: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        self.available_A = self.C
        self.available_B = self.C
        self.committed_A = 0.0
        self.committed_B = 0.0
        self.pending_A: List[PendingRelease] = []
        self.pending_B: List[PendingRelease] = []

        self.settled_value = 0.0
        self.dropped_value = 0.0
        self.total_value = 0.0
        self.drops = 0
        self.flushes = 0
        self.flush_A_count = 0
        self.flush_B_count = 0
        self.drop_A_count = 0
        self.drop_B_count = 0
        self.settled_A_value = 0.0
        self.settled_B_value = 0.0
        self.time = 0

        if tx_values is None:
            self._tx_values = [
                float(self.rng.integers(1, int(self.max_transaction) + 1))
                for _ in range(self.max_steps)
            ]
        else:
            self._tx_values = [float(x) for x in tx_values]
            if len(self._tx_values) < self.max_steps:
                raise ValueError(
                    f"tx_values has {len(self._tx_values)} steps, expected at least {self.max_steps}."
                )

        if tx_types is None:
            self._tx_types = [int(x) for x in self.rng.integers(0, 2, size=self.max_steps)]
        else:
            self._tx_types = [int(x) for x in tx_types]
            if len(self._tx_types) < self.max_steps:
                raise ValueError(
                    f"tx_types has {len(self._tx_types)} steps, expected at least {self.max_steps}."
                )
            if any(x not in (POOL_A, POOL_B) for x in self._tx_types[: self.max_steps]):
                raise ValueError("tx_types must contain only 0 for Pool A or 1 for Pool B.")

        self.current_tx_value = float(self._tx_values[0])
        self.current_tx_type = int(self._tx_types[0])
        self._release_due_collateral()
        return self.get_state()

    def _release_one_queue(self, queue: List[PendingRelease]) -> Tuple[List[PendingRelease], float]:
        returned = 0.0
        still_pending: List[PendingRelease] = []
        for item in queue:
            if item.return_time <= self.time:
                returned += float(item.amount)
            else:
                still_pending.append(item)
        return still_pending, returned

    def _release_due_collateral(self) -> Tuple[float, float]:
        self.pending_A, returned_A = self._release_one_queue(self.pending_A)
        self.pending_B, returned_B = self._release_one_queue(self.pending_B)
        if returned_A > 0.0:
            self.available_A = min(self.C, self.available_A + returned_A)
        if returned_B > 0.0:
            self.available_B = min(self.C, self.available_B + returned_B)
        return float(returned_A), float(returned_B)

    def _pending_release_total(self, queue: List[PendingRelease]) -> float:
        return float(sum(float(item.amount) for item in queue))

    def _next_release_features(self, queue: List[PendingRelease]) -> Tuple[float, float]:
        if not queue:
            return 0.0, 0.0
        next_return_time = min(int(item.return_time) for item in queue)
        next_amount = sum(
            float(item.amount)
            for item in queue
            if int(item.return_time) == next_return_time
        )
        time_to_next = max(0.0, float(next_return_time - self.time))
        return float(next_amount), float(time_to_next)

    def get_action_mask(self) -> Optional[Dict[str, np.ndarray]]:
        if self.mask_mode == "none":
            return None
        settle_mask = np.ones(2, dtype=np.float32)
        flush_mask = np.ones(self.num_flush_choices, dtype=np.float32)
        flush_mask[0] = 1.0
        if self.committed_A <= SAFE_MASK_EPS:
            flush_mask[1:self.flush_levels] = 0.0
        if self.committed_B <= SAFE_MASK_EPS:
            flush_mask[self.flush_levels:self.num_flush_choices] = 0.0
        return {"settle_mask": settle_mask, "flush_mask": flush_mask}

    def get_state(self) -> np.ndarray:
        tx_type_is_A = 1.0 if self.current_tx_type == POOL_A else 0.0
        tx_type_is_B = 1.0 if self.current_tx_type == POOL_B else 0.0
        time_norm = self.time / max(1, self.max_steps - 1)
        state = [
            self.current_tx_value / max(1.0, self.max_transaction),
            tx_type_is_A,
            tx_type_is_B,
            self.available_A / self.C,
            self.committed_A / self.C,
            self.available_B / self.C,
            self.committed_B / self.C,
            time_norm,
        ]
        if self.state_feature_mode in {"pressure", "pressure_release"}:
            flush_need_A = (
                max(0.0, self.current_tx_value - self.available_A) / self.C
                if self.current_tx_type == POOL_A
                else 0.0
            )
            flush_need_B = (
                max(0.0, self.current_tx_value - self.available_B) / self.C
                if self.current_tx_type == POOL_B
                else 0.0
            )
            state.extend([
                self.committed_A / self.C,
                self.committed_B / self.C,
                self.available_A / self.C,
                self.available_B / self.C,
                flush_need_A,
                flush_need_B,
            ])
        if self.state_feature_mode == "pressure_release":
            pending_release_A = self._pending_release_total(self.pending_A)
            pending_release_B = self._pending_release_total(self.pending_B)
            next_release_A, time_to_next_release_A = self._next_release_features(self.pending_A)
            next_release_B, time_to_next_release_B = self._next_release_features(self.pending_B)
            f_norm = max(1, self.F)
            state.extend([
                pending_release_A / self.C,
                pending_release_B / self.C,
                next_release_A / self.C,
                next_release_B / self.C,
                time_to_next_release_A / f_norm,
                time_to_next_release_B / f_norm,
                (self.available_A + pending_release_A) / self.C,
                (self.available_B + pending_release_B) / self.C,
            ])
        arr = np.asarray(state, dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            raise ValueError("Non-finite state features in TwoPoolCollateralEnv.")
        return np.clip(arr, 0.0, 5.0).astype(np.float32, copy=False)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        settle_choice, flush_action = self.decode_action(action)
        return self.step_decision(settle_choice=settle_choice, flush_action=flush_action)

    def step_decision(
        self,
        settle_choice: int,
        flush_action: int,
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if int(settle_choice) not in (SETTLE_DISCARD, SETTLE_ACCEPT):
            raise ValueError(f"settle_choice must be 0 or 1, got {settle_choice}")
        if self.time >= self.max_steps:
            raise RuntimeError("Cannot call step after episode is done.")

        tx = float(self.current_tx_value)
        tx_type = int(self.current_tx_type)
        self.total_value += tx

        settled_this_step = 0.0
        dropped = False
        if int(settle_choice) == SETTLE_ACCEPT:
            if tx_type == POOL_A and self.available_A >= tx:
                self.available_A -= tx
                self.committed_A += tx
                self.settled_A_value += tx
                settled_this_step = tx
            elif tx_type == POOL_B and self.available_B >= tx:
                self.available_B -= tx
                self.committed_B += tx
                self.settled_B_value += tx
                settled_this_step = tx
            else:
                dropped = True
        else:
            dropped = True

        if settled_this_step > 0.0:
            self.settled_value += settled_this_step
        if dropped:
            self.drops += 1
            self.dropped_value += tx
            if tx_type == POOL_A:
                self.drop_A_count += 1
            else:
                self.drop_B_count += 1

        flush_pool, flush_fraction = self.decode_flush_action(flush_action)
        flush_amount = 0.0
        if flush_pool == POOL_A:
            flush_amount = max(0.0, min(self.committed_A, flush_fraction * self.committed_A))
            if flush_amount > 0.0:
                self.committed_A -= flush_amount
                self.pending_A.append(PendingRelease(self.time + self.F, flush_amount))
                self.flushes += 1
                self.flush_A_count += 1
        elif flush_pool == POOL_B:
            flush_amount = max(0.0, min(self.committed_B, flush_fraction * self.committed_B))
            if flush_amount > 0.0:
                self.committed_B -= flush_amount
                self.pending_B.append(PendingRelease(self.time + self.F, flush_amount))
                self.flushes += 1
                self.flush_B_count += 1

        reward = (
            self.money_p * settled_this_step
            - self.money_tau * float(flush_amount > 0.0)
            - self.drop_penalty * float(dropped)
        )

        self.time += 1
        done = self.time >= self.max_steps
        returned_A = 0.0
        returned_B = 0.0
        if not done:
            self.current_tx_value = float(self._tx_values[self.time])
            self.current_tx_type = int(self._tx_types[self.time])
            returned_A, returned_B = self._release_due_collateral()

        info = {
            "tx_value": tx,
            "tx_type": tx_type,
            "settle_choice": int(settle_choice),
            "flush_action": int(flush_action),
            "flush_pool": flush_pool,
            "flush_fraction": float(flush_fraction),
            "flush_amount": float(flush_amount),
            "flushes_this_step": int(flush_amount > 0.0),
            "settled_value": float(settled_this_step),
            "accepted": bool(settled_this_step > 0.0),
            "dropped": bool(dropped),
            "returned_A_next_state": float(returned_A),
            "returned_B_next_state": float(returned_B),
            "available_A": float(self.available_A),
            "available_B": float(self.available_B),
            "committed_A": float(self.committed_A),
            "committed_B": float(self.committed_B),
        }
        return self.get_state(), float(reward), bool(done), info

    def get_metrics(self) -> Dict[str, float]:
        value_accept_ratio = self.settled_value / self.total_value if self.total_value > 0.0 else 0.0
        money = self.money_p * self.settled_value - self.money_tau * self.flushes
        return {
            "settled_value": float(self.settled_value),
            "dropped_value": float(self.dropped_value),
            "total_value": float(self.total_value),
            "total_transaction_value": float(self.total_value),
            "value_accept_ratio": float(value_accept_ratio),
            "drops": float(self.drops),
            "drop_rate": float(self.drops / max(1, self.max_steps)),
            "flushes": float(self.flushes),
            "money": float(money),
            "flush_A_count": float(self.flush_A_count),
            "flush_B_count": float(self.flush_B_count),
            "drop_A_count": float(self.drop_A_count),
            "drop_B_count": float(self.drop_B_count),
            "settled_A_value": float(self.settled_A_value),
            "settled_B_value": float(self.settled_B_value),
            "available_A": float(self.available_A),
            "available_B": float(self.available_B),
            "committed_A": float(self.committed_A),
            "committed_B": float(self.committed_B),
        }
