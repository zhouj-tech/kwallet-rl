from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_FLUSH_LEVELS = 5
DEFAULT_FLUSH_GRID = "uniform"
FLUSH_FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
NONUNIFORM_V1_FLUSH_FRACTIONS = [
    0.0,
    0.03,
    0.05,
    0.08,
    0.10,
    0.15,
    0.20,
    0.30,
    0.45,
    0.60,
    0.80,
    1.00,
]
SETTLE_DISCARD = 0
SETTLE_ACCEPT = 1
SAFE_MASK_EPS = 1e-8


def build_flush_fractions(
    flush_levels: int,
    flush_grid: str = DEFAULT_FLUSH_GRID,
) -> List[float]:
    flush_grid = str(flush_grid)
    if flush_grid == "nonuniform_v1":
        return list(NONUNIFORM_V1_FLUSH_FRACTIONS)
    if flush_grid != "uniform":
        raise ValueError(
            "flush_grid must be one of {'uniform', 'nonuniform_v1'}, "
            f"got {flush_grid!r}."
        )
    flush_levels = int(flush_levels)
    if flush_levels < 2:
        raise ValueError("flush_levels must be at least 2.")
    return [float(i / (flush_levels - 1)) for i in range(flush_levels)]


@dataclass
class PendingFlush:
    return_time: int
    amount: float


class GeneralCollateralEnv:
    """Single-pool general collateral environment.

    This environment differs from the K-wallet setting:
    - There are no per-wallet balances.
    - The system has one collateral pool with total size C.
    - The action decides whether to settle the current transaction.
    - The action also decides how much committed-but-unflushed collateral to flush.

    Step semantics:
    - At the start of a visible state, all collateral due at the current time has already returned.
    - The agent observes this state and acts.
    - The environment settles/discards the current transaction.
    - The environment then applies the flush decision.
    - Time advances.
    - Before returning the next state, collateral due at the next time is released.

    This avoids a state-action mismatch where the agent sees pre-release collateral
    but the environment executes the action after releasing collateral.
    """

    def __init__(
        self,
        C: float = 1200.0,
        F: int = 3,
        max_transaction: int = 1000,
        max_steps: int = 1000,
        seed: int = 123,
        money_p: float = 1.0,
        money_tau: float = 100.0,
        drop_penalty: float = 0.0,
        flush_levels: int = DEFAULT_FLUSH_LEVELS,
        flush_grid: str = DEFAULT_FLUSH_GRID,
        state_feature_mode: str = "base",
        mask_mode: str = "none",
    ) -> None:
        if C <= 0:
            raise ValueError("C must be positive.")
        if F < 0:
            raise ValueError("F must be non-negative.")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        if money_p < 0:
            raise ValueError("money_p must be non-negative.")
        if money_tau < 0:
            raise ValueError("money_tau must be non-negative.")
        if drop_penalty < 0:
            raise ValueError("drop_penalty must be non-negative.")

        self.C = float(C)
        self.F = int(F)
        self.max_transaction = int(max_transaction)
        self.max_steps = int(max_steps)

        self.money_p = float(money_p)
        self.money_tau = float(money_tau)
        self.drop_penalty = float(drop_penalty)

        self.rng = np.random.default_rng(seed)

        self.flush_levels = int(flush_levels)
        self.flush_grid = str(flush_grid)
        self.flush_fractions = build_flush_fractions(
            self.flush_levels,
            self.flush_grid,
        )
        self.num_flush_choices = len(self.flush_fractions)
        self.action_size = 2 * self.num_flush_choices
        self.num_actions = self.action_size

        self.state_feature_mode = str(state_feature_mode)
        if self.state_feature_mode not in {"base", "pressure"}:
            raise ValueError(
                "state_feature_mode must be one of {'base', 'pressure'}, "
                f"got {self.state_feature_mode!r}."
            )

        self.mask_mode = str(mask_mode)
        if self.mask_mode not in {"none", "safe"}:
            raise ValueError(
                "mask_mode must be one of {'none', 'safe'}, "
                f"got {self.mask_mode!r}."
            )

        # For F=3, bucket 0 means returning in 1 step,
        # bucket 1 means returning in 2 steps, etc.
        self.pending_bucket_count = max(1, self.F)

        self._tx_stream: List[float] = []
        self.reset()

    @property
    def state_size(self) -> int:
        size = self.base_state_size
        if self.state_feature_mode == "pressure":
            size += 6
        return size

    @property
    def base_state_size(self) -> int:
        return 5 + self.pending_bucket_count

    def decode_action(self, action_id: int) -> Tuple[int, int]:
        if not 0 <= int(action_id) < self.action_size:
            raise ValueError(
                f"Action out of range for general collateral model: {action_id}"
            )
        action_id = int(action_id)
        settle_choice = action_id // self.num_flush_choices
        flush_choice = action_id % self.num_flush_choices
        return settle_choice, flush_choice

    def encode_action(self, settle_choice: int, flush_choice: int) -> int:
        if settle_choice not in (SETTLE_DISCARD, SETTLE_ACCEPT):
            raise ValueError(
                f"settle_choice must be 0 or 1, got {settle_choice}"
            )
        if not 0 <= int(flush_choice) < self.num_flush_choices:
            raise ValueError(
                f"flush_choice must be in [0, {self.num_flush_choices - 1}], got {flush_choice}"
            )
        return int(settle_choice) * self.num_flush_choices + int(flush_choice)

    def get_action_mask(self) -> Optional[Dict[str, np.ndarray]]:
        """Return optional settle/flush masks for safe action selection.

        The safe mask only removes meaningless nonzero flush choices when
        there is no committed-unflushed collateral to flush. It does not alter
        rewards or environment dynamics.
        """
        if self.mask_mode == "none":
            return None

        settle_mask = np.ones(2, dtype=np.float32)
        flush_mask = np.ones(self.num_flush_choices, dtype=np.float32)
        flushable_amount = float(self.committed_unflushed)
        if flushable_amount <= SAFE_MASK_EPS:
            flush_mask[1:] = 0.0
        return {
            "settle_mask": settle_mask,
            "flush_mask": flush_mask,
        }

    def reset(self, tx_stream: Optional[Sequence[float]] = None) -> np.ndarray:
        self.available_collateral = self.C
        self.committed_unflushed = 0.0
        self.pending_flush: List[PendingFlush] = []

        self.settled_value = 0.0
        self.total_transaction_value = 0.0

        # Total drops are split into two interpretable components.
        self.policy_discards = 0
        self.capacity_drops = 0
        self.drops = 0

        self.flushes = 0
        self.time = 0

        if tx_stream is None:
            self._tx_stream = [
                float(self.rng.integers(1, self.max_transaction + 1))
                for _ in range(self.max_steps)
            ]
        else:
            self._tx_stream = [float(x) for x in tx_stream]
            if len(self._tx_stream) < self.max_steps:
                raise ValueError(
                    f"tx_stream has {len(self._tx_stream)} steps, "
                    f"expected at least {self.max_steps}"
                )

        self.current_tx = float(self._tx_stream[0])

        # Usually no pending collateral exists at reset, but this keeps
        # the state convention explicit: visible state is already post-release.
        self._release_due_collateral()

        return self._get_state()

    def _release_due_collateral(self) -> float:
        returned = 0.0
        still_pending: List[PendingFlush] = []

        for item in self.pending_flush:
            if item.return_time <= self.time:
                returned += float(item.amount)
            else:
                still_pending.append(item)

        if returned > 0.0:
            self.available_collateral = min(
                self.C, self.available_collateral + returned
            )

        self.pending_flush = still_pending
        return float(returned)

    def _pending_buckets(self) -> List[float]:
        buckets = [0.0 for _ in range(self.pending_bucket_count)]

        for item in self.pending_flush:
            remaining = max(1, int(item.return_time - self.time))
            idx = min(self.pending_bucket_count, remaining) - 1
            buckets[idx] += float(item.amount) / self.C

        return buckets

    def _get_state(self) -> np.ndarray:
        pending_total = sum(float(item.amount) for item in self.pending_flush)
        time_norm = self.time / max(1, self.max_steps - 1)

        state = [
            self.available_collateral / self.C,
            self.committed_unflushed / self.C,
            pending_total / self.C,
            self.current_tx / max(1.0, float(self.max_transaction)),
            time_norm,
        ]
        state.extend(self._pending_buckets())

        if self.state_feature_mode == "pressure":
            available_ratio = self.available_collateral / self.C
            used_ratio = self.committed_unflushed / self.C
            pending_flush_ratio = pending_total / self.C
            incoming_ratio = self.current_tx / self.C
            pressure_ratio = 1.0 - available_ratio
            flush_need_ratio = max(
                0.0,
                float(self.current_tx) - float(self.available_collateral),
            ) / self.C
            pressure_features = np.asarray(
                [
                    available_ratio,
                    used_ratio,
                    pending_flush_ratio,
                    incoming_ratio,
                    pressure_ratio,
                    flush_need_ratio,
                ],
                dtype=np.float32,
            )
            if not np.all(np.isfinite(pressure_features)):
                raise ValueError(
                    "Non-finite pressure state features encountered in "
                    "GeneralCollateralEnv."
                )
            pressure_features = np.clip(pressure_features, 0.0, 5.0)
            state.extend(float(x) for x in pressure_features)

        return np.asarray(state, dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Step with a discrete RL action.

        Important:
        flush_fraction is applied after settlement, so a choice like
        "settle + flush 100%" flushes the committed collateral after the
        current transaction has been added.
        """
        settle_choice, flush_choice = self.decode_action(action)
        return self.step_decision(
            settle_choice=settle_choice,
            flush_choice=flush_choice,
            flush_amount=None,
        )

    def step_decision(
        self,
        settle_choice: int,
        flush_amount: Optional[float] = None,
        flush_choice: Optional[int] = None,
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Step with explicit settle and flush decisions.

        Parameters
        ----------
        settle_choice:
            0 = discard, 1 = attempt to settle.

        flush_amount:
            Optional exact amount of committed collateral to flush.
            This is mainly for threshold baselines such as A_eta.

        flush_choice:
            Optional discrete flush choice in [0, num_flush_choices - 1].
            If flush_amount is None and flush_choice is provided,
            the environment computes:
                flush_amount = flush_fractions[flush_choice] * committed_unflushed
            after the settlement decision has been applied.
        """
        if settle_choice not in (SETTLE_DISCARD, SETTLE_ACCEPT):
            raise ValueError(
                f"settle_choice must be 0 or 1, got {settle_choice}"
            )

        if flush_choice is not None:
            if not 0 <= int(flush_choice) < self.num_flush_choices:
                raise ValueError(
                    f"flush_choice must be in [0, {self.num_flush_choices - 1}], got {flush_choice}"
                )

        if self.time >= self.max_steps:
            raise RuntimeError("Cannot call step after episode is done.")

        tx = float(self.current_tx)
        self.total_transaction_value += tx

        settled_this_step = 0.0
        dropped = False
        drop_reason = "none"

        # 1. Settle or discard current transaction.
        if settle_choice == SETTLE_ACCEPT:
            if self.available_collateral >= tx:
                self.available_collateral -= tx
                self.committed_unflushed += tx
                self.settled_value += tx
                settled_this_step = tx
            else:
                self.capacity_drops += 1
                self.drops += 1
                dropped = True
                drop_reason = "insufficient_collateral"
        else:
            self.policy_discards += 1
            self.drops += 1
            dropped = True
            drop_reason = "discarded_by_policy"

        # 2. Compute flush amount after settlement.
        #    This is the key correction: RL flush fractions should refer to
        #    post-settlement committed_unflushed collateral.
        if flush_amount is None:
            if flush_choice is None:
                raw_flush_amount = 0.0
                flush_fraction = None
            else:
                flush_fraction = self.flush_fractions[int(flush_choice)]
                raw_flush_amount = flush_fraction * self.committed_unflushed
        else:
            raw_flush_amount = float(flush_amount)
            flush_fraction = None if flush_choice is None else self.flush_fractions[int(flush_choice)]

        applied_flush = max(
            0.0,
            min(float(raw_flush_amount), self.committed_unflushed),
        )

        # 3. Apply flush.
        if applied_flush > 0.0:
            self.committed_unflushed -= applied_flush

            # Current convention: a flush at time t returns at time t + F.
            # This matches many simulation-style implementations.
            # If strict paper interval (t, t+F] is required, use t + F + 1.
            self.pending_flush.append(
                PendingFlush(
                    return_time=self.time + self.F,
                    amount=applied_flush,
                )
            )
            self.flushes += 1

        reward = (
            self.money_p * settled_this_step
            - self.money_tau * float(applied_flush > 0.0)
            - self.drop_penalty * float(dropped)
        )

        # 4. Advance time.
        self.time += 1
        done = self.time >= self.max_steps

        # 5. Prepare next visible state.
        #    Release due collateral BEFORE returning next state, so the agent
        #    observes a state that is consistent with the next action execution.
        returned_next_state = 0.0
        if not done:
            self.current_tx = float(self._tx_stream[self.time])
            returned_next_state = self._release_due_collateral()

        pending_total = sum(float(item.amount) for item in self.pending_flush)

        info = {
            "tx": tx,
            "settle_choice": int(settle_choice),
            "flush_choice": None if flush_choice is None else int(flush_choice),
            "flush_fraction": flush_fraction,
            "raw_flush_amount": float(raw_flush_amount),
            "flush_amount": float(applied_flush),
            "flushes_this_step": int(applied_flush > 0.0),
            "settled_value": float(settled_this_step),
            "accepted": bool(settled_this_step > 0.0),
            "dropped": bool(dropped),
            "drop_reason": drop_reason,
            "policy_discards": int(self.policy_discards),
            "capacity_drops": int(self.capacity_drops),
            "returned_collateral_next_state": float(returned_next_state),
            "available_collateral": float(self.available_collateral),
            "committed_unflushed": float(self.committed_unflushed),
            "pending_flush": float(pending_total),
        }

        return self._get_state(), float(reward), bool(done), info

    def get_metrics(self) -> Dict[str, float]:
        total_value = float(self.total_transaction_value)
        value_accept_ratio = (
            self.settled_value / total_value if total_value > 0.0 else 0.0
        )

        money = self.money_p * self.settled_value - self.money_tau * self.flushes

        return {
            "settled_value": float(self.settled_value),
            "total_transaction_value": float(total_value),
            "value_accept_ratio": float(value_accept_ratio),

            # Main drop metrics.
            "drops": float(self.drops),
            "drop_rate": float(self.drops / max(1, self.max_steps)),

            # More interpretable drop decomposition.
            "policy_discards": float(self.policy_discards),
            "policy_discard_rate": float(self.policy_discards / max(1, self.max_steps)),
            "capacity_drops": float(self.capacity_drops),
            "capacity_drop_rate": float(self.capacity_drops / max(1, self.max_steps)),

            "flushes": float(self.flushes),
            "money": float(money),

            "available_collateral": float(self.available_collateral),
            "committed_unflushed": float(self.committed_unflushed),
            "pending_flush": float(
                sum(float(item.amount) for item in self.pending_flush)
            ),
        }
