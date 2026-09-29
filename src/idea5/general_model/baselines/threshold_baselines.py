from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from general_model.envs.general_collateral_env import (
    GeneralCollateralEnv,
    SETTLE_ACCEPT,
    SETTLE_DISCARD,
)
from general_model.utils.metrics import (
    compact_summary,
    compute_cross_regime_aggregate,
    summarize_episode_metrics,
)


ETA_GRID = [round(float(x), 2) for x in np.arange(0.05, 1.0, 0.05)]


def theoretical_eta(
    money_tau: float,
    money_p: float,
    C: float,
    T_max: float,
) -> float:
    """Compute the theoretical threshold eta for the general collateral model.

    Formula:
        beta = tau / (pC)
        eta* = sqrt((1 - T_max / C) * beta)

    The returned value is clamped to [0.01, 0.95] for simulation safety.

    Notes:
    - If T_max >= C, the theoretical expression becomes non-positive.
      In that case, this function returns the lower clamp value.
    - If money_p <= 0, the utility from settlement is invalid for this formula.
    """
    money_tau = float(money_tau)
    money_p = float(money_p)
    C = float(C)
    T_max = float(T_max)

    if C <= 0:
        raise ValueError("C must be positive.")
    if money_p <= 0:
        raise ValueError("money_p must be positive for theoretical_eta().")
    if money_tau < 0:
        raise ValueError("money_tau must be non-negative.")

    beta = money_tau / max(1e-12, money_p * C)
    eta_raw = math.sqrt(max(0.0, (1.0 - T_max / C) * beta))

    return float(min(0.95, max(0.01, eta_raw)))


class FixedThresholdPolicy:
    """Fixed-threshold A_eta policy for the general collateral model.

    Policy rule:
    1. Settle the current transaction if there is enough available collateral.
    2. After the hypothetical settlement, if committed-unflushed collateral
       reaches eta * C, flush eta * C, capped by the projected committed amount.

    This matches the intended A_eta timing:
        settle first, then check whether R >= eta C.
    """

    def __init__(self, eta: float, C: float) -> None:
        if eta <= 0:
            raise ValueError("eta must be positive.")
        if C <= 0:
            raise ValueError("C must be positive.")

        self.eta = float(eta)
        self.C = float(C)

    def decide(self, env: GeneralCollateralEnv) -> Tuple[int, float]:
        threshold = self.eta * self.C

        current_tx = float(env.current_tx)
        can_settle = float(env.available_collateral) >= current_tx
        settle_choice = SETTLE_ACCEPT if can_settle else SETTLE_DISCARD

        # Critical correction:
        # The threshold should be checked after the possible settlement.
        projected_committed = float(env.committed_unflushed)
        if can_settle:
            projected_committed += current_tx

        if projected_committed >= threshold:
            flush_amount = min(threshold, projected_committed)
        else:
            flush_amount = 0.0

        return settle_choice, float(flush_amount)


def evaluate_threshold_on_pool(
    eta: float,
    config: Dict[str, Any],
    tx_pool: np.ndarray,
    num_episodes: int,
    label: str,
) -> Dict[str, Any]:
    """Evaluate a fixed-threshold policy on one transaction pool."""

    if tx_pool.ndim != 2:
        raise ValueError(
            f"tx_pool must be 2D with shape episodes x T, got shape {tx_pool.shape}"
        )

    env_cfg = config["env"]
    reward_cfg = config["reward"]

    policy = FixedThresholdPolicy(
        eta=float(eta),
        C=float(env_cfg["C"]),
    )

    eval_episodes = min(int(num_episodes), int(tx_pool.shape[0]))
    max_steps = int(env_cfg["T"])

    if tx_pool.shape[1] < max_steps:
        raise ValueError(
            f"tx_pool has width {tx_pool.shape[1]}, but env T={max_steps}."
        )

    all_results: List[Dict[str, float]] = []

    for ep in range(eval_episodes):
        env = GeneralCollateralEnv(
            C=float(env_cfg["C"]),
            F=int(env_cfg["F"]),
            max_transaction=int(env_cfg["T_max"]),
            max_steps=max_steps,
            seed=int(config["seed"]) + ep,
            money_p=float(reward_cfg["money_p"]),
            money_tau=float(reward_cfg["money_tau"]),
            drop_penalty=float(reward_cfg.get("drop_penalty", 0.0)),
            flush_levels=int(env_cfg.get("flush_levels", 5)),
            flush_grid=str(env_cfg.get("flush_grid", "uniform")),
            state_feature_mode=str(env_cfg.get("state_feature_mode", "base")),
            mask_mode=str(env_cfg.get("mask_mode", "none")),
        )

        env.reset(tx_stream=tx_pool[ep])

        for _ in range(max_steps):
            settle_choice, flush_amount = policy.decide(env)

            _, _, done, _ = env.step_decision(
                settle_choice=settle_choice,
                flush_amount=flush_amount,
                flush_choice=None,
            )

            if done:
                break

        all_results.append(env.get_metrics())

    return {
        "label": label,
        "eta": float(eta),
        "num_episodes": eval_episodes,
        "summary": summarize_episode_metrics(all_results),
    }


def select_best_eta(
    eta_grid: Iterable[float],
    config: Dict[str, Any],
    val_pool: np.ndarray,
    num_episodes: int,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Select the eta with the highest validation money."""

    rows: List[Dict[str, Any]] = []
    best_eta: float | None = None
    best_money = -1e30

    for eta in eta_grid:
        result = evaluate_threshold_on_pool(
            eta=float(eta),
            config=config,
            tx_pool=val_pool,
            num_episodes=num_episodes,
            label="VAL",
        )

        summary = result["summary"]

        row = {
            "eta": float(eta),
            "money": float(summary["money"]["mean"]),
            "value_accept_ratio": float(summary["value_accept_ratio"]["mean"]),
            "drops": float(summary["drops"]["mean"]),
            "drop_rate": float(summary["drop_rate"]["mean"]),
            "flushes": float(summary["flushes"]["mean"]),
            "settled_value": float(summary["settled_value"]["mean"]),
        }

        # These fields exist after the environment fix.
        # Keep them optional so this file remains compatible with older metrics utilities.
        if "policy_discards" in summary:
            row["policy_discards"] = float(summary["policy_discards"]["mean"])
        if "capacity_drops" in summary:
            row["capacity_drops"] = float(summary["capacity_drops"]["mean"])
        if "policy_discard_rate" in summary:
            row["policy_discard_rate"] = float(summary["policy_discard_rate"]["mean"])
        if "capacity_drop_rate" in summary:
            row["capacity_drop_rate"] = float(summary["capacity_drop_rate"]["mean"])

        rows.append(row)

        if row["money"] > best_money:
            best_money = float(row["money"])
            best_eta = float(eta)

    if best_eta is None:
        raise RuntimeError("eta grid was empty.")

    return best_eta, rows


def evaluate_threshold_cross_regime(
    eta: float,
    config: Dict[str, Any],
    test_pools: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Evaluate one fixed-threshold eta on all test regimes."""

    test_results: Dict[str, Any] = {}

    for regime, tx_pool in test_pools.items():
        result = evaluate_threshold_on_pool(
            eta=float(eta),
            config=config,
            tx_pool=tx_pool,
            num_episodes=int(config["eval"]["num_episodes"]),
            label=regime,
        )

        result["test_regime"] = regime
        result["summary"] = compact_summary(result["summary"])
        test_results[regime] = result

    return {
        "eta": float(eta),
        "test_results": test_results,
        "aggregate": compute_cross_regime_aggregate(test_results),
    }
