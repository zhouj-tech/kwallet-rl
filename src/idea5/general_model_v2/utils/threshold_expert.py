from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from general_model_v2.envs.two_pool_collateral_env import (
    POOL_A,
    SETTLE_ACCEPT,
    SETTLE_DISCARD,
    TwoPoolCollateralEnv,
)
from general_model_v2.utils.metrics import (
    compact_summary,
    compute_cross_regime_aggregate,
    summarize_episode_metrics,
)


ETA_GRID = [round(float(x), 2) for x in np.arange(0.05, 1.0, 0.05)]
THRESHOLD_MODES = {"pool_specific", "global"}


def get_threshold_mode(config: Dict[str, Any] | None = None) -> str:
    if config is None:
        return "pool_specific"
    mode = str(config.get("threshold_mode", "pool_specific"))
    if mode == "grid":
        mode = "pool_specific"
    if mode not in THRESHOLD_MODES:
        raise ValueError("threshold_mode must be 'pool_specific' or 'global'.")
    return mode


def make_threshold_env(config: Dict[str, Any]) -> TwoPoolCollateralEnv:
    env_cfg = config["env"]
    reward_cfg = config["reward"]
    return TwoPoolCollateralEnv(
        C=float(env_cfg["C"]),
        F=int(env_cfg["F"]),
        max_transaction=float(env_cfg["T_max"]),
        max_steps=int(env_cfg["T"]),
        seed=int(config["seed"]),
        money_p=float(reward_cfg["money_p"]),
        money_tau=float(reward_cfg["money_tau"]),
        drop_penalty=float(reward_cfg["drop_penalty"]),
        flush_levels=int(env_cfg.get("flush_levels", 17)),
        flush_grid=str(env_cfg.get("flush_grid", "uniform")),
        state_feature_mode=str(env_cfg.get("state_feature_mode", "base")),
        mask_mode=str(env_cfg.get("mask_mode", "none")),
    )


def threshold_decision(
    env: TwoPoolCollateralEnv,
    eta: float,
    threshold_mode: str = "pool_specific",
) -> Tuple[int, int]:
    tx = float(env.current_tx_value)
    if env.current_tx_type == POOL_A:
        can_settle = env.available_A >= tx
    else:
        can_settle = env.available_B >= tx
    settle_choice = SETTLE_ACCEPT if can_settle else SETTLE_DISCARD
    pressure_A = float(env.committed_A / env.C)
    pressure_B = float(env.committed_B / env.C)
    if threshold_mode == "pool_specific":
        trigger_pressure = max(pressure_A, pressure_B)
    elif threshold_mode == "global":
        trigger_pressure = float((env.committed_A + env.committed_B) / (2.0 * env.C))
    else:
        raise ValueError("threshold_mode must be 'pool_specific' or 'global'.")
    if trigger_pressure >= float(eta):
        if pressure_A >= pressure_B and env.committed_A > 0.0:
            return settle_choice, env.flush_levels - 1
        if env.committed_B > 0.0:
            return settle_choice, env.num_flush_choices - 1
    return settle_choice, 0


def evaluate_eta_on_pool(
    eta: float,
    config: Dict[str, Any],
    pool: Dict[str, np.ndarray],
    num_episodes: int,
    label: str,
) -> Dict[str, Any]:
    values = pool["values"]
    types = pool["types"]
    eval_episodes = min(int(num_episodes), int(values.shape[0]))
    all_results: List[Dict[str, Any]] = []
    threshold_mode = get_threshold_mode(config)
    for ep in range(eval_episodes):
        env = make_threshold_env(config)
        env.reset(tx_values=values[ep], tx_types=types[ep])
        for _ in range(int(config["eval"]["max_steps"])):
            settle_choice, flush_action = threshold_decision(env, eta, threshold_mode)
            _, _, done, _ = env.step_decision(settle_choice, flush_action)
            if done:
                break
        all_results.append(env.get_metrics())
    return {
        "label": label,
        "eta": float(eta),
        "threshold_mode": threshold_mode,
        "num_episodes": eval_episodes,
        "summary": summarize_episode_metrics(all_results),
    }


def select_best_eta(
    config: Dict[str, Any],
    val_pool: Dict[str, np.ndarray],
    num_episodes: int | None = None,
) -> Tuple[float, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    best_eta = ETA_GRID[0]
    best_money = -1e30
    if num_episodes is None:
        num_episodes = int(config["eval"]["num_episodes"])
    threshold_mode = get_threshold_mode(config)
    for eta in ETA_GRID:
        result = evaluate_eta_on_pool(
            eta=eta,
            config=config,
            pool=val_pool,
            num_episodes=int(num_episodes),
            label="VAL",
        )
        summary = result["summary"]
        row = {
            "threshold_mode": threshold_mode,
            "eta": float(eta),
            "val_money": float(summary["money"]["mean"]),
            "val_value_accept_ratio": float(summary["value_accept_ratio"]["mean"]),
            "val_drops": float(summary["drops"]["mean"]),
            "val_flushes": float(summary["flushes"]["mean"]),
        }
        rows.append(row)
        if row["val_money"] > best_money:
            best_money = row["val_money"]
            best_eta = float(eta)
    return best_eta, rows


def evaluate_cross_regime_threshold(
    eta: float,
    config: Dict[str, Any],
    test_pools: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, Any]:
    test_results: Dict[str, Any] = {}
    for regime, pool in test_pools.items():
        print(f"[Eval ] {regime}")
        result = evaluate_eta_on_pool(
            eta=eta,
            config=config,
            pool=pool,
            num_episodes=int(config["eval"]["num_episodes"]),
            label=regime,
        )
        result["test_regime"] = regime
        result["summary"] = compact_summary(result["summary"])
        test_results[regime] = result
    return {
        "eta": float(eta),
        "threshold_mode": get_threshold_mode(config),
        "test_results": test_results,
        "aggregate": compute_cross_regime_aggregate(test_results),
    }
