# =========================================================
# 5/15 dual critic
# Dual-Branch Factorized Actor-Critic / PPO for K-Wallet RL
#
# Independent benchmark script.
#
# Does NOT modify:
# - KWalletEnv
# - reward function
# - action semantics
# - data pool protocol
# - existing DQN benchmark
# - existing factorized AC benchmark
#
# Core idea:
# Factorized AC splits action into settle / flush heads.
# Dual-Branch Factorized AC further splits state reasoning into:
#   1. Capacity branch: immediate wallet feasibility
#   2. Risk/context branch: future pressure / temporal risk
# A learned gate fuses both branches.
# =========================================================

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/kwallet_cache")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/kwallet_matplotlib_cache")
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT_FOR_IMPORT = THIS_FILE.parents[4]
IDEA3_CONTEXT_DIR = PROJECT_ROOT_FOR_IMPORT / "src" / "idea3" / "context_attention"
if str(IDEA3_CONTEXT_DIR) not in sys.path:
    sys.path.insert(0, str(IDEA3_CONTEXT_DIR))


# =========================================================
# Import shared environment / data protocol from existing benchmark
# =========================================================

try:
    from kwallet_ctx_attn_fair_benchmark import (
        DATA_POOL_DIR,
        IDEA3_DIR,
        PROJECT_ROOT,
        DEFAULT_MIX_EQ_MASTER,
        DEFAULT_MIX_EQ_VAL,
        DEFAULT_STATIC_EVAL_FILES,
        KWalletEnv,
        build_pool_fingerprints,
        compute_cross_regime_aggregate,
        count_parameters,
        load_tx_pool,
        train_pool_file_for_regime,
        verify_data_integrity,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from kwallet_ctx_attn_fair_benchmark import (  # type: ignore
        DATA_POOL_DIR,
        IDEA3_DIR,
        PROJECT_ROOT,
        DEFAULT_MIX_EQ_MASTER,
        DEFAULT_MIX_EQ_VAL,
        DEFAULT_STATIC_EVAL_FILES,
        KWalletEnv,
        build_pool_fingerprints,
        compute_cross_regime_aggregate,
        count_parameters,
        load_tx_pool,
        train_pool_file_for_regime,
        verify_data_integrity,
    )


# =========================================================
# Global config
# =========================================================

RESULT_ROOT = PROJECT_ROOT / "src" / "idea4" / "ac" / "results" / "dual_critic"
LOG_EVERY_N = 25

ORIGINAL_REWARD_FORMULA = "reward_t = env_reward_t"
RAW_MONEY_REWARD_FORMULA = "reward_t = money_p * settled_t - money_tau * flush_indicator_t"
MONEY_NORMALIZED_REWARD_FORMULA = "reward_t = settled_t / settled_scale - tau_scaled * flush_indicator_t"
HYBRID_MONEY_REWARD_FORMULA = (
    "reward_t = env_reward_t + hybrid_alpha * "
    "(settled_t / settled_scale - tau_scaled * flush_indicator_t)"
)
EVALUATION_MONEY_FORMULA = "eval_money = money_p * settled - money_tau * flushes"


CONFIG: Dict[str, Any] = {
    "seed": 123,
    "model_mode": "dual_branch_factorized_ac",
    "debug_mode": False,
    "save_mode": "full",
    "env": {
        "C": 1200.0,
        "k": 3,
        "T": 1000,
        "F": 3,
        "enable_shaping": False,
    },
    "data": {
        "train_regime": "MIX12_EQ",
        "train_pool_file": DEFAULT_MIX_EQ_MASTER,
        "val_pool_file": DEFAULT_MIX_EQ_VAL,
        "test_pool_files": DEFAULT_STATIC_EVAL_FILES,
    },
    "train": {
        "episodes": 1000,
        "train_use_episodes": 3000,
        "max_steps": 1000,
        "device": "cpu",
        "learning_rate": 3e-4,
        "gamma": 0.98,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "update_epochs": 4,
        "minibatch_size": 256,
        "value_coef": 0.5,
        "entropy_coef_start": 0.03,
        "entropy_coef_end": 0.003,
        "max_grad_norm": 1.0,
        "val_every": 50,
        "val_num_episodes": 200,
        "val_metric": "value_accept_ratio",
        "use_best_model_for_final_eval": True,
        "hidden_size": 128,
        "branch_hidden_size": 128,
        "risk_feature_dim": 10,
    },
    "eval": {
        "num_episodes": 200,
        "max_steps": 1000,
    },
    "attention_context": {
        "window_size": 50,
    },
    "gate": {
        "gate_temperature": 2.0,
        "gate_min": 0.1,
        "gate_max": 0.9,
        "gate_target": 0.7,
        "gate_reg_coef": 0.01,
    },
    "aux_risk": {
        "aux_risk_coef": 0.01,
        "aux_risk_window": 20,
    },
    "residual_risk": {
        "risk_residual_scale": 0.2,
        "value_residual_scale": 0.1,
    },
    "dual_critic": {
        # If enabled, the capacity and risk value heads are each trained
        # against the PPO return in addition to the gated final critic.
        "enable_dual_critic": False,
        "dual_critic_coef": 0.1,
        "enable_value_diagnostics": True,
    },
    "reward": {
        "alpha_drop": 0.02,
        "beta_flush": 0.01,
        "reward_mode": "original",

        # Raw evaluation money:
        # eval_money = money_p * settled - money_tau * flushes
        "money_p": 1.0,
        "money_tau": 10.0,

        # PPO-friendly scaled money reward:
        # money_normalized = settled_t / settled_scale - tau_scaled * flush_indicator_t
        "settled_scale": 50.0,
        "tau_scaled": 0.2,

        # Hybrid reward:
        # env_reward_t + hybrid_alpha * money_normalized
        "hybrid_alpha": 0.1,

        "reward_formula": ORIGINAL_REWARD_FORMULA,
        "objective_label": "original_reward",
        "training_objective": "original_reward",
    },
    "plot": {
        "window": 50,
    },
    "output": {
        "output_dir": str(RESULT_ROOT),
    },
}


# =========================================================
# Basic utilities
# =========================================================

def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def dual_env_mode(model_mode: str) -> str:
    if model_mode in {
        "dual_branch_factorized_ac",
        "dual_branch_factorized_ac_gate_balanced",
        "dual_branch_factorized_ac_gate_regularized",
        "dual_branch_factorized_ac_auxrisk",
        "dual_branch_capacity_only",
        "dual_branch_residual_risk",
    }:
        return "baseline"
    raise ValueError(f"Unsupported model_mode={model_mode}")


def build_run_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def format_c_value(c_value: float) -> str:
    return str(int(c_value)) if float(c_value).is_integer() else str(c_value)


def format_reward_value(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def update_reward_metadata(config: Dict[str, Any]) -> None:
    reward_cfg = config["reward"]
    mode = reward_cfg["reward_mode"]

    if mode == "original":
        reward_cfg["reward_formula"] = ORIGINAL_REWARD_FORMULA
        reward_cfg["objective_label"] = "original_reward"
        reward_cfg["training_objective"] = "original_reward"

    elif mode == "money":
        reward_cfg["reward_formula"] = RAW_MONEY_REWARD_FORMULA
        reward_cfg["objective_label"] = (
            f"money_p{format_reward_value(reward_cfg['money_p'])}_"
            f"tau{format_reward_value(reward_cfg['money_tau'])}"
        )
        reward_cfg["training_objective"] = "money_reward"

    elif mode == "money_normalized":
        reward_cfg["reward_formula"] = MONEY_NORMALIZED_REWARD_FORMULA
        reward_cfg["objective_label"] = (
            f"moneyN_scale{format_reward_value(reward_cfg['settled_scale'])}_"
            f"tauS{format_reward_value(reward_cfg['tau_scaled'])}"
        )
        reward_cfg["training_objective"] = "money_normalized_reward"

    elif mode == "hybrid_money":
        reward_cfg["reward_formula"] = HYBRID_MONEY_REWARD_FORMULA
        reward_cfg["objective_label"] = (
            f"hybridMoney_alpha{format_reward_value(reward_cfg['hybrid_alpha'])}_"
            f"scale{format_reward_value(reward_cfg['settled_scale'])}_"
            f"tauS{format_reward_value(reward_cfg['tau_scaled'])}"
        )
        reward_cfg["training_objective"] = "hybrid_money_reward"

    else:
        raise ValueError(f"Unsupported reward_mode={mode}")

    config.update(
        {
            "reward_mode": reward_cfg["reward_mode"],
            "objective_label": reward_cfg["objective_label"],
            "money_p": float(reward_cfg["money_p"]),
            "money_tau": float(reward_cfg["money_tau"]),
            "settled_scale": float(reward_cfg["settled_scale"]),
            "tau_scaled": float(reward_cfg["tau_scaled"]),
            "hybrid_alpha": float(reward_cfg["hybrid_alpha"]),
            "reward_formula": reward_cfg["reward_formula"],
            "training_objective": reward_cfg["training_objective"],
        }
    )

def reward_metadata(config: Dict[str, Any]) -> Dict[str, Any]:
    reward_cfg = config["reward"]
    return {
        "reward_mode": reward_cfg["reward_mode"],
        "objective_label": reward_cfg["objective_label"],
        "money_p": float(reward_cfg["money_p"]),
        "money_tau": float(reward_cfg["money_tau"]),
        "settled_scale": float(reward_cfg["settled_scale"]),
        "tau_scaled": float(reward_cfg["tau_scaled"]),
        "hybrid_alpha": float(reward_cfg["hybrid_alpha"]),
        "reward_formula": reward_cfg["reward_formula"],
        "training_objective": reward_cfg["training_objective"],
    }

def select_training_reward(
    env_reward: float,
    info: Dict[str, Any],
    config: Dict[str, Any],
) -> Tuple[float, float, float, float]:
    reward_cfg = config["reward"]

    if "settled_value" not in info:
        info["settled_value"] = float(info.get("tx", 0.0) if info.get("accepted", False) else 0.0)

    if "flushes_this_step" not in info:
        flush_choice = info.get("flush_choice", None)
        info["flushes_this_step"] = 1 if flush_choice is not None and flush_choice < int(config["env"]["k"]) else 0

    settled_value = float(info["settled_value"])
    flushes_this_step = float(info["flushes_this_step"])

    # Raw money is always recorded for diagnostics and final evaluation consistency.
    money_reward = (
        float(reward_cfg["money_p"]) * settled_value
        - float(reward_cfg["money_tau"]) * flushes_this_step
    )

    # PPO-friendly scaled money reward.
    normalized_money_reward = (
        settled_value / float(reward_cfg["settled_scale"])
        - float(reward_cfg["tau_scaled"]) * flushes_this_step
    )

    mode = reward_cfg["reward_mode"]

    if mode == "original":
        selected_reward = float(env_reward)

    elif mode == "money":
        selected_reward = money_reward

    elif mode == "money_normalized":
        selected_reward = normalized_money_reward

    elif mode == "hybrid_money":
        selected_reward = (
            float(env_reward)
            + float(reward_cfg["hybrid_alpha"]) * normalized_money_reward
        )

    else:
        raise ValueError(f"Unsupported reward_mode={mode}")

    return (
        float(selected_reward),
        float(money_reward),
        float(settled_value),
        float(flushes_this_step),
    )

def add_eval_money_metrics(metrics: Dict[str, float], config: Dict[str, Any]) -> None:
    reward_cfg = config["reward"]
    metrics["eval_money_p"] = float(reward_cfg["money_p"])
    metrics["eval_money_tau"] = float(reward_cfg["money_tau"])
    metrics["eval_money"] = (
        float(reward_cfg["money_p"]) * float(metrics.get("settled", 0.0))
        - float(reward_cfg["money_tau"]) * float(metrics.get("flushes", 0.0))
    )


def add_reward_summary_metadata(summary: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    summary["training_objective"] = {"value": config["reward"]["training_objective"]}
    summary["evaluation_money_formula"] = {"value": EVALUATION_MONEY_FORMULA}
    summary["money_method"] = {"value": "true_money_via_settled"}
    return summary


def build_scenario_name(config: Dict[str, Any]) -> str:
    env = config["env"]
    c_value = format_c_value(float(env["C"]))

    scenario = (
        f"{config['model_mode']}_train{config['data']['train_regime']}_"
        f"C{c_value}_k{env['k']}_T{env['T']}_F{env['F']}_seed{config['seed']}"
    )

    mode = config["reward"]["reward_mode"]

    if mode == "money":
        scenario += (
            f"_rewardMONEY_p{format_reward_value(config['reward']['money_p'])}"
            f"_tau{format_reward_value(config['reward']['money_tau'])}"
        )

    elif mode == "money_normalized":
        scenario += (
            f"_rewardMN_scale{format_reward_value(config['reward']['settled_scale'])}"
            f"_tauS{format_reward_value(config['reward']['tau_scaled'])}"
        )

    elif mode == "hybrid_money":
        scenario += (
            f"_rewardHYBRID_alpha{format_reward_value(config['reward']['hybrid_alpha'])}"
            f"_scale{format_reward_value(config['reward']['settled_scale'])}"
            f"_tauS{format_reward_value(config['reward']['tau_scaled'])}"
        )

    dual_critic_cfg = config.get("dual_critic", {})
    if dual_critic_cfg.get("enable_dual_critic", False):
        scenario += f"_dualCritic_coef{format_reward_value(dual_critic_cfg.get('dual_critic_coef', 0.1))}"

    return scenario

def build_title_tag(config: Dict[str, Any], run_stamp: str) -> str:
    env = config["env"]
    c_value = format_c_value(float(env["C"]))

    return (
        f"{run_stamp} | mode={config['model_mode']} | "
        f"train={config['data']['train_regime']} | "
        f"C={c_value} k={env['k']} T={env['T']} F={env['F']}"
    )


DROP_FLAG_KEYS = (
    "dropped",
    "drop",
    "is_drop",
    "tx_dropped",
    "transaction_dropped",
)


def infer_drop_flag(info: Dict[str, Any]) -> Tuple[float, str]:
    for key in DROP_FLAG_KEYS:
        if key in info:
            return float(bool(info[key])), f"explicit info drop key: {key}"

    return float(not bool(info.get("accepted", False))), "inferred from acceptance signal"


def build_future_drop_risk_labels(
    drop_flags: List[float],
    aux_risk_window: int,
) -> List[float]:
    labels: List[float] = []
    n = len(drop_flags)
    window = max(0, int(aux_risk_window))

    for t in range(n):
        start = t + 1
        end = min(n, t + 1 + window)
        future_window = drop_flags[start:end]
        labels.append(float(any(flag > 0.0 for flag in future_window)))

    return labels


def build_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    run_stamp = build_run_stamp()
    scenario = build_scenario_name(config)

    result_root = Path(config["output"]["output_dir"]).expanduser().resolve()

    result_run_dir = result_root / "runs" / scenario / run_stamp
    checkpoint_run_dir = result_root / "checkpoints" / scenario / run_stamp
    aggregate_dir = result_root / "aggregates"
    plot_dir = result_root / "plots" / scenario / run_stamp

    return {
        "project_root": str(PROJECT_ROOT),
        "data_pool_dir": str(DATA_POOL_DIR),
        "result_root": str(result_root),
        "scenario": scenario,
        "run_stamp": run_stamp,
        "title_tag": build_title_tag(config, run_stamp),
        "train_regime": config["data"]["train_regime"],
        "train_pool_path": str(DATA_POOL_DIR / config["data"]["train_pool_file"]),
        "val_pool_path": str(DATA_POOL_DIR / config["data"]["val_pool_file"]),
        "test_pool_paths": {
            regime: str(DATA_POOL_DIR / file_name)
            for regime, file_name in config["data"]["test_pool_files"].items()
        },
        "result_run_dir": str(result_run_dir),
        "checkpoint_run_dir": str(checkpoint_run_dir),
        "aggregate_dir": str(aggregate_dir),
        "plot_dir": str(plot_dir),
        "run_info_path": str(result_run_dir / "run_info.json"),
        "run_config_path": str(result_run_dir / "run_config.json"),
        "results_json_path": str(result_run_dir / "cross_regime_results.json"),
        "summary_txt_path": str(result_run_dir / "summary_table.txt"),
        "training_history_path": str(result_run_dir / "training_history.json"),
        "validation_history_path": str(result_run_dir / "validation_history.json"),
        "eval_plot_path": str(plot_dir / "cross_regime_plot.png"),
        "training_plot_path": str(plot_dir / "training_curve.png"),
        "best_model_path": str(checkpoint_run_dir / "best_model.pth"),
        "last_model_path": str(checkpoint_run_dir / "last_model.pth"),
    }


def ensure_dirs(paths: Dict[str, Any], config: Dict[str, Any]) -> None:
    if config["debug_mode"] or config["save_mode"] == "none":
        return

    for key in ["result_run_dir", "checkpoint_run_dir", "aggregate_dir", "plot_dir"]:
        os.makedirs(paths[key], exist_ok=True)


def save_json(payload: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def make_env(config: Dict[str, Any], max_steps: int) -> KWalletEnv:
    env_cfg = config["env"]
    reward_cfg = config["reward"]

    return KWalletEnv(
        C=env_cfg["C"],
        k=env_cfg["k"],
        F=env_cfg["F"],
        max_transaction=env_cfg["T"],
        max_steps=max_steps,
        seed=config["seed"],
        model_mode=dual_env_mode(config["model_mode"]),
        attention_window_size=config["attention_context"]["window_size"],
        enable_shaping=env_cfg["enable_shaping"],
        alpha_drop=reward_cfg["alpha_drop"],
        beta_flush=reward_cfg["beta_flush"],
    )


# =========================================================
# Dual-Branch Actor-Critic model
# =========================================================

class DualBranchFactorizedActorCritic(nn.Module):
    """
    Dual-Branch Factorized Actor-Critic.

    Existing factorized AC:
        full state -> shared encoder -> settle head + flush head + value head

    This model:
        full state -> capacity branch -> settle/flush/value proposals
        full state -> risk/context branch -> settle/flush/value proposals
        full state -> gate g in [0,1]
        final output = gated fusion of both branches
    """

    def __init__(
        self,
        state_size: int,
        base_state_size: int,
        k: int,
        hidden_size: int = 128,
        branch_hidden_size: int = 128,
        risk_feature_dim: int = 10,
        gate_mode: str = "dual_branch_factorized_ac",
        gate_temperature: float = 2.0,
        gate_min: float = 0.1,
        gate_max: float = 0.9,
        risk_residual_scale: float = 0.2,
        value_residual_scale: float = 0.1,
    ):
        super().__init__()

        self.state_size = int(state_size)
        self.base_state_size = int(base_state_size)
        self.k = int(k)
        self.risk_feature_dim = int(risk_feature_dim)
        self.gate_mode = str(gate_mode)
        self.gate_temperature = float(gate_temperature)
        self.gate_min = float(gate_min)
        self.gate_max = float(gate_max)
        self.risk_residual_scale = float(risk_residual_scale)
        self.value_residual_scale = float(value_residual_scale)
        self.use_aux_risk = self.gate_mode == "dual_branch_factorized_ac_auxrisk"

        # Capacity branch uses the full raw state.
        # It keeps wallet balance / usable / freeze / current tx information intact.
        self.capacity_encoder = nn.Sequential(
            nn.Linear(self.state_size, branch_hidden_size),
            nn.ReLU(),
            nn.Linear(branch_hidden_size, branch_hidden_size),
            nn.ReLU(),
        )

        # Risk branch uses engineered context/risk summaries from the same state.
        # This makes it different from simply duplicating the capacity branch.
        self.risk_encoder = nn.Sequential(
            nn.Linear(self.risk_feature_dim, branch_hidden_size),
            nn.ReLU(),
            nn.Linear(branch_hidden_size, branch_hidden_size),
            nn.ReLU(),
        )

        # Gate sees the full state and decides how much to trust capacity branch.
        self.gate_net = nn.Sequential(
            nn.Linear(self.state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        # Capacity branch proposals.
        self.capacity_settle_head = nn.Linear(branch_hidden_size, self.k + 1)
        self.capacity_flush_head = nn.Linear(branch_hidden_size, self.k + 1)
        self.capacity_value_head = nn.Linear(branch_hidden_size, 1)

        # Risk branch proposals.
        self.risk_settle_head = nn.Linear(branch_hidden_size, self.k + 1)
        self.risk_flush_head = nn.Linear(branch_hidden_size, self.k + 1)
        self.risk_value_head = nn.Linear(branch_hidden_size, 1)
        self.risk_aux_head = (
            nn.Linear(branch_hidden_size, 1)
            if self.use_aux_risk
            else None
        )

    def compute_gate(self, gate_logits: torch.Tensor) -> torch.Tensor:
        if self.gate_mode == "dual_branch_factorized_ac":
            return torch.sigmoid(gate_logits)

        if self.gate_mode in {
            "dual_branch_factorized_ac_gate_balanced",
            "dual_branch_factorized_ac_gate_regularized",
            "dual_branch_factorized_ac_auxrisk",
        "dual_branch_capacity_only",
        "dual_branch_residual_risk",
        }:
            gate_raw = torch.sigmoid(gate_logits / self.gate_temperature)
            return self.gate_min + (self.gate_max - self.gate_min) * gate_raw

        raise ValueError(f"Unsupported gate_mode={self.gate_mode}")

    def build_risk_features(self, state: torch.Tensor) -> torch.Tensor:
        """
        Build compact risk/context features from baseline state.

        This is deterministic feature engineering. It does not change env state,
        reward, action semantics, or evaluation protocol.

        Assumption:
        baseline state is usually length 3k + 2.
        First k entries often contain wallet-related continuous capacity values.
        Last two entries are treated as transaction/time-like context when present.
        """

        batch_size = state.shape[0]
        k = self.k

        if state.shape[1] >= k:
            wallet_part = state[:, :k]
        else:
            pad = torch.zeros(
                batch_size,
                k - state.shape[1],
                device=state.device,
                dtype=state.dtype,
            )
            wallet_part = torch.cat([state, pad], dim=1)

        wallet_mean = wallet_part.mean(dim=1, keepdim=True)
        wallet_min = wallet_part.min(dim=1, keepdim=True).values
        wallet_max = wallet_part.max(dim=1, keepdim=True).values
        wallet_std = wallet_part.std(dim=1, keepdim=True, unbiased=False)

        if state.shape[1] >= 2:
            last_two = state[:, -2:]
        else:
            last_two = torch.zeros(
                batch_size,
                2,
                device=state.device,
                dtype=state.dtype,
            )

        current_like = last_two[:, :1]
        time_like = last_two[:, 1:2]

        eps = 1e-6
        pressure_vs_mean = current_like / (wallet_mean.abs() + eps)
        pressure_vs_max = current_like / (wallet_max.abs() + eps)

        if state.shape[1] >= 2 * k:
            mid_part = state[:, k:2 * k]
            mid_mean = mid_part.mean(dim=1, keepdim=True)
        else:
            mid_mean = torch.zeros(batch_size, 1, device=state.device, dtype=state.dtype)

        if state.shape[1] >= 3 * k:
            third_part = state[:, 2 * k:3 * k]
            third_mean = third_part.mean(dim=1, keepdim=True)
        else:
            third_mean = torch.zeros(batch_size, 1, device=state.device, dtype=state.dtype)

        risk = torch.cat(
            [
                current_like,
                time_like,
                wallet_mean,
                wallet_min,
                wallet_max,
                wallet_std,
                pressure_vs_mean,
                pressure_vs_max,
                mid_mean,
                third_mean,
            ],
            dim=1,
        )

        if risk.shape[1] != self.risk_feature_dim:
            raise RuntimeError(
                f"risk feature dim mismatch: got {risk.shape[1]}, "
                f"expected {self.risk_feature_dim}"
            )

        return risk

    def critic_components(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return branch-specific critic values and their gated fusion.

        This is used for dual-critic training and value diagnostics.
        It does not change the actor logits or action semantics.
        """
        h_capacity = self.capacity_encoder(state)
        value_capacity = self.capacity_value_head(h_capacity).squeeze(-1)

        risk_features = self.build_risk_features(state)
        h_risk = self.risk_encoder(risk_features)
        value_risk = self.risk_value_head(h_risk).squeeze(-1)

        if self.gate_mode == "dual_branch_capacity_only":
            gates = torch.ones_like(value_capacity)
            value_final = value_capacity

        elif self.gate_mode == "dual_branch_residual_risk":
            value_delta = torch.tanh(value_risk)
            value_final = value_capacity + self.value_residual_scale * value_delta
            gates = torch.full_like(
                value_capacity,
                1.0 - min(max(self.risk_residual_scale, 0.0), 1.0),
            )

        else:
            gate_logits = self.gate_net(state)
            gates = self.compute_gate(gate_logits).squeeze(-1)
            value_final = gates * value_capacity + (1.0 - gates) * value_risk

        return value_capacity, value_risk, value_final, gates

    def forward(
        self,
        state: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        # ---------------------------------------------------------
        # 1. Compute capacity branch outputs first
        # ---------------------------------------------------------
        h_capacity = self.capacity_encoder(state)

        settle_capacity = self.capacity_settle_head(h_capacity)
        flush_capacity = self.capacity_flush_head(h_capacity)
        value_capacity = self.capacity_value_head(h_capacity).squeeze(-1)

        # ---------------------------------------------------------
        # 2. Compute risk / future-pressure branch outputs
        # ---------------------------------------------------------
        risk_features = self.build_risk_features(state)
        h_risk = self.risk_encoder(risk_features)

        settle_risk = self.risk_settle_head(h_risk)
        flush_risk = self.risk_flush_head(h_risk)
        value_risk = self.risk_value_head(h_risk).squeeze(-1)

        risk_logit = (
            self.risk_aux_head(h_risk).squeeze(-1)
            if self.risk_aux_head is not None
            else None
        )

        # ---------------------------------------------------------
        # Variant A: capacity-only ablation
        # ---------------------------------------------------------
        if self.gate_mode == "dual_branch_capacity_only":
            gate_proxy = torch.ones_like(value_capacity)
            return (
                settle_capacity,
                flush_capacity,
                value_capacity,
                gate_proxy,
                risk_logit,
            )

        # ---------------------------------------------------------
        # Variant B: capacity-dominant future-pressure residual correction
        # ---------------------------------------------------------
        if self.gate_mode == "dual_branch_residual_risk":
            settle_delta = torch.tanh(settle_risk)
            flush_delta = torch.tanh(flush_risk)
            value_delta = torch.tanh(value_risk)

            settle_logits = (
                settle_capacity
                + self.risk_residual_scale * settle_delta
            )
            flush_logits = (
                flush_capacity
                + self.risk_residual_scale * flush_delta
            )
            value = (
                value_capacity
                + self.value_residual_scale * value_delta
            )

            # Diagnostic proxy, not a learned gate.
            # Example: residual_scale=0.2 means approximately 80% capacity-dominant.
            gate_proxy = torch.full_like(
                value_capacity,
                1.0 - min(max(self.risk_residual_scale, 0.0), 1.0),
            )

            return settle_logits, flush_logits, value, gate_proxy, risk_logit

        # ---------------------------------------------------------
        # Original gate-based variants
        # ---------------------------------------------------------
        gate_logits = self.gate_net(state)
        g = self.compute_gate(gate_logits)

        settle_logits = g * settle_capacity + (1.0 - g) * settle_risk
        flush_logits = g * flush_capacity + (1.0 - g) * flush_risk

        value = (
            g.squeeze(-1) * value_capacity
            + (1.0 - g.squeeze(-1)) * value_risk
        )

        return settle_logits, flush_logits, value, g.squeeze(-1), risk_logit


# =========================================================
# PPO Agent
# =========================================================

class DualBranchPPOAgent:
    def __init__(
        self,
        config: Dict[str, Any],
        state_size: int,
        base_state_size: int,
        k: int,
    ):
        train_cfg = config["train"]

        self.config = config
        self.k = int(k)
        self.device = torch.device(train_cfg["device"])
        gate_cfg = config["gate"]
        self.gate_target = float(gate_cfg["gate_target"])
        self.gate_reg_coef = float(gate_cfg["gate_reg_coef"])
        self.use_gate_regularization = (
            config["model_mode"] == "dual_branch_factorized_ac_gate_regularized"
        )
        aux_risk_cfg = config["aux_risk"]
        self.use_aux_risk = config["model_mode"] == "dual_branch_factorized_ac_auxrisk"
        self.aux_risk_coef = float(aux_risk_cfg["aux_risk_coef"])

        dual_critic_cfg = config.get("dual_critic", {})
        self.enable_dual_critic = bool(dual_critic_cfg.get("enable_dual_critic", False))
        self.dual_critic_coef = float(dual_critic_cfg.get("dual_critic_coef", 0.1))
        self.enable_value_diagnostics = bool(dual_critic_cfg.get("enable_value_diagnostics", True))

        residual_cfg = config.get("residual_risk", {})

        self.model = DualBranchFactorizedActorCritic(
            state_size=state_size,
            base_state_size=base_state_size,
            k=k,
            hidden_size=int(train_cfg["hidden_size"]),
            branch_hidden_size=int(train_cfg["branch_hidden_size"]),
            risk_feature_dim=int(train_cfg["risk_feature_dim"]),
            gate_mode=str(config["model_mode"]),
            gate_temperature=float(gate_cfg["gate_temperature"]),
            gate_min=float(gate_cfg["gate_min"]),
            gate_max=float(gate_cfg["gate_max"]),
            risk_residual_scale=float(residual_cfg.get("risk_residual_scale", 0.2)),
            value_residual_scale=float(residual_cfg.get("value_residual_scale", 0.1)),
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=float(train_cfg["learning_rate"]),
        )

    def distributions(
        self,
        states: torch.Tensor,
    ) -> Tuple[Categorical, Categorical, torch.Tensor, torch.Tensor]:
        settle_logits, flush_logits, values, gates, _ = self.model(states)
        settle_dist = Categorical(logits=settle_logits)
        flush_dist = Categorical(logits=flush_logits)
        return settle_dist, flush_dist, values, gates

    @torch.no_grad()
    def act(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, int, int, float, float, float]:
        state_t = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        settle_dist, flush_dist, value, gate = self.distributions(state_t)

        if deterministic:
            settle = torch.argmax(settle_dist.logits, dim=-1)
            flush = torch.argmax(flush_dist.logits, dim=-1)
        else:
            settle = settle_dist.sample()
            flush = flush_dist.sample()

        logp = settle_dist.log_prob(settle) + flush_dist.log_prob(flush)

        settle_int = int(settle.item())
        flush_int = int(flush.item())
        action_int = settle_int * (self.k + 1) + flush_int

        return (
            action_int,
            settle_int,
            flush_int,
            float(logp.item()),
            float(value.item()),
            float(gate.item()),
        )

    def update(self, batch: Dict[str, List[Any]]) -> Dict[str, float]:
        train_cfg = self.config["train"]

        states = torch.tensor(
            np.array(batch["states"]),
            dtype=torch.float32,
            device=self.device,
        )
        settle_actions = torch.tensor(
            batch["settle_actions"],
            dtype=torch.int64,
            device=self.device,
        )
        flush_actions = torch.tensor(
            batch["flush_actions"],
            dtype=torch.int64,
            device=self.device,
        )
        old_logp = torch.tensor(
            batch["logp"],
            dtype=torch.float32,
            device=self.device,
        )
        returns = torch.tensor(
            batch["returns"],
            dtype=torch.float32,
            device=self.device,
        )
        advantages = torch.tensor(
            batch["advantages"],
            dtype=torch.float32,
            device=self.device,
        )
        if self.use_aux_risk:
            future_drop_risk = torch.tensor(
                batch["future_drop_risk"],
                dtype=torch.float32,
                device=self.device,
            )
        else:
            future_drop_risk = torch.zeros(
                states.shape[0],
                dtype=torch.float32,
                device=self.device,
            )

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = states.shape[0]
        mb_size = min(int(train_cfg["minibatch_size"]), n)

        last_metrics = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "gate_mean": 0.0,
            "gate_balance_loss": 0.0,
            "ppo_loss": 0.0,
            "total_loss": 0.0,
            "aux_risk_loss": 0.0,
            "aux_risk_pos_rate": 0.0,
            "aux_risk_pred_mean": 0.0,
            "dual_critic_loss": 0.0,
            "value_capacity_mean": 0.0,
            "value_risk_mean": 0.0,
            "value_final_mean": 0.0,
            "value_disagreement": 0.0,
        }

        for _ in range(int(train_cfg["update_epochs"])):
            order = torch.randperm(n, device=self.device)

            for start in range(0, n, mb_size):
                idx = order[start:start + mb_size]

                settle_logits, flush_logits, values, gates, risk_logits = self.model(states[idx])
                settle_dist = Categorical(logits=settle_logits)
                flush_dist = Categorical(logits=flush_logits)

                logp = (
                    settle_dist.log_prob(settle_actions[idx])
                    + flush_dist.log_prob(flush_actions[idx])
                )

                ratio = torch.exp(logp - old_logp[idx])

                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(
                    ratio,
                    1.0 - float(train_cfg["clip_eps"]),
                    1.0 + float(train_cfg["clip_eps"]),
                ) * advantages[idx]

                policy_loss = -torch.min(surr1, surr2).mean()

                # Final gated critic loss used by PPO/GAE.
                final_value_loss = (returns[idx] - values).pow(2).mean()

                # Branch-specific critic values for dual-critic training and diagnostics.
                value_capacity, value_risk, value_final_check, _ = self.model.critic_components(states[idx])
                dual_critic_loss = 0.5 * (
                    (returns[idx] - value_capacity).pow(2).mean()
                    + (returns[idx] - value_risk).pow(2).mean()
                )
                value_disagreement = torch.mean(torch.abs(value_capacity - value_risk))

                value_loss = final_value_loss
                if self.enable_dual_critic:
                    value_loss = value_loss + self.dual_critic_coef * dual_critic_loss

                entropy = settle_dist.entropy().mean() + flush_dist.entropy().mean()
                gate_balance_loss = (gates - self.gate_target).pow(2).mean()

                original_ppo_loss = (
                    policy_loss
                    + float(train_cfg["value_coef"]) * value_loss
                    - float(batch["entropy_coef"]) * entropy
                )
                loss = original_ppo_loss

                if self.use_gate_regularization:
                    loss = loss + self.gate_reg_coef * gate_balance_loss

                aux_risk_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                aux_risk_pred_mean = torch.zeros((), dtype=torch.float32, device=self.device)
                aux_risk_pos_rate = future_drop_risk[idx].mean()

                if self.use_aux_risk:
                    if risk_logits is None:
                        raise RuntimeError("Aux-risk mode requires risk logits.")
                    aux_risk_loss = F.binary_cross_entropy_with_logits(
                        risk_logits,
                        future_drop_risk[idx],
                    )
                    aux_risk_pred_mean = torch.sigmoid(risk_logits).mean()
                    loss = loss + self.aux_risk_coef * aux_risk_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    float(train_cfg["max_grad_norm"]),
                )
                self.optimizer.step()

                last_metrics = {
                    "loss": float(loss.item()),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.item()),
                    "gate_mean": float(gates.mean().item()),
                    "gate_balance_loss": float(gate_balance_loss.item()),
                    "ppo_loss": float(original_ppo_loss.item()),
                    "total_loss": float(loss.item()),
                    "aux_risk_loss": float(aux_risk_loss.item()),
                    "aux_risk_pos_rate": float(aux_risk_pos_rate.item()),
                    "aux_risk_pred_mean": float(aux_risk_pred_mean.item()),
                    "dual_critic_loss": float(dual_critic_loss.item()),
                    "value_capacity_mean": float(value_capacity.mean().item()),
                    "value_risk_mean": float(value_risk.mean().item()),
                    "value_final_mean": float(values.mean().item()),
                    "value_disagreement": float(value_disagreement.item()),
                }

        return last_metrics


# =========================================================
# GAE
# =========================================================

def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[List[float], List[float]]:
    advantages: List[float] = []
    gae = 0.0
    next_value = float(last_value)

    for t in reversed(range(len(rewards))):
        nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * gae_lambda * nonterminal * gae

        advantages.insert(0, float(gae))
        next_value = values[t]

    returns = [
        float(adv + val)
        for adv, val in zip(advantages, values)
    ]

    return returns, advantages


# =========================================================
# Evaluation
# =========================================================

def summarize_episode_metrics(
    all_results: List[Dict[str, float]],
) -> Dict[str, Dict[str, Any]]:
    if not all_results:
        raise ValueError("No episode metrics to summarize.")

    summary: Dict[str, Dict[str, Any]] = {}

    for metric in all_results[0].keys():
        values = [float(r[metric]) for r in all_results]
        summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "values": values,
        }

    return summary


@torch.no_grad()
def evaluate_agent_on_array(
    agent: DualBranchPPOAgent,
    config: Dict[str, Any],
    tx_pool: np.ndarray,
    label: str,
    num_eval_episodes: int,
    max_steps: int,
) -> Dict[str, Any]:
    num_eval_episodes = min(num_eval_episodes, tx_pool.shape[0])
    env = make_env(config, max_steps=max_steps)

    all_results: List[Dict[str, float]] = []

    # Debug counters for deterministic evaluation.
    # These show whether argmax evaluation collapses to one bad action.
    eval_action_counts: Dict[int, int] = {}
    eval_settle_counts: Dict[int, int] = {}
    eval_flush_counts: Dict[int, int] = {}
    eval_first_infos_debug: List[Dict[str, Any]] = []

    for ep in range(num_eval_episodes):
        state = env.reset(tx_stream=tx_pool[ep])

        total_requested_value = 0.0
        total_tx_count = 0
        accepted_count = 0
        episode_gates: List[float] = []
        episode_value_capacity: List[float] = []
        episode_value_risk: List[float] = []
        episode_value_disagreement: List[float] = []

        for _ in range(max_steps):
            total_requested_value += float(env.current_tx)
            total_tx_count += 1

            action, settle, flush, _, _, gate = agent.act(
                state,
                deterministic=True,
            )

            if agent.enable_value_diagnostics:
                state_t = torch.tensor(
                    state,
                    dtype=torch.float32,
                    device=agent.device,
                ).unsqueeze(0)
                value_capacity_t, value_risk_t, _, _ = agent.model.critic_components(state_t)
                value_capacity_scalar = float(value_capacity_t.item())
                value_risk_scalar = float(value_risk_t.item())
                episode_value_capacity.append(value_capacity_scalar)
                episode_value_risk.append(value_risk_scalar)
                episode_value_disagreement.append(abs(value_capacity_scalar - value_risk_scalar))

            eval_action_counts[action] = eval_action_counts.get(action, 0) + 1
            eval_settle_counts[settle] = eval_settle_counts.get(settle, 0) + 1
            eval_flush_counts[flush] = eval_flush_counts.get(flush, 0) + 1

            episode_gates.append(float(gate))

            state, _, done, info = env.step(action)

            if config["debug_mode"] and label == "VAL" and ep == 0 and len(eval_first_infos_debug) < 10:
                eval_first_infos_debug.append(
                    {
                        "step": total_tx_count - 1,
                        "settle": settle,
                        "flush": flush,
                        "action": action,
                        "accepted": info.get("accepted", None),
                        "info": dict(info),
                    }
                )

            if info.get("accepted", False):
                accepted_count += 1

            if done:
                break

        metrics = env.get_metrics()

        metrics["value_accept_ratio"] = (
            metrics["settled"] / total_requested_value
            if total_requested_value > 0
            else 0.0
        )
        metrics["count_accept_ratio"] = (
            accepted_count / total_tx_count
            if total_tx_count > 0
            else 0.0
        )
        metrics["total_requested_value"] = total_requested_value
        metrics["total_tx_count"] = total_tx_count
        metrics["accepted_count"] = accepted_count
        metrics["gate_mean"] = float(np.mean(episode_gates)) if episode_gates else 0.0
        metrics["gate_std"] = float(np.std(episode_gates)) if episode_gates else 0.0
        metrics["gate_min"] = float(np.min(episode_gates)) if episode_gates else 0.0
        metrics["gate_max"] = float(np.max(episode_gates)) if episode_gates else 0.0
        metrics["value_capacity_mean"] = float(np.mean(episode_value_capacity)) if episode_value_capacity else 0.0
        metrics["value_risk_mean"] = float(np.mean(episode_value_risk)) if episode_value_risk else 0.0
        metrics["value_disagreement"] = float(np.mean(episode_value_disagreement)) if episode_value_disagreement else 0.0
        add_eval_money_metrics(metrics, config)

        all_results.append(metrics)

    if config["debug_mode"] and label == "VAL":
        print("\n[Eval Action Debug]")
        print(f"label={label}")
        print(f"num_eval_episodes={num_eval_episodes}")
        print(f"eval_settle_counts={eval_settle_counts}")
        print(f"eval_flush_counts={eval_flush_counts}")
        print(f"eval_action_counts={eval_action_counts}")
        print(f"eval_first_infos_debug={eval_first_infos_debug}")

    summary = summarize_episode_metrics(all_results)
    add_reward_summary_metadata(summary, config)
    return {
        "label": label,
        "num_episodes": num_eval_episodes,
        "summary": summary,
        "raw_results": all_results,
    }


def evaluate_agent_on_pool(
    agent: DualBranchPPOAgent,
    config: Dict[str, Any],
    test_pool_path: str,
    test_regime: str,
) -> Dict[str, Any]:
    eval_cfg = config["eval"]

    if not verify_data_integrity(
        test_pool_path,
        expected_steps=eval_cfg["max_steps"],
        label=test_regime,
    ):
        raise RuntimeError(f"Data verification failed for {test_regime}")

    tx_pool = load_tx_pool(
        test_pool_path,
        expected_steps=eval_cfg["max_steps"],
    )

    result = evaluate_agent_on_array(
        agent=agent,
        config=config,
        tx_pool=tx_pool,
        label=test_regime,
        num_eval_episodes=int(eval_cfg["num_episodes"]),
        max_steps=int(eval_cfg["max_steps"]),
    )

    result["test_regime"] = test_regime
    result["test_pool_path"] = test_pool_path

    return result


def add_money_to_cross_regime_aggregate(
    aggregate: Dict[str, Any],
    cross_results: Dict[str, Any],
) -> Dict[str, Any]:
    eval_money_values: List[float] = []

    for regime_result in cross_results.values():
        summary = regime_result.get("summary", {})
        eval_money_data = summary.get("eval_money", {})
        if isinstance(eval_money_data, dict) and "mean" in eval_money_data:
            eval_money_values.append(float(eval_money_data["mean"]))

    if eval_money_values:
        aggregate["mean_eval_money"] = float(np.mean(eval_money_values))
        aggregate["worst_regime_eval_money"] = float(np.min(eval_money_values))
        aggregate["std_eval_money_across_regimes"] = float(np.std(eval_money_values))

    return aggregate


def evaluate_agent_cross_regime(
    agent: DualBranchPPOAgent,
    config: Dict[str, Any],
    paths: Dict[str, Any],
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("Dual-Branch AC Cross-Regime Evaluation")
    print("=" * 70)

    cross_results: Dict[str, Any] = {}

    for regime_name, pool_path in paths["test_pool_paths"].items():
        cross_results[regime_name] = evaluate_agent_on_pool(
            agent=agent,
            config=config,
            test_pool_path=pool_path,
            test_regime=regime_name,
        )

    aggregate = compute_cross_regime_aggregate(cross_results)
    aggregate = add_money_to_cross_regime_aggregate(aggregate, cross_results)

    return {
        "config": config,
        "scenario": paths["scenario"],
        "train_regime": paths["train_regime"],
        "model_mode": config["model_mode"],
        "seed": config["seed"],
        "timestamp": datetime.now().isoformat(),
        "test_results": cross_results,
        "aggregate": aggregate,
        **reward_metadata(config),
    }


# =========================================================
# Training
# =========================================================

def train_agent(
    config: Dict[str, Any],
    paths: Dict[str, Any],
) -> Tuple[
    DualBranchPPOAgent,
    List[float],
    List[float],
    List[Dict[str, float]],
    List[Dict[str, float]],
    List[Dict[str, Any]],
]:
    train_cfg = config["train"]

    tx_pool_full = load_tx_pool(
        paths["train_pool_path"],
        expected_steps=int(train_cfg["max_steps"]),
    )
    tx_pool_val = load_tx_pool(
        paths["val_pool_path"],
        expected_steps=int(train_cfg["max_steps"]),
    )

    train_use_episodes = int(train_cfg["train_use_episodes"])

    if train_use_episodes > tx_pool_full.shape[0]:
        raise ValueError(
            f"train_use_episodes={train_use_episodes} exceeds pool size={tx_pool_full.shape[0]}"
        )

    if int(train_cfg["episodes"]) > train_use_episodes:
        raise ValueError(
            f"episodes={train_cfg['episodes']} exceeds train_use_episodes={train_use_episodes}"
        )

    tx_pool_train = tx_pool_full[:train_use_episodes]

    env = make_env(
        config,
        max_steps=int(train_cfg["max_steps"]),
    )

    agent = DualBranchPPOAgent(
        config=config,
        state_size=env.state_size,
        base_state_size=env.base_state_size,
        k=env.k,
    )

    param_counts = count_parameters(agent.model)

    print("\n" + "=" * 70)
    print("Train Dual-Branch Factorized Actor-Critic / PPO")
    print("=" * 70)
    print(f"mode={config['model_mode']} train_regime={paths['train_regime']} seed={config['seed']}")
    print(f"env C={env.C} k={env.k} F={env.F} T={env.max_transaction}")
    print(f"state={env.state_size} base_state={env.base_state_size} action={(env.k + 1) ** 2}")
    print(f"train_pool={tx_pool_train.shape} val_pool={tx_pool_val.shape}")
    print(f"model parameters: total={param_counts['total']} trainable={param_counts['trainable']}")
    if config["model_mode"] == "dual_branch_capacity_only":
        print("dual logic: capacity-only ablation, no risk branch fusion")
    elif config["model_mode"] == "dual_branch_residual_risk":
        print("dual logic: capacity branch + bounded future-pressure residual correction")
    else:
        print("dual logic: capacity branch + risk/context branch + learned gate")

    returns_history: List[float] = []
    loss_history: List[float] = []
    validation_history: List[Dict[str, float]] = []
    gate_history: List[Dict[str, float]] = []
    reward_history: List[Dict[str, Any]] = []

    best_val_score = -1e18
    best_state_dict = None
    best_val_snapshot = None

    t_start = time.perf_counter()
    use_aux_risk = config["model_mode"] == "dual_branch_factorized_ac_auxrisk"
    aux_risk_window = int(config["aux_risk"]["aux_risk_window"])
    drop_source_announced = False

    for ep in range(int(train_cfg["episodes"])):
        state = env.reset(tx_stream=tx_pool_train[ep])
        episode_return = 0.0
        episode_original_reward = 0.0
        episode_settled_value = 0.0
        episode_flushes = 0.0
        episode_money_reward = 0.0

        states: List[np.ndarray] = []
        settle_actions: List[int] = []
        flush_actions: List[int] = []
        logps: List[float] = []
        values: List[float] = []
        rewards: List[float] = []
        dones: List[bool] = []
        episode_gates: List[float] = []
        drop_flags: List[float] = []
        step_rewards: List[float] = []
        step_money_rewards: List[float] = []
        settle_action_counts = [0 for _ in range(env.k + 1)]
        flush_action_counts = [0 for _ in range(env.k + 1)]

        entropy_coef = float(train_cfg["entropy_coef_start"]) + (
            float(train_cfg["entropy_coef_end"])
            - float(train_cfg["entropy_coef_start"])
        ) * min(ep / max(1, int(train_cfg["episodes"])), 1.0)

        action_counts = {}
        settle_counts = {}
        flush_counts = {}
        accepted_count_debug = 0
        reward_sum_debug = 0.0
        first_infos_debug = []

        for step_i in range(int(train_cfg["max_steps"])):
            action, settle, flush, logp, value, gate = agent.act(
                state,
                deterministic=False,
            )

            action_counts[action] = action_counts.get(action, 0) + 1
            settle_counts[settle] = settle_counts.get(settle, 0) + 1
            flush_counts[flush] = flush_counts.get(flush, 0) + 1
            settle_action_counts[int(settle)] += 1
            flush_action_counts[int(flush)] += 1

            next_state, reward, done, info = env.step(action)
            if ep == 0 and step_i == 0:
                print(f"[RewardInfo] first env.step info keys: {sorted(info.keys())}")
            selected_reward, money_reward, settled_value, flushes_this_step = select_training_reward(
                reward,
                info,
                config,
            )

            reward_sum_debug += float(reward)

            if info.get("accepted", False):
                accepted_count_debug += 1

            if use_aux_risk:
                drop_flag, drop_source = infer_drop_flag(info)
                drop_flags.append(drop_flag)

                if not drop_source_announced:
                    print(f"[AuxRisk] future_drop_risk source: {drop_source}")
                    drop_source_announced = True

            if ep == 0 and step_i < 10:
                first_infos_debug.append(
                    {
                        "step": step_i,
                        "settle": settle,
                        "flush": flush,
                        "action": action,
                        "reward": float(reward),
                        "selected_training_reward": float(selected_reward),
                        "accepted": info.get("accepted", None),
                        "info": dict(info),
                    }
                )

            states.append(state)
            settle_actions.append(settle)
            flush_actions.append(flush)
            logps.append(logp)
            values.append(value)
            rewards.append(float(selected_reward))
            dones.append(bool(done))
            episode_gates.append(float(gate))
            step_rewards.append(float(selected_reward))
            step_money_rewards.append(float(money_reward))

            episode_original_reward += float(reward)
            episode_return += float(selected_reward)
            episode_settled_value += settled_value
            episode_flushes += flushes_this_step
            episode_money_reward += money_reward
            state = next_state

            if done:
                break

        with torch.no_grad():
            state_t = torch.tensor(
                state,
                dtype=torch.float32,
                device=agent.device,
            ).unsqueeze(0)
            _, _, last_value_t, _ = agent.distributions(state_t)
            last_value = 0.0 if dones[-1] else float(last_value_t.item())

        returns, advantages = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            last_value=last_value,
            gamma=float(train_cfg["gamma"]),
            gae_lambda=float(train_cfg["gae_lambda"]),
        )
        future_drop_risk = (
            build_future_drop_risk_labels(drop_flags, aux_risk_window)
            if use_aux_risk
            else []
        )

        if use_aux_risk and len(future_drop_risk) != len(states):
            raise RuntimeError(
                "future_drop_risk label count must match PPO rollout states: "
                f"labels={len(future_drop_risk)} states={len(states)}"
            )

        metrics = agent.update(
            {
                "states": states,
                "settle_actions": settle_actions,
                "flush_actions": flush_actions,
                "logp": logps,
                "returns": returns,
                "advantages": advantages,
                "entropy_coef": entropy_coef,
                "future_drop_risk": future_drop_risk,
            }
        )

        returns_history.append(float(episode_return))
        loss_history.append(float(metrics["loss"]))
        episode_steps = max(1, len(step_rewards))
        reward_history.append(
            {
                "episode": ep + 1,
                "episode_original_reward": float(episode_original_reward),
                "episode_training_reward": float(episode_return),
                "episode_settled_value": float(episode_settled_value),
                "episode_flushes": float(episode_flushes),
                "episode_money_reward": float(episode_money_reward),
                "mean_step_reward": float(np.mean(step_rewards)) if step_rewards else 0.0,
                "min_step_reward": float(np.min(step_rewards)) if step_rewards else 0.0,
                "max_step_reward": float(np.max(step_rewards)) if step_rewards else 0.0,
                "mean_step_money_reward": float(np.mean(step_money_rewards)) if step_money_rewards else 0.0,
                "min_step_money_reward": float(np.min(step_money_rewards)) if step_money_rewards else 0.0,
                "max_step_money_reward": float(np.max(step_money_rewards)) if step_money_rewards else 0.0,
                "settle_noop_ratio": float(settle_action_counts[env.k] / episode_steps),
                "flush_noop_ratio": float(flush_action_counts[env.k] / episode_steps),
                "settle_action_counts": settle_action_counts,
                "flush_action_counts": flush_action_counts,
            }
        )

        gate_row = {
            "episode": ep + 1,
            "gate_mean": float(np.mean(episode_gates)) if episode_gates else 0.0,
            "gate_min": float(np.min(episode_gates)) if episode_gates else 0.0,
            "gate_max": float(np.max(episode_gates)) if episode_gates else 0.0,
            "gate_balance_loss": float(metrics.get("gate_balance_loss", 0.0)),
            "ppo_loss": float(metrics.get("ppo_loss", metrics["loss"])),
            "total_loss": float(metrics.get("total_loss", metrics["loss"])),
            "aux_risk_loss": float(metrics.get("aux_risk_loss", 0.0)),
            "aux_risk_pos_rate": float(metrics.get("aux_risk_pos_rate", 0.0)),
            "aux_risk_pred_mean": float(metrics.get("aux_risk_pred_mean", 0.0)),
            "dual_critic_loss": float(metrics.get("dual_critic_loss", 0.0)),
            "value_capacity_mean": float(metrics.get("value_capacity_mean", 0.0)),
            "value_risk_mean": float(metrics.get("value_risk_mean", 0.0)),
            "value_disagreement": float(metrics.get("value_disagreement", 0.0)),
        }
        gate_history.append(gate_row)
        if config["debug_mode"] and ep < 3:
            ep_metrics_debug = env.get_metrics()

            print("\n[Action Debug]")
            print(f"episode={ep + 1}")
            print(f"settle_counts={settle_counts}")
            print(f"flush_counts={flush_counts}")
            print(f"action_counts={action_counts}")
            print(f"accepted_count_debug={accepted_count_debug}")
            print(f"reward_sum_debug={reward_sum_debug:.4f}")
            print(f"env_metrics={ep_metrics_debug}")
            print(f"first_infos_debug={first_infos_debug}")

        if (ep + 1) % LOG_EVERY_N == 0 or ep == 0:
            recent_mean = float(np.mean(returns_history[-LOG_EVERY_N:]))
            elapsed = time.perf_counter() - t_start

            print(
                f"[Train] ep={ep + 1:4d}/{train_cfg['episodes']} "
                f"return={episode_return:10.2f} recent={recent_mean:10.2f} "
                f"money={episode_money_reward:10.2f} "
                f"loss={metrics['loss']:.4f} ent={metrics['entropy']:.4f} "
                f"ppo={gate_row['ppo_loss']:.4f} "
                f"gate={gate_row['gate_mean']:.3f} "
                f"gate_bal={gate_row['gate_balance_loss']:.6f} "
                f"aux_loss={gate_row['aux_risk_loss']:.6f} "
                f"aux_pos={gate_row['aux_risk_pos_rate']:.3f} "
                f"aux_pred={gate_row['aux_risk_pred_mean']:.3f} "
                f"dualV={gate_row['dual_critic_loss']:.4f} "
                f"Vdiff={gate_row['value_disagreement']:.4f} "
                f"elapsed={elapsed:.1f}s"
            )

        val_every = int(train_cfg["val_every"])

        if val_every > 0 and (
            (ep + 1) % val_every == 0
            or (ep + 1) == int(train_cfg["episodes"])
        ):
            val_result = evaluate_agent_on_array(
                agent=agent,
                config=config,
                tx_pool=tx_pool_val,
                label="VAL",
                num_eval_episodes=int(train_cfg["val_num_episodes"]),
                max_steps=int(train_cfg["max_steps"]),
            )

            metric_name = train_cfg["val_metric"]
            val_score = float(val_result["summary"][metric_name]["mean"])

            val_row = {
                "episode": ep + 1,
                "metric": metric_name,
                "value_accept_ratio": float(
                    val_result["summary"]["value_accept_ratio"]["mean"]
                ),
                "drop_rate": float(
                    val_result["summary"]["drop_rate"]["mean"]
                ),
                "drops": float(
                    val_result["summary"]["drops"]["mean"]
                ),
                "flushes": float(
                    val_result["summary"]["flushes"]["mean"]
                ),
                "settled": float(
                    val_result["summary"]["settled"]["mean"]
                ),
                "count_accept_ratio": float(
                    val_result["summary"]["count_accept_ratio"]["mean"]
                ),
                "eval_money": float(
                    val_result["summary"]["eval_money"]["mean"]
                ),
                "gate_mean": float(
                    val_result["summary"].get("gate_mean", {}).get("mean", 0.0)
                ),
                "value_disagreement": float(
                    val_result["summary"].get("value_disagreement", {}).get("mean", 0.0)
                ),
                "value_capacity_mean": float(
                    val_result["summary"].get("value_capacity_mean", {}).get("mean", 0.0)
                ),
                "value_risk_mean": float(
                    val_result["summary"].get("value_risk_mean", {}).get("mean", 0.0)
                ),
                "score": val_score,
            }

            validation_history.append(val_row)

            print(
                f"[Val  ] ep={ep + 1:4d} {metric_name}={val_score:.4f} "
                f"val_acc={val_row['value_accept_ratio']:.4f} "
                f"eval_money={val_row['eval_money']:.2f} "
                f"drops={val_row['drops']:.2f} flushes={val_row['flushes']:.2f} "
                f"gate={val_row['gate_mean']:.3f} "
                f"Vdiff={val_row['value_disagreement']:.4f}"
            )

            if val_score > best_val_score:
                best_val_score = val_score
                best_val_snapshot = val_row
                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in agent.model.state_dict().items()
                }

                if not config["debug_mode"] and config["save_mode"] == "full":
                    torch.save(best_state_dict, paths["best_model_path"])

                print(
                    f"[Val  ] best checkpoint updated: "
                    f"ep={ep + 1}, score={val_score:.4f}"
                )

    if train_cfg["use_best_model_for_final_eval"] and best_state_dict is not None:
        agent.model.load_state_dict(best_state_dict)
        print(
            f"Loaded best checkpoint: "
            f"ep={best_val_snapshot['episode']} score={best_val_snapshot['score']:.4f}"
        )

    if not config["debug_mode"] and config["save_mode"] == "full":
        torch.save(agent.model.state_dict(), paths["last_model_path"])
        print(f"Last model saved to: {paths['last_model_path']}")

    return agent, returns_history, loss_history, validation_history, gate_history, reward_history


# =========================================================
# Reporting and saving
# =========================================================

def build_cross_regime_report_text(results: Dict[str, Any]) -> str:
    lines: List[str] = []

    lines.append("=" * 120)
    lines.append("Dual-Branch Factorized AC Cross-Regime Evaluation")
    lines.append("=" * 120)
    lines.append(f"timestamp    : {results['timestamp']}")
    lines.append(f"scenario     : {results['scenario']}")
    lines.append(f"model_mode   : {results['model_mode']}")
    lines.append(f"train_regime : {results['train_regime']}")
    lines.append(f"seed         : {results['seed']}")
    lines.append(f"reward_mode  : {results.get('reward_mode')}")
    lines.append(f"money_p      : {results.get('money_p')}")
    lines.append(f"money_tau    : {results.get('money_tau')}")
    lines.append(f"settled_scale: {results.get('settled_scale')}")
    lines.append(f"tau_scaled   : {results.get('tau_scaled')}")
    lines.append(f"hybrid_alpha : {results.get('hybrid_alpha')}")
    lines.append("-" * 120)

    lines.append(
        f"{'Test':<8}"
        f"{'ValAcc(%)':>14}"
        f"{'EvalMoney':>14}"
        f"{'Drops':>12}"
        f"{'Flushes':>12}"
        f"{'DropRate(%)':>14}"
        f"{'CntAcc(%)':>14}"
        f"{'Gate':>10}"
        f"{'VDiff':>10}"
    )

    lines.append("-" * 120)

    for regime_name, regime_result in results["test_results"].items():
        s = regime_result["summary"]
        gate_mean = s.get("gate_mean", {}).get("mean", 0.0)
        value_disagreement = s.get("value_disagreement", {}).get("mean", 0.0)
        eval_money = s.get("eval_money", {}).get("mean", 0.0)

        lines.append(
            f"{regime_name:<8}"
            f"{100 * s['value_accept_ratio']['mean']:>14.2f}"
            f"{eval_money:>14.2f}"
            f"{s['drops']['mean']:>12.2f}"
            f"{s['flushes']['mean']:>12.2f}"
            f"{100 * s['drop_rate']['mean']:>14.2f}"
            f"{100 * s['count_accept_ratio']['mean']:>14.2f}"
            f"{gate_mean:>10.3f}"
            f"{value_disagreement:>10.3f}"
        )

    agg = results["aggregate"]

    lines.append("-" * 120)
    lines.append(f"Mean ValAcc (%)        : {100 * agg['mean_value_accept_ratio']:.4f}")
    lines.append(f"Worst-Regime ValAcc (%): {100 * agg['worst_regime_value_accept_ratio']:.4f}")
    lines.append(f"Std Across Regimes     : {agg['std_value_accept_ratio_across_regimes']:.6f}")
    lines.append(f"Mean Drops             : {agg['mean_drops']:.4f}")
    lines.append(f"Mean Flushes           : {agg['mean_flushes']:.4f}")

    if "mean_eval_money" in agg:
        lines.append(f"Mean EvalMoney         : {agg['mean_eval_money']:.4f}")
        lines.append(f"Worst-Regime EvalMoney : {agg['worst_regime_eval_money']:.4f}")
        lines.append(f"Std EvalMoney          : {agg['std_eval_money_across_regimes']:.6f}")

    lines.append("=" * 120)

    return "\n".join(lines)


def save_results(results: Dict[str, Any], save_path: str) -> None:
    compact: Dict[str, Any] = {
        "config": results["config"],
        "scenario": results["scenario"],
        "train_regime": results["train_regime"],
        "model_mode": results["model_mode"],
        "seed": results["seed"],
        "timestamp": results["timestamp"],
        "aggregate": results["aggregate"],
        "test_results": {},
        **reward_metadata(results["config"]),
    }

    for regime_name, regime_result in results["test_results"].items():
        compact["test_results"][regime_name] = {
            "num_episodes": regime_result["num_episodes"],
            "summary": {},
        }

        for metric, data in regime_result["summary"].items():
            if isinstance(data, dict) and "mean" in data:
                compact["test_results"][regime_name]["summary"][metric] = {
                    "mean": data["mean"],
                    "std": data["std"],
                    "min": data["min"],
                    "max": data["max"],
                    "median": data["median"],
                }
            else:
                compact["test_results"][regime_name]["summary"][metric] = data

    save_json(compact, save_path)
    print(f"Results saved to: {save_path}")


def moving_average(values: List[float], window: int) -> np.ndarray:
    if not values:
        return np.array([])

    arr = np.array(values, dtype=float)
    out = np.zeros_like(arr)

    for i in range(len(arr)):
        left = max(0, i - window + 1)
        out[i] = np.mean(arr[left:i + 1])

    return out


def plot_training_curves(
    returns: List[float],
    losses: List[float],
    save_path: str,
    title_tag: str,
    window: int,
) -> None:
    fig = plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(returns, alpha=0.35, label="Return")
    plt.plot(
        moving_average(returns, window),
        linewidth=2,
        label=f"MA({window})",
    )
    plt.title(f"Dual-Branch AC Training Curves\n{title_tag}")
    plt.ylabel("Episode Return")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(losses, alpha=0.8)
    plt.xlabel("Episode")
    plt.ylabel("PPO Loss")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Training plot saved to: {save_path}")


def plot_evaluation_results(
    results: Dict[str, Any],
    save_path: str,
    title_tag: str,
) -> None:
    regimes = list(results["test_results"].keys())

    val_acc = [
        100 * results["test_results"][r]["summary"]["value_accept_ratio"]["mean"]
        for r in regimes
    ]
    eval_money = [
        results["test_results"][r]["summary"].get("eval_money", {}).get("mean", 0.0)
        for r in regimes
    ]
    drops = [
        results["test_results"][r]["summary"]["drops"]["mean"]
        for r in regimes
    ]
    flushes = [
        results["test_results"][r]["summary"]["flushes"]["mean"]
        for r in regimes
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    fig.suptitle(
        f"Dual-Branch AC Cross-Regime Evaluation\n{title_tag}",
        fontsize=14,
        fontweight="bold",
    )

    chart_items = [
        (axes[0, 0], val_acc, "Value Accept Ratio (%)"),
        (axes[0, 1], eval_money, "Eval Money"),
        (axes[1, 0], drops, "Drops"),
        (axes[1, 1], flushes, "Flushes"),
    ]

    for ax, values, label in chart_items:
        ax.bar(regimes, values)
        ax.set_title(label)
        ax.grid(True, alpha=0.3, axis="y")

        for i, v in enumerate(values):
            ax.text(
                i,
                v,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Evaluation plot saved to: {save_path}")


def write_run_outputs(
    config: Dict[str, Any],
    paths: Dict[str, Any],
    results: Dict[str, Any],
    returns: List[float],
    losses: List[float],
    validation_history: List[Dict[str, float]],
    gate_history: List[Dict[str, float]],
    reward_history: List[Dict[str, Any]],
) -> None:
    if config["debug_mode"] or config["save_mode"] == "none":
        print("Save skipped because debug_mode=True or save_mode=none.")
        return

    save_json(
        {
            "config": config,
            "paths": paths,
            "pool_fingerprints": build_pool_fingerprints(config, paths),
            "timestamp": datetime.now().isoformat(),
            **reward_metadata(config),
        },
        paths["run_info_path"],
    )

    save_json(config, paths["run_config_path"])
    save_results(results, paths["results_json_path"])

    with open(paths["summary_txt_path"], "w", encoding="utf-8") as f:
        f.write(build_cross_regime_report_text(results))

    save_json(
        {
            "returns": returns,
            "loss_history": losses,
            "gate_history": gate_history,
            "reward_history": reward_history,
            **reward_metadata(config),
        },
        paths["training_history_path"],
    )

    save_json(
        {
            "validation_history": validation_history,
            **reward_metadata(config),
        },
        paths["validation_history_path"],
    )

    plot_training_curves(
        returns=returns,
        losses=losses,
        save_path=paths["training_plot_path"],
        title_tag=paths["title_tag"],
        window=int(config["plot"]["window"]),
    )

    plot_evaluation_results(
        results=results,
        save_path=paths["eval_plot_path"],
        title_tag=paths["title_tag"],
    )


# =========================================================
# Aggregation
# =========================================================

def safe_summary_mean(summary: Dict[str, Any], metric: str, default: Any = 0.0) -> Any:
    data = summary.get(metric)
    if isinstance(data, dict) and "mean" in data:
        return float(data["mean"])
    if default is None:
        return None
    return float(default)


def flatten_result_for_csv(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    config = payload["config"]
    aggregate = payload.get("aggregate", {})

    rows: List[Dict[str, Any]] = []

    reward_mode = payload.get("reward_mode", config.get("reward", {}).get("reward_mode"))
    training_objective = payload.get(
        "training_objective",
        config.get("reward", {}).get("training_objective"),
    )
    money_p = float(payload.get("money_p", config.get("reward", {}).get("money_p", 1.0)))
    money_tau = float(payload.get("money_tau", config.get("reward", {}).get("money_tau", 10.0)))
    settled_scale = float(payload.get("settled_scale", config.get("reward", {}).get("settled_scale", 50.0)))
    tau_scaled = float(payload.get("tau_scaled", config.get("reward", {}).get("tau_scaled", 0.2)))
    hybrid_alpha = float(payload.get("hybrid_alpha", config.get("reward", {}).get("hybrid_alpha", 0.1)))

    for regime, regime_result in payload["test_results"].items():
        s = regime_result["summary"]
        settled = safe_summary_mean(s, "settled")
        flushes = safe_summary_mean(s, "flushes")
        eval_money = safe_summary_mean(s, "eval_money", money_p * settled - money_tau * flushes)

        rows.append(
            {
                "scenario": payload["scenario"],
                "model_mode": payload["model_mode"],
                "train_regime": payload["train_regime"],
                "seed": payload["seed"],
                "test_regime": regime,

                "reward_mode": reward_mode,
                "training_objective": training_objective,
                "money_p": money_p,
                "money_tau": money_tau,
                "settled_scale": settled_scale,
                "tau_scaled": tau_scaled,
                "hybrid_alpha": hybrid_alpha,

                "value_accept_ratio": safe_summary_mean(s, "value_accept_ratio"),
                "settled": settled,
                "eval_money": eval_money,
                "drops": safe_summary_mean(s, "drops"),
                "flushes": flushes,
                "drop_rate": safe_summary_mean(s, "drop_rate"),
                "count_accept_ratio": safe_summary_mean(s, "count_accept_ratio"),

                "gate_mean": safe_summary_mean(s, "gate_mean", None) if "gate_mean" in s else None,
                "gate_std": safe_summary_mean(s, "gate_std", None) if "gate_std" in s else None,
                "gate_min": safe_summary_mean(s, "gate_min", None) if "gate_min" in s else None,
                "gate_max": safe_summary_mean(s, "gate_max", None) if "gate_max" in s else None,

                "mean_value_accept_ratio": aggregate.get("mean_value_accept_ratio"),
                "worst_regime_value_accept_ratio": aggregate.get("worst_regime_value_accept_ratio"),
                "std_value_accept_ratio_across_regimes": aggregate.get(
                    "std_value_accept_ratio_across_regimes"
                ),
                "mean_eval_money": aggregate.get("mean_eval_money"),
                "worst_regime_eval_money": aggregate.get("worst_regime_eval_money"),
                "std_eval_money_across_regimes": aggregate.get("std_eval_money_across_regimes"),

                "C": config["env"]["C"],
                "k": config["env"]["k"],
                "F": config["env"]["F"],
                "T": config["env"]["T"],
            }
        )

    return rows


def aggregate_results(result_root: Path = RESULT_ROOT) -> Path:
    result_files = sorted(
        (result_root / "runs").glob("**/cross_regime_results.json")
    )

    rows: List[Dict[str, Any]] = []

    for path in result_files:
        rows.extend(flatten_result_for_csv(path))

    aggregate_dir = result_root / "aggregates"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    out_path = aggregate_dir / "dual_ac_fair_benchmark_aggregated.csv"

    fieldnames = [
        "scenario",
        "model_mode",
        "train_regime",
        "seed",
        "test_regime",

        "reward_mode",
        "training_objective",
        "money_p",
        "money_tau",
        "settled_scale",
        "tau_scaled",
        "hybrid_alpha",

        "value_accept_ratio",
        "settled",
        "eval_money",
        "drops",
        "flushes",
        "drop_rate",
        "count_accept_ratio",

        "gate_mean",
        "gate_std",
        "gate_min",
        "gate_max",
        "value_capacity_mean",
        "value_risk_mean",
        "value_disagreement",

        "enable_dual_critic",
        "dual_critic_coef",
        "enable_value_diagnostics",

        "mean_value_accept_ratio",
        "worst_regime_value_accept_ratio",
        "std_value_accept_ratio_across_regimes",
        "mean_eval_money",
        "worst_regime_eval_money",
        "std_eval_money_across_regimes",

        "C",
        "k",
        "F",
        "T",
    ]

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Aggregated CSV saved to: {out_path} ({len(rows)} rows)")

    return out_path


# =========================================================
# CLI
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dual-Branch Factorized AC fair benchmark for K-wallet RL."
    )

    parser.add_argument(
        "--model_mode",
        choices=[
            "dual_branch_factorized_ac",
            "dual_branch_factorized_ac_gate_balanced",
            "dual_branch_factorized_ac_gate_regularized",
            "dual_branch_factorized_ac_auxrisk",
            "dual_branch_capacity_only",
            "dual_branch_residual_risk",
        ],
        default=CONFIG["model_mode"],
    )

    parser.add_argument("--train_regime", type=str, default=CONFIG["data"]["train_regime"])
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--C", type=float, default=CONFIG["env"]["C"])
    parser.add_argument("--k", type=int, default=CONFIG["env"]["k"])
    parser.add_argument("--F", type=int, default=CONFIG["env"]["F"])
    parser.add_argument("--T", type=int, default=CONFIG["env"]["T"])

    parser.add_argument("--output_dir", type=str, default=CONFIG["output"]["output_dir"])
    parser.add_argument("--device", type=str, default=CONFIG["train"]["device"])
    parser.add_argument("--episodes", type=int, default=CONFIG["train"]["episodes"])
    parser.add_argument("--eval_episodes", type=int, default=CONFIG["eval"]["num_episodes"])

    parser.add_argument(
        "--val_metric",
        choices=["value_accept_ratio", "eval_money"],
        default=None,
        help=(
            "Validation metric for best checkpoint selection. "
            "If omitted, money-family reward modes use eval_money; original uses value_accept_ratio."
        ),
    )

    parser.add_argument("--save_mode", choices=["none", "full"], default=CONFIG["save_mode"])

    parser.add_argument(
        "--reward_mode",
        choices=["original", "money", "money_normalized", "hybrid_money"],
        default=CONFIG["reward"]["reward_mode"],
    )
    parser.add_argument("--money_p", type=float, default=CONFIG["reward"]["money_p"])
    parser.add_argument("--money_tau", type=float, default=CONFIG["reward"]["money_tau"])
    parser.add_argument("--settled_scale", type=float, default=CONFIG["reward"]["settled_scale"])
    parser.add_argument("--tau_scaled", type=float, default=CONFIG["reward"]["tau_scaled"])
    parser.add_argument("--hybrid_alpha", type=float, default=CONFIG["reward"]["hybrid_alpha"])

    parser.add_argument("--gate_temperature", type=float, default=CONFIG["gate"]["gate_temperature"])
    parser.add_argument("--gate_min", type=float, default=CONFIG["gate"]["gate_min"])
    parser.add_argument("--gate_max", type=float, default=CONFIG["gate"]["gate_max"])
    parser.add_argument("--gate_target", type=float, default=CONFIG["gate"]["gate_target"])
    parser.add_argument("--gate_reg_coef", type=float, default=CONFIG["gate"]["gate_reg_coef"])

    parser.add_argument("--aux_risk_coef", type=float, default=CONFIG["aux_risk"]["aux_risk_coef"])
    parser.add_argument("--aux_risk_window", type=int, default=CONFIG["aux_risk"]["aux_risk_window"])

    parser.add_argument(
        "--risk_residual_scale",
        type=float,
        default=CONFIG["residual_risk"]["risk_residual_scale"],
        help="Scale for bounded future-pressure residual correction.",
    )
    parser.add_argument(
        "--value_residual_scale",
        type=float,
        default=CONFIG["residual_risk"]["value_residual_scale"],
        help="Scale for value residual correction.",
    )

    parser.add_argument(
        "--enable_dual_critic",
        action="store_true",
        help="Train capacity and risk value heads with branch-specific critic losses.",
    )
    parser.add_argument(
        "--dual_critic_coef",
        type=float,
        default=CONFIG["dual_critic"]["dual_critic_coef"],
        help="Weight for branch-specific dual critic loss.",
    )
    parser.add_argument(
        "--disable_value_diagnostics",
        action="store_true",
        help="Disable value disagreement diagnostics during evaluation.",
    )

    parser.add_argument("--debug_mode", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")

    return parser.parse_args()

def apply_args_to_config(args: argparse.Namespace) -> Dict[str, Any]:
    config = json.loads(json.dumps(CONFIG))

    config["model_mode"] = args.model_mode
    config["seed"] = int(args.seed)
    config["save_mode"] = args.save_mode
    config["debug_mode"] = bool(args.debug_mode)

    config["env"]["C"] = float(args.C)
    config["env"]["k"] = int(args.k)
    config["env"]["F"] = int(args.F)
    config["env"]["T"] = int(args.T)

    config["train"]["device"] = args.device
    config["train"]["episodes"] = int(args.episodes)
    config["train"]["max_steps"] = int(args.T)
    config["train"]["val_num_episodes"] = int(args.eval_episodes)

    config["eval"]["num_episodes"] = int(args.eval_episodes)
    config["eval"]["max_steps"] = int(args.T)

    config["output"]["output_dir"] = args.output_dir

    config["data"]["train_regime"] = args.train_regime

    if args.train_regime == "MIX12_EQ":
        config["data"]["train_pool_file"] = DEFAULT_MIX_EQ_MASTER
        config["data"]["val_pool_file"] = DEFAULT_MIX_EQ_VAL
    else:
        train_file = train_pool_file_for_regime(args.train_regime)
        config["data"]["train_pool_file"] = train_file
        config["data"]["val_pool_file"] = DEFAULT_MIX_EQ_VAL

    config["data"]["test_pool_files"] = DEFAULT_STATIC_EVAL_FILES

    config["gate"]["gate_temperature"] = float(args.gate_temperature)
    config["gate"]["gate_min"] = float(args.gate_min)
    config["gate"]["gate_max"] = float(args.gate_max)
    config["gate"]["gate_target"] = float(args.gate_target)
    config["gate"]["gate_reg_coef"] = float(args.gate_reg_coef)

    config["aux_risk"]["aux_risk_coef"] = float(args.aux_risk_coef)
    config["aux_risk"]["aux_risk_window"] = int(args.aux_risk_window)

    config["residual_risk"]["risk_residual_scale"] = float(args.risk_residual_scale)
    config["residual_risk"]["value_residual_scale"] = float(args.value_residual_scale)

    config["dual_critic"]["enable_dual_critic"] = bool(args.enable_dual_critic)
    config["dual_critic"]["dual_critic_coef"] = float(args.dual_critic_coef)
    config["dual_critic"]["enable_value_diagnostics"] = not bool(args.disable_value_diagnostics)

    config["reward"]["reward_mode"] = args.reward_mode
    config["reward"]["money_p"] = float(args.money_p)
    config["reward"]["money_tau"] = float(args.money_tau)
    config["reward"]["settled_scale"] = float(args.settled_scale)
    config["reward"]["tau_scaled"] = float(args.tau_scaled)
    config["reward"]["hybrid_alpha"] = float(args.hybrid_alpha)

    if args.val_metric is not None:
        config["train"]["val_metric"] = args.val_metric
    elif args.reward_mode in {"money", "money_normalized", "hybrid_money"}:
        config["train"]["val_metric"] = "eval_money"
    else:
        config["train"]["val_metric"] = "value_accept_ratio"

    update_reward_metadata(config)
    return config

def main() -> None:
    args = parse_args()

    if args.aggregate_only:
        aggregate_results(Path(args.output_dir))
        return

    config = apply_args_to_config(args)

    set_seed(int(config["seed"]))

    paths = build_paths(config)
    ensure_dirs(paths, config)

    print("\n" + "=" * 70)
    print("Phase-Dual-AC Fair Benchmark")
    print("=" * 70)
    print(f"scenario: {paths['scenario']}")
    print(f"results : {paths['result_run_dir']}")
    print(f"train   : {paths['train_pool_path']}")
    print(f"val     : {paths['val_pool_path']}")
    print(f"reward_mode = {config['reward']['reward_mode']}")
    print(f"money_p = {config['reward']['money_p']}")
    print(f"money_tau = {config['reward']['money_tau']}")
    print(f"settled_scale = {config['reward']['settled_scale']}")
    print(f"tau_scaled = {config['reward']['tau_scaled']}")
    print(f"hybrid_alpha = {config['reward']['hybrid_alpha']}")
    print(f"val_metric = {config['train']['val_metric']}")
    print(f"training_reward_formula = {config['reward']['reward_formula']}")
    print(f"risk_residual_scale = {config['residual_risk']['risk_residual_scale']}")
    print(f"value_residual_scale = {config['residual_risk']['value_residual_scale']}")
    print(f"enable_dual_critic = {config['dual_critic']['enable_dual_critic']}")
    print(f"dual_critic_coef = {config['dual_critic']['dual_critic_coef']}")
    print(f"enable_value_diagnostics = {config['dual_critic']['enable_value_diagnostics']}")

    agent, returns, losses, validation_history, gate_history, reward_history = train_agent(
        config,
        paths,
    )

    results = evaluate_agent_cross_regime(
        agent,
        config,
        paths,
    )

    report = build_cross_regime_report_text(results)
    print("\n" + report)

    write_run_outputs(
        config=config,
        paths=paths,
        results=results,
        returns=returns,
        losses=losses,
        validation_history=validation_history,
        gate_history=gate_history,
        reward_history=reward_history,
    )

    if not config["debug_mode"] and config["save_mode"] != "none":
        aggregate_results(Path(config["output"]["output_dir"]))


if __name__ == "__main__":
    main()