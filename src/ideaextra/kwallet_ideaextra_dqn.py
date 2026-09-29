# K-Wallet DQN for ideaextra
# Unified fair-benchmark DQN baseline
#
# Main purpose:
# 1) Use the same 12-regime ideaextra pool protocol as AC/PPO models
# 2) Support original / raw money / normalized money / hybrid money reward
# 3) Support validation-selected best checkpoint
# 4) Output unified metrics / aggregate CSV for comparison with:
#    - Basic PPO
#    - Factorized AC
#    - Dual-Branch AC
#
# This script keeps DQN algorithm unchanged:
# - flat joint-action DQN
# - action_size = (k + 1)^2
# - replay buffer
# - Double DQN target
# - epsilon-greedy exploration
# - target network update

import os
import argparse
import csv
import json
import random
import hashlib
from datetime import datetime
from pathlib import Path
from collections import deque
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


# =========================================================
# Paths
# Default: this script is placed under src/ideaextra/
# =========================================================

IDEA_ROOT = Path(__file__).resolve().parent
DATA_POOL_DIR = IDEA_ROOT / "data" / "pools"
RESULTS_ROOT = IDEA_ROOT / "results" / "dqn_baseline"
CHECKPOINTS_ROOT = IDEA_ROOT / "checkpoints" / "dqn_baseline"


# =========================================================
# Regime constants
# =========================================================

REGIME_ORDER = [
    "US", "TLS", "LNS", "TLNS", "TPLS", "PLS",
    "UB", "TLB", "LNB", "TLNB", "TPLB", "PLB",
]

DEFAULT_MIX_EQ_MASTER = (
    "MIX12_EQ_US_TLS_LNS_TLNS_TPLS_PLS_UB_TLB_LNB_TLNB_TPLB_PLB_master_T1000.npy"
)
DEFAULT_MIX_EQ_VAL = (
    "MIX12_EQ_US_TLS_LNS_TLNS_TPLS_PLS_UB_TLB_LNB_TLNB_TPLB_PLB_val_T1000.npy"
)
DEFAULT_STATIC_EVAL_FILES = {
    r: f"{r}_static_eval_T1000.npy" for r in REGIME_ORDER
}


# =========================================================
# Global config
# =========================================================

CONFIG: Dict[str, Any] = {
    "seed": 123,
    "model_mode": "baseline",
    "debug_mode": False,
    "save_mode": "full",  # none / brief / full

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
        "max_steps": 1000,
        "batch_size": 256,
        "target_update_every": 20,
        "device": "cpu",

        # validation / best checkpoint
        "validate_every": 20,
        "val_num_episodes": 100,
        "val_metric": "value_accept_ratio",
        "use_best_model_for_final_eval": True,
    },

    "eval": {
        "num_episodes": 100,
        "max_steps": 1000,
    },

    "plot": {
        "window": 20,
    },

    "reward": {
        "reward_mode": "original",

        # Raw evaluation-money parameters.
        # Evaluation always uses:
        # eval_money = money_p * settled - money_tau * flushes
        "money_p": 1.0,
        "money_tau": 10.0,

        # PPO/AC-friendly money scaling parameters.
        # money_normalized:
        # reward_t = settled_t / settled_scale - tau_scaled * flush_indicator_t
        "settled_scale": 50.0,
        "tau_scaled": 0.2,

        # hybrid_money:
        # reward_t = env_reward_t + hybrid_alpha * normalized_money_reward_t
        "hybrid_alpha": 0.1,

        "reward_formula": "reward_t = env_reward_t",
        "objective_label": "original_reward",
        "training_objective": "original_reward",
    },

    "output": {
        "output_dir": str(RESULTS_ROOT),
        "checkpoint_dir": str(CHECKPOINTS_ROOT),
    },
}


# =========================================================
# Reward constants
# =========================================================

REFRESH_COST = 0.01
IMBALANCE_PENALTY = 0.02
WASTEFUL_REFRESH_PENALTY = 0.02
WASTEFUL_REFRESH_THRESH = 0.6

LOG_EVERY_N = 50

RAW_MONEY_REWARD_FORMULA = "reward_t = money_p * settled_t - money_tau * flush_indicator_t"
MONEY_NORMALIZED_REWARD_FORMULA = "reward_t = settled_t / settled_scale - tau_scaled * flush_indicator_t"
HYBRID_MONEY_REWARD_FORMULA = (
    "reward_t = env_reward_t + hybrid_alpha * "
    "(settled_t / settled_scale - tau_scaled * flush_indicator_t)"
)
ORIGINAL_REWARD_FORMULA = "reward_t = env_reward_t"
EVALUATION_MONEY_FORMULA = "eval_money = money_p * settled - money_tau * flushes"


# =========================================================
# Utility functions
# =========================================================

def set_seed(seed: int = 123) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_run_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def format_reward_value(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def format_c_value(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def build_mix_eq_master_filename(T: int) -> str:
    regimes = "_".join(REGIME_ORDER)
    return f"MIX12_EQ_{regimes}_master_T{T}.npy"


def build_mix_eq_val_filename(T: int) -> str:
    regimes = "_".join(REGIME_ORDER)
    return f"MIX12_EQ_{regimes}_val_T{T}.npy"


def build_static_eval_files(T: int) -> Dict[str, str]:
    return {r: f"{r}_static_eval_T{T}.npy" for r in REGIME_ORDER}


def train_pool_file_for_regime(regime: str, T: int) -> str:
    if regime == "MIX12_EQ":
        return build_mix_eq_master_filename(T)
    return f"{regime}_static_master_T{T}.npy"


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
        info["flushes_this_step"] = (
            1 if flush_choice is not None and flush_choice < int(config["env"]["k"]) else 0
        )

    settled_value = float(info["settled_value"])
    flushes_this_step = float(info["flushes_this_step"])

    # Raw money is always recorded and used for evaluation diagnostics.
    money_reward = (
        float(reward_cfg["money_p"]) * settled_value
        - float(reward_cfg["money_tau"]) * flushes_this_step
    )

    # Normalized money uses a PPO/AC-friendly scale.
    normalized_money_reward = (
        settled_value / float(reward_cfg["settled_scale"])
        - float(reward_cfg["tau_scaled"]) * flushes_this_step
    )

    reward_mode = reward_cfg["reward_mode"]

    if reward_mode == "original":
        selected_reward = float(env_reward)

    elif reward_mode == "money":
        selected_reward = money_reward

    elif reward_mode == "money_normalized":
        selected_reward = normalized_money_reward

    elif reward_mode == "hybrid_money":
        selected_reward = (
            float(env_reward)
            + float(reward_cfg["hybrid_alpha"]) * normalized_money_reward
        )

    else:
        raise ValueError(f"Unsupported reward_mode={reward_mode}")

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
    train_regime = config["data"]["train_regime"]
    test_regimes = "_".join(config["data"]["test_pool_files"].keys())

    scenario = (
        f"{config['model_mode']}_train{train_regime}_cross{test_regimes}_"
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

    return scenario


def build_title_tag(config: Dict[str, Any], run_stamp: str) -> str:
    env = config["env"]
    c_value = format_c_value(float(env["C"]))
    train_regime = config["data"]["train_regime"]

    return (
        f"{run_stamp} | mode={config['model_mode']} | train={train_regime} | "
        f"C={c_value} k={env['k']} T={env['T']} F={env['F']}"
    )


def build_paths(config: Dict[str, Any]) -> Dict[str, str]:
    scenario = build_scenario_name(config)
    run_stamp = build_run_stamp()

    result_root = Path(config["output"]["output_dir"]).expanduser().resolve()
    checkpoint_root = Path(config["output"]["checkpoint_dir"]).expanduser().resolve()

    result_run_dir = result_root / "runs" / scenario / run_stamp
    checkpoint_run_dir = checkpoint_root / scenario / run_stamp
    aggregate_dir = result_root / "aggregates"
    plot_dir = result_root / "plots" / scenario / run_stamp

    train_pool_file = config["data"]["train_pool_file"]
    val_pool_file = config["data"]["val_pool_file"]
    test_pool_files = config["data"]["test_pool_files"]

    return {
        "idea_root": str(IDEA_ROOT),
        "data_pool_dir": str(DATA_POOL_DIR),
        "result_root": str(result_root),
        "checkpoint_root": str(checkpoint_root),

        "scenario": scenario,
        "run_stamp": run_stamp,
        "title_tag": build_title_tag(config, run_stamp),

        "train_regime": config["data"]["train_regime"],
        "train_pool_path": str(DATA_POOL_DIR / train_pool_file),
        "val_pool_path": str(DATA_POOL_DIR / val_pool_file),
        "test_pool_paths": {k: str(DATA_POOL_DIR / v) for k, v in test_pool_files.items()},

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
        "validation_plot_path": str(plot_dir / "validation_curve.png"),

        "last_model_path": str(checkpoint_run_dir / "last_model.pth"),
        "best_model_path": str(checkpoint_run_dir / "best_model.pth"),
    }


def ensure_dirs(paths: Dict[str, str], config: Dict[str, Any]) -> None:
    os.makedirs(paths["data_pool_dir"], exist_ok=True)

    if config["debug_mode"] or config["save_mode"] == "none":
        return

    for key in ["result_run_dir", "checkpoint_run_dir", "aggregate_dir", "plot_dir"]:
        os.makedirs(paths[key], exist_ok=True)


def save_json(payload: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_tx_pool(pool_path: str, expected_steps: int) -> np.ndarray:
    if not os.path.exists(pool_path):
        raise FileNotFoundError(f"Pool file not found: {pool_path}")

    if pool_path.endswith(".json"):
        with open(pool_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tx_pool = np.array(data, dtype=np.int32)
    elif pool_path.endswith(".npy"):
        tx_pool = np.load(pool_path)
    else:
        raise ValueError(f"Unsupported file format: {pool_path}")

    if tx_pool.ndim != 2:
        raise ValueError(
            f"tx_pool must be 2D [num_episodes, steps], got ndim={tx_pool.ndim}"
        )

    if tx_pool.shape[1] != expected_steps:
        raise ValueError(
            f"Expected each episode to have {expected_steps} steps, "
            f"but tx_pool.shape[1]={tx_pool.shape[1]}"
        )

    return tx_pool.astype(np.int32)


def verify_data_integrity(pool_path: str, expected_steps: int, label: str = "") -> bool:
    print("\n" + "=" * 70)
    print(f"Data integrity check {label}".strip())
    print("=" * 70)

    try:
        tx_pool = load_tx_pool(pool_path, expected_steps=expected_steps)
        file_size = os.path.getsize(pool_path) / 1024
        pool_hash = hashlib.md5(tx_pool.tobytes()).hexdigest()

        print(f"Loaded file: {pool_path}")
        print(f"Shape: {tx_pool.shape}")
        print(f"Size: {file_size:.2f} KB")
        print(f"MD5: {pool_hash}")
        print(f"First episode first 5 tx: {tx_pool[0, :5].tolist()}")
        print("=" * 70 + "\n")
        return True

    except Exception as e:
        print(f"Data verification failed: {str(e)}")
        return False


def build_pool_fingerprints(config: Dict[str, Any], paths: Dict[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    pool_items = {
        "train": paths["train_pool_path"],
        "val": paths["val_pool_path"],
        **{f"test_{k}": v for k, v in paths["test_pool_paths"].items()},
    }

    expected_steps = int(config["train"]["max_steps"])

    for name, path in pool_items.items():
        if not os.path.exists(path):
            out[name] = {"path": path, "exists": False}
            continue

        arr = load_tx_pool(path, expected_steps=expected_steps)
        out[name] = {
            "path": path,
            "exists": True,
            "shape": list(arr.shape),
            "md5": hashlib.md5(arr.tobytes()).hexdigest(),
        }

    return out


# =========================================================
# DQN network
# =========================================================

class DQN(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


# =========================================================
# K-Wallet environment
# Local copy matching ideaextra DQN protocol
# =========================================================

class KWalletEnv:
    def __init__(
        self,
        C: float = 1200,
        k: int = 3,
        F: int = 3,
        max_transaction: int = 1000,
        max_steps: int = 1000,
        seed: int = 123,
        enable_shaping: bool = False,
    ):
        self.C = float(C)
        self.k = int(k)
        self.F = int(F)
        self.max_transaction = int(max_transaction)
        self.max_steps = int(max_steps)
        self.wallet_size = self.C / self.k
        self.num_actions = (self.k + 1) ** 2

        self.alpha_drop = 0.02
        self.beta_flush = REFRESH_COST
        self.enable_shaping = enable_shaping

        if self.enable_shaping:
            self.IMBALANCE_PENALTY = IMBALANCE_PENALTY
            self.WASTEFUL_REFRESH_PENALTY = WASTEFUL_REFRESH_PENALTY
            self.WASTEFUL_REFRESH_THRESH = WASTEFUL_REFRESH_THRESH
            self.INVALID_ACTION_PENALTY = 0.05

        self.rng = np.random.default_rng(seed)
        self._tx_stream = None
        self.reset()

    def reset(self, tx_stream: Optional[np.ndarray] = None) -> np.ndarray:
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

        if tx_stream is not None:
            self._tx_stream = list(tx_stream)
        else:
            self._tx_stream = [
                int(self.rng.integers(1, self.max_transaction + 1))
                for _ in range(self.max_steps)
            ]

        self.current_tx = self._tx_stream[self.time]
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        state = []

        for w in self.wallets:
            state.append(w / self.wallet_size)

        for i in range(self.k):
            state.append(0.0 if self._usable(i) else 1.0)

        for i in range(self.k):
            rem = max(0, self.freeze_until[i] - self.time)
            state.append((rem / self.F) if self.F > 0 else 0.0)

        state.append(self.current_tx / self.max_transaction)
        return np.array(state, dtype=np.float32)

    def _usable(self, i: int) -> bool:
        return self.time > self.freeze_until[i]

    def _decode_action(self, action_int: int) -> Tuple[int, int]:
        if not (0 <= action_int < self.num_actions):
            raise ValueError(
                f"Action out of range: action={action_int}, "
                f"valid range=[0, {self.num_actions - 1}]"
            )

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

        # 1) flush
        if flush_choice < self.k:
            if self._usable(flush_choice):
                self.pending_refill[flush_choice] = True
                self.wallets[flush_choice] = 0.0
                self.freeze_until[flush_choice] = self.time + self.F - 1
                self.num_flushes += 1
                flushes_this_step += 1
                refresh_targets.append(flush_choice)
            else:
                if self.enable_shaping:
                    reward -= getattr(self, "INVALID_ACTION_PENALTY", 0.05)

        fit_idx = None

        # 2) settle
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

        # 3) flush cost
        reward -= self.beta_flush * flushes_this_step

        # 4) optional shaping
        if self.enable_shaping:
            usable_balances = [
                self.wallets[i] for i in range(self.k)
                if self._usable(i)
            ]

            if len(usable_balances) >= 2:
                std_norm = float(np.std(np.array(usable_balances)) / self.wallet_size)
                reward -= IMBALANCE_PENALTY * std_norm

            for i in refresh_targets:
                if (pre_refresh_balances[i] / self.wallet_size) >= WASTEFUL_REFRESH_THRESH:
                    reward -= WASTEFUL_REFRESH_PENALTY

        # 5) advance time
        self.time += 1

        # 6) refill when freeze ends
        for i in range(self.k):
            if self.pending_refill[i] and self._usable(i):
                self.wallets[i] = self.wallet_size
                self.pending_refill[i] = False

        # 7) next tx
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


# =========================================================
# DQN Agent
# =========================================================

class DQNAgent:
    def __init__(self, state_size: int, action_size: int, device: str = "cpu"):
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.device = torch.device(device)

        self.memory = deque(maxlen=20000)
        self.gamma = 0.98
        self.epsilon = 0.8
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.999

        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        self.update_target_network()

    def update_target_network(self) -> None:
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, s, a, r, s2, done) -> None:
        self.memory.append((s, a, r, s2, done))

    def act(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q = self.model(s)
            return int(torch.argmax(q, dim=1).item())

    def replay(self, batch_size: int = 128) -> Optional[Dict[str, float]]:
        if len(self.memory) < batch_size:
            return None

        batch = random.sample(self.memory, batch_size)
        s, a, r, s2, d = zip(*batch)

        s = torch.tensor(np.array(s), dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.int64, device=self.device)
        r = torch.tensor(r, dtype=torch.float32, device=self.device)
        s2 = torch.tensor(np.array(s2), dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device)

        # Double DQN target
        with torch.no_grad():
            next_online_q = self.model(s2)
            next_act = next_online_q.argmax(dim=1)
            next_target_q = self.target_model(s2)
            q_next = next_target_q.gather(1, next_act.unsqueeze(1)).squeeze(1)
            y = r + self.gamma * (1.0 - d) * q_next

        q = self.model(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = nn.SmoothL1Loss()(q, y)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()

        return {"loss": float(loss.item())}

    def decay_epsilon(self) -> None:
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# =========================================================
# Evaluation
# =========================================================

def summarize_episode_metrics(all_results: List[Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
    if not all_results:
        raise ValueError("No episode metrics to summarize.")

    summary: Dict[str, Dict[str, Any]] = {}
    metric_names = all_results[0].keys()

    for metric in metric_names:
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


def evaluate_agent_on_array(
    agent: DQNAgent,
    config: Dict[str, Any],
    tx_pool: np.ndarray,
    label: str,
    num_eval_episodes: int,
    max_steps: int,
) -> Dict[str, Any]:
    env_cfg = config["env"]

    num_eval_episodes = min(int(num_eval_episodes), tx_pool.shape[0])

    env = KWalletEnv(
        C=env_cfg["C"],
        k=env_cfg["k"],
        F=env_cfg["F"],
        max_transaction=env_cfg["T"],
        max_steps=max_steps,
        seed=config["seed"],
        enable_shaping=env_cfg["enable_shaping"],
    )

    old_eps = agent.epsilon
    agent.epsilon = 0.0

    all_results: List[Dict[str, float]] = []

    for ep in range(num_eval_episodes):
        current_tx_stream = tx_pool[ep]
        state = env.reset(tx_stream=current_tx_stream)

        episode_total_requested_value = 0.0
        episode_total_tx_count = 0
        episode_accepted_count = 0

        for _ in range(max_steps):
            current_tx = env.current_tx
            episode_total_requested_value += float(current_tx)
            episode_total_tx_count += 1

            action = agent.act(state)
            state, _, done, info = env.step(action)

            if info.get("accepted", False):
                episode_accepted_count += 1

            if done:
                break

        metrics = env.get_metrics()
        metrics["value_accept_ratio"] = (
            metrics["settled"] / episode_total_requested_value
            if episode_total_requested_value > 0
            else 0.0
        )
        metrics["count_accept_ratio"] = (
            episode_accepted_count / episode_total_tx_count
            if episode_total_tx_count > 0
            else 0.0
        )
        metrics["total_requested_value"] = episode_total_requested_value
        metrics["total_tx_count"] = episode_total_tx_count
        metrics["accepted_count"] = episode_accepted_count

        add_eval_money_metrics(metrics, config)
        all_results.append(metrics)

    summary = summarize_episode_metrics(all_results)
    add_reward_summary_metadata(summary, config)

    agent.epsilon = old_eps

    return {
        "label": label,
        "num_episodes": num_eval_episodes,
        "summary": summary,
        "raw_results": all_results,
    }


def evaluate_agent_on_pool(
    agent: DQNAgent,
    config: Dict[str, Any],
    test_pool_path: str,
    test_regime: str,
) -> Dict[str, Any]:
    eval_cfg = config["eval"]

    if not verify_data_integrity(
        test_pool_path,
        expected_steps=int(eval_cfg["max_steps"]),
        label=f"(regime={test_regime})",
    ):
        raise RuntimeError(f"Data verification failed for regime={test_regime}")

    tx_pool = load_tx_pool(test_pool_path, expected_steps=int(eval_cfg["max_steps"]))

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


def compute_cross_regime_aggregate_local(cross_results: Dict[str, Any]) -> Dict[str, float]:
    value_accept_ratios = []
    drops = []
    flushes = []
    eval_money_values = []

    for regime_result in cross_results.values():
        summary = regime_result["summary"]
        value_accept_ratios.append(float(summary["value_accept_ratio"]["mean"]))
        drops.append(float(summary["drops"]["mean"]))
        flushes.append(float(summary["flushes"]["mean"]))
        eval_money_values.append(float(summary["eval_money"]["mean"]))

    return {
        "mean_value_accept_ratio": float(np.mean(value_accept_ratios)),
        "worst_regime_value_accept_ratio": float(np.min(value_accept_ratios)),
        "std_value_accept_ratio_across_regimes": float(np.std(value_accept_ratios)),
        "mean_drops": float(np.mean(drops)),
        "mean_flushes": float(np.mean(flushes)),
        "mean_eval_money": float(np.mean(eval_money_values)),
        "worst_regime_eval_money": float(np.min(eval_money_values)),
        "std_eval_money_across_regimes": float(np.std(eval_money_values)),
    }


def evaluate_agent_cross_regime(
    agent: DQNAgent,
    config: Dict[str, Any],
    paths: Dict[str, str],
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("DQN Cross-Regime Evaluation")
    print("=" * 70)

    cross_results = {}

    for regime_name, pool_path in paths["test_pool_paths"].items():
        cross_results[regime_name] = evaluate_agent_on_pool(
            agent=agent,
            config=config,
            test_pool_path=pool_path,
            test_regime=regime_name,
        )

    aggregate = compute_cross_regime_aggregate_local(cross_results)

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
    paths: Dict[str, str],
):
    print("\n" + "=" * 70)
    print("Train DQN Baseline")
    print("=" * 70)

    train_cfg = config["train"]
    env_cfg = config["env"]

    train_pool = load_tx_pool(
        pool_path=paths["train_pool_path"],
        expected_steps=int(train_cfg["max_steps"]),
    )
    val_pool = load_tx_pool(
        pool_path=paths["val_pool_path"],
        expected_steps=int(train_cfg["max_steps"]),
    )

    if int(train_cfg["episodes"]) > train_pool.shape[0]:
        raise ValueError(
            f"episodes={train_cfg['episodes']} exceeds train pool rows={train_pool.shape[0]}"
        )

    print(f"train_pool={train_pool.shape}")
    print(f"val_pool={val_pool.shape}")
    print(f"train_pool_path={paths['train_pool_path']}")
    print(f"val_pool_path={paths['val_pool_path']}")

    env = KWalletEnv(
        C=env_cfg["C"],
        k=env_cfg["k"],
        F=env_cfg["F"],
        max_transaction=env_cfg["T"],
        max_steps=int(train_cfg["max_steps"]),
        seed=config["seed"],
        enable_shaping=env_cfg["enable_shaping"],
    )

    state_size = len(env._get_state())
    action_size = env.num_actions
    agent = DQNAgent(state_size, action_size, device=train_cfg["device"])

    print(f"model_mode={config['model_mode']}")
    print(f"env C={env.C} k={env.k} F={env.F} T={env.max_transaction}")
    print(f"state={state_size} action={action_size}")
    print(f"episodes={train_cfg['episodes']}")
    print(f"val_metric={train_cfg['val_metric']}")

    returns: List[float] = []
    loss_history: List[float] = []
    epsilons: List[float] = []
    validation_history: List[Dict[str, float]] = []
    reward_history: List[Dict[str, Any]] = []

    best_val_score = -1e18
    best_state_dict = None
    best_val_snapshot = None

    for ep in range(int(train_cfg["episodes"])):
        current_tx_stream = train_pool[ep]
        state = env.reset(tx_stream=current_tx_stream)

        episode_return = 0.0
        episode_original_reward = 0.0
        episode_settled_value = 0.0
        episode_flushes = 0.0
        episode_money_reward = 0.0

        step_rewards: List[float] = []
        step_money_rewards: List[float] = []

        settle_action_counts = [0 for _ in range(env.k + 1)]
        flush_action_counts = [0 for _ in range(env.k + 1)]

        for step_i in range(int(train_cfg["max_steps"])):
            action = agent.act(state)
            settle = int(action // (env.k + 1))
            flush = int(action % (env.k + 1))

            settle_action_counts[settle] += 1
            flush_action_counts[flush] += 1

            next_state, env_reward, done, info = env.step(action)

            if ep == 0 and step_i == 0:
                print(f"[RewardInfo] first env.step info keys: {sorted(info.keys())}")

            selected_reward, money_reward, settled_value, flushes_this_step = select_training_reward(
                env_reward,
                info,
                config,
            )

            agent.remember(state, action, selected_reward, next_state, done)

            state = next_state
            episode_return += selected_reward
            episode_original_reward += float(env_reward)
            episode_settled_value += settled_value
            episode_flushes += flushes_this_step
            episode_money_reward += money_reward

            step_rewards.append(float(selected_reward))
            step_money_rewards.append(float(money_reward))

            metrics = agent.replay(batch_size=int(train_cfg["batch_size"]))
            if metrics is not None and "loss" in metrics:
                loss_history.append(metrics["loss"])

            if done:
                break

        agent.decay_epsilon()

        if (ep + 1) % int(train_cfg["target_update_every"]) == 0:
            agent.update_target_network()

        returns.append(float(episode_return))
        epsilons.append(float(agent.epsilon))

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

        validate_every = int(train_cfg["validate_every"])
        if validate_every > 0 and (
            (ep + 1) % validate_every == 0
            or (ep + 1) == int(train_cfg["episodes"])
        ):
            val_result = evaluate_agent_on_array(
                agent=agent,
                config=config,
                tx_pool=val_pool,
                label="VAL",
                num_eval_episodes=int(train_cfg["val_num_episodes"]),
                max_steps=int(train_cfg["max_steps"]),
            )

            metric_name = train_cfg["val_metric"]
            val_score = float(val_result["summary"][metric_name]["mean"])

            val_row = {
                "episode": ep + 1,
                "metric": metric_name,
                "score": val_score,
                "value_accept_ratio": float(val_result["summary"]["value_accept_ratio"]["mean"]),
                "eval_money": float(val_result["summary"]["eval_money"]["mean"]),
                "settled": float(val_result["summary"]["settled"]["mean"]),
                "drops": float(val_result["summary"]["drops"]["mean"]),
                "flushes": float(val_result["summary"]["flushes"]["mean"]),
                "drop_rate": float(val_result["summary"]["drop_rate"]["mean"]),
                "count_accept_ratio": float(val_result["summary"]["count_accept_ratio"]["mean"]),
                "utilization": float(val_result["summary"]["utilization"]["mean"]),
            }
            validation_history.append(val_row)

            print(
                f"[Val  ] ep={ep + 1:4d} "
                f"{metric_name}={val_score:.4f} "
                f"val_acc={val_row['value_accept_ratio']:.4f} "
                f"eval_money={val_row['eval_money']:.2f} "
                f"drops={val_row['drops']:.2f} flushes={val_row['flushes']:.2f}"
            )

            if val_score > best_val_score:
                best_val_score = val_score
                best_val_snapshot = val_row
                best_state_dict = {
                    k: v.detach().cpu().clone()
                    for k, v in agent.model.state_dict().items()
                }

                if (not config["debug_mode"]) and config["save_mode"] == "full":
                    torch.save(best_state_dict, paths["best_model_path"])
                    print(f"[Val  ] best checkpoint saved: ep={ep + 1}, score={val_score:.4f}")

        if (ep + 1) % LOG_EVERY_N == 0 or ep == 0:
            recent_returns = returns[max(0, len(returns) - LOG_EVERY_N):]
            mean_recent_return = float(np.mean(recent_returns))
            print(
                f"[Train] ep={ep + 1:4d}/{train_cfg['episodes']} "
                f"return={episode_return:10.2f} "
                f"recent={mean_recent_return:10.2f} "
                f"money={episode_money_reward:10.2f} "
                f"eps={agent.epsilon:.4f}"
            )

    if best_state_dict is not None and train_cfg["use_best_model_for_final_eval"]:
        agent.model.load_state_dict(best_state_dict)
        agent.update_target_network()
        print(
            f"Loaded best checkpoint: "
            f"ep={best_val_snapshot['episode']} score={best_val_snapshot['score']:.4f}"
        )

    if (not config["debug_mode"]) and config["save_mode"] == "full":
        torch.save(agent.model.state_dict(), paths["last_model_path"])
        print(f"Last model saved to: {paths['last_model_path']}")

    return agent, returns, loss_history, epsilons, validation_history, reward_history


# =========================================================
# Reporting / saving / plotting
# =========================================================

def build_cross_regime_report_text(results: Dict[str, Any]) -> str:
    lines = []

    lines.append("=" * 112)
    lines.append("DQN Baseline Cross-Regime Evaluation")
    lines.append("=" * 112)
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
    lines.append("-" * 112)

    lines.append(
        f"{'Test':<8}"
        f"{'ValAcc(%)':>14}"
        f"{'EvalMoney':>14}"
        f"{'Drops':>12}"
        f"{'Flushes':>12}"
        f"{'DropRate(%)':>14}"
        f"{'CntAcc(%)':>14}"
    )
    lines.append("-" * 112)

    for regime_name, regime_result in results["test_results"].items():
        summary = regime_result["summary"]
        lines.append(
            f"{regime_name:<8}"
            f"{100 * summary['value_accept_ratio']['mean']:>14.2f}"
            f"{summary['eval_money']['mean']:>14.2f}"
            f"{summary['drops']['mean']:>12.2f}"
            f"{summary['flushes']['mean']:>12.2f}"
            f"{100 * summary['drop_rate']['mean']:>14.2f}"
            f"{100 * summary['count_accept_ratio']['mean']:>14.2f}"
        )

    agg = results["aggregate"]

    lines.append("-" * 112)
    lines.append(f"Mean ValAcc (%)        : {100 * agg['mean_value_accept_ratio']:.4f}")
    lines.append(f"Worst-Regime ValAcc (%): {100 * agg['worst_regime_value_accept_ratio']:.4f}")
    lines.append(f"Std Across Regimes     : {agg['std_value_accept_ratio_across_regimes']:.6f}")
    lines.append(f"Mean Drops             : {agg['mean_drops']:.4f}")
    lines.append(f"Mean Flushes           : {agg['mean_flushes']:.4f}")
    lines.append(f"Mean EvalMoney         : {agg['mean_eval_money']:.4f}")
    lines.append(f"Worst-Regime EvalMoney : {agg['worst_regime_eval_money']:.4f}")
    lines.append(f"Std EvalMoney          : {agg['std_eval_money_across_regimes']:.6f}")
    lines.append("=" * 112)

    return "\n".join(lines)


def print_cross_regime_report(results: Dict[str, Any]) -> None:
    print("\n" + build_cross_regime_report_text(results))


def save_results(results: Dict[str, Any], save_path: str) -> None:
    compact_results = {
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
        compact_results["test_results"][regime_name] = {
            "num_episodes": regime_result["num_episodes"],
            "summary": {},
        }

        for metric, data in regime_result["summary"].items():
            if isinstance(data, dict) and "mean" in data:
                compact_results["test_results"][regime_name]["summary"][metric] = {
                    "mean": data["mean"],
                    "std": data["std"],
                    "min": data["min"],
                    "max": data["max"],
                    "median": data["median"],
                }
            else:
                compact_results["test_results"][regime_name]["summary"][metric] = data

    save_json(compact_results, save_path)
    print(f"Results saved to: {save_path}")


def save_training_history(
    returns: List[float],
    loss_history: List[float],
    epsilons: List[float],
    validation_history: List[Dict[str, float]],
    reward_history: List[Dict[str, Any]],
    save_path: str,
    config: Dict[str, Any],
) -> None:
    payload = {
        "returns": returns,
        "loss_history": loss_history,
        "epsilons": epsilons,
        "validation_history": validation_history,
        "reward_history": reward_history,
        **reward_metadata(config),
    }
    save_json(payload, save_path)
    print(f"Training history saved to: {save_path}")


def moving_average(values: List[float], window: int) -> np.ndarray:
    if len(values) == 0:
        return np.array([])

    arr = np.array(values, dtype=float)
    out = np.zeros_like(arr)

    for i in range(len(arr)):
        left = max(0, i - window + 1)
        out[i] = np.mean(arr[left:i + 1])

    return out


def plot_training_curves(
    returns: List[float],
    loss_history: List[float],
    epsilons: List[float],
    save_path: str,
    title_tag: str,
    window: int = 20,
) -> None:
    fig = plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(returns, alpha=0.35, label="Return")
    if len(returns) > 0:
        plt.plot(moving_average(returns, window), linewidth=2, label=f"MA({window})")
    plt.title(f"DQN Training Curves\n{title_tag}")
    plt.ylabel("Episode Return")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(3, 1, 2)
    if len(loss_history) > 0:
        plt.plot(loss_history, alpha=0.8)
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 3)
    plt.plot(epsilons)
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Training plot saved to: {save_path}")


def plot_validation_curve(
    validation_history: List[Dict[str, float]],
    save_path: str,
    title_tag: str,
) -> None:
    if len(validation_history) == 0:
        return

    xs = [d["episode"] for d in validation_history]
    scores = [d["score"] for d in validation_history]
    metric_name = validation_history[-1].get("metric", "score")

    plt.figure(figsize=(10, 5))
    plt.plot(xs, scores, marker="o")
    plt.title(f"DQN Validation Curve ({metric_name})\n{title_tag}")
    plt.xlabel("Episode")
    plt.ylabel(metric_name)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Validation plot saved to: {save_path}")


def plot_evaluation_results(results: Dict[str, Any], save_path: str, title_tag: str) -> None:
    test_results = results["test_results"]
    regimes = list(test_results.keys())

    chart_items = [
        ("value_accept_ratio", "Value Accept Ratio (%)", True),
        ("eval_money", "Eval Money", False),
        ("drops", "Drops", False),
        ("flushes", "Flushes", False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"DQN Cross-Regime Evaluation\n{title_tag}", fontsize=15, fontweight="bold")

    for idx, (metric, label, as_percent) in enumerate(chart_items):
        ax = axes[idx // 2, idx % 2]
        values = []

        for regime in regimes:
            value = test_results[regime]["summary"][metric]["mean"]
            if as_percent:
                value *= 100
            values.append(value)

        ax.bar(regimes, values)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Test Regime", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Evaluation plot saved to: {save_path}")


def write_run_outputs(
    config: Dict[str, Any],
    paths: Dict[str, str],
    results: Dict[str, Any],
    returns: List[float],
    loss_history: List[float],
    epsilons: List[float],
    validation_history: List[Dict[str, float]],
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
    print(f"Summary table saved to: {paths['summary_txt_path']}")

    save_training_history(
        returns=returns,
        loss_history=loss_history,
        epsilons=epsilons,
        validation_history=validation_history,
        reward_history=reward_history,
        save_path=paths["training_history_path"],
        config=config,
    )

    save_json(
        {
            "validation_history": validation_history,
            **reward_metadata(config),
        },
        paths["validation_history_path"],
    )

    if config["save_mode"] == "full":
        plot_training_curves(
            returns,
            loss_history,
            epsilons,
            save_path=paths["training_plot_path"],
            title_tag=paths["title_tag"],
            window=int(config["plot"]["window"]),
        )

        plot_validation_curve(
            validation_history,
            save_path=paths["validation_plot_path"],
            title_tag=paths["title_tag"],
        )

        plot_evaluation_results(
            results,
            save_path=paths["eval_plot_path"],
            title_tag=paths["title_tag"],
        )


# =========================================================
# Aggregation
# =========================================================

def safe_summary_mean(summary: Dict[str, Any], metric: str, default: float = 0.0) -> float:
    data = summary.get(metric)
    if isinstance(data, dict) and "mean" in data:
        return float(data["mean"])
    return float(default)


def flatten_result_for_csv(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    config = payload["config"]
    aggregate = payload.get("aggregate", {})

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

    rows: List[Dict[str, Any]] = []

    for regime, regime_result in payload["test_results"].items():
        summary = regime_result["summary"]

        settled = safe_summary_mean(summary, "settled")
        flushes = safe_summary_mean(summary, "flushes")
        eval_money = safe_summary_mean(
            summary,
            "eval_money",
            money_p * settled - money_tau * flushes,
        )

        rows.append(
            {
                "scenario": payload["scenario"],
                "model_mode": payload.get("model_mode", config.get("model_mode", "baseline")),
                "train_regime": payload["train_regime"],
                "seed": payload.get("seed", config.get("seed")),
                "test_regime": regime,

                "reward_mode": reward_mode,
                "training_objective": training_objective,
                "money_p": money_p,
                "money_tau": money_tau,
                "settled_scale": settled_scale,
                "tau_scaled": tau_scaled,
                "hybrid_alpha": hybrid_alpha,

                "value_accept_ratio": safe_summary_mean(summary, "value_accept_ratio"),
                "settled": settled,
                "eval_money": eval_money,
                "drops": safe_summary_mean(summary, "drops"),
                "flushes": flushes,
                "drop_rate": safe_summary_mean(summary, "drop_rate"),
                "count_accept_ratio": safe_summary_mean(summary, "count_accept_ratio"),

                "mean_value_accept_ratio": aggregate.get("mean_value_accept_ratio"),
                "worst_regime_value_accept_ratio": aggregate.get("worst_regime_value_accept_ratio"),
                "std_value_accept_ratio_across_regimes": aggregate.get(
                    "std_value_accept_ratio_across_regimes"
                ),

                "mean_eval_money": aggregate.get("mean_eval_money"),
                "worst_regime_eval_money": aggregate.get("worst_regime_eval_money"),
                "std_eval_money_across_regimes": aggregate.get(
                    "std_eval_money_across_regimes"
                ),

                "C": config["env"]["C"],
                "k": config["env"]["k"],
                "F": config["env"]["F"],
                "T": config["env"]["T"],
            }
        )

    return rows


def aggregate_results(result_root: Path = RESULTS_ROOT) -> Path:
    result_files = sorted((result_root / "runs").glob("**/cross_regime_results.json"))

    rows: List[Dict[str, Any]] = []
    for path in result_files:
        rows.extend(flatten_result_for_csv(path))

    aggregate_dir = result_root / "aggregates"
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    out_path = aggregate_dir / "dqn_fair_benchmark_aggregated.csv"

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
    parser = argparse.ArgumentParser(description="Unified DQN fair benchmark for K-wallet RL.")

    parser.add_argument("--model_mode", choices=["baseline"], default=CONFIG["model_mode"])
    parser.add_argument("--train_regime", type=str, default=None)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--C", type=float, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--F", type=int, default=None)
    parser.add_argument("--T", type=int, default=None)

    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--eval_episodes", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--save_mode", choices=["none", "brief", "full"], default=None)
    parser.add_argument("--debug_mode", action="store_true")

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

    parser.add_argument(
        "--val_metric",
        choices=["value_accept_ratio", "eval_money"],
        default=None,
        help=(
            "Validation metric for best checkpoint selection. "
            "If omitted, money mode uses eval_money and original mode uses value_accept_ratio."
        ),
    )

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--aggregate_only", action="store_true")

    return parser.parse_args()


def apply_args_to_config(args: argparse.Namespace) -> Dict[str, Any]:
    config = json.loads(json.dumps(CONFIG))

    config["model_mode"] = args.model_mode

    if args.seed is not None:
        config["seed"] = int(args.seed)

    if args.C is not None:
        config["env"]["C"] = float(args.C)

    if args.k is not None:
        config["env"]["k"] = int(args.k)

    if args.F is not None:
        config["env"]["F"] = int(args.F)

    if args.T is not None:
        config["env"]["T"] = int(args.T)
        config["train"]["max_steps"] = int(args.T)
        config["eval"]["max_steps"] = int(args.T)

    if args.episodes is not None:
        config["train"]["episodes"] = int(args.episodes)

    if args.eval_episodes is not None:
        config["eval"]["num_episodes"] = int(args.eval_episodes)
        config["train"]["val_num_episodes"] = int(args.eval_episodes)

    if args.device is not None:
        config["train"]["device"] = args.device

    if args.save_mode is not None:
        config["save_mode"] = args.save_mode

    if args.debug_mode:
        config["debug_mode"] = True

    if args.output_dir is not None:
        config["output"]["output_dir"] = args.output_dir

    if args.checkpoint_dir is not None:
        config["output"]["checkpoint_dir"] = args.checkpoint_dir

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

    T = int(config["env"]["T"])

    train_regime = args.train_regime if args.train_regime is not None else config["data"]["train_regime"]
    config["data"]["train_regime"] = train_regime

    if train_regime == "MIX12_EQ":
        config["data"]["train_pool_file"] = build_mix_eq_master_filename(T)
        config["data"]["val_pool_file"] = build_mix_eq_val_filename(T)
    else:
        config["data"]["train_pool_file"] = train_pool_file_for_regime(train_regime, T)
        config["data"]["val_pool_file"] = build_mix_eq_val_filename(T)

    config["data"]["test_pool_files"] = build_static_eval_files(T)

    update_reward_metadata(config)
    return config


# =========================================================
# Main
# =========================================================

def main() -> None:
    args = parse_args()
    config = apply_args_to_config(args)

    if args.aggregate_only:
        aggregate_results(Path(config["output"]["output_dir"]).expanduser().resolve())
        return

    print("\n" + "=" * 70)
    print("K-Wallet DQN Fair Benchmark")
    print("=" * 70)

    set_seed(int(config["seed"]))

    paths = build_paths(config)
    ensure_dirs(paths, config)

    print(f"scenario: {paths['scenario']}")
    print(f"model_mode = {config['model_mode']}")
    print(f"train_regime = {paths['train_regime']}")
    print(f"run_stamp = {paths['run_stamp']}")
    print(f"idea_root = {paths['idea_root']}")
    print(f"data_pool_dir = {paths['data_pool_dir']}")
    print(f"train_pool = {paths['train_pool_path']}")
    print(f"val_pool = {paths['val_pool_path']}")
    print(f"save_mode = {config['save_mode']}")
    print(f"debug_mode = {config['debug_mode']}")
    print(f"reward_mode = {config['reward']['reward_mode']}")
    print(f"money_p = {config['reward']['money_p']}")
    print(f"money_tau = {config['reward']['money_tau']}")
    print(f"settled_scale = {config['reward']['settled_scale']}")
    print(f"tau_scaled = {config['reward']['tau_scaled']}")
    print(f"hybrid_alpha = {config['reward']['hybrid_alpha']}")
    print(f"val_metric = {config['train']['val_metric']}")
    print(f"training_reward_formula = {config['reward']['reward_formula']}")
    print(f"output_dir = {config['output']['output_dir']}")
    print(f"checkpoint_dir = {config['output']['checkpoint_dir']}")

    try:
        print("\n[Stage 1/2] Train DQN")
        print("-" * 70)

        agent, returns, loss_history, epsilons, validation_history, reward_history = train_agent(
            config=config,
            paths=paths,
        )

        print("\n[Stage 2/2] Cross-Regime Evaluation")
        print("-" * 70)

        results = evaluate_agent_cross_regime(
            agent=agent,
            config=config,
            paths=paths,
        )

        print_cross_regime_report(results)

        write_run_outputs(
            config=config,
            paths=paths,
            results=results,
            returns=returns,
            loss_history=loss_history,
            epsilons=epsilons,
            validation_history=validation_history,
            reward_history=reward_history,
        )

        if not config["debug_mode"] and config["save_mode"] != "none":
            aggregate_results(Path(config["output"]["output_dir"]).expanduser().resolve())

        print("\nFinished.")
        print("=" * 70)

        if config["save_mode"] in ["brief", "full"] and not config["debug_mode"]:
            print(f"result_dir: {paths['result_run_dir']}")
        if config["save_mode"] == "full" and not config["debug_mode"]:
            print(f"checkpoint_dir: {paths['checkpoint_run_dir']}")

    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()