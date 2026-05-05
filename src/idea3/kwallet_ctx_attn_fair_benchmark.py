# =========================================================
# Phase-1 Fair Benchmark
# Attention-Context DQN vs MIX12 Generalist DQN
#
# This script intentionally does not modify or import the currently
# running kwallet_attention_context12_dqn.py experiment.
# =========================================================

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import tempfile
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/kwallet_cache")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/kwallet_matplotlib_cache")
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

THIS_FILE = Path(__file__).resolve()
IDEA3_DIR = THIS_FILE.parent
SRC_DIR = IDEA3_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
DATA_POOL_DIR = PROJECT_ROOT / "src" / "ideaextra" / "data" / "pools"
RESULT_ROOT = IDEA3_DIR / "fair_benchmark_results"


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
DEFAULT_STATIC_EVAL_FILES = {r: f"{r}_static_eval_T1000.npy" for r in REGIME_ORDER}


CONFIG: Dict[str, Any] = {
    "seed": 123,
    "model_mode": "attn_context",
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
        "batch_size": 256,
        "target_update_every": 10,
        "device": "cpu",
        "learning_rate": 5e-4,
        "gamma": 0.98,
        "replay_size": 20000,
        "epsilon_start": 0.8,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.9995,
        "val_every": 50,
        "val_num_episodes": 200,
        "val_metric": "value_accept_ratio",
        "use_best_model_for_final_eval": True,
        "hidden_size": 128,
    },
    "eval": {
        "num_episodes": 200,
        "max_steps": 1000,
    },
    "attention_context": {
        "window_size": 50,
        "d_model": 32,
        "n_heads": 2,
        "context_dim": 32,
        "dropout": 0.05,
    },
    "reward": {
        "alpha_drop": 0.02,
        "beta_flush": 0.01,
    },
    "plot": {
        "window": 50,
    },
    "output": {
        "output_dir": str(RESULT_ROOT),
    },
}
LOG_EVERY_N = 100


class TimingStats:
    def __init__(self) -> None:
        self.totals: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def add(self, key: str, elapsed: float) -> None:
        self.totals[key] = self.totals.get(key, 0.0) + float(elapsed)
        self.counts[key] = self.counts.get(key, 0) + 1

    def summary_lines(self) -> List[str]:
        keys = [
            "env_step_state",
            "replay_sampling",
            "model_forward",
            "backward_update",
            "validation",
            "cross_regime_eval",
            "episode_total",
        ]
        lines = []
        for key in keys:
            total = self.totals.get(key, 0.0)
            count = self.counts.get(key, 0)
            avg = total / count if count else 0.0
            lines.append(f"{key}: total={total:.3f}s count={count} avg={avg:.6f}s")
        return lines

    def print_summary(self, prefix: str) -> None:
        print(prefix)
        for line in self.summary_lines():
            print(f"  {line}")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    return {
        "total": int(sum(p.numel() for p in model.parameters())),
        "trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }


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


def build_run_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def train_pool_file_for_regime(train_regime: str) -> str:
    if train_regime == "MIX12_EQ":
        return DEFAULT_MIX_EQ_MASTER
    if train_regime in REGIME_ORDER:
        return f"{train_regime}_static_master_T1000.npy"
    raise ValueError(f"Unknown train_regime={train_regime}. Use MIX12_EQ or one of {REGIME_ORDER}")


def build_scenario_name(config: Dict[str, Any]) -> str:
    env = config["env"]
    train_regime = config["data"]["train_regime"]
    mode = config["model_mode"]
    c_value = int(env["C"]) if float(env["C"]).is_integer() else env["C"]
    return f"{mode}_train{train_regime}_C{c_value}_k{env['k']}_T{env['T']}_F{env['F']}_seed{config['seed']}"


def build_title_tag(config: Dict[str, Any], run_stamp: str) -> str:
    env = config["env"]
    c_value = int(env["C"]) if float(env["C"]).is_integer() else env["C"]
    return (
        f"{run_stamp} | mode={config['model_mode']} | train={config['data']['train_regime']} | "
        f"C={c_value} k={env['k']} T={env['T']} F={env['F']}"
    )


def build_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    run_stamp = build_run_stamp()
    scenario = build_scenario_name(config)
    result_root = Path(config.get("output", {}).get("output_dir", str(RESULT_ROOT))).expanduser().resolve()
    result_run_dir = result_root / "runs" / scenario / run_stamp
    checkpoint_run_dir = result_root / "checkpoints" / scenario / run_stamp
    aggregate_dir = result_root / "aggregates"

    paths = {
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
        "run_info_path": str(result_run_dir / "run_info.json"),
        "results_json_path": str(result_run_dir / "cross_regime_results.json"),
        "summary_txt_path": str(result_run_dir / "summary_table.txt"),
        "training_history_path": str(result_run_dir / "training_history.json"),
        "validation_history_path": str(result_run_dir / "validation_history.json"),
        "eval_plot_path": str(result_run_dir / "cross_regime_plot.png"),
        "training_plot_path": str(result_run_dir / "training_curve.png"),
        "best_model_path": str(checkpoint_run_dir / "best_model.pth"),
        "last_model_path": str(checkpoint_run_dir / "last_model.pth"),
    }
    return paths


def ensure_dirs(paths: Dict[str, Any], config: Dict[str, Any]) -> None:
    if config["debug_mode"] or config["save_mode"] == "none":
        return
    for key in ["result_run_dir", "checkpoint_run_dir", "aggregate_dir"]:
        os.makedirs(paths[key], exist_ok=True)


def load_tx_pool(pool_path: str, expected_steps: int) -> np.ndarray:
    if not os.path.exists(pool_path):
        raise FileNotFoundError(f"Pool file not found: {pool_path}")
    if pool_path.endswith(".npy"):
        tx_pool = np.load(pool_path)
    elif pool_path.endswith(".json"):
        with open(pool_path, "r", encoding="utf-8") as f:
            tx_pool = np.array(json.load(f), dtype=np.int32)
    else:
        raise ValueError(f"Unsupported pool format: {pool_path}")
    if tx_pool.ndim != 2:
        raise ValueError(f"tx_pool must be 2D [episodes, steps], got ndim={tx_pool.ndim}")
    if tx_pool.shape[1] != expected_steps:
        raise ValueError(f"Expected steps={expected_steps}, got {tx_pool.shape[1]}")
    return tx_pool


def verify_data_integrity(pool_path: str, expected_steps: int, label: str = "") -> bool:
    try:
        tx_pool = load_tx_pool(pool_path, expected_steps)
        pool_hash = hashlib.md5(tx_pool.tobytes()).hexdigest()
        print(f"[data] {label} shape={tx_pool.shape} md5={pool_hash} path={pool_path}")
        return True
    except Exception as exc:
        print(f"[data] failed {label}: {exc}")
        return False


def compute_pool_fingerprint(pool_path: str, expected_steps: int) -> Dict[str, Any]:
    tx_pool = load_tx_pool(pool_path, expected_steps)
    return {
        "path": pool_path,
        "shape": [int(x) for x in tx_pool.shape],
        "md5": hashlib.md5(tx_pool.tobytes()).hexdigest(),
    }


def build_pool_fingerprints(config: Dict[str, Any], paths: Dict[str, Any]) -> Dict[str, Any]:
    train_steps = int(config["train"]["max_steps"])
    eval_steps = int(config["eval"]["max_steps"])
    return {
        "train": compute_pool_fingerprint(paths["train_pool_path"], train_steps),
        "validation": compute_pool_fingerprint(paths["val_pool_path"], train_steps),
        "eval": {
            regime: compute_pool_fingerprint(pool_path, eval_steps)
            for regime, pool_path in paths["test_pool_paths"].items()
        },
    }


class KWalletEnv:
    """
    Shared fixed-setting K-wallet environment for both benchmark modes.

    Leakage rule:
    - At decision step t, current_tx is tx[t].
    - recent_tx_window is exactly tx[max(0, t-W):t], i.e. only transactions
      already processed before the current decision.
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
        model_mode: str = "attn_context",
        attention_window_size: int = 50,
        enable_shaping: bool = False,
        alpha_drop: float = 0.02,
        beta_flush: float = 0.01,
    ):
        if model_mode not in {"baseline", "attn_context"}:
            raise ValueError(f"Unsupported model_mode={model_mode}")
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
        self._recent_tx_cache: Optional[np.ndarray] = None
        self._recent_mask_cache: Optional[np.ndarray] = None
        self.reset()

    @property
    def base_state_size(self) -> int:
        return 3 * self.k + 2

    @property
    def state_size(self) -> int:
        if self.model_mode == "baseline":
            return self.base_state_size
        return self.base_state_size + 2 * self.attention_window_size

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
        self._build_recent_context_cache()

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

    def _build_recent_context_cache(self) -> None:
        if self.model_mode != "attn_context":
            self._recent_tx_cache = None
            self._recent_mask_cache = None
            return

        n_steps = len(self._tx_stream)
        window = self.attention_window_size
        values = np.zeros((n_steps, window), dtype=np.float32)
        masks = np.zeros((n_steps, window), dtype=np.float32)
        tx = np.asarray(self._tx_stream, dtype=np.float32) / float(self.max_transaction)

        for t in range(n_steps):
            start = max(0, t - window)
            hist = tx[start:t]
            hist_len = int(hist.shape[0])
            if hist_len > 0:
                values[t, window - hist_len:] = hist
                masks[t, window - hist_len:] = 1.0

        self._recent_tx_cache = values
        self._recent_mask_cache = masks

    def _get_recent_tx_window_and_mask(self) -> Tuple[np.ndarray, np.ndarray]:
        if (
            self._recent_tx_cache is not None
            and self._recent_mask_cache is not None
            and self.time < self._recent_tx_cache.shape[0]
        ):
            return self._recent_tx_cache[self.time], self._recent_mask_cache[self.time]

        hist = list(self.tx_history)
        if len(hist) > self.attention_window_size:
            hist = hist[-self.attention_window_size:]
        pad_len = self.attention_window_size - len(hist)
        padded = [0.0] * pad_len + hist
        valid_mask = [0.0] * pad_len + [1.0] * len(hist)
        values = np.array([float(x) / self.max_transaction for x in padded], dtype=np.float32)
        return values, np.array(valid_mask, dtype=np.float32)

    def get_recent_tx_debug(self) -> Tuple[List[int], List[float]]:
        hist = list(self.tx_history)
        pad_len = self.attention_window_size - len(hist)
        padded = [0] * pad_len + [int(x) for x in hist]
        valid_mask = [0.0] * pad_len + [1.0] * len(hist)
        return padded, valid_mask

    def _get_state(self) -> np.ndarray:
        state = self._get_base_state()
        if self.model_mode == "attn_context":
            recent_tx, recent_mask = self._get_recent_tx_window_and_mask()
            return np.concatenate([np.array(state, dtype=np.float32), recent_tx, recent_mask]).astype(
                np.float32,
                copy=False,
            )
        return np.array(state, dtype=np.float32)

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


def run_leakage_sanity_checks(config: Dict[str, Any]) -> None:
    env_cfg = config["env"]
    attn_cfg = config["attention_context"]
    window_size = 5
    toy_stream = [101, 202, 303, 404, 505, 606, 707]
    env = KWalletEnv(
        C=env_cfg["C"],
        k=env_cfg["k"],
        F=env_cfg["F"],
        max_transaction=env_cfg["T"],
        max_steps=len(toy_stream),
        seed=config["seed"],
        model_mode="attn_context",
        attention_window_size=window_size,
        enable_shaping=env_cfg["enable_shaping"],
        alpha_drop=config["reward"]["alpha_drop"],
        beta_flush=config["reward"]["beta_flush"],
    )
    env.reset(tx_stream=toy_stream)
    padded, mask = env.get_recent_tx_debug()
    if padded != [0] * window_size or mask != [0.0] * window_size:
        raise AssertionError("Leakage check failed after reset: recent window must be all padding.")

    for t in range(len(toy_stream)):
        padded, mask = env.get_recent_tx_debug()
        expected_hist = toy_stream[max(0, t - window_size):t]
        expected = [0] * (window_size - len(expected_hist)) + expected_hist
        expected_mask = [0.0] * (window_size - len(expected_hist)) + [1.0] * len(expected_hist)
        if padded != expected:
            raise AssertionError(f"Leakage check failed at t={t}: got {padded}, expected {expected}")
        if mask != expected_mask:
            raise AssertionError(f"Mask check failed at t={t}: got {mask}, expected {expected_mask}")
        future_values = set(toy_stream[t + 1:])
        observed_nonzero = {x for x in padded if x != 0}
        if observed_nonzero.intersection(future_values):
            raise AssertionError(f"Future value leaked into state at t={t}: {observed_nonzero}")
        if t < len(toy_stream) - 1:
            env.step(env.k * (env.k + 1) + env.k)

    if attn_cfg.get("window_size", 50) <= 0:
        raise AssertionError("Attention window_size must be positive.")
    print("[leakage] sanity checks passed: recent window uses only tx[max(0,t-W):t].")


class BaselineDQN(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class RecentTxAttentionEncoder(nn.Module):
    def __init__(
        self,
        window_size: int,
        d_model: int = 32,
        n_heads: int = 2,
        context_dim: int = 32,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.window_size = int(window_size)
        self.tx_projection = nn.Linear(1, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, window_size, d_model))
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.out = nn.Sequential(
            nn.Linear(d_model, context_dim),
            nn.ReLU(),
        )

    def forward(self, recent_tx: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        # recent_tx: [batch, window], valid_mask: [batch, window], 1=real tx, 0=padding
        valid_bool = valid_mask > 0.5
        key_padding_mask = ~valid_bool
        all_padding = ~valid_bool.any(dim=1)
        if all_padding.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_padding] = False

        x = self.tx_projection(recent_tx.unsqueeze(-1)) + self.pos_embedding
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))

        masked = x * valid_bool.unsqueeze(-1).float()
        denom = valid_bool.sum(dim=1, keepdim=True).clamp(min=1).float()
        pooled = masked.sum(dim=1) / denom
        context = self.out(pooled)
        context = context * (~all_padding).unsqueeze(1).float()
        return context


class AttentionContextDQN(nn.Module):
    def __init__(
        self,
        base_state_size: int,
        window_size: int,
        action_size: int,
        hidden_size: int = 128,
        d_model: int = 32,
        n_heads: int = 2,
        context_dim: int = 32,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.base_state_size = int(base_state_size)
        self.window_size = int(window_size)
        self.context_encoder = RecentTxAttentionEncoder(
            window_size=window_size,
            d_model=d_model,
            n_heads=n_heads,
            context_dim=context_dim,
            dropout=dropout,
        )
        self.q_net = nn.Sequential(
            nn.Linear(base_state_size + context_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        base_state = state[:, :self.base_state_size]
        tx_start = self.base_state_size
        tx_end = tx_start + self.window_size
        recent_tx = state[:, tx_start:tx_end]
        recent_mask = state[:, tx_end:tx_end + self.window_size]
        context = self.context_encoder(recent_tx, recent_mask)
        return self.q_net(torch.cat([base_state, context], dim=1))


class DQNAgent:
    def __init__(
        self,
        model_mode: str,
        state_size: int,
        base_state_size: int,
        window_size: int,
        action_size: int,
        train_cfg: Dict[str, Any],
        attn_cfg: Dict[str, Any],
        device: str = "cpu",
    ):
        self.model_mode = model_mode
        self.state_size = int(state_size)
        self.base_state_size = int(base_state_size)
        self.window_size = int(window_size)
        self.action_size = int(action_size)
        self.device = torch.device(device)

        self.memory = deque(maxlen=int(train_cfg["replay_size"]))
        self.gamma = float(train_cfg["gamma"])
        self.epsilon = float(train_cfg["epsilon_start"])
        self.epsilon_min = float(train_cfg["epsilon_min"])
        self.epsilon_decay = float(train_cfg["epsilon_decay"])
        self.timing: Optional[TimingStats] = None

        self.model = self._build_model(train_cfg, attn_cfg).to(self.device)
        self.target_model = self._build_model(train_cfg, attn_cfg).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=float(train_cfg["learning_rate"]))
        self.update_target_network()

    def _build_model(self, train_cfg: Dict[str, Any], attn_cfg: Dict[str, Any]) -> nn.Module:
        hidden_size = int(train_cfg["hidden_size"])
        if self.model_mode == "baseline":
            return BaselineDQN(self.state_size, self.action_size, hidden_size=hidden_size)
        if self.model_mode == "attn_context":
            return AttentionContextDQN(
                base_state_size=self.base_state_size,
                window_size=self.window_size,
                action_size=self.action_size,
                hidden_size=hidden_size,
                d_model=int(attn_cfg["d_model"]),
                n_heads=int(attn_cfg["n_heads"]),
                context_dim=int(attn_cfg["context_dim"]),
                dropout=float(attn_cfg["dropout"]),
            )
        raise ValueError(f"Unsupported model_mode={self.model_mode}")

    def update_target_network(self) -> None:
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, s, a, r, s2, done) -> None:
        self.memory.append((s, a, r, s2, done))

    def act(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            t0 = time.perf_counter()
            q = self.model(s)
            if self.timing is not None:
                self.timing.add("model_forward", time.perf_counter() - t0)
            return int(torch.argmax(q, dim=1).item())

    def replay(self, batch_size: int) -> Optional[Dict[str, float]]:
        if len(self.memory) < batch_size:
            return None
        t0 = time.perf_counter()
        batch = random.sample(self.memory, batch_size)
        if self.timing is not None:
            self.timing.add("replay_sampling", time.perf_counter() - t0)
        s, a, r, s2, d = zip(*batch)

        s = torch.tensor(np.array(s), dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.int64, device=self.device)
        r = torch.tensor(r, dtype=torch.float32, device=self.device)
        s2 = torch.tensor(np.array(s2), dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            t0 = time.perf_counter()
            next_online_q = self.model(s2)
            next_act = next_online_q.argmax(dim=1)
            next_target_q = self.target_model(s2)
            q_next = next_target_q.gather(1, next_act.unsqueeze(1)).squeeze(1)
            y = r + self.gamma * (1.0 - d) * q_next
            if self.timing is not None:
                self.timing.add("model_forward", time.perf_counter() - t0)

        t0 = time.perf_counter()
        q = self.model(s).gather(1, a.unsqueeze(1)).squeeze(1)
        if self.timing is not None:
            self.timing.add("model_forward", time.perf_counter() - t0)
        loss = nn.SmoothL1Loss()(q, y)
        t0 = time.perf_counter()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()
        if self.timing is not None:
            self.timing.add("backward_update", time.perf_counter() - t0)
        return {"loss": float(loss.item())}

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def make_env(config: Dict[str, Any], max_steps: int) -> KWalletEnv:
    env_cfg = config["env"]
    attn_cfg = config["attention_context"]
    reward_cfg = config["reward"]
    return KWalletEnv(
        C=env_cfg["C"],
        k=env_cfg["k"],
        F=env_cfg["F"],
        max_transaction=env_cfg["T"],
        max_steps=max_steps,
        seed=config["seed"],
        model_mode=config["model_mode"],
        attention_window_size=attn_cfg["window_size"],
        enable_shaping=env_cfg["enable_shaping"],
        alpha_drop=reward_cfg["alpha_drop"],
        beta_flush=reward_cfg["beta_flush"],
    )


def summarize_episode_metrics(all_results: List[Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
    summary = {}
    for metric in all_results[0].keys():
        values = [r[metric] for r in all_results]
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
    num_eval_episodes = min(num_eval_episodes, tx_pool.shape[0])
    env = make_env(config, max_steps=max_steps)
    old_eps = agent.epsilon
    agent.epsilon = 0.0
    all_results = []

    try:
        with torch.inference_mode():
            for ep in range(num_eval_episodes):
                s = env.reset(tx_stream=tx_pool[ep])
                total_requested_value = 0.0
                total_tx_count = 0
                accepted_count = 0
                for _ in range(max_steps):
                    current_tx = env.current_tx
                    total_requested_value += float(current_tx)
                    total_tx_count += 1
                    a = agent.act(s)
                    t0 = time.perf_counter()
                    s, _, done, info = env.step(a)
                    if agent.timing is not None:
                        agent.timing.add("env_step_state", time.perf_counter() - t0)
                    if info.get("accepted", False):
                        accepted_count += 1
                    if done:
                        break

                metrics = env.get_metrics()
                metrics["value_accept_ratio"] = metrics["settled"] / total_requested_value if total_requested_value > 0 else 0.0
                metrics["count_accept_ratio"] = accepted_count / total_tx_count if total_tx_count > 0 else 0.0
                metrics["total_requested_value"] = total_requested_value
                metrics["total_tx_count"] = total_tx_count
                metrics["accepted_count"] = accepted_count
                all_results.append(metrics)
    finally:
        agent.epsilon = old_eps
    return {
        "label": label,
        "num_episodes": num_eval_episodes,
        "summary": summarize_episode_metrics(all_results),
        "raw_results": all_results,
    }


def evaluate_agent_on_pool(
    agent: DQNAgent,
    config: Dict[str, Any],
    test_pool_path: str,
    test_regime: str,
) -> Dict[str, Any]:
    eval_cfg = config["eval"]
    if not verify_data_integrity(test_pool_path, expected_steps=eval_cfg["max_steps"], label=test_regime):
        raise RuntimeError(f"Data verification failed for {test_regime}")
    tx_pool = load_tx_pool(test_pool_path, expected_steps=eval_cfg["max_steps"])
    result = evaluate_agent_on_array(
        agent=agent,
        config=config,
        tx_pool=tx_pool,
        label=test_regime,
        num_eval_episodes=eval_cfg["num_episodes"],
        max_steps=eval_cfg["max_steps"],
    )
    result["test_regime"] = test_regime
    result["test_pool_path"] = test_pool_path
    return result


def evaluate_agent_cross_regime(agent: DQNAgent, config: Dict[str, Any], paths: Dict[str, Any]) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("Cross-Regime Evaluation")
    print("=" * 70)
    t_eval_start = time.perf_counter()
    cross_results = {}
    for regime_name, pool_path in paths["test_pool_paths"].items():
        cross_results[regime_name] = evaluate_agent_on_pool(agent, config, pool_path, regime_name)
    if agent.timing is not None:
        agent.timing.add("cross_regime_eval", time.perf_counter() - t_eval_start)
    return {
        "config": config,
        "scenario": paths["scenario"],
        "train_regime": paths["train_regime"],
        "model_mode": config["model_mode"],
        "seed": config["seed"],
        "timestamp": datetime.now().isoformat(),
        "test_results": cross_results,
        "aggregate": compute_cross_regime_aggregate(cross_results),
    }


def train_agent(config: Dict[str, Any], paths: Dict[str, Any]) -> Tuple[DQNAgent, List[float], List[float], List[float], List[Dict[str, float]]]:
    train_cfg = config["train"]
    tx_pool_full = load_tx_pool(paths["train_pool_path"], expected_steps=train_cfg["max_steps"])
    train_use_episodes = int(train_cfg["train_use_episodes"])
    if train_use_episodes > tx_pool_full.shape[0]:
        raise ValueError(f"train_use_episodes={train_use_episodes} exceeds pool size={tx_pool_full.shape[0]}")
    if train_cfg["episodes"] > train_use_episodes:
        raise ValueError("Training episodes exceed selected train pool rows.")
    tx_pool_train = tx_pool_full[:train_use_episodes]
    tx_pool_val = load_tx_pool(paths["val_pool_path"], expected_steps=train_cfg["max_steps"])

    env = make_env(config, max_steps=train_cfg["max_steps"])
    agent = DQNAgent(
        model_mode=config["model_mode"],
        state_size=env.state_size,
        base_state_size=env.base_state_size,
        window_size=config["attention_context"]["window_size"],
        action_size=env.num_actions,
        train_cfg=train_cfg,
        attn_cfg=config["attention_context"],
        device=train_cfg["device"],
    )
    timing_stats = TimingStats()
    if config["debug_mode"]:
        agent.timing = timing_stats
    param_counts = count_parameters(agent.model)

    print("\n" + "=" * 70)
    print("Train Fair Benchmark DQN")
    print("=" * 70)
    print(f"mode={config['model_mode']} train_regime={paths['train_regime']} seed={config['seed']}")
    print(f"env C={env.C} k={env.k} F={env.F} T={env.max_transaction}")
    print(f"state={env.state_size} base_state={env.base_state_size} action={env.num_actions}")
    print(f"train_pool={tx_pool_train.shape} val_pool={tx_pool_val.shape}")
    print(f"shared train cfg: replay={train_cfg['replay_size']} hidden={train_cfg['hidden_size']} lr={train_cfg['learning_rate']}")
    print(f"model parameters: total={param_counts['total']} trainable={param_counts['trainable']}")

    returns: List[float] = []
    loss_history: List[float] = []
    epsilons: List[float] = []
    validation_history: List[Dict[str, float]] = []
    best_val_score = -1e18
    best_val_snapshot = None

    if config["debug_mode"] or config["save_mode"] == "none":
        tmp = tempfile.NamedTemporaryFile(prefix="kwallet_fair_best_", suffix=".pth", delete=False)
        best_ckpt_path = tmp.name
        tmp.close()
    else:
        best_ckpt_path = paths["best_model_path"]

    for ep in range(train_cfg["episodes"]):
        episode_t0 = time.perf_counter()
        state = env.reset(tx_stream=tx_pool_train[ep])
        episode_return = 0.0
        for _ in range(train_cfg["max_steps"]):
            action = agent.act(state)
            t0 = time.perf_counter()
            next_state, reward, done, _ = env.step(action)
            if agent.timing is not None:
                agent.timing.add("env_step_state", time.perf_counter() - t0)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            episode_return += reward
            metrics = agent.replay(batch_size=train_cfg["batch_size"])
            if metrics is not None:
                loss_history.append(metrics["loss"])
            if done:
                break
        if agent.timing is not None:
            agent.timing.add("episode_total", time.perf_counter() - episode_t0)

        if (ep + 1) % train_cfg["target_update_every"] == 0:
            agent.update_target_network()

        returns.append(float(episode_return))
        epsilons.append(float(agent.epsilon))
        agent.decay_epsilon()

        if (ep + 1) % LOG_EVERY_N == 0 or ep == 0:
            recent_mean = float(np.mean(returns[-LOG_EVERY_N:]))
            print(
                f"[Train] ep={ep + 1:4d}/{train_cfg['episodes']} "
                f"return={episode_return:10.2f} recent={recent_mean:10.2f} eps={agent.epsilon:.4f}"
            )

        val_every = int(train_cfg["val_every"])
        if val_every > 0 and ((ep + 1) % val_every == 0 or (ep + 1) == train_cfg["episodes"]):
            t0 = time.perf_counter()
            val_result = evaluate_agent_on_array(
                agent=agent,
                config=config,
                tx_pool=tx_pool_val,
                label="VAL",
                num_eval_episodes=train_cfg["val_num_episodes"],
                max_steps=train_cfg["max_steps"],
            )
            if agent.timing is not None:
                agent.timing.add("validation", time.perf_counter() - t0)
            metric_name = train_cfg["val_metric"]
            val_score = float(val_result["summary"][metric_name]["mean"])
            val_row = {
                "episode": ep + 1,
                "metric": metric_name,
                "value_accept_ratio": float(val_result["summary"]["value_accept_ratio"]["mean"]),
                "drop_rate": float(val_result["summary"]["drop_rate"]["mean"]),
                "drops": float(val_result["summary"]["drops"]["mean"]),
                "flushes": float(val_result["summary"]["flushes"]["mean"]),
                "score": val_score,
            }
            validation_history.append(val_row)
            print(f"[Val  ] ep={ep + 1:4d} {metric_name}={val_score:.4f}")

            if val_score > best_val_score:
                best_val_score = val_score
                best_val_snapshot = val_row
                torch.save(agent.model.state_dict(), best_ckpt_path)
                print(f"[Val  ] best checkpoint updated: ep={ep + 1}, score={val_score:.4f}")

        if config["debug_mode"] and (ep + 1) % 5 == 0:
            timing_stats.print_summary(f"[Timing] cumulative summary after episode {ep + 1}")

    if train_cfg["use_best_model_for_final_eval"] and best_val_snapshot is not None and os.path.exists(best_ckpt_path):
        agent.model.load_state_dict(torch.load(best_ckpt_path, map_location=agent.device))
        agent.update_target_network()
        print(f"Loaded best checkpoint: ep={best_val_snapshot['episode']} score={best_val_snapshot['score']:.4f}")

    if not config["debug_mode"] and config["save_mode"] == "full":
        torch.save(agent.model.state_dict(), paths["last_model_path"])
        print(f"Last model saved to: {paths['last_model_path']}")

    return agent, returns, loss_history, epsilons, validation_history


def compute_cross_regime_aggregate(test_results: Dict[str, Any]) -> Dict[str, float]:
    val_accs = [test_results[r]["summary"]["value_accept_ratio"]["mean"] for r in test_results]
    drops = [test_results[r]["summary"]["drops"]["mean"] for r in test_results]
    flushes = [test_results[r]["summary"]["flushes"]["mean"] for r in test_results]
    return {
        "mean_value_accept_ratio": float(np.mean(val_accs)),
        "worst_regime_value_accept_ratio": float(np.min(val_accs)),
        "std_value_accept_ratio_across_regimes": float(np.std(val_accs)),
        "mean_drops": float(np.mean(drops)),
        "mean_flushes": float(np.mean(flushes)),
    }


def build_cross_regime_report_text(results: Dict[str, Any]) -> str:
    lines = []
    lines.append("=" * 96)
    lines.append("Phase-1 Fair Benchmark Cross-Regime Evaluation")
    lines.append("=" * 96)
    lines.append(f"timestamp    : {results['timestamp']}")
    lines.append(f"scenario     : {results['scenario']}")
    lines.append(f"model_mode   : {results['model_mode']}")
    lines.append(f"train_regime : {results['train_regime']}")
    lines.append(f"seed         : {results['seed']}")
    lines.append("-" * 96)
    lines.append(
        f"{'Test':<8}{'ValAcc(%)':>14}{'Drops':>12}{'Flushes':>12}"
        f"{'DropRate(%)':>14}{'CntAcc(%)':>14}"
    )
    lines.append("-" * 96)
    for regime_name, regime_result in results["test_results"].items():
        s = regime_result["summary"]
        lines.append(
            f"{regime_name:<8}"
            f"{100 * s['value_accept_ratio']['mean']:>14.2f}"
            f"{s['drops']['mean']:>12.2f}"
            f"{s['flushes']['mean']:>12.2f}"
            f"{100 * s['drop_rate']['mean']:>14.2f}"
            f"{100 * s['count_accept_ratio']['mean']:>14.2f}"
        )
    agg = results["aggregate"]
    lines.append("-" * 96)
    lines.append(f"Mean ValAcc (%)      : {100 * agg['mean_value_accept_ratio']:.4f}")
    lines.append(f"Worst-Regime ValAcc (%): {100 * agg['worst_regime_value_accept_ratio']:.4f}")
    lines.append(f"Std Across Regimes   : {agg['std_value_accept_ratio_across_regimes']:.6f}")
    lines.append(f"Mean Drops           : {agg['mean_drops']:.4f}")
    lines.append(f"Mean Flushes         : {agg['mean_flushes']:.4f}")
    lines.append("=" * 96)
    return "\n".join(lines)


def save_json(payload: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_results(results: Dict[str, Any], save_path: str) -> None:
    compact = {
        "config": results["config"],
        "scenario": results["scenario"],
        "train_regime": results["train_regime"],
        "model_mode": results["model_mode"],
        "seed": results["seed"],
        "timestamp": results["timestamp"],
        "aggregate": results["aggregate"],
        "test_results": {},
    }
    for regime_name, regime_result in results["test_results"].items():
        compact["test_results"][regime_name] = {
            "num_episodes": regime_result["num_episodes"],
            "summary": {},
        }
        for metric, data in regime_result["summary"].items():
            compact["test_results"][regime_name]["summary"][metric] = {
                "mean": data["mean"],
                "std": data["std"],
                "min": data["min"],
                "max": data["max"],
                "median": data["median"],
            }
    save_json(compact, save_path)
    print(f"Results saved to: {save_path}")


def save_summary_txt(results: Dict[str, Any], save_path: str) -> None:
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(build_cross_regime_report_text(results))
    print(f"Summary saved to: {save_path}")


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
    loss_history: List[float],
    epsilons: List[float],
    save_path: str,
    title_tag: str,
    window: int,
) -> None:
    fig = plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(returns, alpha=0.35, label="Return")
    plt.plot(moving_average(returns, window), linewidth=2, label=f"MA({window})")
    plt.title(f"Training Curves\n{title_tag}")
    plt.ylabel("Episode Return")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(3, 1, 2)
    if loss_history:
        plt.plot(loss_history, alpha=0.8)
    plt.ylabel("Replay Loss")
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 3)
    plt.plot(epsilons)
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Training plot saved to: {save_path}")


def plot_evaluation_results(results: Dict[str, Any], save_path: str, title_tag: str) -> None:
    regimes = list(results["test_results"].keys())
    val_acc = [100 * results["test_results"][r]["summary"]["value_accept_ratio"]["mean"] for r in regimes]
    drops = [results["test_results"][r]["summary"]["drops"]["mean"] for r in regimes]
    flushes = [results["test_results"][r]["summary"]["flushes"]["mean"] for r in regimes]
    drop_rate = [100 * results["test_results"][r]["summary"]["drop_rate"]["mean"] for r in regimes]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Cross-Regime Evaluation\n{title_tag}", fontsize=14, fontweight="bold")
    for ax, values, label in [
        (axes[0, 0], val_acc, "Value Accept Ratio (%)"),
        (axes[0, 1], drops, "Drops"),
        (axes[1, 0], flushes, "Flushes"),
        (axes[1, 1], drop_rate, "Drop Rate (%)"),
    ]:
        ax.bar(regimes, values)
        ax.set_title(label)
        ax.grid(True, alpha=0.3, axis="y")
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Evaluation plot saved to: {save_path}")


def write_run_outputs(
    config: Dict[str, Any],
    paths: Dict[str, Any],
    results: Dict[str, Any],
    returns: List[float],
    loss_history: List[float],
    epsilons: List[float],
    validation_history: List[Dict[str, float]],
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
        },
        paths["run_info_path"],
    )
    save_results(results, paths["results_json_path"])
    save_summary_txt(results, paths["summary_txt_path"])
    save_json(
        {"returns": returns, "loss_history": loss_history, "epsilons": epsilons},
        paths["training_history_path"],
    )
    save_json({"validation_history": validation_history}, paths["validation_history_path"])
    plot_training_curves(
        returns,
        loss_history,
        epsilons,
        save_path=paths["training_plot_path"],
        title_tag=paths["title_tag"],
        window=config["plot"]["window"],
    )
    plot_evaluation_results(results, paths["eval_plot_path"], paths["title_tag"])


def flatten_result_for_csv(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = []
    config = payload["config"]
    aggregate = payload.get("aggregate", {})
    for regime, regime_result in payload["test_results"].items():
        s = regime_result["summary"]
        rows.append({
            "scenario": payload["scenario"],
            "model_mode": payload["model_mode"],
            "train_regime": payload["train_regime"],
            "seed": payload["seed"],
            "test_regime": regime,
            "value_accept_ratio": s["value_accept_ratio"]["mean"],
            "drops": s["drops"]["mean"],
            "flushes": s["flushes"]["mean"],
            "drop_rate": s["drop_rate"]["mean"],
            "count_accept_ratio": s["count_accept_ratio"]["mean"],
            "mean_value_accept_ratio": aggregate.get("mean_value_accept_ratio"),
            "worst_regime_value_accept_ratio": aggregate.get("worst_regime_value_accept_ratio"),
            "std_value_accept_ratio_across_regimes": aggregate.get("std_value_accept_ratio_across_regimes"),
            "C": config["env"]["C"],
            "k": config["env"]["k"],
            "F": config["env"]["F"],
            "T": config["env"]["T"],
        })
    return rows


def aggregate_results(result_root: Path = RESULT_ROOT) -> Path:
    result_files = sorted((result_root / "runs").glob("**/cross_regime_results.json"))
    rows: List[Dict[str, Any]] = []
    for path in result_files:
        rows.extend(flatten_result_for_csv(path))
    aggregate_dir = result_root / "aggregates"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    out_path = aggregate_dir / "fair_benchmark_aggregated.csv"
    fieldnames = [
        "scenario", "model_mode", "train_regime", "seed", "test_regime",
        "value_accept_ratio", "drops", "flushes", "drop_rate", "count_accept_ratio",
        "mean_value_accept_ratio", "worst_regime_value_accept_ratio",
        "std_value_accept_ratio_across_regimes", "C", "k", "F", "T",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Aggregated CSV saved to: {out_path} ({len(rows)} rows)")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-1 fair benchmark for K-wallet DQN.")
    parser.add_argument("--model_mode", choices=["baseline", "attn_context"], default=CONFIG["model_mode"])
    parser.add_argument("--train_regime", type=str, default=CONFIG["data"]["train_regime"])
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--C", type=float, default=CONFIG["env"]["C"])
    parser.add_argument("--k", type=int, default=CONFIG["env"]["k"])
    parser.add_argument("--F", type=int, default=CONFIG["env"]["F"])
    parser.add_argument("--T", type=int, default=CONFIG["env"]["T"])
    parser.add_argument("--window_size", type=int, default=CONFIG["attention_context"]["window_size"])
    parser.add_argument("--output_dir", type=str, default=CONFIG["output"]["output_dir"])
    parser.add_argument("--device", type=str, default=CONFIG["train"]["device"])
    parser.add_argument("--episodes", type=int, default=CONFIG["train"]["episodes"])
    parser.add_argument("--eval_episodes", type=int, default=CONFIG["eval"]["num_episodes"])
    parser.add_argument("--val_every", type=int, default=CONFIG["train"]["val_every"])
    parser.add_argument("--val_episodes", type=int, default=CONFIG["train"]["val_num_episodes"])
    parser.add_argument("--save_mode", choices=["none", "full"], default=CONFIG["save_mode"])
    parser.add_argument("--debug_mode", action="store_true")
    parser.add_argument("--skip_leakage_check", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")
    return parser.parse_args()


def apply_args_to_config(args: argparse.Namespace) -> Dict[str, Any]:
    config = json.loads(json.dumps(CONFIG))
    config["model_mode"] = args.model_mode
    config["seed"] = int(args.seed)
    config["debug_mode"] = bool(args.debug_mode)
    config["save_mode"] = args.save_mode
    config["env"]["C"] = float(args.C)
    config["env"]["k"] = int(args.k)
    config["env"]["F"] = int(args.F)
    config["env"]["T"] = int(args.T)
    config["attention_context"]["window_size"] = int(args.window_size)
    config["plot"]["window"] = int(args.window_size)
    config["output"]["output_dir"] = args.output_dir
    config["data"]["train_regime"] = args.train_regime
    config["data"]["train_pool_file"] = train_pool_file_for_regime(args.train_regime)
    config["train"]["device"] = args.device
    config["train"]["episodes"] = int(args.episodes)
    config["train"]["val_every"] = int(args.val_every)
    config["train"]["val_num_episodes"] = int(args.val_episodes)
    config["eval"]["num_episodes"] = int(args.eval_episodes)
    return config


def main() -> None:
    args = parse_args()
    if args.aggregate_only:
        aggregate_results(Path(args.output_dir).expanduser().resolve())
        return

    config = apply_args_to_config(args)
    set_seed(config["seed"])
    if not args.skip_leakage_check:
        run_leakage_sanity_checks(config)

    paths = build_paths(config)
    ensure_dirs(paths, config)
    print("\n" + "=" * 70)
    print("Phase-1 Fair Benchmark")
    print("=" * 70)
    print(f"scenario: {paths['scenario']}")
    print(f"results : {paths['result_run_dir']}")
    print(f"train   : {paths['train_pool_path']}")
    print(f"val     : {paths['val_pool_path']}")

    agent, returns, loss_history, epsilons, validation_history = train_agent(config, paths)
    results = evaluate_agent_cross_regime(agent, config, paths)
    if config["debug_mode"] and agent.timing is not None:
        agent.timing.print_summary("[Timing] final cumulative summary after cross-regime evaluation")
    print("\n" + build_cross_regime_report_text(results))
    write_run_outputs(config, paths, results, returns, loss_history, epsilons, validation_history)
    if not config["debug_mode"] and config["save_mode"] != "none":
        aggregate_results(Path(paths["result_root"]))


if __name__ == "__main__":
    main()
