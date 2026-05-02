# =========================================================
# 5/2
# K-Wallet Attention Context DQN
# Model name: ATTN_CTX_MIX12_EQ
# Purpose:
#   Train one context-aware generalist policy on MIX12_EQ.
#   The policy uses a lightweight self-attention encoder to learn
#   recent transaction context from a transaction window.
# =========================================================

import os
import json
import hashlib
import tempfile
import random
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


# =========================================================
# Path setup
# =========================================================

THIS_FILE = Path(__file__).resolve()
IDEAEXTRA_DIR = THIS_FILE.parent
SRC_DIR = IDEAEXTRA_DIR.parent
PROJECT_ROOT = SRC_DIR.parent


# =========================================================
# Global config
# =========================================================

CONFIG = {
    "seed": 123,
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
        "train_regime": "ATTN_CTX_MIX12_EQ",
        "train_pool_file": "MIX12_EQ_US_TLS_LNS_TLNS_TPLS_PLS_UB_TLB_LNB_TLNB_TPLB_PLB_master_T1000.npy",
        "val_pool_file": "MIX12_EQ_US_TLS_LNS_TLNS_TPLS_PLS_UB_TLB_LNB_TLNB_TPLB_PLB_val_T1000.npy",
        "test_pool_files": {
            "US": "US_static_eval_T1000.npy",
            "TLS": "TLS_static_eval_T1000.npy",
            "LNS": "LNS_static_eval_T1000.npy",
            "TLNS": "TLNS_static_eval_T1000.npy",
            "TPLS": "TPLS_static_eval_T1000.npy",
            "PLS": "PLS_static_eval_T1000.npy",
            "UB": "UB_static_eval_T1000.npy",
            "TLB": "TLB_static_eval_T1000.npy",
            "LNB": "LNB_static_eval_T1000.npy",
            "TLNB": "TLNB_static_eval_T1000.npy",
            "TPLB": "TPLB_static_eval_T1000.npy",
            "PLB": "PLB_static_eval_T1000.npy",
        },
    },

    "train": {
        "episodes": 1000,
        "train_use_episodes": 3000,
        "val_num_episodes": 200,
        "max_steps": 1000,
        "batch_size": 256,
        "target_update_every": 10,
        "device": "cpu",
        "learning_rate": 5e-4,
        "epsilon_start": 0.8,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.9995,
        "val_every": 50,
        "val_metric": "value_accept_ratio",
        "val_split_episodes": 0,
        "gamma": 0.98,
    },

    "eval": {
        "num_episodes": 200,
        "max_steps": 1000,
    },

    "plot": {
        "window": 50,
    },

    "attention_context": {
        "enabled": True,
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

    "output": {
        "save_under_ideaextra": True,
        "run_root_name": "attention_context_runs",
    },
}

LOG_EVERY_N = 100
REGIME_ORDER = [
    "US", "TLS", "LNS", "TLNS", "TPLS", "PLS",
    "UB", "TLB", "LNB", "TLNB", "TPLB", "PLB"
]


# =========================================================
# Utility functions
# =========================================================

def set_seed(seed: int = 123):
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
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_scenario_name(config: Dict[str, Any]) -> str:
    env = config["env"]
    train_regime = config["data"]["train_regime"]
    test_regimes = "_".join(config["data"]["test_pool_files"].keys())
    C = int(env["C"]) if float(env["C"]).is_integer() else env["C"]
    return f"train{train_regime}_cross{test_regimes}_attnctx_C{C}_k{env['k']}_T{env['T']}_F{env['F']}"


def build_title_tag(config: Dict[str, Any], run_stamp: str) -> str:
    env = config["env"]
    train_regime = config["data"]["train_regime"]
    ctx = config["attention_context"]
    C = int(env["C"]) if float(env["C"]).is_integer() else env["C"]
    return (
        f"{run_stamp} | train={train_regime} | ATTN-CTX "
        f"W={ctx['window_size']} d={ctx['d_model']} heads={ctx['n_heads']} | "
        f"C={C} k={env['k']} T={env['T']} F={env['F']}"
    )


def build_paths(config: Dict[str, Any]) -> Dict[str, str]:
    scenario = build_scenario_name(config)
    run_stamp = build_run_stamp()

    data_pool_dir = PROJECT_ROOT / "src" / "ideaextra" / "data" / "pools"
    data_report_root = PROJECT_ROOT / "src" / "ideaextra" / "data" / "reports"

    C_value = int(config["env"]["C"]) if float(config["env"]["C"]).is_integer() else config["env"]["C"]
    seed_value = config["seed"]
    setting_tag = f"C{C_value}_seed{seed_value}"

    # =========================================================
    # Save all attention-context outputs under src/idea3/attention_results
    # =========================================================
    run_root = PROJECT_ROOT / "src" / "idea3" / "attention_results"

    result_scenario_dir = run_root / "results" / setting_tag / scenario
    checkpoint_scenario_dir = run_root / "checkpoints" / setting_tag / scenario
    log_dir = run_root / "logs"

    result_run_dir = result_scenario_dir / run_stamp
    checkpoint_run_dir = checkpoint_scenario_dir / run_stamp

    train_pool_file = config["data"]["train_pool_file"]
    val_pool_file = config["data"].get("val_pool_file")
    test_pool_files = config["data"]["test_pool_files"]

    paths = {
        "project_root": str(PROJECT_ROOT),
        "scenario": scenario,
        "run_stamp": run_stamp,
        "setting_tag": setting_tag,
        "title_tag": build_title_tag(config, run_stamp),
        "data_pool_dir": str(data_pool_dir),
        "data_report_root": str(data_report_root),
        "train_regime": config["data"]["train_regime"],
        "train_pool_path": str(data_pool_dir / train_pool_file),
        "val_pool_path": str(data_pool_dir / val_pool_file) if val_pool_file else None,
        "test_pool_paths": {r: str(data_pool_dir / f) for r, f in test_pool_files.items()},
        "log_dir": str(log_dir),
        "result_scenario_dir": str(result_scenario_dir),
        "result_run_dir": str(result_run_dir),
        "run_info_path": str(result_run_dir / "run_info.json"),
        "results_json_path": str(result_run_dir / "cross_regime_results.json"),
        "summary_txt_path": str(result_run_dir / "cross_regime_summary.txt"),
        "eval_plot_path": str(result_run_dir / "cross_regime_bar.png"),
        "training_plot_path": str(result_run_dir / "train_curve.png"),
        "val_plot_path": str(result_run_dir / "validation_curve.png"),
        "val_history_json_path": str(result_run_dir / "validation_history.json"),
        "checkpoint_scenario_dir": str(checkpoint_scenario_dir),
        "checkpoint_run_dir": str(checkpoint_run_dir),
        "model_path": str(checkpoint_run_dir / "model.pth"),
        "best_model_path": str(checkpoint_run_dir / "best_model.pth"),
    }
    return paths


def ensure_dirs(paths: Dict[str, str]):
    for key in [
        "data_report_root", "result_scenario_dir", "result_run_dir",
        "checkpoint_scenario_dir", "checkpoint_run_dir", "log_dir"
    ]:
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


def save_run_info(config: Dict[str, Any], paths: Dict[str, str]):
    snapshot = {
        "scenario": paths["scenario"],
        "run_stamp": paths["run_stamp"],
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "paths": paths,
    }
    with open(paths["run_info_path"], "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)
    print(f"Run info saved to: {paths['run_info_path']}")


def verify_data_integrity(pool_path: str, expected_steps: int, label: str = "") -> bool:
    print("\n" + "=" * 70)
    print(f"Data integrity check {label}".strip())
    print("=" * 70)
    try:
        tx_pool = load_tx_pool(pool_path, expected_steps)
        file_size = os.path.getsize(pool_path) / 1024
        pool_hash = hashlib.md5(tx_pool.tobytes()).hexdigest()
        print(f"Loaded: {pool_path}")
        print(f"Shape : {tx_pool.shape}")
        print(f"Size  : {file_size:.2f} KB")
        print(f"MD5   : {pool_hash}")
        print(f"First episode first 5 tx: {tx_pool[0, :5].tolist()}")
        print("=" * 70 + "\n")
        return True
    except Exception as e:
        print(f"Data check failed: {e}")
        return False


# =========================================================
# K-Wallet environment with recent transaction window
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
        attention_context_enabled: bool = True,
        attention_window_size: int = 30,
        alpha_drop: float = 0.02,
        beta_flush: float = 0.01,
    ):
        self.C = float(C)
        self.k = int(k)
        self.F = int(F)
        self.max_transaction = int(max_transaction)
        self.max_steps = int(max_steps)
        self.wallet_size = self.C / self.k
        self.num_actions = (self.k + 1) ** 2

        self.enable_shaping = bool(enable_shaping)
        self.attention_context_enabled = bool(attention_context_enabled)
        self.attention_window_size = int(attention_window_size)
        self.tx_history = deque(maxlen=self.attention_window_size)

        self.alpha_drop = float(alpha_drop)
        self.beta_flush = float(beta_flush)
        self.IMBALANCE_PENALTY = 0.02
        self.WASTEFUL_REFRESH_PENALTY = 0.02
        self.WASTEFUL_REFRESH_THRESH = 0.6
        self.INVALID_ACTION_PENALTY = 0.05

        self.rng = np.random.default_rng(seed)
        self._tx_stream = None
        self.reset()

    def reset(self, tx_stream: List[int] = None) -> np.ndarray:
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
            self._tx_stream = list(tx_stream)
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
        return state

    def _get_recent_tx_window(self) -> List[float]:
        if not self.attention_context_enabled:
            return []

        hist = list(self.tx_history)
        pad_len = self.attention_window_size - len(hist)
        padded = [0.0] * pad_len + hist
        return [float(x) / self.max_transaction for x in padded]

    def _get_state(self) -> np.ndarray:
        state = self._get_base_state()
        state.extend(self._get_recent_tx_window())
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

        # 1. Flush first
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
                    reward -= self.INVALID_ACTION_PENALTY

        fit_idx = None

        # 2. Settle
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

        # 3. Flush cost
        reward -= self.beta_flush * flushes_this_step

        # 4. Optional shaping
        if self.enable_shaping:
            usable_balances = [self.wallets[i] for i in range(self.k) if self._usable(i)]
            if len(usable_balances) >= 2:
                std_norm = float(np.std(np.array(usable_balances)) / self.wallet_size)
                reward -= self.IMBALANCE_PENALTY * std_norm

            for i in refresh_targets:
                if (pre_refresh_balances[i] / self.wallet_size) >= self.WASTEFUL_REFRESH_THRESH:
                    reward -= self.WASTEFUL_REFRESH_PENALTY

        # 5. Update recent transaction history after this transaction is processed
        self.tx_history.append(float(tx))

        # 6. Advance time
        self.time += 1

        # 7. Refill frozen wallets when usable again
        for i in range(self.k):
            if self.pending_refill[i] and self._usable(i):
                self.wallets[i] = self.wallet_size
                self.pending_refill[i] = False

        # 8. Update current transaction
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
# Attention Context DQN
# =========================================================

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
        self.d_model = int(d_model)

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

    def forward(self, recent_tx: torch.Tensor) -> torch.Tensor:
        # recent_tx: [batch, window_size]
        x = recent_tx.unsqueeze(-1)  # [batch, window, 1]
        x = self.tx_projection(x) + self.pos_embedding

        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        # Mean pooling over the recent transaction window
        pooled = x.mean(dim=1)
        context = self.out(pooled)
        return context


class AttentionDQN(nn.Module):
    def __init__(
        self,
        base_state_size: int,
        window_size: int,
        action_size: int,
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
            nn.Linear(base_state_size + context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        base_state = state[:, :self.base_state_size]
        recent_tx = state[:, self.base_state_size:self.base_state_size + self.window_size]
        context = self.context_encoder(recent_tx)
        x = torch.cat([base_state, context], dim=1)
        return self.q_net(x)


class DQNAgent:
    def __init__(
        self,
        base_state_size: int,
        window_size: int,
        state_size: int,
        action_size: int,
        train_cfg: Dict[str, Any],
        attn_cfg: Dict[str, Any],
        device: str = "cpu",
    ):
        self.base_state_size = int(base_state_size)
        self.window_size = int(window_size)
        self.state_size = int(state_size)
        self.action_size = int(action_size)
        self.device = torch.device(device)

        self.memory = deque(maxlen=20000)
        self.gamma = float(train_cfg.get("gamma", 0.98))
        self.epsilon = float(train_cfg.get("epsilon_start", 0.8))
        self.epsilon_min = float(train_cfg.get("epsilon_min", 0.05))
        self.epsilon_decay = float(train_cfg.get("epsilon_decay", 0.9995))

        self.model = AttentionDQN(
            base_state_size=base_state_size,
            window_size=window_size,
            action_size=action_size,
            d_model=attn_cfg.get("d_model", 32),
            n_heads=attn_cfg.get("n_heads", 2),
            context_dim=attn_cfg.get("context_dim", 32),
            dropout=attn_cfg.get("dropout", 0.05),
        ).to(self.device)

        self.target_model = AttentionDQN(
            base_state_size=base_state_size,
            window_size=window_size,
            action_size=action_size,
            d_model=attn_cfg.get("d_model", 32),
            n_heads=attn_cfg.get("n_heads", 2),
            context_dim=attn_cfg.get("context_dim", 32),
            dropout=attn_cfg.get("dropout", 0.05),
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=float(train_cfg.get("learning_rate", 5e-4)))
        self.update_target_network()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def act(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q = self.model(s)
            return int(torch.argmax(q, dim=1).item())

    def replay(self, batch_size: int = 256) -> Dict[str, float] | None:
        if len(self.memory) < batch_size:
            return None

        batch = random.sample(self.memory, batch_size)
        s, a, r, s2, d = zip(*batch)

        s = torch.tensor(np.array(s), dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.int64, device=self.device)
        r = torch.tensor(r, dtype=torch.float32, device=self.device)
        s2 = torch.tensor(np.array(s2), dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device)

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


# =========================================================
# Evaluation and training
# =========================================================

def make_env(config: Dict[str, Any], max_steps: int) -> KWalletEnv:
    env_cfg = config["env"]
    attn_cfg = config["attention_context"]
    reward_cfg = config.get("reward", {})
    return KWalletEnv(
        C=env_cfg["C"],
        k=env_cfg["k"],
        F=env_cfg["F"],
        max_transaction=env_cfg["T"],
        max_steps=max_steps,
        seed=config["seed"],
        enable_shaping=env_cfg["enable_shaping"],
        attention_context_enabled=attn_cfg.get("enabled", True),
        attention_window_size=attn_cfg.get("window_size", 30),
        alpha_drop=reward_cfg.get("alpha_drop", 0.02),
        beta_flush=reward_cfg.get("beta_flush", 0.01),
    )


def summarize_results(all_results: List[Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
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
    label: str = "VAL",
) -> Dict[str, Any]:
    eval_cfg = config["eval"]
    train_cfg = config["train"]

    if label == "VAL":
        num_eval_episodes = min(tx_pool.shape[0], int(train_cfg.get("val_num_episodes", eval_cfg["num_episodes"])))
    else:
        num_eval_episodes = min(tx_pool.shape[0], int(eval_cfg["num_episodes"]))

    env = make_env(config, max_steps=eval_cfg["max_steps"])
    old_eps = agent.epsilon
    agent.epsilon = 0.0
    all_results = []

    for ep in range(num_eval_episodes):
        s = env.reset(tx_stream=tx_pool[ep])
        total_requested_value = 0.0
        total_tx_count = 0
        accepted_count = 0

        for _ in range(eval_cfg["max_steps"]):
            current_tx = env.current_tx
            total_requested_value += float(current_tx)
            total_tx_count += 1

            a = agent.act(s)
            s, _, done, info = env.step(a)
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

    agent.epsilon = old_eps
    summary = summarize_results(all_results)
    metric_name = train_cfg.get("val_metric", "value_accept_ratio")
    values = [r[metric_name] for r in all_results]

    return {
        "label": label,
        "metric": metric_name,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "num_episodes": num_eval_episodes,
        "summary": summary,
    }


def train_agent(config: Dict[str, Any], paths: Dict[str, str]):
    print("\n" + "=" * 70)
    print("Train Attention Context DQN")
    print("=" * 70)

    train_cfg = config["train"]
    attn_cfg = config["attention_context"]

    tx_pool_full = load_tx_pool(paths["train_pool_path"], expected_steps=train_cfg["max_steps"])
    train_use_episodes = int(train_cfg.get("train_use_episodes", tx_pool_full.shape[0]))
    if train_use_episodes > tx_pool_full.shape[0]:
        raise ValueError(f"train_use_episodes={train_use_episodes} exceeds pool size={tx_pool_full.shape[0]}")

    tx_pool_train = tx_pool_full[:train_use_episodes]

    val_pool_path = paths.get("val_pool_path")
    if val_pool_path and os.path.exists(val_pool_path):
        tx_pool_val = load_tx_pool(val_pool_path, expected_steps=train_cfg["max_steps"])
        val_source = f"Independent val pool: {val_pool_path}"
    else:
        tx_pool_val = None
        val_source = "No validation pool"

    print(f"Train pool loaded: {tx_pool_full.shape}")
    print(f"Train path       : {paths['train_pool_path']}")
    print(f"Train regime     : {paths['train_regime']}")
    print(f"Validation source: {val_source}")
    if tx_pool_val is not None:
        print(f"Train/Val shape  : train={tx_pool_train.shape}, val={tx_pool_val.shape}")

    env = make_env(config, max_steps=train_cfg["max_steps"])
    state_size = len(env._get_state())
    base_state_size = 3 * env.k + 1
    window_size = attn_cfg.get("window_size", 30)
    action_size = env.num_actions

    if state_size != base_state_size + window_size:
        raise ValueError(f"State size mismatch: state_size={state_size}, base={base_state_size}, window={window_size}")

    agent = DQNAgent(
        base_state_size=base_state_size,
        window_size=window_size,
        state_size=state_size,
        action_size=action_size,
        train_cfg=train_cfg,
        attn_cfg=attn_cfg,
        device=train_cfg["device"],
    )

    print(f"Env: C={env.C}, k={env.k}, F={env.F}, T={env.max_transaction}")
    print(f"Network: State={state_size}, BaseState={base_state_size}, RecentWindow={window_size}, Action={action_size}")
    print(f"Attention: d_model={attn_cfg['d_model']}, heads={attn_cfg['n_heads']}, context_dim={attn_cfg['context_dim']}")
    print(f"Reward: alpha_drop={env.alpha_drop}, beta_flush={env.beta_flush}")
    print(f"Episodes: {train_cfg['episodes']}\n")

    returns, loss_history, epsilons = [], [], []
    val_history = []
    best_val_score = -1e18
    best_val_snapshot = None
    val_every = int(train_cfg.get("val_every", 0))

    if config.get("debug_mode", False):
        tmp = tempfile.NamedTemporaryFile(prefix="kwallet_attn_best_", suffix=".pth", delete=False)
        best_ckpt_path = tmp.name
        tmp.close()
    else:
        best_ckpt_path = paths["best_model_path"]

    if train_cfg["episodes"] > tx_pool_train.shape[0]:
        raise ValueError("Training episodes exceed available train pool rows.")

    for ep in range(train_cfg["episodes"]):
        state = env.reset(tx_stream=tx_pool_train[ep])
        G = 0.0

        for _ in range(train_cfg["max_steps"]):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            G += reward

            m = agent.replay(batch_size=train_cfg["batch_size"])
            if m is not None:
                loss_history.append(m["loss"])
            if done:
                break

        if (ep + 1) % train_cfg["target_update_every"] == 0:
            agent.update_target_network()

        returns.append(G)
        epsilons.append(agent.epsilon)
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        if (ep + 1) % LOG_EVERY_N == 0 or ep == 0:
            recent_mean = np.mean(returns[-LOG_EVERY_N:]) if len(returns) >= LOG_EVERY_N else np.mean(returns)
            print(f"[Train] Episode {ep+1:4d}/{train_cfg['episodes']} | Return={G:10.2f} | RecentMean={recent_mean:10.2f} | Epsilon={agent.epsilon:.4f}")

        if tx_pool_val is not None and val_every > 0 and ((ep + 1) % val_every == 0 or (ep + 1) == train_cfg["episodes"]):
            val_result = evaluate_agent_on_array(agent, config, tx_pool_val, label="VAL")
            val_score = float(val_result["mean"])
            val_history.append({
                "episode": ep + 1,
                "metric": val_result["metric"],
                "mean": val_result["mean"],
                "std": val_result["std"],
                "num_episodes": val_result["num_episodes"],
            })
            print(f"[Val  ] Episode {ep+1:4d} | metric={val_result['metric']} | mean={val_score:.4f} | std={val_result['std']:.4f}")

            if val_score > best_val_score:
                best_val_score = val_score
                best_val_snapshot = val_history[-1]
                torch.save(agent.model.state_dict(), best_ckpt_path)
                print(f"Best checkpoint updated: episode={ep+1}, val_{val_result['metric']}={val_score:.4f}")

    if tx_pool_val is not None and best_val_snapshot is not None and os.path.exists(best_ckpt_path):
        agent.model.load_state_dict(torch.load(best_ckpt_path, map_location=agent.device))
        agent.target_model.load_state_dict(agent.model.state_dict())
        print(f"Loaded best checkpoint: episode={best_val_snapshot['episode']} | {best_val_snapshot['metric']}={best_val_snapshot['mean']:.4f}")
    else:
        print("No validation best checkpoint used. Using final model.")

    if config["save_mode"] == "full":
        torch.save(agent.model.state_dict(), paths["model_path"])
        print(f"Model saved to: {paths['model_path']}")

    return agent, returns, loss_history, epsilons, val_history


def evaluate_agent_on_pool(agent: DQNAgent, config: Dict[str, Any], test_pool_path: str, test_regime: str) -> Dict[str, Any]:
    eval_cfg = config["eval"]
    if not verify_data_integrity(test_pool_path, expected_steps=eval_cfg["max_steps"], label=f"regime={test_regime}"):
        raise RuntimeError(f"Data verification failed: {test_regime}")

    tx_pool = load_tx_pool(test_pool_path, expected_steps=eval_cfg["max_steps"])
    num_eval_episodes = min(eval_cfg["num_episodes"], tx_pool.shape[0])
    env = make_env(config, max_steps=eval_cfg["max_steps"])

    old_eps = agent.epsilon
    agent.epsilon = 0.0
    all_results = []
    total_action_hist = np.zeros(env.num_actions, dtype=int)

    print(f"Start evaluation: {test_regime} | episodes={num_eval_episodes} | pool={test_pool_path}")

    for ep in range(num_eval_episodes):
        s = env.reset(tx_stream=tx_pool[ep])
        total_requested_value = 0.0
        total_tx_count = 0
        accepted_count = 0
        flush_none_count = 0
        flush_action_count = 0
        settle_none_count = 0
        both_none_count = 0
        action_hist = np.zeros(env.num_actions, dtype=int)

        for _ in range(eval_cfg["max_steps"]):
            current_tx = env.current_tx
            total_requested_value += float(current_tx)
            total_tx_count += 1

            a = agent.act(s)
            action_hist[a] += 1
            settle_choice, flush_choice = env._decode_action(a)
            if flush_choice == env.k:
                flush_none_count += 1
            else:
                flush_action_count += 1
            if settle_choice == env.k:
                settle_none_count += 1
            if settle_choice == env.k and flush_choice == env.k:
                both_none_count += 1

            s, _, done, info = env.step(a)
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

        total_actions = max(int(np.sum(action_hist)), 1)
        metrics["flush_none_ratio"] = flush_none_count / total_actions
        metrics["flush_action_ratio"] = flush_action_count / total_actions
        metrics["settle_none_ratio"] = settle_none_count / total_actions
        metrics["both_none_ratio"] = both_none_count / total_actions

        all_results.append(metrics)
        total_action_hist += action_hist

    agent.epsilon = old_eps
    summary = summarize_results(all_results)
    top_idx = np.argsort(-total_action_hist)[:10]
    action_stats = {
        "top_actions": top_idx.tolist(),
        "top_action_counts": total_action_hist[top_idx].tolist(),
    }

    return {
        "test_regime": test_regime,
        "test_pool_path": test_pool_path,
        "num_episodes": num_eval_episodes,
        "summary": summary,
        "raw_results": all_results,
        "action_stats": action_stats,
    }


def evaluate_agent_cross_regime(agent: DQNAgent, config: Dict[str, Any], paths: Dict[str, str]) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("Cross-Regime Evaluation")
    print("=" * 70)

    cross_results = {}
    for regime_name, pool_path in paths["test_pool_paths"].items():
        cross_results[regime_name] = evaluate_agent_on_pool(agent, config, pool_path, regime_name)

    return {
        "config": config,
        "scenario": paths["scenario"],
        "train_regime": paths["train_regime"],
        "timestamp": datetime.now().isoformat(),
        "test_results": cross_results,
    }


# =========================================================
# Save and plot
# =========================================================

def build_cross_regime_report_text(results: Dict[str, Any]) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("Cross-Regime Evaluation Report")
    lines.append("=" * 70)
    lines.append(f"Train regime: {results['train_regime']}")
    lines.append("-" * 70)
    lines.append(
        f"{'Test':<8}{'Settled':>14}{'Drops':>12}{'Flushes':>12}"
        f"{'Util(%)':>12}{'DropRate(%)':>14}{'ValAcc(%)':>14}{'CntAcc(%)':>14}"
    )
    lines.append("-" * 70)
    for regime_name, regime_result in results["test_results"].items():
        s = regime_result["summary"]
        lines.append(
            f"{regime_name:<8}"
            f"{s['settled']['mean']:>14.2f}"
            f"{s['drops']['mean']:>12.2f}"
            f"{s['flushes']['mean']:>12.2f}"
            f"{100 * s['utilization']['mean']:>12.2f}"
            f"{100 * s['drop_rate']['mean']:>14.2f}"
            f"{100 * s['value_accept_ratio']['mean']:>14.2f}"
            f"{100 * s['count_accept_ratio']['mean']:>14.2f}"
        )
    lines.append("=" * 70)
    return "\n".join(lines)


def print_cross_regime_report(results: Dict[str, Any]):
    print("\n" + build_cross_regime_report_text(results))


def save_results(results: Dict[str, Any], save_path: str):
    compact_results = {
        "config": results["config"],
        "scenario": results["scenario"],
        "train_regime": results["train_regime"],
        "timestamp": results["timestamp"],
        "test_results": {},
    }
    for regime_name, regime_result in results["test_results"].items():
        compact_results["test_results"][regime_name] = {
            "num_episodes": regime_result["num_episodes"],
            "summary": {},
            "action_stats": regime_result.get("action_stats", {}),
        }
        for metric, data in regime_result["summary"].items():
            compact_results["test_results"][regime_name]["summary"][metric] = {
                "mean": data["mean"],
                "std": data["std"],
                "min": data["min"],
                "max": data["max"],
                "median": data["median"],
            }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(compact_results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {save_path}")


def save_results_summary_txt(results: Dict[str, Any], save_path: str):
    text = build_cross_regime_report_text(results)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Summary text saved to: {save_path}")


def plot_training_curves(returns, loss_history, epsilons, save_path, title_tag, window=50):
    ep_x = list(range(1, len(returns) + 1))
    loss_x = list(range(1, len(loss_history) + 1))

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle(f"Training Curves\n{title_tag}", fontsize=14, fontweight="bold")

    axes[0].plot(ep_x, returns, linewidth=1.3)
    axes[0].set_title("Episode Return")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Return")
    axes[0].grid(True, alpha=0.3)
    if len(returns) >= window:
        ma = np.convolve(returns, np.ones(window) / window, mode="valid")
        axes[0].plot(range(window, len(returns) + 1), ma, linewidth=2, label=f"Moving Avg ({window})")
        axes[0].legend()

    axes[1].plot(loss_x, loss_history, linewidth=1.0)
    axes[1].set_title("Replay Loss")
    axes[1].set_xlabel("Update Step")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(ep_x, epsilons, linewidth=1.3)
    axes[2].set_title("Epsilon")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Epsilon")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to: {save_path}")


def plot_validation_curve(val_history: List[Dict[str, Any]], save_path: str, title_tag: str):
    if not val_history:
        print("No validation history. Skip validation curve.")
        return

    episodes = [x["episode"] for x in val_history]
    means = [x["mean"] for x in val_history]
    stds = [x["std"] for x in val_history]
    metric_name = val_history[0].get("metric", "validation_metric")
    best_idx = int(np.argmax(means))

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, means, marker="o", linewidth=2, label=f"Validation {metric_name}")
    lower = np.array(means) - np.array(stds)
    upper = np.array(means) + np.array(stds)
    plt.fill_between(episodes, lower, upper, alpha=0.2, label="±1 std")
    plt.scatter([episodes[best_idx]], [means[best_idx]], s=80, label=f"Best @ ep{episodes[best_idx]}")
    plt.axvline(episodes[best_idx], linestyle="--", alpha=0.7)
    plt.title(f"Validation Curve\n{title_tag}", fontsize=14, fontweight="bold")
    plt.xlabel("Training Episode")
    plt.ylabel(metric_name)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Validation curve saved to: {save_path}")


def save_validation_history(val_history: List[Dict[str, Any]], save_path: str):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(val_history, f, indent=2, ensure_ascii=False)
    print(f"Validation history saved to: {save_path}")


def plot_evaluation_results(results: Dict[str, Any], save_path: str, title_tag: str):
    test_results = results["test_results"]
    regimes = list(test_results.keys())
    metrics_to_plot = [
        ("settled", "Total Settled", 1.0),
        ("drops", "Drops", 1.0),
        ("flushes", "Flushes", 1.0),
        ("drop_rate", "Drop Rate (%)", 100.0),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Cross-Regime Evaluation\n{title_tag}", fontsize=14, fontweight="bold")

    for idx, (metric, label, scale) in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        values = [test_results[r]["summary"][metric]["mean"] * scale for r in regimes]
        ax.bar(regimes, values)
        ax.set_title(label)
        ax.set_xlabel("Test Regime")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3, axis="y")
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Evaluation plot saved to: {save_path}")


# =========================================================
# Main
# =========================================================

def main():
    print("\n" + "=" * 70)
    print("K-Wallet Attention Context DQN Training and Cross-Regime Evaluation")
    print("=" * 70)

    set_seed(CONFIG["seed"])
    paths = build_paths(CONFIG)
    ensure_dirs(paths)
    save_run_info(CONFIG, paths)

    print(f"Scenario       : {paths['scenario']}")
    print(f"Run stamp      : {paths['run_stamp']}")
    print(f"Results dir    : {paths['result_run_dir']}")
    print(f"Checkpoints dir: {paths['checkpoint_run_dir']}")
    print(f"Data pool dir  : {paths['data_pool_dir']}")

    agent, returns, loss_history, epsilons, val_history = train_agent(CONFIG, paths)

    plot_training_curves(
        returns=returns,
        loss_history=loss_history,
        epsilons=epsilons,
        save_path=paths["training_plot_path"],
        title_tag=paths["title_tag"],
        window=CONFIG["plot"]["window"],
    )

    plot_validation_curve(
        val_history=val_history,
        save_path=paths["val_plot_path"],
        title_tag=paths["title_tag"],
    )
    save_validation_history(val_history, paths["val_history_json_path"])

    results = evaluate_agent_cross_regime(agent, CONFIG, paths)
    print_cross_regime_report(results)

    if CONFIG["save_mode"] == "full":
        save_results(results, paths["results_json_path"])
        save_results_summary_txt(results, paths["summary_txt_path"])
        plot_evaluation_results(results, paths["eval_plot_path"], paths["title_tag"])

    print("\nAll done.")


if __name__ == "__main__":
    main()
