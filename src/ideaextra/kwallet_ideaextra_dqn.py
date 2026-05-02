# K-Wallet DQN for ideaextra
# 说明：
# 1) 直接读取你新 generator 产出的 12-regime .npy pools
# 2) 训练时可选择 specialist 或 generalist
# 3) 结果统一保存到 ideaextra/results 和 ideaextra/checkpoints
# 4) 支持用 mixed_equal_val pool 做验证并保存 best checkpoint

import os
import argparse
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
# 路径：默认假设这个脚本放在 src/ideaextra/ 目录下
# =========================================================
IDEA_ROOT = Path(__file__).resolve().parent
DATA_POOL_DIR = IDEA_ROOT / "data" / "pools"
RESULTS_ROOT = IDEA_ROOT / "results"
CHECKPOINTS_ROOT = IDEA_ROOT / "checkpoints"


# =========================================================
# 常量：12 个 regime 名称与默认 pools
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
# 统一配置
# 你最常改这块
# =========================================================
CONFIG: Dict[str, Any] = {
    "seed": 123,
    "debug_mode": False,        # True 只在终端看，不写文件
    "save_mode": "full",        # none / brief / full

    "env": {
        "C": 1200.0,
        "k": 3,
        "T": 1000,
        "F": 3,
        "enable_shaping": False,
    },

    "data": {
        # ===== 训练池 =====
        # specialist 例子：
        # "train_regime": "LNB",
        # "train_pool_file": "LNB_static_master_T1000.npy",
        #
        # generalist 例子（当前默认）：
        "train_regime": "UB",
        "train_pool_file": "UB_static_master_T1000.npy",

        # ===== 验证池：建议始终使用 mixed-equal val =====
        "val_pool_file": DEFAULT_MIX_EQ_VAL,

        # ===== 测试池：12 个 static eval =====
        "test_pool_files": DEFAULT_STATIC_EVAL_FILES,
    },

    "train": {
        "episodes": 1000,
        "max_steps": 1000,
        "batch_size": 256,
        "target_update_every": 20,
        "device": "cpu",

        # 验证与 best checkpoint
        "validate_every": 20,
        "val_num_episodes": 60,
        "use_best_model_for_final_eval": True,
    },

    "eval": {
        "num_episodes": 100,
        "max_steps": 1000,
    },

    "plot": {
        "window": 20,
    },
}


# =========================================================
# 奖励塑形参数
# =========================================================
REFRESH_COST = 0.01
IMBALANCE_PENALTY = 0.02
WASTEFUL_REFRESH_PENALTY = 0.02
WASTEFUL_REFRESH_THRESH = 0.6
LOG_EVERY_N = 50


# =========================================================
# 工具函数
# =========================================================
def set_seed(seed: int = 123):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_run_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_scenario_name(config: Dict[str, Any]) -> str:
    env = config["env"]
    train_regime = config["data"]["train_regime"]
    test_regimes = "_".join(config["data"]["test_pool_files"].keys())
    C = int(env["C"]) if float(env["C"]).is_integer() else env["C"]
    return f"train{train_regime}_cross{test_regimes}_C{C}_k{env['k']}_T{env['T']}_F{env['F']}"


def build_title_tag(config: Dict[str, Any], run_stamp: str) -> str:
    env = config["env"]
    train_regime = config["data"]["train_regime"]
    return f"{run_stamp} | train={train_regime} | C={int(env['C'])} k={env['k']} T={env['T']} F={env['F']}"


def build_paths(config: Dict[str, Any]) -> Dict[str, str]:
    scenario = build_scenario_name(config)
    run_stamp = build_run_stamp()

    result_scenario_dir = RESULTS_ROOT / scenario
    result_run_dir = result_scenario_dir / run_stamp

    checkpoint_scenario_dir = CHECKPOINTS_ROOT / scenario
    checkpoint_run_dir = checkpoint_scenario_dir / run_stamp

    train_pool_file = config["data"]["train_pool_file"]
    val_pool_file = config["data"]["val_pool_file"]
    test_pool_files = config["data"]["test_pool_files"]

    return {
        "idea_root": str(IDEA_ROOT),
        "data_pool_dir": str(DATA_POOL_DIR),
        "results_root": str(RESULTS_ROOT),
        "checkpoints_root": str(CHECKPOINTS_ROOT),

        "scenario": scenario,
        "run_stamp": run_stamp,
        "title_tag": build_title_tag(config, run_stamp),

        "train_regime": config["data"]["train_regime"],
        "train_pool_path": str(DATA_POOL_DIR / train_pool_file),
        "val_pool_path": str(DATA_POOL_DIR / val_pool_file),
        "test_pool_paths": {k: str(DATA_POOL_DIR / v) for k, v in test_pool_files.items()},

        "result_scenario_dir": str(result_scenario_dir),
        "result_run_dir": str(result_run_dir),
        "run_info_path": str(result_run_dir / "run_info.json"),
        "results_json_path": str(result_run_dir / "cross_regime_results.json"),
        "summary_txt_path": str(result_run_dir / "cross_regime_summary.txt"),
        "eval_plot_path": str(result_run_dir / "cross_regime_bar.png"),
        "training_plot_path": str(result_run_dir / "train_curve.png"),
        "validation_plot_path": str(result_run_dir / "validation_curve.png"),
        "training_history_path": str(result_run_dir / "training_history.json"),

        "checkpoint_scenario_dir": str(checkpoint_scenario_dir),
        "checkpoint_run_dir": str(checkpoint_run_dir),
        "last_model_path": str(checkpoint_run_dir / "last_model.pth"),
        "best_model_path": str(checkpoint_run_dir / "best_model_by_val.pth"),
    }


def ensure_dirs(paths: Dict[str, str], config: Dict[str, Any]):
    os.makedirs(paths["data_pool_dir"], exist_ok=True)

    if not config["debug_mode"] and config["save_mode"] in ["brief", "full"]:
        os.makedirs(paths["result_scenario_dir"], exist_ok=True)
        os.makedirs(paths["result_run_dir"], exist_ok=True)

    if not config["debug_mode"] and config["save_mode"] == "full":
        os.makedirs(paths["checkpoint_scenario_dir"], exist_ok=True)
        os.makedirs(paths["checkpoint_run_dir"], exist_ok=True)


def save_run_info(config: Dict[str, Any], paths: Dict[str, str]):
    if config["debug_mode"] or config["save_mode"] == "none":
        return

    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "scenario": paths["scenario"],
        "run_stamp": paths["run_stamp"],
        "title_tag": paths["title_tag"],
        "config": config,
        "paths": paths,
    }

    with open(paths["run_info_path"], "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)


def load_tx_pool(pool_path: str, expected_steps: int) -> np.ndarray:
    if not os.path.exists(pool_path):
        raise FileNotFoundError(f"找不到 pool 文件: {pool_path}")

    if pool_path.endswith(".json"):
        with open(pool_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tx_pool = np.array(data, dtype=np.int32)
    elif pool_path.endswith(".npy"):
        tx_pool = np.load(pool_path)
    else:
        raise ValueError(f"暂不支持的文件格式: {pool_path}")

    if tx_pool.ndim != 2:
        raise ValueError(
            f"tx_pool 必须是二维数组 [num_episodes, steps]，当前 ndim={tx_pool.ndim}"
        )

    if tx_pool.shape[1] != expected_steps:
        raise ValueError(
            f"每个 episode 的交易数应为 {expected_steps}，"
            f"但当前 tx_pool.shape[1]={tx_pool.shape[1]}"
        )

    return tx_pool.astype(np.int32)


def verify_data_integrity(pool_path: str, expected_steps: int, label: str = "") -> bool:
    print("\n" + "=" * 70)
    print(f"🔍 数据完整性验证 {label}".strip())
    print("=" * 70)

    try:
        tx_pool = load_tx_pool(pool_path, expected_steps=expected_steps)
        file_size = os.path.getsize(pool_path) / 1024
        pool_hash = hashlib.md5(tx_pool.tobytes()).hexdigest()

        print(f"✅ 成功加载文件: {pool_path}")
        print(f"📊 矩阵形状: {tx_pool.shape}")
        print(f"💾 文件大小: {file_size:.2f} KB")
        print(f"🔑 数据指纹 (MD5): {pool_hash}")
        print(f"🎲 首个 episode 前5笔交易: {tx_pool[0, :5].tolist()}")
        print("=" * 70 + "\n")
        return True
    except Exception as e:
        print(f"❌ 数据加载失败: {str(e)}")
        return False


# =========================================================
# Q 网络
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
# K-Wallet 环境
# =========================================================
class KWalletEnv:
    def __init__(
        self,
        C: float = 3000,
        k: int = 4,
        F: int = 1,
        max_transaction: int = 1000,
        max_steps: int = 1000,
        seed: int = 123,
        enable_shaping: bool = True,
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
            raise ValueError(f"动作越界: action={action_int}, 合法范围应为 [0, {self.num_actions - 1}]")

        base = self.k + 1
        settle_choice = action_int // base
        flush_choice = action_int % base
        return settle_choice, flush_choice

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
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
            else:
                if self.enable_shaping:
                    reward -= getattr(self, "INVALID_ACTION_PENALTY", 0.05)

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

        self.time += 1

        for i in range(self.k):
            if self.pending_refill[i] and self._usable(i):
                self.wallets[i] = self.wallet_size
                self.pending_refill[i] = False

        if self.time < len(self._tx_stream):
            self.current_tx = self._tx_stream[self.time]

        done = (self.time >= self.max_steps)

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
        self.state_size = state_size
        self.action_size = action_size
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

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# =========================================================
# 评估核心：既能用于 val，也能用于 final test
# =========================================================
def summarize_episode_metrics(all_results: List[Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
    summary = {}
    metric_names = all_results[0].keys()

    for metric in metric_names:
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
    env_cfg = config["env"]

    num_eval_episodes = min(num_eval_episodes, tx_pool.shape[0])

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

    all_results = []

    for ep in range(num_eval_episodes):
        current_tx_stream = tx_pool[ep]
        s = env.reset(tx_stream=current_tx_stream)

        episode_total_requested_value = 0.0
        episode_total_tx_count = 0
        episode_accepted_count = 0

        for _ in range(max_steps):
            current_tx = env.current_tx
            episode_total_requested_value += float(current_tx)
            episode_total_tx_count += 1

            a = agent.act(s)
            s, r, done, info = env.step(a)

            if info.get("accepted", False):
                episode_accepted_count += 1

            if done:
                break

        metrics = env.get_metrics()
        metrics["value_accept_ratio"] = (
            metrics["settled"] / episode_total_requested_value
            if episode_total_requested_value > 0 else 0.0
        )
        metrics["count_accept_ratio"] = (
            episode_accepted_count / episode_total_tx_count
            if episode_total_tx_count > 0 else 0.0
        )
        metrics["total_requested_value"] = episode_total_requested_value
        metrics["total_tx_count"] = episode_total_tx_count
        metrics["accepted_count"] = episode_accepted_count

        all_results.append(metrics)

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

    if not verify_data_integrity(
        test_pool_path,
        expected_steps=eval_cfg["max_steps"],
        label=f"(regime={test_regime})",
    ):
        raise RuntimeError(f"数据验证失败，终止评估: {test_regime}")

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


def evaluate_agent_cross_regime(
    agent: DQNAgent,
    config: Dict[str, Any],
    paths: Dict[str, str],
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("🎯 开始 Cross-Regime 评估")
    print("=" * 70)

    cross_results = {}
    for regime_name, pool_path in paths["test_pool_paths"].items():
        cross_results[regime_name] = evaluate_agent_on_pool(
            agent=agent,
            config=config,
            test_pool_path=pool_path,
            test_regime=regime_name,
        )

    return {
        "config": config,
        "scenario": paths["scenario"],
        "train_regime": paths["train_regime"],
        "timestamp": datetime.now().isoformat(),
        "test_results": cross_results,
    }


# =========================================================
# 训练
# =========================================================
def train_agent(
    config: Dict[str, Any],
    paths: Dict[str, str],
):
    print("\n" + "=" * 70)
    print("🚀 开始训练 DQN 智能体")
    print("=" * 70)

    train_cfg = config["train"]
    env_cfg = config["env"]

    train_pool = load_tx_pool(
        pool_path=paths["train_pool_path"],
        expected_steps=train_cfg["max_steps"],
    )
    val_pool = load_tx_pool(
        pool_path=paths["val_pool_path"],
        expected_steps=train_cfg["max_steps"],
    )

    if train_cfg["episodes"] > train_pool.shape[0]:
        raise ValueError(
            f"训练 episodes={train_cfg['episodes']} 超过训练池行数 {train_pool.shape[0]}"
        )

    print(f"✅ 训练池: {train_pool.shape}")
    print(f"✅ 验证池: {val_pool.shape}")
    print(f"📂 train pool: {paths['train_pool_path']}")
    print(f"📂 val pool  : {paths['val_pool_path']}")

    env = KWalletEnv(
        C=env_cfg["C"],
        k=env_cfg["k"],
        F=env_cfg["F"],
        max_transaction=env_cfg["T"],
        max_steps=train_cfg["max_steps"],
        seed=config["seed"],
        enable_shaping=env_cfg["enable_shaping"],
    )

    state_size = len(env._get_state())
    action_size = env.num_actions
    agent = DQNAgent(state_size, action_size, device=train_cfg["device"])

    print(f"📊 环境配置: C={env.C}, k={env.k}, F={env.F}, T={env.max_transaction}")
    print(f"🧠 网络结构: state={state_size}, action={action_size}")
    print(f"🎮 动作空间: (settle_target, flush_target)")
    print(f"🎯 训练回合数: {train_cfg['episodes']}")

    returns, loss_history, epsilons = [], [], []
    validation_history = []
    best_val_score = -1e18
    best_state_dict = None

    for ep in range(train_cfg["episodes"]):
        current_tx_stream = train_pool[ep]
        state = env.reset(tx_stream=current_tx_stream)
        G = 0.0

        for _ in range(train_cfg["max_steps"]):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            G += reward

            metrics = agent.replay(batch_size=train_cfg["batch_size"])
            if metrics is not None and "loss" in metrics:
                loss_history.append(metrics["loss"])

            if done:
                break

        agent.decay_epsilon()
        returns.append(G)
        epsilons.append(agent.epsilon)

        if (ep + 1) % train_cfg["target_update_every"] == 0:
            agent.update_target_network()

        if (ep + 1) % train_cfg["validate_every"] == 0:
            val_result = evaluate_agent_on_array(
                agent=agent,
                config=config,
                tx_pool=val_pool,
                label="val_pool",
                num_eval_episodes=train_cfg["val_num_episodes"],
                max_steps=train_cfg["max_steps"],
            )

            val_score = val_result["summary"]["value_accept_ratio"]["mean"]
            validation_history.append({
                "episode": ep + 1,
                "value_accept_ratio": float(val_score),
                "drop_rate": float(val_result["summary"]["drop_rate"]["mean"]),
                "utilization": float(val_result["summary"]["utilization"]["mean"]),
            })

            if val_score > best_val_score:
                best_val_score = val_score
                best_state_dict = {k: v.detach().cpu().clone() for k, v in agent.model.state_dict().items()}

                if (not config["debug_mode"]) and config["save_mode"] == "full":
                    torch.save(best_state_dict, paths["best_model_path"])
                    print(f"💾 保存新的 best model: episode={ep+1}, val_score={val_score:.4f}")

        if (ep + 1) % LOG_EVERY_N == 0 or ep == 0:
            recent_returns = returns[max(0, len(returns) - LOG_EVERY_N):]
            mean_recent_return = np.mean(recent_returns)
            print(
                f"[Train] Episode {ep + 1:>4}/{train_cfg['episodes']} | "
                f"Return={G:>10.2f} | RecentMean={mean_recent_return:>10.2f} | "
                f"Epsilon={agent.epsilon:.4f}"
            )

    if best_state_dict is not None and train_cfg["use_best_model_for_final_eval"]:
        agent.model.load_state_dict(best_state_dict)
        agent.update_target_network()
        print(f"✅ 已切换到验证集最佳模型，best val score = {best_val_score:.4f}")

    if (not config["debug_mode"]) and config["save_mode"] == "full":
        torch.save(agent.model.state_dict(), paths["last_model_path"])
        print(f"💾 last model 已保存至: {paths['last_model_path']}")

    return agent, returns, loss_history, epsilons, validation_history


# =========================================================
# 输出与可视化
# =========================================================
def build_cross_regime_report_text(results: Dict[str, Any]) -> str:
    lines = []
    lines.append("=" * 90)
    lines.append("Cross-Regime Evaluation Report")
    lines.append("=" * 90)
    lines.append(f"timestamp    : {results['timestamp']}")
    lines.append(f"scenario     : {results['scenario']}")
    lines.append(f"train_regime : {results['train_regime']}")
    lines.append("-" * 90)
    lines.append(
        f"{'Test':<8}"
        f"{'Settled':>14}"
        f"{'Drops':>12}"
        f"{'Flushes':>12}"
        f"{'Util(%)':>12}"
        f"{'DropRate(%)':>14}"
        f"{'ValAcc(%)':>14}"
        f"{'CntAcc(%)':>14}"
    )
    lines.append("-" * 90)

    for regime_name, regime_result in results["test_results"].items():
        summary = regime_result["summary"]
        lines.append(
            f"{regime_name:<8}"
            f"{summary['settled']['mean']:>14.2f}"
            f"{summary['drops']['mean']:>12.2f}"
            f"{summary['flushes']['mean']:>12.2f}"
            f"{100 * summary['utilization']['mean']:>12.2f}"
            f"{100 * summary['drop_rate']['mean']:>14.2f}"
            f"{100 * summary['value_accept_ratio']['mean']:>14.2f}"
            f"{100 * summary['count_accept_ratio']['mean']:>14.2f}"
        )

    lines.append("=" * 90)
    return "\n".join(lines)


def print_cross_regime_report(results: Dict[str, Any]):
    print("\n" + build_cross_regime_report_text(results))


def save_results(results: Dict[str, Any], save_path: str):
    compact_results = {
        "config": results["config"],
        "scenario": results["scenario"],
        "train_regime": results["train_regime"],
        "timestamp": results["timestamp"],
        "test_results": {}
    }

    for regime_name, regime_result in results["test_results"].items():
        compact_results["test_results"][regime_name] = {
            "num_episodes": regime_result["num_episodes"],
            "summary": {}
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

    print(f"💾 Cross-regime 评估结果已保存至: {save_path}")


def save_results_summary_txt(results: Dict[str, Any], save_path: str):
    report_text = build_cross_regime_report_text(results)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"📝 文本摘要已保存至: {save_path}")


def save_training_history(
    returns: List[float],
    loss_history: List[float],
    epsilons: List[float],
    validation_history: List[Dict[str, float]],
    save_path: str,
):
    payload = {
        "returns": returns,
        "loss_history": loss_history,
        "epsilons": epsilons,
        "validation_history": validation_history,
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"📝 训练历史已保存至: {save_path}")


def moving_average(values: List[float], window: int) -> np.ndarray:
    if len(values) == 0:
        return np.array([])
    arr = np.array(values, dtype=float)
    out = np.zeros_like(arr)
    for i in range(len(arr)):
        left = max(0, i - window + 1)
        out[i] = np.mean(arr[left:i+1])
    return out


def plot_training_curves(
    returns: List[float],
    loss_history: List[float],
    epsilons: List[float],
    save_path: str,
    title_tag: str,
    window: int = 20,
):
    fig = plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(returns, alpha=0.35, label="Return")
    if len(returns) > 0:
        plt.plot(moving_average(returns, window), linewidth=2, label=f"MA({window})")
    plt.title(f"Training Curves\n{title_tag}")
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
    print(f"📈 训练曲线已保存至: {save_path}")


def plot_validation_curve(
    validation_history: List[Dict[str, float]],
    save_path: str,
    title_tag: str,
):
    if len(validation_history) == 0:
        return

    xs = [d["episode"] for d in validation_history]
    ys = [100 * d["value_accept_ratio"] for d in validation_history]

    plt.figure(figsize=(10, 5))
    plt.plot(xs, ys, marker="o")
    plt.title(f"Validation Curve\n{title_tag}")
    plt.xlabel("Episode")
    plt.ylabel("Value Accept Ratio (%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📈 验证曲线已保存至: {save_path}")


def plot_evaluation_results(results: Dict[str, Any], save_path: str, title_tag: str):
    test_results = results["test_results"]
    regimes = list(test_results.keys())

    metrics_to_plot = [
        ("value_accept_ratio", "Value Accept Ratio (%)"),
        ("count_accept_ratio", "Count Accept Ratio (%)"),
        ("drop_rate", "Drop Rate (%)"),
        ("utilization", "Utilization (%)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Cross-Regime Evaluation\n{title_tag}", fontsize=15, fontweight="bold")

    for idx, (metric, label) in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        values = []
        for regime in regimes:
            value = test_results[regime]["summary"][metric]["mean"]
            if metric in ["value_accept_ratio", "count_accept_ratio", "drop_rate", "utilization"]:
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
    print(f"📈 Cross-regime 对比图已保存至: {save_path}")


def save_brief_outputs(
    results: Dict[str, Any],
    config: Dict[str, Any],
    paths: Dict[str, str],
):
    os.makedirs(paths["result_run_dir"], exist_ok=True)

    summary_text_path = os.path.join(paths["result_run_dir"], "summary_table.txt")
    report_text = build_cross_regime_report_text(results)
    with open(summary_text_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    brief_info = {
        "timestamp": datetime.now().isoformat(),
        "scenario": paths["scenario"],
        "train_regime": config["data"]["train_regime"],
        "train_pool_file": config["data"]["train_pool_file"],
        "val_pool_file": config["data"]["val_pool_file"],
        "test_pool_files": config["data"]["test_pool_files"],
        "seed": config["seed"],
        "env": config["env"],
        "train": config["train"],
        "eval": config["eval"],
    }

    brief_json_path = os.path.join(paths["result_run_dir"], "run_brief.json")
    with open(brief_json_path, "w", encoding="utf-8") as f:
        json.dump(brief_info, f, indent=2, ensure_ascii=False)

    print(f"📝 已保存轻量结果: {summary_text_path}")
    print(f"📝 已保存运行摘要: {brief_json_path}")


# =========================================================
# 主流程
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_regime", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()

def main():
    args = parse_args()

    if args.train_regime is not None:
        CONFIG["data"]["train_regime"] = args.train_regime

        if args.train_regime == "MIX12_EQ":
            CONFIG["data"]["train_pool_file"] = DEFAULT_MIX_EQ_MASTER
        else:
            CONFIG["data"]["train_pool_file"] = f"{args.train_regime}_static_master_T1000.npy"

    if args.seed is not None:
        CONFIG["seed"] = args.seed
        
    print("\n" + "=" * 70)
    print("🎯 K-Wallet DQN 训练与 Cross-Regime 评估系统（ideaextra版）")
    print("=" * 70)

    set_seed(CONFIG["seed"])

    paths = build_paths(CONFIG)
    ensure_dirs(paths, CONFIG)

    print(f"🧪 当前场景: {paths['scenario']}")
    print(f"🧪 训练 regime: {paths['train_regime']}")
    print(f"🕒 本次运行: {paths['run_stamp']}")
    print(f"📂 idea 根目录: {paths['idea_root']}")
    print(f"📂 数据目录: {paths['data_pool_dir']}")
    print(f"📂 训练数据文件: {paths['train_pool_path']}")
    print(f"📂 验证数据文件: {paths['val_pool_path']}")
    print(f"💾 当前保存模式: {CONFIG['save_mode']}")

    if CONFIG["save_mode"] in ["brief", "full"] and not CONFIG["debug_mode"]:
        print(f"📂 本次结果目录: {paths['result_run_dir']}")
    if CONFIG["save_mode"] == "full" and not CONFIG["debug_mode"]:
        print(f"📂 本次模型目录: {paths['checkpoint_run_dir']}")

    try:
        save_run_info(CONFIG, paths)

        print("\n【阶段 1/2】训练 DQN 智能体")
        print("-" * 70)

        agent, returns, loss_history, epsilons, validation_history = train_agent(
            config=CONFIG,
            paths=paths,
        )

        print("\n【阶段 2/2】Cross-Regime 评估")
        print("-" * 70)

        results = evaluate_agent_cross_regime(
            agent=agent,
            config=CONFIG,
            paths=paths,
        )

        print_cross_regime_report(results)

        if CONFIG["save_mode"] == "none" or CONFIG["debug_mode"]:
            print("🛠️ 当前不保存文件，仅终端输出。")

        elif CONFIG["save_mode"] == "brief":
            save_brief_outputs(results, CONFIG, paths)

        elif CONFIG["save_mode"] == "full":
            save_results(results, paths["results_json_path"])
            save_results_summary_txt(results, paths["summary_txt_path"])
            save_training_history(
                returns, loss_history, epsilons, validation_history,
                paths["training_history_path"],
            )
            plot_training_curves(
                returns, loss_history, epsilons,
                save_path=paths["training_plot_path"],
                title_tag=paths["title_tag"],
                window=CONFIG["plot"]["window"],
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
        else:
            raise ValueError(f"未知 save_mode: {CONFIG['save_mode']}")

        print("\n✅ 所有任务完成!")
        print("=" * 70)

        if CONFIG["save_mode"] in ["brief", "full"] and not CONFIG["debug_mode"]:
            print(f"📁 结果目录: {paths['result_run_dir']}")
        if CONFIG["save_mode"] == "full" and not CONFIG["debug_mode"]:
            print(f"📁 模型目录: {paths['checkpoint_run_dir']}")

    except Exception as e:
        print(f"\n❌ 执行过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
