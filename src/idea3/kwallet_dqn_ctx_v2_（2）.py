#4/3
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Tuple, Dict, Any, List
import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from scipy import stats
import tempfile


THIS_FILE = Path(__file__).resolve()
IDEA3_DIR = THIS_FILE.parent
SRC_DIR = IDEA3_DIR.parent
PROJECT_ROOT = SRC_DIR.parent


def generator_alias(name: str) -> str:
    """
    给 generator 一个简短别名，避免路径和标题太长
    """
    alias_map = {
        "mix_lognorm_small_mid_uniform_tail_v1": "mixTail",
        "mixture": "mix",
        "uniform": "uni",
        "lognormal": "logn",
        "exponential": "exp",
        "pareto": "par",
    }
    return alias_map.get(name, name[:12])


def build_run_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# =========================================================
# ✅ 统一参数设置
# =========================================================

CONFIG = {
    "seed":532,
    "debug_mode": True,
    "save_mode": "brief",

    # ==============================
    # 主实验环境参数（当前默认：k=6, F=3 主 setting）
    # ==============================
    "env": {
        "C": 3000.0,
        "k": 6,
        "T": 1000,
        "F": 3,
        "enable_shaping": False,
    },

    # ==============================
    # 数据配置
    # train / val / test 分开：
    # - train_pool_file: 训练池
    # - val_pool_file:   可选，若为空则从训练池尾部切出验证集
    # - test_pool_files: 独立 held-out 测试池
    # ==============================
    "data": {
        "train_regime": "MIXED_EQ",
        "train_pool_file": "MIXED_EQ_SLSHBLBH_master_T1000.npy",
        "val_pool_file": "MIXED_EQ_SLSHBLBH_val_T1000.npy",
        "test_pool_files": {
            "SL": "SL_static_eval_T1000.npy",
            "SH": "SH_static_eval_T1000.npy",
            "BL": "BL_static_eval_T1000.npy",
            "BH": "BH_static_eval_T1000.npy",
            "SW": "SL_BH_SH_20_50_30_switch_eval_T1000.npy",
        }
    },

    # ==============================
    # 训练参数（优先提升稳定性）
    # ==============================
    "train": {
        #训练循环总共跑多少轮。
        "episodes": 3000,
        #这次训练，最多从 master train pool 里取多少条 episode 数据出来用。
        "train_use_episodes": 3000,
        "val_num_episodes": 100,
        "max_steps": 1000,
        "batch_size": 256,
        "target_update_every": 10,
        "device": "cpu",
        "learning_rate": 5e-4,
        "epsilon_start": 0.8,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.9995,
        "val_every": 50,
        "val_metric": "count_accept_ratio",
        "val_split_episodes": 0,
    },
    # 评估参数（独立 test pools）
    "eval": {
        "num_episodes": 200,
        "max_steps": 1000,
    },

    "output": {
        "save_results": False,
    },

    "plot": {
        "window": 20,
    },

    # ==============================
    # Context-aware v2 配置
    # 4 个特征：
    # [recent_mean_tx, recent_large_ratio, usable_wallet_fraction, pressure_score]
    # ==============================
    "context": {
        "enabled": False,
        "window_size": 20,
        "large_tx_ratio_threshold": 0.6,
    },
}

LOG_EVERY_N = 100


def set_seed(seed: int = 123):
    """设置全局随机种子"""
    import os, random as _random, numpy as _np, torch as _torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    _random.seed(seed)
    _np.random.seed(seed)
    _torch.manual_seed(seed)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(seed)



# =========================================================
# ✅ 场景与路径管理
# =========================================================
def build_scenario_name(config: Dict[str, Any]) -> str:
    env = config["env"]
    train_regime = config["data"]["train_regime"]
    test_regimes = "".join(config["data"]["test_pool_files"].keys())
    C = int(env["C"]) if float(env["C"]).is_integer() else env["C"]
    ctx_tag = "ctx" if config.get("context", {}).get("enabled", False) else "static"
    return f"train{train_regime}_cross{test_regimes}_{ctx_tag}_C{C}_k{env['k']}_T{env['T']}_F{env['F']}"


def build_title_tag(config: Dict[str, Any], run_stamp: str) -> str:
    env = config["env"]
    train_regime = config["data"]["train_regime"]
    ctx_tag = "context" if config.get("context", {}).get("enabled", False) else "static"
    return f"{run_stamp} | train={train_regime} | mode={ctx_tag} | C={int(env['C'])} k={env['k']} T={env['T']} F={env['F']}"


def build_paths(config: Dict[str, Any]) -> Dict[str, str]:
    """
    统一使用项目根目录，并接入新的 regime pool 路径
    """
    scenario = build_scenario_name(config)
    run_stamp = build_run_stamp()

    data_pool_dir = PROJECT_ROOT / "src" / "idea3" / "data" / "pools"
    data_report_root = PROJECT_ROOT / "src" / "idea3" / "data" / "reports"

    result_scenario_dir = PROJECT_ROOT / "results" / scenario
    result_run_dir = result_scenario_dir / run_stamp

    checkpoint_scenario_dir = PROJECT_ROOT / "checkpoints" / scenario
    checkpoint_run_dir = checkpoint_scenario_dir / run_stamp

    train_pool_file = config["data"]["train_pool_file"]
    val_pool_file = config["data"].get("val_pool_file")
    test_pool_files = config["data"]["test_pool_files"]

    paths = {
        "project_root": str(PROJECT_ROOT),
        "scenario": scenario,
        "run_stamp": run_stamp,
        "title_tag": build_title_tag(config, run_stamp),

        # data
        "data_pool_dir": str(data_pool_dir),
        "data_report_root": str(data_report_root),
        "train_regime": config["data"]["train_regime"],
        "train_pool_path": str(data_pool_dir / train_pool_file),
        "val_pool_path": str(data_pool_dir / val_pool_file) if val_pool_file else None,
        "test_pool_paths": {regime: str(data_pool_dir / file_name) for regime, file_name in test_pool_files.items()},

        # results
        "result_scenario_dir": str(result_scenario_dir),
        "result_run_dir": str(result_run_dir),
        "run_info_path": str(result_run_dir / "run_info.json"),
        "results_json_path": str(result_run_dir / "cross_regime_results.json"),
        "summary_txt_path": str(result_run_dir / "cross_regime_summary.txt"),
        "eval_plot_path": str(result_run_dir / "cross_regime_bar.png"),
        "training_plot_path": str(result_run_dir / "train_curve.png"),

        # checkpoints
        "checkpoint_scenario_dir": str(checkpoint_scenario_dir),
        "checkpoint_run_dir": str(checkpoint_run_dir),
        "model_path": str(checkpoint_run_dir / "model.pth"),
        "best_model_path": str(checkpoint_run_dir / "best_model.pth"),
    }

    return paths


def ensure_dirs(paths: Dict[str, str], config: Dict[str, Any]):
    os.makedirs(paths["data_pool_dir"], exist_ok=True)
    os.makedirs(paths["data_report_root"], exist_ok=True)

    if not config["debug_mode"]:
        os.makedirs(paths["result_scenario_dir"], exist_ok=True)
        os.makedirs(paths["result_run_dir"], exist_ok=True)
        os.makedirs(paths["checkpoint_scenario_dir"], exist_ok=True)
        os.makedirs(paths["checkpoint_run_dir"], exist_ok=True)


def load_tx_pool(pool_path: str, expected_steps: int) -> np.ndarray:
    """
    加载新的 tx pool。
    期望格式：shape = [num_episodes, steps_per_episode]
    """
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
            f"tx_pool 必须是二维数组 [num_episodes, steps]，但当前 ndim={tx_pool.ndim}。"
            f" 你当前的新 pool 很可能还是一维单条 flow，需要先生成二维 episode pool。"
        )

    if tx_pool.shape[1] != expected_steps:
        raise ValueError(
            f"每个 episode 的交易数应为 {expected_steps}，"
            f"但当前 tx_pool.shape[1]={tx_pool.shape[1]}"
        )

    return tx_pool


def save_run_info(config: Dict[str, Any], paths: Dict[str, str]):
    if config["debug_mode"]:
        print("🛠️ debug_mode=True，跳过保存 run_info.json")
        return

    snapshot = {
        "scenario": paths["scenario"],
        "run_stamp": paths["run_stamp"],
        "title_tag": paths["title_tag"],
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "paths": paths,
    }

    with open(paths["run_info_path"], "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)

    print(f"📝 本次运行信息已保存至: {paths['run_info_path']}")


# =========================================================
# Q 网络定义
# =========================================================
class DQN(nn.Module):
    """深度Q网络"""
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
# K-Wallet 环境 (对齐指标计算)
# =========================================================
class KWalletEnv:
    """
    K-Wallet 环境

    关键特点:
    - 指标计算与 baseline 对齐
    - 添加详细业务指标跟踪
    - 可选 context-aware state：在当前状态后追加 recent-window 特征
    """

    def __init__(
        self,
        C: float = 3000,
        k: int = 4,
        F: int = 1,
        max_transaction: int = 1000,
        max_steps: int = 1000,
        seed: int = 123,
        enable_shaping: bool = True,
        context_enabled: bool = False,
        context_window_size: int = 20,
        large_tx_ratio_threshold: float = 0.6,
    ):
        self.C = float(C)
        self.k = int(k)
        self.F = int(F)
        self.max_transaction = int(max_transaction)
        self.max_steps = int(max_steps)
        self.wallet_size = self.C / self.k
        self.num_actions = (self.k + 1) ** 2  # 动作 = (结算目标, 刷新目标)

        # Context-aware 配置
        self.context_enabled = bool(context_enabled)
        self.context_window_size = int(context_window_size)
        self.large_tx_ratio_threshold = float(large_tx_ratio_threshold)
        self.large_tx_threshold = self.large_tx_ratio_threshold * self.wallet_size
        self.tx_history = deque(maxlen=self.context_window_size)

        # 奖励参数
        # 奖励参数
        self.alpha_drop = 0.02
        self.beta_flush = 0.01
        self.enable_shaping = enable_shaping

        # 这些参数统一挂到 self 上，避免引用未定义的全局常量
        self.IMBALANCE_PENALTY = 0.02
        self.WASTEFUL_REFRESH_PENALTY = 0.02
        self.WASTEFUL_REFRESH_THRESH = 0.6
        self.INVALID_ACTION_PENALTY = 0.05

        self.rng = np.random.default_rng(seed)
        self._tx_stream = None
        self.reset()

    def reset(self, tx_stream: List[int] = None) -> np.ndarray:
        """
        重置环境

        参数:
            tx_stream: 外部交易流(如果为None则随机生成)
        """
        self.wallets = [self.wallet_size] * self.k
        self.freeze_until = [-1] * self.k
        self.pending_refill = [False] * self.k

        # 业务指标
        self.total_settled = 0.0
        self.total_accepted = 0.0
        self.num_flushes = 0
        self.drops = 0
        self.oversize_drops = 0
        self.insufficient_drops = 0

        self.time = 0
        self.tx_history.clear()  # recent-window history 只记录过去已处理的交易

        # 加载或生成交易流
        if tx_stream is not None:
            self._tx_stream = list(tx_stream)
        else:
            self._tx_stream = [
                int(self.rng.integers(1, self.max_transaction + 1))
                for _ in range(self.max_steps)
            ]

        self.current_tx = self._tx_stream[self.time]
        return self._get_state()

    def _get_context_features(self) -> List[float]:
        """
        context v2：更“决策化”的 4 个特征
        返回顺序固定为：
        [recent_mean_tx, recent_large_ratio, usable_wallet_fraction, pressure_score]

        设计思想：
        - 不再只描述“最近交易流长什么样”
        - 还描述“最近交易流相对于当前可用资源有多大压力”
        """
        if (not self.context_enabled) or len(self.tx_history) == 0:
            return [0.0, 0.0, 0.0, 0.0]

        hist = np.array(self.tx_history, dtype=np.float32)

        # 1) 最近流量强度
        recent_mean_raw = float(np.mean(hist))
        recent_mean_tx = recent_mean_raw / self.max_transaction

        # 2) 最近大额交易比例
        recent_large_ratio = float(np.mean(hist > self.large_tx_threshold))

        # 3) 当前可用钱包比例
        usable_indices = [i for i in range(self.k) if self._usable(i)]
        usable_wallet_fraction = len(usable_indices) / self.k

        # 4) 当前“流量相对资源”的压力
        if len(usable_indices) == 0:
            avg_usable_balance = 1e-6
        else:
            avg_usable_balance = float(np.mean([self.wallets[i] for i in usable_indices]))
            avg_usable_balance = max(avg_usable_balance, 1e-6)

        pressure_raw = recent_mean_raw / avg_usable_balance
        pressure_score = min(pressure_raw, 3.0) / 3.0

        return [
            float(recent_mean_tx),
            float(recent_large_ratio),
            float(usable_wallet_fraction),
            float(pressure_score),
        ]

    def _get_state(self) -> np.ndarray:
        """构建状态向量"""
        state = []

        # 钱包余额(归一化)
        for w in self.wallets:
            state.append(w / self.wallet_size)

        # 可用性标志
        for i in range(self.k):
            state.append(0.0 if self._usable(i) else 1.0)

        # 剩余冻结时间(归一化)
        for i in range(self.k):
            rem = max(0, self.freeze_until[i] - self.time)
            state.append((rem / self.F) if self.F > 0 else 0.0)

        # 当前交易金额(归一化)
        state.append(self.current_tx / self.max_transaction)

        # 可选：在 state 后面追加 context 特征
        if self.context_enabled:
            state.extend(self._get_context_features())

        return np.array(state, dtype=np.float32)

    def _usable(self, i: int) -> bool:
        """检查钱包 i 是否可用"""
        return self.time > self.freeze_until[i]

    def _decode_action(self, action_int: int) -> Tuple[int, int]:
        """
        将整数动作解码为二元动作:
        - settle_choice: 0..k-1 表示结算到某个钱包，k 表示本步不结算
        - flush_choice:  0..k-1 表示刷新某个钱包，k 表示本步不刷新
        """
        if not (0 <= action_int < self.num_actions):
            raise ValueError(f"动作越界: action={action_int}, 合法范围应为 [0, {self.num_actions - 1}]")

        base = self.k + 1
        settle_choice = action_int // base
        flush_choice = action_int % base
        return settle_choice, flush_choice

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步

        动作语义:
        - 先执行 0 或 1 个刷新动作
        - 再执行 0 或 1 个结算动作
        """
        reward = 0.0
        flushes_this_step = 0
        refresh_targets = []
        tx = self.current_tx

        settle_choice, flush_choice = self._decode_action(action)

        # 记录刷新前余额，用于浪费性刷新惩罚
        pre_refresh_balances = {i: self.wallets[i] for i in range(self.k)}

        # 1) 先执行刷新
        if flush_choice < self.k:
            if self._usable(flush_choice):
                self.pending_refill[flush_choice] = True
                self.wallets[flush_choice] = 0.0
                self.freeze_until[flush_choice] = self.time + self.F - 1
                self.num_flushes += 1
                flushes_this_step += 1
                refresh_targets.append(flush_choice)
            else:
                # 刷新被冻结的钱包，给一个小惩罚
                if self.enable_shaping:
                    reward -= getattr(self, "INVALID_ACTION_PENALTY", 0.05)

        fit_idx = None

        # 2) 再执行结算
        if tx > self.wallet_size:
            # 单笔交易大于单个钱包容量，必丢
            self.drops += 1
            self.oversize_drops += 1
            reward -= self.alpha_drop
        elif settle_choice < self.k:
            # 尝试将交易放入指定钱包
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
            # settle_choice == k: 本步主动不结算
            self.drops += 1
            self.insufficient_drops += 1
            reward -= self.alpha_drop

        # 3) 刷新成本
        reward -= self.beta_flush * flushes_this_step

        # 4) 奖励塑形
        if self.enable_shaping:
            usable_balances = [
                self.wallets[i] for i in range(self.k)
                if self._usable(i)
            ]
            if len(usable_balances) >= 2:
                std_norm = float(np.std(np.array(usable_balances)) / self.wallet_size)
                reward -= self.IMBALANCE_PENALTY * std_norm

                for i in refresh_targets:
                    if (pre_refresh_balances[i] / self.wallet_size) >= self.WASTEFUL_REFRESH_THRESH:
                        reward -= self.WASTEFUL_REFRESH_PENALTY

        # 5) 当前交易已处理完，把它写入 history（供下一步 state 使用）
        self.tx_history.append(float(tx))

        # 6) 时间推进
        self.time += 1

        # 7) 冻结结束后补满钱包
        for i in range(self.k):
            if self.pending_refill[i] and self._usable(i):
                self.wallets[i] = self.wallet_size
                self.pending_refill[i] = False

        # 8) 更新当前交易
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
        """
        获取业务指标(与 baseline 对齐)
        """
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
    """DQN智能体"""

    def __init__(self, state_size: int, action_size: int, train_cfg: Dict[str, Any], device: str = "cpu"):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device(device)

        self.memory = deque(maxlen=20000)
        self.gamma = 0.98
        self.epsilon = float(train_cfg.get("epsilon_start", 0.8))
        self.epsilon_min = float(train_cfg.get("epsilon_min", 0.05))
        self.epsilon_decay = float(train_cfg.get("epsilon_decay", 0.999))

        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=float(train_cfg.get("learning_rate", 1e-3)))

        self.update_target_network()

    def update_target_network(self):
        """更新目标网络"""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, s, a, r, s2, done):
        """存储经验"""
        self.memory.append((s, a, r, s2, done))

    def act(self, state: np.ndarray) -> int:
        """选择动作(epsilon-greedy)"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q = self.model(s)
            return int(torch.argmax(q, dim=1).item())

    def replay(self, batch_size: int = 128) -> Dict[str, float] | None:
        """经验回放"""
        if len(self.memory) < batch_size:
            return None

        batch = random.sample(self.memory, batch_size)
        s, a, r, s2, d = zip(*batch)

        s = torch.tensor(np.array(s), dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.int64, device=self.device)
        r = torch.tensor(r, dtype=torch.float32, device=self.device)
        s2 = torch.tensor(np.array(s2), dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device)

        # Double DQN
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
        """衰减 epsilon"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# =========================================================
# 训练函数
# =========================================================

def evaluate_agent_on_array(
    agent: DQNAgent,
    config: Dict[str, Any],
    tx_pool: np.ndarray,
    label: str = "VAL",
) -> Dict[str, Any]:
    eval_cfg = config["eval"]
    env_cfg = config["env"]

    val_num_episodes = int(config["train"].get("val_num_episodes", eval_cfg["num_episodes"]))
    num_eval_episodes = min(tx_pool.shape[0], val_num_episodes)
    test_segment = tx_pool[:num_eval_episodes]

    env = KWalletEnv(
        C=env_cfg["C"],
        k=env_cfg["k"],
        F=env_cfg["F"],
        max_transaction=env_cfg["T"],
        max_steps=eval_cfg["max_steps"],
        seed=config["seed"],
        enable_shaping=env_cfg["enable_shaping"],
        context_enabled=config.get("context", {}).get("enabled", False),
        context_window_size=config.get("context", {}).get("window_size", 20),
        large_tx_ratio_threshold=config.get("context", {}).get("large_tx_ratio_threshold", 0.6),
    )

    old_eps = agent.epsilon
    agent.epsilon = 0.0
    all_results = []

    for ep in range(num_eval_episodes):
        current_tx_stream = test_segment[ep]
        s = env.reset(tx_stream=current_tx_stream)
        episode_total_requested_value = 0.0
        episode_total_tx_count = 0
        episode_accepted_count = 0

        for _ in range(eval_cfg["max_steps"]):
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
        metrics["value_accept_ratio"] = metrics["settled"] / episode_total_requested_value if episode_total_requested_value > 0 else 0.0
        metrics["count_accept_ratio"] = episode_accepted_count / episode_total_tx_count if episode_total_tx_count > 0 else 0.0
        all_results.append(metrics)

    agent.epsilon = old_eps
    metric_name = config["train"].get("val_metric", "count_accept_ratio")
    values = [r[metric_name] for r in all_results]
    return {
        "label": label,
        "metric": metric_name,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "num_episodes": num_eval_episodes,
    }


def train_agent(
    config: Dict[str, Any],
    paths: Dict[str, str]
):
    """训练 DQN 智能体（支持 validation + best checkpoint 选择）"""
    print("\n" + "=" * 70)
    print("🚀 开始训练 DQN 智能体")
    print("=" * 70)

    train_cfg = config["train"]
    env_cfg = config["env"]

    train_pool_path = paths["train_pool_path"]
    tx_pool_full = load_tx_pool(pool_path=train_pool_path, expected_steps=train_cfg["max_steps"])

    train_use_episodes = int(train_cfg.get("train_use_episodes", tx_pool_full.shape[0]))
    if train_use_episodes > tx_pool_full.shape[0]:
        raise ValueError(
            f"train_use_episodes={train_use_episodes} 超过训练池可用行数 {tx_pool_full.shape[0]}"
        )

    tx_pool_train = tx_pool_full[:train_use_episodes]

    val_pool_path = paths.get("val_pool_path")
    if val_pool_path and os.path.exists(val_pool_path):
        tx_pool_val = load_tx_pool(pool_path=val_pool_path, expected_steps=train_cfg["max_steps"])
        val_source = f"独立 val pool: {val_pool_path}"
    else:
        val_split_episodes = int(train_cfg.get("val_split_episodes", 0))
        if val_split_episodes > 0 and tx_pool_train.shape[0] > val_split_episodes:
            tx_pool_val = tx_pool_train[-val_split_episodes:]
            tx_pool_train = tx_pool_train[:-val_split_episodes]
            val_source = f"从训练池尾部切分 val episodes={val_split_episodes}"
        else:
            tx_pool_val = None
            val_source = "未启用 validation"

    if train_cfg["episodes"] > tx_pool_train.shape[0]:
        raise ValueError(
            f"训练 episodes={train_cfg['episodes']} 超过可用训练池行数 {tx_pool_train.shape[0]}。"
            f" 请增大 pool 中的 episode 数，或调小 train.episodes。"
        )

    print(f"✅ 成功加载训练池: {tx_pool_full.shape}")
    print(f"📂 训练数据路径: {train_pool_path}")
    print(f"🧪 当前训练场景: {paths['train_regime']}")
    print(f"🧪 Validation 来源: {val_source}")
    if tx_pool_val is not None:
        print(f"📊 Train/Val 形状: train={tx_pool_train.shape}, val={tx_pool_val.shape}")

    env = KWalletEnv(
        C=env_cfg["C"],
        k=env_cfg["k"],
        F=env_cfg["F"],
        max_transaction=env_cfg["T"],
        max_steps=train_cfg["max_steps"],
        seed=config["seed"],
        enable_shaping=env_cfg["enable_shaping"],
        context_enabled=config.get("context", {}).get("enabled", False),
        context_window_size=config.get("context", {}).get("window_size", 20),
        large_tx_ratio_threshold=config.get("context", {}).get("large_tx_ratio_threshold", 0.6),
    )

    state_size = len(env._get_state())
    action_size = env.num_actions
    agent = DQNAgent(state_size, action_size, train_cfg=train_cfg, device=train_cfg["device"])

    print(f"📊 环境配置: C={env.C}, k={env.k}, F={env.F}, T={env.max_transaction}")
    print(f"🧠 网络结构: State={state_size}, Action={action_size}")
    print(f"🧩 Context-aware: {env.context_enabled} | window={env.context_window_size} | large_tx_threshold={env.large_tx_threshold:.1f}")
    print(f"🎮 动作定义: (settle_target, flush_target)，每个维度取值 0..k-1 或 k=none")
    print(f"🎯 训练回合数: {train_cfg['episodes']}\n")

    returns, loss_history, epsilons = [], [], []
    best_val_score = -1e18
    best_val_snapshot = None
    val_every = int(train_cfg.get("val_every", 0))

    # 保存 best checkpoint：debug 模式也允许保存到临时文件
    if config.get("debug_mode", False):
        tmp = tempfile.NamedTemporaryFile(prefix="kwallet_best_", suffix=".pth", delete=False)
        best_ckpt_path = tmp.name
        tmp.close()
    else:
        os.makedirs(paths["checkpoint_run_dir"], exist_ok=True)
        best_ckpt_path = paths["best_model_path"]

    for ep in range(train_cfg["episodes"]):
        current_tx_stream = tx_pool_train[ep]
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

        if (ep + 1) % train_cfg["target_update_every"] == 0:
            agent.update_target_network()

        returns.append(G)
        epsilons.append(agent.epsilon)
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        if (ep + 1) % LOG_EVERY_N == 0 or ep == 0:
            recent_mean = np.mean(returns[-LOG_EVERY_N:]) if len(returns) >= LOG_EVERY_N else np.mean(returns)
            print(f"[Train] Episode {ep+1:4d}/{train_cfg['episodes']} | Return={G:10.2f} | RecentMean={recent_mean:10.2f} | Epsilon={agent.epsilon:.4f}")

        # validation + best checkpoint
        if tx_pool_val is not None and val_every > 0 and ((ep + 1) % val_every == 0 or (ep + 1) == train_cfg["episodes"]):
            val_result = evaluate_agent_on_array(agent, config, tx_pool_val, label="VAL")
            val_score = float(val_result["mean"])
            print(f"[Val ] Episode {ep+1:4d} | metric={val_result['metric']} | mean={val_score:.4f} | std={val_result['std']:.4f}")
            if val_score > best_val_score:
                best_val_score = val_score
                best_val_snapshot = {
                    "episode": ep + 1,
                    "metric": val_result["metric"],
                    "mean": val_result["mean"],
                    "std": val_result["std"],
                }
                torch.save(agent.model.state_dict(), best_ckpt_path)
                print(f"✅ 更新 best checkpoint: episode={ep+1}, val_{val_result['metric']}={val_score:.4f}")

    # 若启用了 validation，则加载 best checkpoint；否则保留最后一个模型
    if tx_pool_val is not None and best_val_snapshot is not None and os.path.exists(best_ckpt_path):
        agent.model.load_state_dict(torch.load(best_ckpt_path, map_location=agent.device))
        agent.target_model.load_state_dict(agent.model.state_dict())
        print(f"🎯 已加载 best checkpoint: episode={best_val_snapshot['episode']} | {best_val_snapshot['metric']}={best_val_snapshot['mean']:.4f}")
    else:
        print("📝 未启用 validation best checkpoint，使用最后一个模型。")

    if config["save_mode"] == "full":
        torch.save(agent.model.state_dict(), paths["model_path"])
        print(f"📝 模型已保存至: {paths['model_path']}")
    else:
        print("🛠️ 当前不是 full 模式，跳过保存模型")

    return agent, returns, loss_history, epsilons


def verify_data_integrity(pool_path: str, expected_steps: int, label: str = "") -> bool:
    """验证单个 pool 文件完整性"""
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


def evaluate_agent_on_pool(
    agent: DQNAgent,
    config: Dict[str, Any],
    test_pool_path: str,
    test_regime: str
) -> Dict[str, Any]:
    """在单个 regime pool 上评估 DQN 智能体，并记录动作统计。"""
    eval_cfg = config["eval"]
    env_cfg = config["env"]

    if not verify_data_integrity(
        test_pool_path,
        expected_steps=eval_cfg["max_steps"],
        label=f"(regime={test_regime})"
    ):
        raise RuntimeError(f"数据验证失败，终止评估: {test_regime}")

    tx_pool = load_tx_pool(pool_path=test_pool_path, expected_steps=eval_cfg["max_steps"])
    num_eval_episodes = min(eval_cfg["num_episodes"], tx_pool.shape[0])
    test_segment = tx_pool[:num_eval_episodes]

    env = KWalletEnv(
        C=env_cfg["C"],
        k=env_cfg["k"],
        F=env_cfg["F"],
        max_transaction=env_cfg["T"],
        max_steps=eval_cfg["max_steps"],
        seed=config["seed"],
        enable_shaping=env_cfg["enable_shaping"],
        context_enabled=config.get("context", {}).get("enabled", False),
        context_window_size=config.get("context", {}).get("window_size", 20),
        large_tx_ratio_threshold=config.get("context", {}).get("large_tx_ratio_threshold", 0.6),
    )

    old_eps = agent.epsilon
    agent.epsilon = 0.0

    print(f"🚀 开始评估 regime={test_regime}")
    print(f"   - 回合数: {num_eval_episodes}")
    print(f"   - 每回合步数: {eval_cfg['max_steps']}")
    print(f"   - 测试池: {test_pool_path}\n")

    all_results = []
    total_action_hist = np.zeros(env.num_actions, dtype=int)
    total_flush_none_count = 0
    total_flush_action_count = 0
    total_settle_none_count = 0
    total_both_none_count = 0

    for ep in range(num_eval_episodes):
        current_tx_stream = test_segment[ep]
        s = env.reset(tx_stream=current_tx_stream)
        episode_total_requested_value = 0.0
        episode_total_tx_count = 0
        episode_accepted_count = 0

        flush_none_count = 0
        flush_action_count = 0
        settle_none_count = 0
        both_none_count = 0
        action_hist = np.zeros(env.num_actions, dtype=int)

        for _ in range(eval_cfg["max_steps"]):
            current_tx = env.current_tx
            episode_total_requested_value += float(current_tx)
            episode_total_tx_count += 1

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

            s, r, done, info = env.step(a)
            if info.get("accepted", False):
                episode_accepted_count += 1
            if done:
                break

        metrics = env.get_metrics()
        metrics["value_accept_ratio"] = metrics["settled"] / episode_total_requested_value if episode_total_requested_value > 0 else 0.0
        metrics["count_accept_ratio"] = episode_accepted_count / episode_total_tx_count if episode_total_tx_count > 0 else 0.0
        metrics["total_requested_value"] = episode_total_requested_value
        metrics["total_tx_count"] = episode_total_tx_count
        metrics["accepted_count"] = episode_accepted_count

        total_actions = max(int(np.sum(action_hist)), 1)
        metrics["flush_none_ratio"] = flush_none_count / total_actions
        metrics["flush_action_ratio"] = flush_action_count / total_actions
        metrics["settle_none_ratio"] = settle_none_count / total_actions
        metrics["both_none_ratio"] = both_none_count / total_actions

        all_results.append(metrics)
        total_action_hist += action_hist
        total_flush_none_count += flush_none_count
        total_flush_action_count += flush_action_count
        total_settle_none_count += settle_none_count
        total_both_none_count += both_none_count

    agent.epsilon = old_eps

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
            "values": values
        }

    total_action_count = max(int(np.sum(total_action_hist)), 1)
    action_stats = {
        "flush_none_ratio": total_flush_none_count / total_action_count,
        "flush_action_ratio": total_flush_action_count / total_action_count,
        "settle_none_ratio": total_settle_none_count / total_action_count,
        "both_none_ratio": total_both_none_count / total_action_count,
        "top_actions": np.argsort(-total_action_hist)[:10].tolist(),
        "top_action_counts": total_action_hist[np.argsort(-total_action_hist)[:10]].tolist(),
    }

    return {
        "test_regime": test_regime,
        "test_pool_path": test_pool_path,
        "num_episodes": num_eval_episodes,
        "summary": summary,
        "raw_results": all_results,
        "action_stats": action_stats,
    }


def evaluate_agent_cross_regime(
    agent: DQNAgent,
    config: Dict[str, Any],
    paths: Dict[str, str]
) -> Dict[str, Any]:
    """
    训练一次后，在多个 regime 上统一评估
    """
    print("\n" + "=" * 70)
    print("🎯 开始 Cross-Regime 评估")
    print("=" * 70)

    cross_results = {}
    for regime_name, pool_path in paths["test_pool_paths"].items():
        cross_results[regime_name] = evaluate_agent_on_pool(
            agent=agent,
            config=config,
            test_pool_path=pool_path,
            test_regime=regime_name
        )

    return {
        "config": config,
        "scenario": paths["scenario"],
        "train_regime": paths["train_regime"],
        "timestamp": datetime.now().isoformat(),
        "test_results": cross_results
    }


def print_cross_regime_report(results: Dict[str, Any]):
    """
    打印 cross-regime 评估报告
    新增两列：
    - ValAcc(%): value_accept_ratio_mean * 100
    - CntAcc(%): count_accept_ratio_mean * 100
    """
    print("\n" + "=" * 70)
    print("📊 Cross-Regime Evaluation Report")
    print("=" * 70)
    print(f"⏰ 评估时间: {datetime.now().isoformat()}")
    print(f"🧪 Train regime: {results['train_regime']}")
    print("-" * 70)

    print(
        f"{'Test':<8}"
        f"{'Settled':>14}"
        f"{'Drops':>12}"
        f"{'Flushes':>12}"
        f"{'Util(%)':>12}"
        f"{'DropRate(%)':>14}"
        f"{'ValAcc(%)':>14}"
        f"{'CntAcc(%)':>14}"
    )
    print("-" * 70)

    for regime_name, regime_result in results["test_results"].items():
        summary = regime_result["summary"]

        print(
            f"{regime_name:<8}"
            f"{summary['settled']['mean']:>14.2f}"
            f"{summary['drops']['mean']:>12.2f}"
            f"{summary['flushes']['mean']:>12.2f}"
            f"{100 * summary['utilization']['mean']:>12.2f}"
            f"{100 * summary['drop_rate']['mean']:>14.2f}"
            f"{100 * summary['value_accept_ratio']['mean']:>14.2f}"
            f"{100 * summary['count_accept_ratio']['mean']:>14.2f}"
        )

    print("=" * 70)
#保存结果
def build_cross_regime_report_text(results: Dict[str, Any]) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("Cross-Regime Evaluation Report")
    lines.append("=" * 70)
    lines.append(f"Train regime: {results['train_regime']}")
    lines.append("-" * 70)
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
    lines.append("-" * 70)

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

    lines.append("=" * 70)
    return "\n".join(lines)

def save_brief_outputs(results: Dict[str, Any], config: Dict[str, Any], paths: Dict[str, str]):
    os.makedirs(paths["result_run_dir"], exist_ok=True)

    # 1. 保存表格文本
    summary_text_path = os.path.join(paths["result_run_dir"], "summary_table.txt")
    report_text = build_cross_regime_report_text(results)
    with open(summary_text_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # 2. 保存基础配置摘要
    brief_info = {
        "timestamp": datetime.now().isoformat(),
        "train_regime": config["data"]["train_regime"],
        "train_pool_file": config["data"]["train_pool_file"],
        "test_pool_files": config["data"]["test_pool_files"],
        "val_pool_file": config["data"].get("val_pool_file"),
        "seed": config["seed"],
        "context_enabled": config.get("context", {}).get("enabled", False),
        "env": {
            "C": config["env"]["C"],
            "k": config["env"]["k"],
            "T": config["env"]["T"],
            "F": config["env"]["F"],
        },
        "train": {
            "episodes": config["train"]["episodes"],
            "max_steps": config["train"]["max_steps"],
            "batch_size": config["train"]["batch_size"],
            "learning_rate": config["train"].get("learning_rate"),
            "epsilon_start": config["train"].get("epsilon_start"),
            "epsilon_min": config["train"].get("epsilon_min"),
            "epsilon_decay": config["train"].get("epsilon_decay"),
            "val_every": config["train"].get("val_every"),
            "val_metric": config["train"].get("val_metric"),
            "val_split_episodes": config["train"].get("val_split_episodes"),
        },
        "eval": {
            "num_episodes": config["eval"]["num_episodes"],
            "max_steps": config["eval"]["max_steps"],
        },
    }

    brief_json_path = os.path.join(paths["result_run_dir"], "run_brief.json")
    with open(brief_json_path, "w", encoding="utf-8") as f:
        json.dump(brief_info, f, indent=2, ensure_ascii=False)

    print(f"📝 已保存轻量结果: {summary_text_path}")
    print(f"📝 已保存运行摘要: {brief_json_path}")

def save_results(results: Dict[str, Any], save_path: str):
    """保存 cross-regime 结果为 JSON 格式"""
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
                "median": data["median"]
            }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(compact_results, f, indent=2, ensure_ascii=False)

    print(f"💾 Cross-regime 评估结果已保存至: {save_path}")


def save_results_summary_txt(results: Dict[str, Any], save_path: str):
    lines = []
    lines.append("Cross-Regime Evaluation Summary")
    lines.append("=" * 60)
    lines.append(f"timestamp    : {results['timestamp']}")
    lines.append(f"scenario     : {results['scenario']}")
    lines.append(f"train_regime : {results['train_regime']}")
    lines.append("")

    for regime_name, regime_result in results["test_results"].items():
        lines.append(f"[Test Regime: {regime_name}]")
        summary = regime_result["summary"]
        key_metrics = ["settled", "drops", "flushes", "utilization", "drop_rate",
                       "avg_tx_value", "oversize_drops", "insufficient_drops"]

        for metric in key_metrics:
            if metric in summary:
                s = summary[metric]
                lines.append(
                    f"{metric:<20} mean={s['mean']:.6f}  std={s['std']:.6f}  "
                    f"min={s['min']:.6f}  max={s['max']:.6f}  median={s['median']:.6f}"
                )
        lines.append("")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"📝 文本摘要已保存至: {save_path}")


def plot_evaluation_results(results: Dict[str, Any], save_path: str, title_tag: str):
    """Cross-regime 柱状对比图"""
    test_results = results["test_results"]
    regimes = list(test_results.keys())

    metrics_to_plot = [
        ("settled", "Total Settled"),
        ("drops", "Drops"),
        ("flushes", "Flushes"),
        ("utilization", "Utilization"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Cross-Regime Evaluation\n{title_tag}", fontsize=15, fontweight="bold")

    for idx, (metric, label) in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        values = []
        for regime in regimes:
            value = test_results[regime]["summary"][metric]["mean"]
            if metric == "utilization":
                value *= 100
            values.append(value)

        ax.bar(regimes, values)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Test Regime", fontsize=11)
        ax.set_ylabel(label + (" (%)" if metric == "utilization" else ""), fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"📈 Cross-regime 对比图已保存至: {save_path}")
    plt.close()


def compare_with_baseline(
    dqn_results: Dict[str, Any],
    baseline_results_path: str,
    comparison_plot_path: str
):
    """
    与 Baseline 结果进行对比分析
    """
    if not os.path.exists(baseline_results_path):
        print(f"\n💡 提示: 未找到 Baseline 结果文件 ({baseline_results_path})")
        print("   若要进行对比分析，请先在同一场景下运行 Baseline 并保存结果\n")
        return

    try:
        with open(baseline_results_path, "r", encoding="utf-8") as f:
            baseline_results = json.load(f)

        print("\n" + "=" * 70)
        print("⚔️  DQN vs Baseline (FWF) 对比分析")
        print("=" * 70)

        dqn_summary = dqn_results["summary"]
        baseline_summary = baseline_results["summary"]

        print(f"{'指标':<25} {'Baseline':>15} {'DQN':>15} {'提升率':>15}")
        print("-" * 70)

        compare_metrics = ["settled", "drops", "flushes", "utilization"]

        for metric in compare_metrics:
            if metric in baseline_summary and metric in dqn_summary:
                baseline_val = baseline_summary[metric]["mean"]
                dqn_val = dqn_summary[metric]["mean"]

                if metric in ["drops", "flushes"]:
                    improvement = (baseline_val - dqn_val) / baseline_val * 100 if baseline_val != 0 else 0.0
                else:
                    improvement = (dqn_val - baseline_val) / baseline_val * 100 if baseline_val != 0 else 0.0

                if metric == "utilization":
                    print(
                        f"{'资金利用率':<25} {baseline_val*100:>14.2f}% "
                        f"{dqn_val*100:>14.2f}% {improvement:>14.2f}%"
                    )
                else:
                    label_map = {
                        "settled": "总处理金额",
                        "drops": "丢包数",
                        "flushes": "刷新次数"
                    }
                    label = label_map.get(metric, metric)
                    print(
                        f"{label:<25} {baseline_val:>15.2f} {dqn_val:>15.2f} "
                        f"{improvement:>14.2f}%"
                    )

                if "values" in baseline_summary.get(metric, {}) and "values" in dqn_summary.get(metric, {}):
                    baseline_values = baseline_summary[metric]["values"]
                    dqn_values = dqn_summary[metric]["values"]
                    t_stat, p_value = stats.ttest_ind(baseline_values, dqn_values)

                    sig_mark = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else ""))
                    print(f"  └─ 统计检验: t={t_stat:.3f}, p={p_value:.4f} {sig_mark}")

        print("=" * 70)
        print("注: *** p<0.001, ** p<0.01, * p<0.05")
        print("=" * 70 + "\n")

        plot_comparison(
            dqn_results,
            baseline_results,
            save_path=comparison_plot_path,
            title_tag=f"{datetime.now().strftime('%Y%m%d_%H%M%S')} | {dqn_results['scenario']}"
        )

    except Exception as e:
        print(f"⚠️  对比分析失败: {str(e)}\n")


def plot_comparison(
    dqn_results: Dict[str, Any],
    baseline_results: Dict[str, Any],
    save_path: str,
    title_tag: str
):
    """生成 DQN 与 Baseline 的对比可视化"""
    dqn_summary = dqn_results["summary"]
    baseline_summary = baseline_results["summary"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"DQN vs Baseline\n{title_tag}", fontsize=15, fontweight="bold")

    metrics_to_compare = [
        ("settled", "总处理金额"),
        ("drops", "丢包数"),
        ("flushes", "刷新次数"),
        ("utilization", "资金利用率")
    ]

    for idx, (metric, label) in enumerate(metrics_to_compare):
        ax = axes[idx // 2, idx % 2]

        if metric in baseline_summary and metric in dqn_summary:
            baseline_vals = baseline_summary[metric].get("values", [])
            dqn_vals = dqn_summary[metric].get("values", [])

            if baseline_vals and dqn_vals:
                bp = ax.boxplot(
                    [baseline_vals, dqn_vals],
                    labels=["Baseline", "DQN"],
                    patch_artist=True,
                    showmeans=True
                )

                colors = ["lightblue", "lightgreen"]
                for patch, color in zip(bp["boxes"], colors):
                    patch.set_facecolor(color)

                baseline_mean = baseline_summary[metric]["mean"]
                dqn_mean = dqn_summary[metric]["mean"]

                if metric == "utilization":
                    ax.text(1, baseline_mean, f"{baseline_mean*100:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
                    ax.text(2, dqn_mean, f"{dqn_mean*100:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
                else:
                    ax.text(1, baseline_mean, f"{baseline_mean:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
                    ax.text(2, dqn_mean, f"{dqn_mean:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

                ax.set_ylabel(label, fontsize=11)
                ax.set_title(f"{label} 对比", fontsize=12, fontweight="bold")
                ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"📊 对比图表已保存至: {save_path}")
    plt.close()


# =========================================================
# 主执行流程
# =========================================================
def main():
    """主函数 - 新的训练 + cross-regime 评估流程"""
    print("\n" + "=" * 70)
    print("🎯 K-Wallet DQN 训练与 Cross-Regime 评估系统")
    print("=" * 70)

    set_seed(CONFIG["seed"])

    paths = build_paths(CONFIG)
    ensure_dirs(paths, CONFIG)

    print(f"🧪 当前场景: {paths['scenario']}")
    print(f"🧪 训练 regime: {paths['train_regime']}")
    print(f"🕒 本次运行: {paths['run_stamp']}")
    print(f"📂 项目根目录: {paths['project_root']}")
    print(f"📂 训练数据文件: {paths['train_pool_path']}")
    if paths.get("val_pool_path"):
        print(f"📂 验证数据文件: {paths['val_pool_path']}")
    print(f"💾 当前保存模式: {CONFIG['save_mode']}")

    if CONFIG["save_mode"] in ["brief", "full"]:
        print(f"📂 本次结果目录: {paths['result_run_dir']}")
    if CONFIG["save_mode"] == "full":
        print(f"📂 本次模型目录: {paths['checkpoint_run_dir']}")
    if CONFIG["save_mode"] == "none":
        print("🛠️ save_mode=none：不保存任何文件")
    elif CONFIG["save_mode"] == "brief":
        print("🛠️ save_mode=brief：仅保存 summary_table.txt 和 run_brief.json")
    elif CONFIG["save_mode"] == "full":
        print("🛠️ save_mode=full：保存完整结果、图表、模型")

    try:
        # 阶段 1: 训练
        print("\n【阶段 1/2】训练 DQN 智能体")
        print("-" * 70)

        agent, returns, loss_history, epsilons = train_agent(
            config=CONFIG,
            paths=paths
        )

        # 训练曲线：只有 full 模式才保存
        if CONFIG["save_mode"] == "full":
            plot_training_curves(
                returns,
                loss_history,
                epsilons,
                save_path=paths["training_plot_path"],
                title_tag=paths["title_tag"],
                window=CONFIG["plot"]["window"]
            )
        else:
            print("🛠️ 当前不是 full 模式，跳过保存训练曲线")

        # 阶段 2: Cross-Regime 评估
        print("\n【阶段 2/2】Cross-Regime 评估")
        print("-" * 70)

        results = evaluate_agent_cross_regime(
            agent=agent,
            config=CONFIG,
            paths=paths
        )

        print_cross_regime_report(results)

        # 保存逻辑
        if CONFIG["save_mode"] == "none":
            print("🛠️ save_mode=none，跳过所有文件保存")

        elif CONFIG["save_mode"] == "brief":
            save_brief_outputs(results, CONFIG, paths)

        elif CONFIG["save_mode"] == "full":
            save_run_info(CONFIG, paths)
            save_results(results, paths["results_json_path"])
            save_results_summary_txt(results, paths["summary_txt_path"])
            plot_evaluation_results(
                results,
                paths["eval_plot_path"],
                title_tag=paths["title_tag"]
            )

        else:
            raise ValueError(f"未知 save_mode: {CONFIG['save_mode']}")

        print("\n✅ 所有任务完成!")
        print("=" * 70)

        if CONFIG["save_mode"] == "none":
            print("\n🛠️ 本次仅在终端输出结果，不写任何文件。")
            print("=" * 70 + "\n")

        elif CONFIG["save_mode"] == "brief":
            print("\n📁 本次轻量产物:")
            print(f"   [结果目录]   {paths['result_run_dir']}")
            print(f"   - {os.path.join(paths['result_run_dir'], 'summary_table.txt')}")
            print(f"   - {os.path.join(paths['result_run_dir'], 'run_brief.json')}")
            print("=" * 70 + "\n")

        elif CONFIG["save_mode"] == "full":
            print("\n📁 本次完整产物:")
            print(f"   [结果目录]   {paths['result_run_dir']}")
            print(f"   [模型目录]   {paths['checkpoint_run_dir']}")
            print(f"   - {paths['run_info_path']}")
            print(f"   - {paths['results_json_path']}")
            print(f"   - {paths['summary_txt_path']}")
            print(f"   - {paths['training_plot_path']}")
            print(f"   - {paths['eval_plot_path']}")
            print(f"   - {paths['model_path']}")
            print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ 执行过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()