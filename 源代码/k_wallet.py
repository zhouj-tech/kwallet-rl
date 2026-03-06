##1/3
##1/22

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
from scipy import stats

# =========================================================
# ✅ 统一参数设置 (与 Baseline 完全一致)
# =========================================================
CONFIG = {
    "seed": 123,
    
    # 环境参数(训练 + 评估 共用)
    "env": {
        "C": 3000.0,
        "k": 10,
        "T": 100,
        "F": 1,
        "enable_shaping": True,
    },
    
    # 训练参数
    "train": {
        "episodes": 3000,
        "max_steps": 1000,
        "batch_size": 256,
        "target_update_every": 20,
        "device": "cpu",
    },
    
    # 评估参数(与baseline完全一致)
    "eval": {
        "num_episodes": 100,
        "max_steps": 1000,
        "tx_pool_path": "shared_tx_pool.npy",
        "test_start_idx": 4000  # 与baseline使用相同的测试段
    },
    
    # 输出配置
    "output": {
        "save_results": True,
        "results_path": "dqn_results.json",
        "plot_path": "dqn_evaluation.png",
        "model_path": "k_wallet_dqn_aligned.pth"
    },
    
    # 绘图参数
    "plot": {
        "window": 100,
    },
}

# 奖励塑形参数
REFRESH_COST = 0.01
IMBALANCE_PENALTY = 0.02
WASTEFUL_REFRESH_PENALTY = 0.02
WASTEFUL_REFRESH_THRESH = 0.6

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
# K-Wallet 环境 (修改版 - 对齐指标计算)
# =========================================================
class KWalletEnv:
    """
    K-Wallet 环境
    
    关键修改:
    - 指标计算与baseline完全对齐
    - 添加详细的业务指标跟踪
    """
    
    def __init__(
        self, 
        C: float = 3000, 
        k: int = 4, 
        F: int = 1, 
        max_transaction: int = 1000,
        max_steps: int = 1000, 
        seed: int = 123, 
        enable_shaping: bool = True
    ):
        self.C = float(C)
        self.k = int(k)
        self.F = int(F)
        self.max_transaction = int(max_transaction)
        self.max_steps = int(max_steps)
        self.wallet_size = self.C / self.k
        
        # 奖励参数
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
    
    def reset(self, tx_stream: List[int] = None) -> np.ndarray:
        """
        重置环境
        
        参数:
            tx_stream: 外部交易流(如果为None则随机生成)
        """
        self.wallets = [self.wallet_size] * self.k
        self.freeze_until = [-1] * self.k
        self.pending_refill = [False] * self.k
        
        # 业务指标(与baseline对齐)
        self.total_settled = 0.0
        self.total_accepted = 0.0  # 用于计算利用率
        self.num_flushes = 0
        self.drops = 0
        self.oversize_drops = 0
        self.insufficient_drops = 0
        
        self.time = 0
        
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
        
        return np.array(state, dtype=np.float32)
    
    def _usable(self, i: int) -> bool:
        """检查钱包i是否可用"""
        return self.time > self.freeze_until[i]
    
    def _decode_action(self, action_int: int) -> Tuple[List[int], List[int]]:
        """
        解码动作
        
        返回:
            settle_mask: 结算钱包掩码
            flush_mask: 刷新钱包掩码
        """
        num_refresh_combos = 1 << self.k
        refresh_idx = action_int % num_refresh_combos
        flush_mask = [(refresh_idx >> i) & 1 for i in range(self.k)]
        
        settle_choice = action_int // num_refresh_combos
        settle_mask = [0] * self.k
        if settle_choice < self.k:
            settle_mask[settle_choice] = 1
        
        return settle_mask, flush_mask
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步
        
        返回:
            next_state: 下一状态
            reward: 奖励(用于训练)
            done: 是否结束
            info: 额外信息
        """
        reward = 0.0
        flushes_this_step = 0
        refresh_targets = []
        tx = self.current_tx
        
        settle_mask, flush_mask = self._decode_action(action)
        
        # 记录刷新前的余额(用于惩罚浪费性刷新)
        pre_refresh_balances = {i: self.wallets[i] for i in range(self.k)}
        
        # 1. 执行刷新动作
        for i in range(self.k):
            if flush_mask[i] == 1:
                if self._usable(i):
                    self.pending_refill[i] = True
                    self.wallets[i] = 0.0
                    self.freeze_until[i] = self.time + self.F - 1
                    self.num_flushes += 1
                    flushes_this_step += 1
                    refresh_targets.append(i)
                else:
                    # 无效刷新(钱包仍在冻结期)
                    if self.enable_shaping:
                        reward -= getattr(self, "INVALID_ACTION_PENALTY", 0.05)
        
        # 2. 执行结算动作
        chosen_idx = -1
        for i in range(self.k):
            if settle_mask[i] == 1:
                chosen_idx = i
                break
        
        fit_idx = None
        
        if tx > self.wallet_size:
            # 交易超过单个钱包容量
            self.drops += 1
            self.oversize_drops += 1
            reward -= self.alpha_drop
        elif chosen_idx != -1:
            # 尝试在指定钱包结算
            if (self._usable(chosen_idx) and 
                (chosen_idx not in refresh_targets) and 
                self.wallets[chosen_idx] >= tx):
                # 成功结算
                self.wallets[chosen_idx] -= tx
                self.total_settled += tx
                self.total_accepted += tx
                reward += float(tx) / self.max_transaction
                fit_idx = chosen_idx
            else:
                # 结算失败
                self.drops += 1
                self.insufficient_drops += 1
                reward -= self.alpha_drop
        else:
            # 未选择钱包(相当于丢包)
            self.drops += 1
            self.insufficient_drops += 1
            reward -= self.alpha_drop
        
        # 3. 刷新惩罚
        reward -= self.beta_flush * flushes_this_step
        
        # 4. 奖励塑形
        if self.enable_shaping:
            # 不平衡惩罚
            usable_balances = [
                self.wallets[i] for i in range(self.k) 
                if self._usable(i)
            ]
            if len(usable_balances) >= 2:
                std_norm = float(np.std(np.array(usable_balances)) / self.wallet_size)
                reward -= IMBALANCE_PENALTY * std_norm
            
            # 浪费性刷新惩罚
            for i in refresh_targets:
                if (pre_refresh_balances[i] / self.wallet_size) >= WASTEFUL_REFRESH_THRESH:
                    reward -= WASTEFUL_REFRESH_PENALTY
        
        # 5. 时间推进
        self.time += 1
        
        # 6. 处理待补充的钱包
        for i in range(self.k):
            if self.pending_refill[i] and self._usable(i):
                self.wallets[i] = self.wallet_size
                self.pending_refill[i] = False
        
        # 7. 更新当前交易
        if self.time < len(self._tx_stream):
            self.current_tx = self._tx_stream[self.time]
        
        done = (self.time >= self.max_steps)
        
        return self._get_state(), float(reward), bool(done), {"fit_idx": fit_idx}
    
    def get_metrics(self) -> Dict[str, float]:
        """
        获取业务指标(与baseline完全一致)
        """
        return {
            'settled': self.total_settled,
            'drops': self.drops,
            'oversize_drops': self.oversize_drops,
            'insufficient_drops': self.insufficient_drops,
            'flushes': self.num_flushes,
            'utilization': self.total_accepted / (self.C * self.max_steps),
            'avg_tx_value': self.total_settled / max(1, self.max_steps - self.drops),
            'drop_rate': self.drops / self.max_steps,
        }


# =========================================================
# DQN Agent
# =========================================================
class DQNAgent:
    """DQN智能体"""
    
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
    
    def replay(self, batch_size: int = 128) -> Dict:
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
        """衰减epsilon"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# =========================================================
# 训练函数
# =========================================================
def train_agent(
    episodes: int = 3000,
    max_steps: int = 1000,
    batch_size: int = 256,
    target_update_every: int = 20,
    device: str = "cpu",
    k: int = 4
):
    """
    训练DQN智能体
    
    返回:
        agent: 训练好的智能体
        returns: 每个episode的回报
        loss_history: 损失历史
        epsilons: epsilon历史
    """
    print("\n" + "="*70)
    print("🚀 开始训练 DQN 智能体")
    print("="*70)
    
    # 1. 检查数据文件
    if not os.path.exists("shared_tx_pool.npy"):
        raise FileNotFoundError(
            "找不到 shared_tx_pool.npy，请先运行 generate_shared_data.py"
        )
    
    tx_pool = np.load("shared_tx_pool.npy")
    print(f"✅ 成功加载交易流数据: {tx_pool.shape}")
    
    # 2. 初始化环境和智能体
    env_cfg = CONFIG["env"]
    env = KWalletEnv(
        C=env_cfg["C"],
        k=env_cfg["k"],
        F=env_cfg["F"],
        max_transaction=env_cfg["T"],
        max_steps=max_steps,
        seed=CONFIG["seed"],
        enable_shaping=env_cfg["enable_shaping"]
    )
    
    state_size = len(env._get_state())
    action_size = (env.k + 1) * (1 << env.k)
    
    agent = DQNAgent(state_size, action_size, device=device)
    
    print(f"📊 环境配置: C={env.C}, k={env.k}, F={env.F}, T={env.max_transaction}")
    print(f"🧠 网络结构: State={state_size}, Action={action_size}")
    print(f"🎯 训练回合数: {episodes}\n")
    
    # 3. 训练循环
    returns, loss_history, epsilons = [], [], []
    
    for ep in range(episodes):
        # 从池子中取出该回合对应的序列
        current_tx_stream = tx_pool[ep]
        state = env.reset(tx_stream=current_tx_stream)
        G = 0.0
        
        for t in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            G += reward
            
            # 经验回放
            metrics = agent.replay(batch_size=batch_size)
            if metrics:
                loss_history.append(metrics["loss"])
            
            if done:
                break
        
        # epsilon衰减
        agent.decay_epsilon()
        
        # 更新目标网络
        if (ep + 1) % target_update_every == 0:
            agent.update_target_network()
        
        returns.append(G)
        epsilons.append(agent.epsilon)
        
        # 打印进度
        if (ep + 1) % LOG_EVERY_N == 0:
            avg = np.mean(returns[-100:])
            print(f"EP={ep+1:04d} | G={G:7.2f} | avg100={avg:7.2f} | "
                  f"drops={env.drops} | util={env.get_metrics()['utilization']:.3f} | "
                  f"eps={agent.epsilon:.3f}")
    
    # 4. 保存模型
    model_path = CONFIG["output"]["model_path"]
    torch.save(agent.model.state_dict(), model_path)
    print(f"\n💾 模型已保存至: {model_path}")
    print("="*70 + "\n")
    
    return agent, returns, loss_history, epsilons


# =========================================================
# 评估函数 (与 Baseline 完全对齐)
# =========================================================
def verify_data_integrity(tx_pool_path: str, config: Dict) -> bool:
    """验证数据文件的完整性(与baseline一致)"""
    print("\n" + "="*70)
    print("🔍 数据完整性验证")
    print("="*70)
    
    if not os.path.exists(tx_pool_path):
        print(f"❌ 错误: 找不到文件 {tx_pool_path}")
        return False
    
    try:
        tx_pool = np.load(tx_pool_path)
        file_size = os.path.getsize(tx_pool_path) / 1024
        
        print(f"✅ 成功加载文件: {tx_pool_path}")
        print(f"📊 矩阵形状: {tx_pool.shape}")
        print(f"💾 文件大小: {file_size:.2f} KB")
        
        # 验证测试段
        test_start = config["eval"]["test_start_idx"]
        test_end = test_start + config["eval"]["num_episodes"]
        
        if test_end > tx_pool.shape[0]:
            print(f"❌ 错误: 测试范围超出数据集")
            return False
        
        test_segment = tx_pool[test_start:test_end]
        segment_hash = hashlib.md5(test_segment.tobytes()).hexdigest()
        
        print(f"\n🔑 测试段数据指纹 (MD5): {segment_hash}")
        print(f"📍 测试段范围: Episodes [{test_start}, {test_end})")
        print(f"🎲 测试段首行前5笔交易: {test_segment[0, :5].tolist()}")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载失败: {str(e)}")
        return False


def evaluate_agent(
    agent: DQNAgent,
    num_episodes: int = 100,
    max_steps: int = 1000
) -> Dict[str, Any]:
    """
    评估DQN智能体(与baseline输出格式完全一致)
    
    返回:
        results: 包含所有评估结果的字典
    """
    print("\n" + "="*70)
    print("🎯 开始评估 DQN 智能体")
    print("="*70)
    
    # 1. 验证数据
    if not verify_data_integrity(CONFIG["eval"]["tx_pool_path"], CONFIG):
        raise RuntimeError("数据验证失败，终止评估")
    
    # 2. 加载测试数据
    tx_pool = np.load(CONFIG["eval"]["tx_pool_path"])
    test_start = CONFIG["eval"]["test_start_idx"]
    test_end = test_start + num_episodes
    test_segment = tx_pool[test_start:test_end]
    
    # 3. 初始化环境
    env_cfg = CONFIG["env"]
    env = KWalletEnv(
        C=env_cfg["C"],
        k=env_cfg["k"],
        F=env_cfg["F"],
        max_transaction=env_cfg["T"],
        max_steps=max_steps,
        seed=CONFIG["seed"],
        enable_shaping=env_cfg["enable_shaping"]
    )
    
    # 4. 设置贪婪策略
    old_eps = agent.epsilon
    agent.epsilon = 0.0
    
    print(f"🚀 评估配置:")
    print(f"   - 回合数: {num_episodes}")
    print(f"   - 每回合步数: {max_steps}")
    print(f"   - 策略: 贪婪 (epsilon=0.0)\n")
    
    # 5. 执行评估
    all_results = []
    
    for ep in range(num_episodes):
        # 使用测试段的交易流
        current_tx_stream = test_segment[ep]
        s = env.reset(tx_stream=current_tx_stream)
        
        for t in range(max_steps):
            a = agent.act(s)
            s, r, done, _ = env.step(a)
            if done:
                break
        
        # 收集业务指标
        metrics = env.get_metrics()
        all_results.append(metrics)
        
        # 进度显示
        if (ep + 1) % 20 == 0:
            print(f"   进度: {ep + 1}/{num_episodes} 回合完成")
    
    # 6. 恢复epsilon
    agent.epsilon = old_eps
    
    print("✅ 评估完成!\n")
    
    # 7. 汇总统计
    summary = {}
    metric_names = all_results[0].keys()
    
    for metric in metric_names:
        values = [r[metric] for r in all_results]
        summary[metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'values': values
        }
    
    return {
        'config': CONFIG,
        'timestamp': datetime.now().isoformat(),
        'num_episodes': num_episodes,
        'summary': summary,
        'raw_results': all_results
    }


def print_evaluation_report(results: Dict[str, Any]):
    """打印格式化的评估报告(与baseline一致)"""
    summary = results['summary']
    
    print("\n" + "="*70)
    print("📊 DQN 评估报告")
    print("="*70)
    print(f"⏰ 评估时间: {results['timestamp']}")
    print(f"🎯 测试回合数: {results['num_episodes']}")
    print("-"*70)
    
    # 主要业务指标
    print("\n【核心业务指标】")
    print(f"{'指标':<30} {'均值':>12} {'标准差':>12} {'范围':>20}")
    print("-"*70)
    
    key_metrics = ['settled', 'drops', 'flushes', 'utilization', 'drop_rate']
    
    for metric in key_metrics:
        if metric in summary:
            data = summary[metric]
            mean = data['mean']
            std = data['std']
            min_val = data['min']
            max_val = data['max']
            
            if metric == 'utilization':
                print(f"{'资金利用率 (%)':<30} {mean*100:>12.2f} {std*100:>12.2f} "
                      f"[{min_val*100:.2f}, {max_val*100:.2f}]")
            elif metric == 'drop_rate':
                print(f"{'丢包率 (%)':<30} {mean*100:>12.2f} {std*100:>12.2f} "
                      f"[{min_val*100:.2f}, {max_val*100:.2f}]")
            else:
                label_map = {
                    'settled': '总处理金额',
                    'drops': '丢包数',
                    'flushes': '刷新次数'
                }
                label = label_map.get(metric, metric)
                print(f"{label:<30} {mean:>12.2f} {std:>12.2f} "
                      f"[{min_val:.2f}, {max_val:.2f}]")
                print("-"*70)
                    # 扩展指标
    print("\n【扩展分析指标】")
    print(f"{'指标':<30} {'值':>12}")
    print("-"*70)
    
    extended_metrics = ['avg_tx_value', 'oversize_drops', 'insufficient_drops']
    for metric in extended_metrics:
        if metric in summary:
            mean = summary[metric]['mean']
            label_map = {
                'avg_tx_value': '平均处理交易额',
                'oversize_drops': '超大交易丢包数',
                'insufficient_drops': '余额不足丢包数'
            }
            label = label_map.get(metric, metric)
            print(f"{label:<30} {mean:>12.2f}")
    
    print("="*70 + "\n")


def save_results(results: Dict[str, Any], save_path: str):
    """保存结果为JSON格式"""
    # 创建精简版本(移除原始values数组以减小文件大小)
    compact_results = {
        'config': results['config'],
        'timestamp': results['timestamp'],
        'num_episodes': results['num_episodes'],
        'summary': {}
    }
    
    for metric, data in results['summary'].items():
        compact_results['summary'][metric] = {
            'mean': data['mean'],
            'std': data['std'],
            'min': data['min'],
            'max': data['max'],
            'median': data['median']
        }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(compact_results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 评估结果已保存至: {save_path}")


def plot_evaluation_results(results: Dict[str, Any], save_path: str):
    """生成评估结果可视化图表"""
    summary = results['summary']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('DQN 评估结果分析', fontsize=16, fontweight='bold')
    
    metrics_to_plot = [
        ('settled', '总处理金额', 'C0'),
        ('drops', '丢包数', 'C1'),
        ('flushes', '刷新次数', 'C2'),
        ('utilization', '资金利用率', 'C3'),
        ('drop_rate', '丢包率', 'C4'),
        ('avg_tx_value', '平均处理交易额', 'C5')
    ]
    
    for idx, (metric, label, color) in enumerate(metrics_to_plot):
        if metric not in summary:
            continue
            
        ax = axes[idx // 3, idx % 3]
        values = summary[metric]['values']
        
        # 绘制分布直方图
        ax.hist(values, bins=30, alpha=0.7, color=color, edgecolor='black')
        
        # 添加统计线
        mean = summary[metric]['mean']
        median = summary[metric]['median']
        ax.axvline(mean, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean:.2f}')
        ax.axvline(median, color='green', linestyle=':', linewidth=2, 
                   label=f'Median: {median:.2f}')
        
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('频数', fontsize=11)
        ax.set_title(f'{label} 分布', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📈 可视化图表已保存至: {save_path}")
    plt.close()


def plot_training_curves(returns: List[float], loss_history: List[float], 
                         epsilons: List[float], save_path: str = "training_curves.png"):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('DQN 训练过程分析', fontsize=16, fontweight='bold')
    
    # 1. Returns曲线
    ax = axes[0, 0]
    ax.plot(returns, alpha=0.3, color='blue', label='Episode Return')
    if len(returns) >= 100:
        window = 100
        moving_avg = np.convolve(returns, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(returns)), moving_avg, 
                color='red', linewidth=2, label=f'{window}-Episode Moving Avg')
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Return', fontsize=11)
    ax.set_title('训练回报曲线', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 2. Loss曲线
    ax = axes[0, 1]
    if loss_history:
        ax.plot(loss_history, alpha=0.5, color='orange')
        if len(loss_history) >= 1000:
            window = 1000
            moving_avg = np.convolve(loss_history, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(loss_history)), moving_avg, 
                    color='red', linewidth=2, label=f'{window}-Step Moving Avg')
        ax.set_xlabel('Training Step', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('训练损失曲线', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # 3. Epsilon衰减曲线
    ax = axes[1, 0]
    ax.plot(epsilons, color='green', linewidth=2)
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Epsilon', fontsize=11)
    ax.set_title('探索率衰减曲线', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 4. 最后1000个episodes的returns分布
    ax = axes[1, 1]
    if len(returns) >= 1000:
        recent_returns = returns[-1000:]
        ax.hist(recent_returns, bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax.axvline(np.mean(recent_returns), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(recent_returns):.2f}')
        ax.set_xlabel('Return', fontsize=11)
        ax.set_ylabel('频数', fontsize=11)
        ax.set_title('最近1000回合回报分布', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📈 训练曲线已保存至: {save_path}")
    plt.close()


def compare_with_baseline(
    dqn_results: Dict[str, Any], 
    baseline_results_path: str = "baseline_results.json"
):
    """
    与Baseline结果进行对比分析
    
    参数:
        dqn_results: DQN评估结果
        baseline_results_path: Baseline结果文件路径
    """
    if not os.path.exists(baseline_results_path):
        print(f"\n💡 提示: 未找到Baseline结果文件 ({baseline_results_path})")
        print("   若要进行对比分析，请先运行Baseline评估并保存结果\n")
        return
    
    try:
        with open(baseline_results_path, 'r', encoding='utf-8') as f:
            baseline_results = json.load(f)
        
        print("\n" + "="*70)
        print("⚔️  DQN vs Baseline (FWF) 对比分析")
        print("="*70)
        
        dqn_summary = dqn_results['summary']
        baseline_summary = baseline_results['summary']
        
        print(f"{'指标':<25} {'Baseline':>15} {'DQN':>15} {'提升率':>15}")
        print("-"*70)
        
        compare_metrics = ['settled', 'drops', 'flushes', 'utilization']
        
        for metric in compare_metrics:
            if metric in baseline_summary and metric in dqn_summary:
                baseline_val = baseline_summary[metric]['mean']
                dqn_val = dqn_summary[metric]['mean']
                
                # 计算提升率(注意drops和flushes越低越好)
                if metric in ['drops', 'flushes']:
                    improvement = (baseline_val - dqn_val) / baseline_val * 100
                else:
                    improvement = (dqn_val - baseline_val) / baseline_val * 100
                
                # 格式化输出
                if metric == 'utilization':
                    print(f"{'资金利用率':<25} {baseline_val*100:>14.2f}% "
                          f"{dqn_val*100:>14.2f}% {improvement:>14.2f}%")
                else:
                    label_map = {
                        'settled': '总处理金额',
                        'drops': '丢包数',
                        'flushes': '刷新次数'
                    }
                    label = label_map.get(metric, metric)
                    print(f"{label:<25} {baseline_val:>15.2f} {dqn_val:>15.2f} "
                          f"{improvement:>14.2f}%")
                
                # 统计显著性检验 (如果有原始数据)
                if 'values' in baseline_summary.get(metric, {}) and \
                   'values' in dqn_summary.get(metric, {}):
                    baseline_values = baseline_summary[metric]['values']
                    dqn_values = dqn_summary[metric]['values']
                    t_stat, p_value = stats.ttest_ind(baseline_values, dqn_values)
                    
                    sig_mark = "***" if p_value < 0.001 else \
                               ("**" if p_value < 0.01 else \
                                ("*" if p_value < 0.05 else ""))
                    print(f"  └─ 统计检验: t={t_stat:.3f}, p={p_value:.4f} {sig_mark}")
        
        print("="*70)
        print("注: *** p<0.001, ** p<0.01, * p<0.05")
        print("="*70 + "\n")
        
        # 生成对比可视化
        plot_comparison(dqn_results, baseline_results)
        
    except Exception as e:
        print(f"⚠️  对比分析失败: {str(e)}\n")


def plot_comparison(
    dqn_results: Dict[str, Any], 
    baseline_results: Dict[str, Any],
    save_path: str = "dqn_vs_baseline.png"
):
    """生成DQN与Baseline的对比可视化"""
    dqn_summary = dqn_results['summary']
    baseline_summary = baseline_results['summary']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('DQN vs Baseline 对比分析', fontsize=16, fontweight='bold')
    
    metrics_to_compare = [
        ('settled', '总处理金额'),
        ('drops', '丢包数'),
        ('flushes', '刷新次数'),
        ('utilization', '资金利用率')
    ]
    
    for idx, (metric, label) in enumerate(metrics_to_compare):
        ax = axes[idx // 2, idx % 2]
        
        if metric in baseline_summary and metric in dqn_summary:
            # 提取数据
            baseline_vals = baseline_summary[metric].get('values', [])
            dqn_vals = dqn_summary[metric].get('values', [])
            
            if baseline_vals and dqn_vals:
                # 绘制箱线图
                data_to_plot = [baseline_vals, dqn_vals]
                bp = ax.boxplot(data_to_plot, labels=['Baseline', 'DQN'],
                               patch_artist=True, showmeans=True)
                
                # 着色
                colors = ['lightblue', 'lightgreen']
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                
                # 添加均值文本
                baseline_mean = baseline_summary[metric]['mean']
                dqn_mean = dqn_summary[metric]['mean']
                
                if metric == 'utilization':
                    ax.text(1, baseline_mean, f'{baseline_mean*100:.2f}%', 
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
                    ax.text(2, dqn_mean, f'{dqn_mean*100:.2f}%', 
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
                else:
                    ax.text(1, baseline_mean, f'{baseline_mean:.2f}', 
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
                    ax.text(2, dqn_mean, f'{dqn_mean:.2f}', 
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
                
                ax.set_ylabel(label, fontsize=11)
                ax.set_title(f'{label} 对比', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 对比图表已保存至: {save_path}")
    plt.close()


# =========================================================
# 主执行流程
# =========================================================
def main():
    """主函数 - 完整的训练和评估流程"""
    print("\n" + "="*70)
    print("🎯 K-Wallet DQN 训练与评估系统")
    print("="*70)
    
    # 设置随机种子
    set_seed(CONFIG["seed"])
    
    try:
        # ============================================
        # 阶段 1: 训练
        # ============================================
        print("\n【阶段 1/3】训练DQN智能体")
        print("-"*70)
        
        agent, returns, loss_history, epsilons = train_agent(
            episodes=CONFIG["train"]["episodes"],
            max_steps=CONFIG["train"]["max_steps"],
            batch_size=CONFIG["train"]["batch_size"],
            target_update_every=CONFIG["train"]["target_update_every"],
            device=CONFIG["train"]["device"],
            k=CONFIG["env"]["k"]
        )
        
        # 绘制训练曲线
        plot_training_curves(returns, loss_history, epsilons, 
                           save_path="training_curves.png")
        
        # ============================================
        # 阶段 2: 评估
        # ============================================
        print("\n【阶段 2/3】评估DQN智能体")
        print("-"*70)
        
        results = evaluate_agent(
            agent,
            num_episodes=CONFIG["eval"]["num_episodes"],
            max_steps=CONFIG["eval"]["max_steps"]
        )
        
        # 打印评估报告
        print_evaluation_report(results)
        
        # 保存结果
        if CONFIG["output"]["save_results"]:
            save_results(results, CONFIG["output"]["results_path"])
        
        # 生成可视化
        plot_evaluation_results(results, CONFIG["output"]["plot_path"])
        
        # ============================================
        # 阶段 3: 与Baseline对比
        # ============================================
        print("\n【阶段 3/3】与Baseline对比分析")
        print("-"*70)
        
        compare_with_baseline(results, baseline_results_path="baseline_results.json")
        
        print("\n✅ 所有任务完成!")
        print("="*70)
        print("\n📁 生成的文件:")
        print(f"   - {CONFIG['output']['model_path']}: 训练好的模型")
        print(f"   - {CONFIG['output']['results_path']}: 评估结果(JSON)")
        print(f"   - {CONFIG['output']['plot_path']}: 评估可视化")
        print(f"   - training_curves.png: 训练曲线")
        print(f"   - dqn_vs_baseline.png: 对比图表(如果baseline结果存在)")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ 执行过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


