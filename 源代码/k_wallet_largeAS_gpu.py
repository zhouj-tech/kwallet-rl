import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Tuple

# --- 全局参数设置 ---
LOG_EVERY_N = 100               # 日志/调试参数：控制打印频率
REFRESH_COST = 0.01             # 每次执行刷新（Flush）动作的固定成本
IMBALANCE_PENALTY = 0.02        # 不平衡惩罚
WASTEFUL_REFRESH_PENALTY = 0.02 # 浪费刷新惩罚
WASTEFUL_REFRESH_THRESH = 0.6   # 浪费刷新的阈值

def set_seed(seed: int = 123):
    """设置随机种子，保证实验可复现"""
    import os, random, numpy as _np, torch as _torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    _np.random.seed(seed)
    _torch.manual_seed(seed)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(seed)

# ===== Q 网络 (神经网络结构) =====
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        # 全连接层网络结构 (MLP)
        self.fc1 = nn.Linear(state_size, 128)  
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        self.relu = nn.ReLU()

    def forward(self, state):
        """前向传播：输入状态，输出 Q 值"""
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# ===== 环境定义 (MDP) (与原版逻辑一致) =====
class KWalletEnv:
    def __init__(self, C=3000, k=10, F=5, max_transaction=100, max_steps=1000, seed=123, enable_shaping=True):
        self.C = C  # 总资金容量
        self.k = k  # 钱包数量
        self.F = F  # 冻结时长
        self.max_transaction = max_transaction
        self.max_steps = max_steps
        self.wallet_size = C / k  # 单个钱包容量
        self.alpha_drop = 0.02    # 掉单惩罚 (交易失败)
        self.beta_flush = REFRESH_COST # 刷新成本
        self.enable_shaping = enable_shaping # 是否启用奖励塑形

        if self.enable_shaping:
           self.IMBALANCE_PENALTY = IMBALANCE_PENALTY
           self.WASTEFUL_REFRESH_PENALTY = WASTEFUL_REFRESH_PENALTY
           self.WASTEFUL_REFRESH_THRESH = WASTEFUL_REFRESH_THRESH
           self.INVALID_ACTION_PENALTY = 0.05
        
        import numpy as _np
        self.rng = _np.random.default_rng(seed)
        self._external_tx_stream = None
        self.reset()

    def reset(self):
        self.wallets = [self.wallet_size] * self.k
        self.freeze_until = [-1] * self.k
        self.total_settled = 0.0
        self.total_accepted = 0.0
        self.num_flushes = 0
        self.drops = 0
        self.oversize_drops = 0
        self.time = 0
        if self._external_tx_stream is not None:
            self.current_tx = self._tx_stream[self.time]
        else:
            self._tx_stream = None
            self.current_tx = self._generate_transaction()
        return self._get_state()

    def _generate_transaction(self):
        return int(self.rng.integers(1, self.max_transaction + 1))

    def _usable(self, i: int) -> bool:
        return self.time > self.freeze_until[i]

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

    def _decode_action(self, action_int: int):
        total_bits = 2 * self.k
        if action_int < 0 or action_int >= (1 << total_bits):
            raise ValueError(f"action must be in [0, { (1<<total_bits)-1 }], got {action_int}")
        bits = [(action_int >> i) & 1 for i in range(total_bits)]
        settle_mask = bits[:self.k]
        flush_mask = bits[self.k:2*self.k]
        return settle_mask, flush_mask

    def step(self, action: int):
        reward = 0.0
        placed = False
        flushes_this_step = 0
        did_refresh = False
        refresh_targets = []
        pre_refresh_balances = {}

        tx = self.current_tx
        settle_mask, flush_mask = self._decode_action(action)
        pre_refresh_balances = {i: self.wallets[i] for i in range(self.k)}

        for i in range(self.k):
            if flush_mask[i] == 1:
                if self._usable(i):
                    self.wallets[i] = self.wallet_size
                    self.freeze_until[i] = self.time + self.F
                    self.num_flushes += 1
                    flushes_this_step += 1
                    did_refresh = True
                    refresh_targets.append(i)
                else:
                    if self.enable_shaping:
                        reward -= getattr(self, "INVALID_ACTION_PENALTY", 0.05)

        candidate_idxs = []
        for i in range(self.k):
            if settle_mask[i] == 1 and self._usable(i) and (i not in refresh_targets):
                if self.wallets[i] > 0:
                    candidate_idxs.append(i)

        total_available = sum(self.wallets[i] for i in candidate_idxs)
        
        if tx > self.wallet_size * self.k:
            self.oversize_drops += 1
            reward -= self.alpha_drop
            placed = False
        else:
            if total_available >= tx and len(candidate_idxs) > 0:
                remaining = tx
                for i in candidate_idxs:
                    take = min(self.wallets[i], remaining)
                    self.wallets[i] -= take
                    remaining -= take
                    if remaining <= 1e-9:
                        break
                self.total_settled += tx
                self.total_accepted += tx
                reward += float(tx) / float(self.C)
                placed = True
            else:
                self.drops += 1
                reward -= self.alpha_drop
                placed = False

        reward -= self.beta_flush * flushes_this_step
        
        if getattr(self, "enable_shaping", False):
            std_norm = float(np.std(np.array(self.wallets)) / self.wallet_size) if self.wallet_size > 0 else 0.0
            reward -= getattr(self, "IMBALANCE_PENALTY", 0.0) * std_norm
            for i in refresh_targets:
                pre_b = pre_refresh_balances.get(i, None)
                if pre_b is not None and (pre_b / self.wallet_size) >= getattr(self, "WASTEFUL_REFRESH_THRESH", 0.6):
                    reward -= getattr(self, "WASTEFUL_REFRESH_PENALTY", 0.0)

        self.time += 1
        self.current_tx = self._generate_transaction() if self._tx_stream is None else self._tx_stream[self.time] if self.time < len(self._tx_stream) else self._generate_transaction()
        done = (self.time >= self.max_steps)

        info = {
           "transaction": tx, "placed": int(placed), "total_settled": self.total_settled,
           "drops": self.drops, "oversize_drops": self.oversize_drops, "num_flushes": self.num_flushes,
           "did_refresh": int(did_refresh), "refresh_targets": refresh_targets,
           "settle_candidates": candidate_idxs, "flushes_this_step": flushes_this_step
        }
        return self._get_state(), float(reward), bool(done), info

    def active_count(self): return sum(1 for i in range(self.k) if self._usable(i))
    def avg_balance(self): return float(np.mean(self.wallets))

# ===== DQN Agent (智能体) (已更新为使用 self.device) =====
class DQNAgent:
    def __init__(self, state_size, action_size, device="cpu"):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device(device) # <--- 关键：设备初始化
        self.memory = deque(maxlen=20000)
        self.gamma = 0.99
        self.epsilon = 0.6
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.997

        # 将模型移动到指定设备
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-4) 
        self.update_target_network()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def act(self, state: np.ndarray) -> int:
        """决策：Epsilon-Greedy 策略"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            # 确保输入张量在正确的设备上
            s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q = self.model(s)
            return int(torch.argmax(q, dim=1).item())

    def replay(self, batch_size=128):
        if len(self.memory) < batch_size:
            return None
        batch = random.sample(self.memory, batch_size)
        s, a, r, s2, d = zip(*batch)
        
        # 确保所有张量都在正确的设备上
        s  = torch.tensor(np.array(s), dtype=torch.float32, device=self.device)
        a  = torch.tensor(a, dtype=torch.int64, device=self.device)
        r  = torch.tensor(r, dtype=torch.float32, device=self.device)
        s2 = torch.tensor(np.array(s2), dtype=torch.float32, device=self.device)
        d  = torch.tensor(d, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_online_q = self.model(s2)
            next_act = next_online_q.argmax(dim=1)
            next_target_q = self.target_model(s2)
            q_next = next_target_q.gather(1, next_act.unsqueeze(1)).squeeze(1)
            y = r + self.gamma * (1.0 - d) * q_next

        q = self.model(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = nn.MSELoss()(q, y)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()
        return float(loss.item())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ===== 日志打印 (与原版逻辑一致) =====
def log_step(env: KWalletEnv, t: int, info: dict, action: int):
    tx = info["transaction"]
    if t == 0:
        print(f"[t={t:03d}] 📥 回合开始 | 🧾 tx={tx:3d} | 活跃:{env.active_count()}/{env.k} | 平均余额:{env.avg_balance():.1f}")
    elif (t == env.max_steps - 1) or info.get("done", False):
        utilization_proxy = env.total_accepted / (env.max_transaction * env.max_steps)
        print(f"[t={t:03d}] 🏁 回合结束 | 累计成交={int(env.total_settled)} | Flush累计:{env.num_flushes} | 利用率代理:{utilization_proxy:.3f}")

# ===== 训练循环 (主流程) (已更新为使用 device) =====
def train_agent(episodes=5000, max_steps=1000, batch_size=128, target_update_every=20, device="cpu", k=3):
    env = KWalletEnv(C=3000, k=k, F=5, max_transaction=1000, max_steps=max_steps, seed=123, enable_shaping=True)
    state_size = len(env._get_state())
    action_size = 1 << (2 * env.k)

    agent = DQNAgent(state_size, action_size, device=device)

    returns = []
    loss_history = []
    epsilons = []

    for ep in range(episodes):
        state = env.reset()
        G = 0.0
        for t in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            G += reward

            loss = agent.replay(batch_size=batch_size)
            if loss is not None:
                loss_history.append(loss)

            is_start = (t == 0)
            is_end = (t == max_steps - 1) or done
            if is_start or is_end:
                log_step(env, t, info, action)

            if done: break
        
        agent.decay_epsilon()

        if (ep + 1) % target_update_every == 0:
            agent.update_target_network()

        returns.append(G)
        epsilons.append(agent.epsilon)
        avg = np.mean(returns[-100:]) if len(returns) >= 100 else np.mean(returns)
        
        print(f"Episode {ep+1:4d} | return={G:7.2f} | avg(last100)={avg:7.2f} | eps={agent.epsilon:5.3f} | buf={len(agent.memory)}")

    torch.save(agent.model.state_dict(), "k_wallet_dqn_bitmask.pth")
    return agent, returns, loss_history, epsilons

# ===== 评估函数 (已更新为使用 device) =====
def evaluate_agent(agent: DQNAgent, num_episodes=10, max_steps=1000, k=3):
    env = KWalletEnv(C=3000, k=k, F=5, max_transaction=1000, max_steps=max_steps)
    old_eps = agent.epsilon
    agent.epsilon = 0.0
    totals = []
    for ep in range(num_episodes):
        s = env.reset()
        G = 0.0
        for t in range(max_steps):
            a = agent.act(s)
            s, r, done, _ = env.step(a)
            G += r
            if done: break
        totals.append(G)
    agent.epsilon = old_eps
    print(f"\n--- 评估结果 ({num_episodes} 回合) ---")
    print(f"Eval: avg return over {num_episodes} eps = {np.mean(totals):.2f} ± {np.std(totals):.2f}")
    return totals

# ===== 绘图 (与原版逻辑一致) =====
def plot_convergence(returns, loss_hist, epsilons, window=100):
    def moving_avg(x, w):
        x = np.asarray(x, dtype=float)
        if len(x) < w: return np.array([])
        c = np.cumsum(np.insert(x, 0, 0.0))
        return (c[w:] - c[:-w]) / w

    episodes = np.arange(1, len(returns) + 1)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(episodes, returns, alpha=0.3, label="Return per episode")
    ma = moving_avg(returns, w=window)
    if ma.size > 1:
        axes[0].plot(episodes[window-1:], ma, label=f"Moving Avg ({window} episodes)", color='r')
    axes[0].set_title("1. Mean Return Convergence")
    axes[0].set_ylabel("Total Reward (G)")
    axes[0].legend(); axes[0].grid(True, alpha=0.5)

    if loss_hist:
        loss_window = max(1, len(loss_hist) // max(1, len(returns)) * 5)
        smoothed_loss = moving_avg(loss_hist, w=loss_window)
        if len(smoothed_loss) > 1:
            loss_episodes = np.linspace(1, len(returns), len(smoothed_loss))
            axes[1].plot(loss_episodes, smoothed_loss, color='g', label="Smoothed Loss")
    axes[1].set_title("2. DQN Training Loss")
    axes[1].set_ylabel("MSE Loss")
    axes[1].grid(True, alpha=0.5)

    axes[2].plot(episodes, epsilons, color='b', label="Epsilon decay")
    axes[2].set_title("3. Epsilon Decay Schedule")
    axes[2].set_ylabel("Epsilon")
    axes[2].set_xlabel("Episode")
    axes[2].grid(True, alpha=0.5)

    plt.tight_layout()
    plt.savefig("full_convergence_analysis_bitmask_gpu.png", dpi=150)
    print("\nSaved full convergence analysis -> full_convergence_analysis_bitmask_gpu.png")

if __name__ == "__main__":
    set_seed(123)

    # --- 关键修改: 检查并使用 GPU ---
    if torch.cuda.is_available():
        DEVICE = "cuda"
        print(f"✅ 找到 CUDA 设备: {torch.cuda.get_device_name(0)}，启用 GPU 加速。")
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
        print("✅ 找到 MPS 设备，启用 Apple Silicon 加速。")
    else:
        DEVICE = "cpu"
        print("❌ 未找到 GPU 设备，使用 CPU 运行。")
    
    K = 3
    NUM_EPISODES = 10000 
    MAX_STEPS = 1000
    print(f"--- 开始 bitmask DQN 训练: k={K}, episodes={NUM_EPISODES} ---")
    
    agent, hist, loss_hist, eps_hist = train_agent(
        episodes=NUM_EPISODES, 
        max_steps=MAX_STEPS, 
        target_update_every=20, 
        device=DEVICE,
        k=K
    )

    evaluate_agent(agent, num_episodes=5, max_steps=MAX_STEPS, k=K)
    plot_convergence(hist, loss_hist, eps_hist, window=100)
