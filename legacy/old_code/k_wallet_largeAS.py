import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Tuple


LOG_EVERY_N = 100               
SIDENOTE_INTERVAL = 20          
REFRESH_COST = 0.01             # 刷新固定成本
IMBALANCE_PENALTY = 0.02        # 不平衡惩罚
WASTEFUL_REFRESH_PENALTY = 0.02 # 浪费刷新惩罚
WASTEFUL_REFRESH_THRESH = 0.6   # 浪费刷新阈值

def set_seed(seed: int = 123):
    import os, random, numpy as _np, torch as _torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    _np.random.seed(seed)
    _torch.manual_seed(seed)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(seed)

# ===== Q 网络  =====
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)  
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        self.relu = nn.ReLU()

    def forward(self, state):
        """前向传播：输入状态，输出 Q 值"""
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.fc3(x) 

# ===== 环境  =====
class KWalletEnv:
    def __init__(self, C=3000, k=3, F=5, max_transaction=1000, max_steps=1000, seed=123, enable_shaping=True):
        self.C = C  
        self.k = k  
        self.F = F  
        self.max_transaction = max_transaction
        self.max_steps = max_steps
        self.wallet_size = C / k  
        self.alpha_drop = 0.02    
        self.beta_flush = REFRESH_COST 
        self.enable_shaping = enable_shaping 

        
        if self.enable_shaping:
           self.IMBALANCE_PENALTY = IMBALANCE_PENALTY
           self.WASTEFUL_REFRESH_PENALTY = WASTEFUL_REFRESH_PENALTY
           self.WASTEFUL_REFRESH_THRESH = WASTEFUL_REFRESH_THRESH
           self.INVALID_ACTION_PENALTY = 0.05 # 对无效操作额外惩罚

        import numpy as _np
        self.rng = _np.random.default_rng(seed)
        self._external_tx_stream = None
        self.reset() 

    def reset(self):
        self.wallets = [self.wallet_size] * self.k 
        self.freeze_until = [-1] * self.k
        self.pending_refill = [False] * self.k         
        self.total_settled = 0.0
        self.total_accepted = 0.0
        self.num_flushes = 0
        self.drops = 0
        self.oversize_drops = 0
        self.time = 0
        
        if self._external_tx_stream is not None:
            self._tx_stream = list(self._external_tx_stream)
            if len(self._tx_stream) < self.max_steps:
                raise ValueError("tx_stream length < max_steps")
            self.current_tx = self._tx_stream[self.time]
        else:
            self._tx_stream = None
            self.current_tx = self._generate_transaction()
        return self._get_state()

    def load_tx_stream(self, seq):
        self._external_tx_stream = list(seq)

    def _generate_transaction(self):
        return int(self.rng.integers(1, self.max_transaction + 1))

    def _usable(self, i: int) -> bool:
        return self.time > self.freeze_until[i]

    def _get_state(self) -> np.ndarray:
        """
        构建状态向量 S。（所有特征进行归一化处理）
        """
        state = []
        # 1. 钱包余额 (归一化到 0-1)
        for w in self.wallets:
            state.append(w / self.wallet_size)
        # 2. 钱包可用状态 (0: 可用, 1: 冻结)
        for i in range(self.k):
            state.append(0.0 if self._usable(i) else 1.0)
        # 3. 剩余冻结时间 (归一化)
        for i in range(self.k):
            rem = max(0, self.freeze_until[i] - self.time)
            state.append((rem / self.F) if self.F > 0 else 0.0)
        # 4. 当前交易额 (归一化)
        state.append(self.current_tx / self.max_transaction)
        return np.array(state, dtype=np.float32)

    def _decode_action(self, action_int: int):
        """
        新解码方式：
        输入是一个整数 action_int
        低 k 位表示刷新掩码 (2^k 种可能)
        高位表示结算选择 (0到k-1为选钱包，k为不结算)
        """
        num_refresh_combos = 1 << self.k
        
        # 提取刷新位掩码
        refresh_idx = action_int % num_refresh_combos
        flush_mask = [(refresh_idx >> i) & 1 for i in range(self.k)]
        
        # 提取结算选择
        settle_choice = action_int // num_refresh_combos
        
        # 为了兼容你原有的代码结构，我们将 settle_choice 转换回原来的 settle_mask
        settle_mask = [0] * self.k
        if settle_choice < self.k:
            settle_mask[settle_choice] = 1
            
        return settle_mask, flush_mask

    def step(self, action: int):
        """
        执行步骤（逻辑基本保持不变，但输入 action 含义已变）
        """
        reward = 0.0
        placed = False
        flushes_this_step = 0
        did_refresh = False
        refresh_targets = []
        
        tx = self.current_tx
        # 1. 解码动作
        settle_mask, flush_mask = self._decode_action(action)
        pre_refresh_balances = {i: self.wallets[i] for i in range(self.k)}

        # 2. 优先处理刷新
        for i in range(self.k):
            if flush_mask[i] == 1:
                if self._usable(i):
                    self.pending_refill[i] = True
                    self.wallets[i] = 0.0     
                    self.freeze_until[i] = self.time + self.F - 1
                    self.num_flushes += 1
                    flushes_this_step += 1
                    did_refresh = True
                    refresh_targets.append(i)
                else:
                    if self.enable_shaping:
                        reward -= getattr(self, "INVALID_ACTION_PENALTY", 0.05)

        # 3. 确定结算钱包 (由新的 settle_mask 决定，此时里面最多只有一个1)
        fit_idx = None
        # 找到被选中的那个钱包索引
        chosen_idx = -1
        for i in range(self.k):
            if settle_mask[i] == 1:
                chosen_idx = i
                break

        # 4. 尝试结算
        if tx > self.wallet_size * self.k:
            self.oversize_drops += 1
            reward -= self.alpha_drop
        elif chosen_idx != -1: # 如果选择了某个钱包
            # 必须满足：可用 + 本步未被刷新 + 余额足够
            if self._usable(chosen_idx) and (chosen_idx not in refresh_targets) and self.wallets[chosen_idx] >= tx:
                self.wallets[chosen_idx] -= tx
                self.total_settled += tx
                self.total_accepted += tx
                reward += float(tx) / self.max_transaction
                fit_idx = chosen_idx
                placed = True
            else:
                # 选了钱包但无法承接（不可用、正在刷、或钱不够），视为 Drop
                self.drops += 1
                reward -= self.alpha_drop
        else:
            # 动作选择了“不结算” (settle_choice == k)
            # 在有交易的情况下不结算，通常也视为一次 Drop
            self.drops += 1
            reward -= self.alpha_drop

        # 5. 奖励塑形 (保持你原有的逻辑)
        reward -= self.beta_flush * flushes_this_step 
        if self.enable_shaping:
            usable_balances = [self.wallets[i] for i in range(self.k) if self._usable(i)]
            if len(usable_balances) >= 2:
                std_norm = float(np.std(np.array(usable_balances)) / self.wallet_size)
                reward -= IMBALANCE_PENALTY * std_norm
            for i in refresh_targets:
                if (pre_refresh_balances[i] / self.wallet_size) >= WASTEFUL_REFRESH_THRESH:
                    reward -= WASTEFUL_REFRESH_PENALTY

        # 6. 时间推进
        self.time += 1
        for i in range(self.k):
            if self.pending_refill[i] and self._usable(i):
                self.wallets[i] = self.wallet_size
                self.pending_refill[i] = False

        if self._tx_stream is not None and self.time < len(self._tx_stream):
            self.current_tx = self._tx_stream[self.time]
        else:
            self.current_tx = self._generate_transaction()
            
        done = (self.time >= self.max_steps)
        return self._get_state(), float(reward), bool(done), {"fit_idx": fit_idx}
    # === 请将以下代码粘贴回 KWalletEnv 类中 ===
    def active_count(self): 
        return sum(1 for i in range(self.k) if self._usable(i))

    def avg_balance(self): 
        return float(np.mean(self.wallets))

    def utilization_proxy(self): 
        # 防止分母为 0
        denominator = (self.max_transaction * self.max_steps)
        if denominator == 0: return 0.0
        return self.total_accepted / denominator

    def snapshot_str(self): 
        return " ".join([f"{i}:{int(self.wallets[i])}" for i in range(self.k)])

# ===== DQN Agent (智能体) =====
class DQNAgent:
    def __init__(self, state_size, action_size, device="cpu"):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device(device)

        # 经验回放缓冲区：存储 (s, a, r, s', done)
        self.memory = deque(maxlen=20000)
        # 超参数
        self.gamma = 0.98       
        self.epsilon = 0.8 
        self.epsilon_min = 0.05 
        self.epsilon_decay = 0.999

        # 双网络结构 (Double DQN 机制)
        # model (online): 用于决策和计算当前 Q 值
        self.model = DQN(state_size, action_size).to(self.device)
        # target_model: 用于计算目标 Q 值 (保持固定，定期更新)
        self.target_model = DQN(state_size, action_size).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3) 
        self.update_target_network()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def act(self, state: np.ndarray) -> int:
        """决策：Epsilon-Greedy 策略"""
        # 探索 (Exploration)：随机选择动作
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        # 利用 (Exploitation)：选择 Q 值最大的动作
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q = self.model(s)
            return int(torch.argmax(q, dim=1).item())

    def replay(self, batch_size=128):
        """
        经验回放：核心学习过程
        从记忆中随机抽取批次，更新 Q 网络以逼近贝尔曼方程
        """
        if len(self.memory) < batch_size:
            return None
        # 1. 随机采样
        batch = random.sample(self.memory, batch_size)
        s, a, r, s2, d = zip(*batch)
        s  = torch.tensor(np.array(s), dtype=torch.float32, device=self.device)
        a  = torch.tensor(a, dtype=torch.int64, device=self.device)
        r  = torch.tensor(r, dtype=torch.float32, device=self.device)
        s2 = torch.tensor(np.array(s2), dtype=torch.float32, device=self.device)
        d  = torch.tensor(d, dtype=torch.float32, device=self.device)

        # 2. 计算目标 Q 值 (Target Q)
        with torch.no_grad():
            # Double DQN: 
            # (a) 使用 Online Network 选择最佳动作
            next_online_q = self.model(s2)
            next_act = next_online_q.argmax(dim=1)
            # (b) 使用 Target Network 评估该动作的 Q 值
            next_target_q = self.target_model(s2)
            q_next = next_target_q.gather(1, next_act.unsqueeze(1)).squeeze(1)
            # 贝尔曼公式: y = R + gamma * Q_next
            y = r + self.gamma * (1.0 - d) * q_next

        # 3. 计算当前 Q 值 (Predicted Q)
        q = self.model(s).gather(1, a.unsqueeze(1)).squeeze(1)

        
        # 4. 计算损失 (Huber / SmoothL1) 并反向传播
        loss = nn.SmoothL1Loss()(q, y)

        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪：防止梯度爆炸
        nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()

        # ===== 额外统计：TD error 与数值尺度（用于解释 loss 走势）=====
        with torch.no_grad():
            td = q - y                       # TD error（带符号）
            td_abs_mean = td.abs().mean().item()
            td_abs_max  = td.abs().max().item()
            q_mean = q.mean().item()
            y_mean = y.mean().item()

        return {
            "loss": float(loss.item()),
            "td_abs_mean": float(td_abs_mean),
            "td_abs_max": float(td_abs_max),
            "q_mean": float(q_mean),
            "y_mean": float(y_mean),
        }


        return float(loss.item())

    def decay_epsilon(self):
        """按回合衰减探索率"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ===== 日志打印 =====
def log_episode_start(env, ep, max_steps):
    print(
        f"\n[EP={ep+1:04d}] 🚦 START (pre-step) | step=0..{max_steps-1} | env.time={env.time} "
        f"| tx0={env.current_tx:4d} | active={env.active_count()}/{env.k} | wallets={env.snapshot_str()}"
    )

def log_episode_end(env, t_last):
    print(
        f"[EP END] 🏁 last_step={t_last:03d} | env.time={env.time} | total_settled={int(env.total_settled)} "
        f"| flush_total={env.num_flushes} | drops={env.drops} | util_proxy={env.utilization_proxy():.3f} "
        f"| wallets={env.snapshot_str()}"
    )


# ===== 训练循环 (主流程) =====
def train_agent(episodes=5000, max_steps=1000, batch_size=256, target_update_every=50, device="cpu", k=3):
    env = KWalletEnv(C=4500, k=3, F=1, max_transaction=1000, max_steps=max_steps, seed=123, enable_shaping=True)
    state_size = len(env._get_state())
    action_size = (env.k + 1) * (1 << env.k)
    agent = DQNAgent(state_size, action_size, device=device)

    returns = []
    loss_history = []
    epsilons = []
    last_td_stats = None   # 记录该回合最后一次 replay 的统计


    # --- 外部循环: 回合 (Episode) ---
    for ep in range(episodes):
        state = env.reset()
        G = 0.0
        # --- 内部循环: 时间步 (Step) ---
        for t in range(max_steps):
            # 1. 决策
            action = agent.act(state)
            # 2. 环境交互
            next_state, reward, done, info = env.step(action)
            # 3. 记忆
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            G += reward

            # 4. 学习 (Replay)
            metrics = agent.replay(batch_size=batch_size)
            if metrics is not None:
                loss_history.append(metrics["loss"])
                last_td_stats = metrics


            # 日志

            if done:
               break

        agent.decay_epsilon()

        if (ep + 1) % target_update_every == 0:
            agent.update_target_network()

        returns.append(G)
        epsilons.append(agent.epsilon)
        avg = np.mean(returns[-100:]) if len(returns) >= 100 else np.mean(returns)
        
        settled = int(env.total_settled)
        accepted = int(env.total_accepted)
        drop_rate = env.drops / max_steps
        flushes = env.num_flushes

    if last_td_stats is None:
        print(
         f"EP={ep+1:04d} | G={G:7.2f} | avg100={avg:7.2f} | "
         f"settled={settled} | accepted={accepted} | drops={env.drops} | drop_rate={drop_rate:.3f} | "
         f"flush={flushes} | util={env.utilization_proxy():.3f} | "
         f"eps={agent.epsilon:5.3f} | buf={len(agent.memory)}"
    )
    else:
        print(
         f"EP={ep+1:04d} | G={G:7.2f} | avg100={avg:7.2f} | "
         f"settled={settled} | accepted={accepted} | drops={env.drops} | drop_rate={drop_rate:.3f} | "
         f"flush={flushes} | util={env.utilization_proxy():.3f} | "
         f"eps={agent.epsilon:5.3f} | buf={len(agent.memory)}"
         f" | loss={last_td_stats['loss']:.5f}"
         f" | |td|_mean={last_td_stats['td_abs_mean']:.5f}"
         f" | |td|_max={last_td_stats['td_abs_max']:.5f}"
         f" | q_mean={last_td_stats['q_mean']:.5f}"
         f" | y_mean={last_td_stats['y_mean']:.5f}"
    )


    torch.save(agent.model.state_dict(), "k_wallet_dqn_bitmask.pth")
    return agent, returns, loss_history, epsilons

# ===== 评估函数 =====
def evaluate_agent(agent: DQNAgent, num_episodes=10, max_steps=1000, k=3):
    env = KWalletEnv(C=4500, k=3, F=1, max_transaction=1000, max_steps=max_steps, seed=123, enable_shaping=True)

    old_eps = agent.epsilon
    agent.epsilon = 0.0

    totals_return = []
    totals_settled = []
    totals_drops = []
    totals_flush = []

    for ep in range(num_episodes):
        s = env.reset()
        G = 0.0
        for t in range(max_steps):
            a = agent.act(s)
            s, r, done, _ = env.step(a)
            G += r
            if done:
                break

        totals_return.append(G)
        totals_settled.append(env.total_settled)
        totals_drops.append(env.drops)
        totals_flush.append(env.num_flushes)

    agent.epsilon = old_eps

    print(f"\n--- 评估结果 ({num_episodes} 回合) ---")
    print(f"Return(G): {np.mean(totals_return):.3f} ± {np.std(totals_return):.3f}")
    print(f"Total Settled: {np.mean(totals_settled):.1f} ± {np.std(totals_settled):.1f}")
    print(f"Drops: {np.mean(totals_drops):.1f} ± {np.std(totals_drops):.1f}")
    print(f"Flushes: {np.mean(totals_flush):.1f} ± {np.std(totals_flush):.1f}")

    return totals_return, totals_settled


# ===== 绘图 =====
def plot_convergence(returns, loss_hist, epsilons, window=100):
    def moving_avg(x, w):
        x = np.asarray(x, dtype=float)
        if len(x) < w: return np.array([])
        c = np.cumsum(np.insert(x, 0, 0.0))
        return (c[w:] - c[:-w]) / w

    episodes = np.arange(1, len(returns) + 1)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # 1. 回报曲线
    axes[0].plot(episodes, returns, alpha=0.3, label="Return per episode")
    ma = moving_avg(returns, w=window)
    if ma.size > 1:
        axes[0].plot(episodes[window-1:], ma, label=f"Moving Avg ({window} episodes)", color='r')
    axes[0].set_title("1. Mean Return Convergence")
    axes[0].set_ylabel("Total Reward (G)")
    axes[0].legend(); axes[0].grid(True, alpha=0.5)

    # 2. 损失曲线
    if loss_hist:
        loss_window = max(1, len(loss_hist) // max(1, len(returns)) * 5)
        smoothed_loss = moving_avg(loss_hist, w=loss_window)
        if len(smoothed_loss) > 1:
            loss_episodes = np.linspace(1, len(returns), len(smoothed_loss))
            axes[1].plot(loss_episodes, smoothed_loss, color='g', label="Smoothed Loss")
    axes[1].set_title("2. DQN Training Loss")
    axes[1].set_ylabel("MSE Loss")
    axes[1].grid(True, alpha=0.5)

    # 3. 探索率曲线
    axes[2].plot(episodes, epsilons, color='b', label="Epsilon decay")
    axes[2].set_title("3. Epsilon Decay Schedule")
    axes[2].set_ylabel("Epsilon")
    axes[2].set_xlabel("Episode")
    axes[2].grid(True, alpha=0.5)

    plt.tight_layout()
    plt.savefig("full_convergence_analysis_bitmask.png", dpi=150)
    print("\nSaved full convergence analysis -> full_convergence_analysis_bitmask.png")

if __name__ == "__main__":
    set_seed(123)

    K = 3
    NUM_EPISODES = 3000
    MAX_STEPS = 1000
    print(f"--- 开始 bitmask DQN 训练: k={K}, episodes={NUM_EPISODES} ---")
    
    agent, hist, loss_hist, eps_hist = train_agent(
        episodes=NUM_EPISODES, 
        max_steps=MAX_STEPS, 
        target_update_every=20, 
        device="cpu", 
        k=K
    )

    evaluate_agent(agent, num_episodes=100, max_steps=MAX_STEPS, k=K)
    plot_convergence(hist, loss_hist, eps_hist, window=100)