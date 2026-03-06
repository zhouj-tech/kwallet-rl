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
REFRESH_COST = 0.01        # 刷新成本 
IMBALANCE_PENALTY = 0.02   # 余额不均衡惩罚（按余额std/容量）
WASTEFUL_REFRESH_PENALTY = 0.02  # 高余额仍刷新扣分
WASTEFUL_REFRESH_THRESH = 0.6    # “高余额”的阈值

def set_seed(seed: int = 123):
    import os, random, numpy as _np, torch as _torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    _np.random.seed(seed)
    _torch.manual_seed(seed)
    if _torch.cuda.is_available():
        _torch.cuda.manual_seed_all(seed)

# ===== Q 网络 (保持不变) =====
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)  
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        self.relu = nn.ReLU()
    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# ===== 环境 KWalletEnv (保持不变) =====
class KWalletEnv:
    def __init__(self, C=3000, k=3, F=5, max_transaction=100, max_steps=1000, seed=123,enable_shaping=True):
        self.C = C
        self.k = k
        self.F = F
        self.max_transaction = max_transaction
        self.max_steps = max_steps
        self.wallet_size = C / k
        self.alpha_drop = 0.02   # 丢单小惩罚
        self.beta_flush = REFRESH_COST  # 刷新微成本

        # --- 启用奖励塑形  ---
        self.enable_shaping = enable_shaping 
        if self.enable_shaping:
           self.IMBALANCE_PENALTY = IMBALANCE_PENALTY
           self.WASTEFUL_REFRESH_PENALTY = WASTEFUL_REFRESH_PENALTY
           self.WASTEFUL_REFRESH_THRESH = WASTEFUL_REFRESH_THRESH
           self.INVALID_ACTION_PENALTY = 0.05 # 确保 Invalid Penalty 可用

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
        # 交易流
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
        return int(self.rng.integers(1, self.max_transaction + 1))  # 均匀分布

    def _usable(self, i: int) -> bool:
        return self.time > self.freeze_until[i]

    def _get_state(self) -> np.ndarray:
        state = []
        # 余额比例
        for w in self.wallets:
            state.append(w / self.wallet_size)
        # 是否在刷新（冻结中）
        for i in range(self.k):
            state.append(0.0 if self._usable(i) else 1.0)
        # 刷新剩余时间比例
        for i in range(self.k):
            rem = max(0, self.freeze_until[i] - self.time)
            state.append((rem / self.F) if self.F > 0 else 0.0)
        # 当前交易 + 时间特征
        state.append(self.current_tx / self.max_transaction)
        state.append(np.sin(2 * np.pi * self.time / 24))
        state.append(np.cos(2 * np.pi * self.time / 24))
        return np.array(state, dtype=np.float32)
    
    
    def step(self, action: int):
     
        reward = 0.0
        placed = False
        flushes_this_step = 0
        did_refresh = False
        refresh_target = None
        pre_refresh_balance = None

        # --- decode 动作 ---
        if action < self.k:
            settle_idx = action
            refresh_self = False
        else:
            settle_idx = action - self.k
            refresh_self = True

        tx = self.current_tx

        # --- 动作有效性检查（统一判定） ---
        valid = True
        if refresh_self:
            if not self._usable(settle_idx): valid = False
        else:
            if tx > self.wallet_size or (not self._usable(settle_idx)) or (self.wallets[settle_idx] < tx): valid = False

        if not valid:
            self.drops += 1
            invalid_pen = getattr(self, "INVALID_ACTION_PENALTY", 0.05) if getattr(self, "enable_shaping", False) else -0.02
            if invalid_pen > 0:
                reward = -float(invalid_pen)
            else:
                reward = float(invalid_pen)

            self.time += 1
            if self._tx_stream is not None and self.time < len(self._tx_stream):
                self.current_tx = self._tx_stream[self.time]
            else:
                self.current_tx = self._generate_transaction()
            done = (self.time >= self.max_steps)

            info = {
            "transaction": tx, "placed": 0, "total_settled": self.total_settled, "drops": self.drops,
            "oversize_drops": self.oversize_drops, "num_flushes": self.num_flushes, "did_refresh": 0,
            "refresh_target": -1, "settle_target": -1, "flushes_this_step": 0
            }
            next_state = self._get_state()
            return next_state, float(reward), bool(done), info

        # --- 如果动作有效：先处理刷新 ---
        if refresh_self and self._usable(settle_idx):
            pre_refresh_balance = self.wallets[settle_idx]
            self.wallets[settle_idx] = self.wallet_size
            self.freeze_until[settle_idx] = self.time + self.F
            self.num_flushes += 1
            flushes_this_step += 1
            did_refresh = True
            refresh_target = settle_idx
            settle_target_for_this_step = None
        else:
            settle_target_for_this_step = settle_idx if not refresh_self else None

        # --- 结算当前交易 ---
        if tx > self.wallet_size:
            self.oversize_drops += 1
            reward -= self.alpha_drop
            placed = False
        else:
            if (settle_target_for_this_step is not None) and self._usable(settle_target_for_this_step) and self.wallets[settle_target_for_this_step] >= tx:
                self.wallets[settle_target_for_this_step] -= tx
                self.total_settled += tx
                self.total_accepted += tx
                if getattr(self, "reward_mode", "norm") == "norm":
                    reward += tx / self.max_transaction
                elif getattr(self, "reward_mode") == "tx":
                    reward += float(tx)
                else:
                    reward += float(tx) / float(self.C)
                placed = True
            else:
                if settle_target_for_this_step is not None:
                    self.drops += 1
                    placed = False

        # --- 奖励塑形 ---
        reward -= self.beta_flush * flushes_this_step

        if getattr(self, "enable_shaping", False):
            std_norm = float(np.std(np.array(self.wallets)) / self.wallet_size) if self.wallet_size > 0 else 0.0
            reward -= getattr(self, "IMBALANCE_PENALTY", 0.0) * std_norm
            if did_refresh and (pre_refresh_balance is not None):
                thresh = getattr(self, "WASTEFUL_REFRESH_THRESH", 0.6)
                if (pre_refresh_balance / self.wallet_size) >= thresh:
                    reward -= getattr(self, "WASTEFUL_REFRESH_PENALTY", 0.0)

        # --- 时间推进与交易更新 ---
        self.time += 1
        if self._tx_stream is not None and self.time < len(self._tx_stream):
            self.current_tx = self._tx_stream[self.time]
        else:
            self.current_tx = self._generate_transaction()
        done = (self.time >= self.max_steps)

        info = {
        "transaction": tx, "placed": int(placed), "total_settled": self.total_settled, "drops": self.drops,
        "oversize_drops": self.oversize_drops, "num_flushes": self.num_flushes, "did_refresh": int(did_refresh),
        "refresh_target": (-1 if refresh_target is None else refresh_target),
        "settle_target": (-1 if settle_target_for_this_step is None else settle_target_for_this_step),
        "flushes_this_step": flushes_this_step
        }

        next_state = self._get_state()
        return next_state, float(reward), bool(done), info

    def active_count(self):
        return sum(1 for i in range(self.k) if self._usable(i))

    def avg_balance(self):
        return float(np.mean(self.wallets))

    def utilization_proxy(self):
        return self.total_accepted / (self.max_transaction * self.max_steps)

    def snapshot_str(self):
        parts = []
        for i in range(self.k):
            status = "on" if self._usable(i) else "fr"
            rem = max(0, self.freeze_until[i] - self.time)
            parts.append(f"{i}:{int(self.wallets[i])},{status},{rem}")
        return " ".join(parts)

# ===== DQN Agent (保持不变) =====
class DQNAgent:
    def __init__(self, state_size, action_size, device="cpu"):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device(device)

        self.memory = deque(maxlen=20000)
        self.gamma = 0.98
        self.epsilon = 0.6
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.997

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

    def replay(self, batch_size=128):
        if len(self.memory) < batch_size:
            return None
        batch = random.sample(self.memory, batch_size)
        s, a, r, s2, d = zip(*batch)
        s  = torch.tensor(np.array(s), dtype=torch.float32, device=self.device)
        a  = torch.tensor(a, dtype=torch.int64, device=self.device)
        r  = torch.tensor(r, dtype=torch.float32, device=self.device)
        s2 = torch.tensor(np.array(s2), dtype=torch.float32, device=self.device)
        d  = torch.tensor(d, dtype=torch.float32, device=self.device)  

        # ===== Double-DQN 目标 =====
        with torch.no_grad():
            next_online_q = self.model(s2)                        
            next_act = next_online_q.argmax(dim=1)               
            next_target_q = self.target_model(s2)                
            q_next = next_target_q.gather(1, next_act.unsqueeze(1)).squeeze(1)
            y = r + self.gamma * (1.0 - d) * q_next

        # 当前Q(s,a)
        q = self.model(s).gather(1, a.unsqueeze(1)).squeeze(1)

        loss = nn.MSELoss()(q, y)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()

        # ε 衰减
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return float(loss.item())

# ===== 日志打印辅助 (确保 SIDENOTE_INTERVAL 足够大或为 0 来减少输出) =====
def log_step(env: KWalletEnv, t: int, info: dict, action: int):
    tx = info["transaction"]
    placed = info["placed"] == 1
    did_refresh = info.get("did_refresh", 0) == 1
    settle_tgt = info.get("settle_target", -1)
    refresh_tgt = info.get("refresh_target", -1)

    if did_refresh and refresh_tgt >= 0 and settle_tgt == -1:
        act_str = f"仅刷新#{refresh_tgt}"
        extra = f" | 冻结至 t={env.freeze_until[refresh_tgt]}"
    elif settle_tgt >= 0:
        act_str = f"结算#{settle_tgt}"
        extra = ""
    else:
        idx = action - env.k if action >= env.k else -1
        act_str = f"仅刷新#{idx}" if idx >= 0 else "空操作"
        extra = f" | 冻结至 t={env.freeze_until[idx]}" if idx >= 0 else ""

    result = "✅成交" if placed else "❌未成"
    
    # 极简日志：只在回合开始和结束打印简洁信息
    if t == 0:
        print(f"[t={t:03d}] 📥 回合开始 | 🧾 tx={tx:3d} | 活跃:{env.active_count()}/{env.k} | 平均余额:{env.avg_balance():.1f}")
    elif (t == env.max_steps - 1) or info.get("done", False):
        print(f"[t={t:03d}] 🏁 回合结束 | 累计成交={int(env.total_settled)} | Flush累计:{env.num_flushes} | 利用率代理:{env.utilization_proxy():.3f}")
    
    # 如果你想看到刷新操作，可以取消注释下面两行
    # elif did_refresh and refresh_tgt >= 0:
    #     print(f"[t={t:03d}] ♻️ 动作={act_str}{extra} | 累计成交={int(env.total_settled)}")


# ===== 训练与评估 (已修复) =====
def train_agent(episodes=500, max_steps=100, batch_size=128, target_update_every=20, device="cpu"):
    env = KWalletEnv(k=10,max_steps=max_steps, enable_shaping=True)
    state_size = len(env._get_state())
    action_size = 2 * env.k

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

            # 📌 FIXED: 确保只调用一次 replay() 并正确记录 loss
            loss = agent.replay(batch_size=batch_size)
            if loss is not None:
                loss_history.append(loss)

            # 📌 极简日志输出：只在回合开始和结束时打印
            is_start = (t == 0)
            is_end = (t == max_steps - 1) or done

            if is_start or is_end:
                log_step(env, t, info, action)

            if done:
                break

        # 回合末做 target 同步与记录
        if (ep + 1) % target_update_every == 0:
            agent.update_target_network()

        returns.append(G)
        epsilons.append(agent.epsilon)
        # 📌 打印每回合的收敛信息
        avg = np.mean(returns[-100:]) if len(returns) >= 100 else np.mean(returns) # 使用 100 回合平均
        print(f"Episode {ep+1:4d} | return={G:7.2f} | avg(last100)={avg:7.2f} | eps={agent.epsilon:5.3f} | buf={len(agent.memory)}")

    torch.save(agent.model.state_dict(), "k_wallet_dqn.pth")
    return agent, returns, loss_history, epsilons


def evaluate_agent(agent: DQNAgent, num_episodes=10, max_steps=1000):
    env = KWalletEnv(k=10, max_steps=max_steps)
    old_eps = agent.epsilon
    agent.epsilon = 0.0 # 评估时关闭探索
    totals = []
    for ep in range(num_episodes):
        s = env.reset()
        G = 0.0
        for t in range(max_steps):
            a = agent.act(s)
            s, r, done, _ = env.step(a)
            G += r
            if done:
                break
        totals.append(G)
    agent.epsilon = old_eps
    print(f"\n--- 评估结果 ({num_episodes} 回合) ---")
    print(f"Eval: avg return over {num_episodes} eps = {np.mean(totals):.2f} ± {np.std(totals):.2f}")
    # 重新运行最后一次评估，以打印详细指标
    env.reset()
    for t in range(max_steps):
        s = env._get_state()
        a = agent.act(s)
        s, r, done, _ = env.step(a)
        if done: break
        
    print(f"total_settled = {env.total_settled:.2f}")
    print(f"drops = {env.drops}")
    print(f"oversize_drops = {env.oversize_drops}")
    print(f"num_flushes = {env.num_flushes}")
    utilization = env.total_accepted / (env.max_transaction * env.max_steps)
    print(f"utilization_proxy = {utilization:.4f}")
    print("---------------------------------")
    return totals

# ===== 核心绘图函数 (已修复，确保绘图不报错) =====
def plot_convergence(returns, loss_hist, epsilons, window=100):
    def moving_avg(x, w):
        x = np.asarray(x, dtype=float)
        if len(x) < w: return np.array([])
        c = np.cumsum(np.insert(x, 0, 0.0))
        return (c[w:] - c[:-w]) / w
    
    episodes = np.arange(1, len(returns) + 1)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # 1. 回报曲线 (最重要的收敛指标)
    axes[0].plot(episodes, returns, alpha=0.3, label="Return per episode")
    ma = moving_avg(returns, w=window)
    if ma.size > 1:
        # X 轴从第一个有效的平滑点开始
        axes[0].plot(episodes[window-1:], ma, label=f"Moving Avg ({window} episodes)", color='r')
    axes[0].set_title("1. Mean Return Convergence")
    axes[0].set_ylabel("Total Reward (G)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.5)

    # 2. DQN 损失 (平滑)
    if loss_hist:
        loss_window = max(1, len(loss_hist) // len(returns) * 5)
        
        # 📌 FIXED: 检查平滑后的数据是否有效，防止 Matplotlib 报错
        smoothed_loss = moving_avg(loss_hist, w=loss_window)
        if len(smoothed_loss) > 1:
            # X 轴也需要截断以匹配平滑后的长度
            loss_episodes = np.linspace(1, len(returns), len(smoothed_loss)) 
            axes[1].plot(loss_episodes, smoothed_loss, color='g', label="Smoothed Loss")
        else:
            print("Warning: Loss history too short or flat to plot moving average.")
    
    axes[1].set_title("2. DQN Training Loss")
    axes[1].set_ylabel("MSE Loss (Smoothed)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.5)
    
    # 3. 探索率 Epsilon
    axes[2].plot(episodes, epsilons, color='b', label="Epsilon decay")
    axes[2].set_title("3. Epsilon Decay Schedule")
    axes[2].set_ylabel("Epsilon ($\epsilon$)")
    axes[2].set_xlabel("Episode")
    axes[2].legend()
    axes[2].grid(True, alpha=0.5)

    plt.tight_layout()
    plt.savefig("full_convergence_analysis.png", dpi=150)
    print("\nSaved full convergence analysis -> full_convergence_analysis.png")


if __name__ == "__main__":
    set_seed(123)
    
    # 📌 增加训练回合数 (5000 回合)，以观察清晰的收敛趋势
    NUM_EPISODES = 500
    MAX_STEPS = 100 
    TARGET_UPDATE_EVERY = 20
    
    print(f"--- 开始训练: {NUM_EPISODES} 回合 x {MAX_STEPS} 步 ---")
    # 运行 train_agent，并接收所有四个返回值
    agent, hist, loss_hist, eps_hist = train_agent(
        episodes=NUM_EPISODES, 
        max_steps=MAX_STEPS, 
        target_update_every=TARGET_UPDATE_EVERY,
        device="cpu"
    )
    
    # 评估最终模型
    evaluate_agent(agent, num_episodes=10, max_steps=MAX_STEPS)

    # 核心绘图调用
    plot_convergence(hist, loss_hist, eps_hist, window=100)