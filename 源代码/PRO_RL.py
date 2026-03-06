# =========================================================
# Research-grade PPO for K-Wallet (含日志 / 评估 / 收敛图)
# =========================================================
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Bernoulli
import random
import matplotlib.pyplot as plt
from dataclasses import dataclass

# =========================
# 全局配置
# =========================
@dataclass
class PPOConfig:
    seed: int = 123
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 环境参数
    C: int = 10000
    k: int = 10
    F: int = 5
    max_transaction: int = 1200
    max_steps: int = 1000
    enable_shaping: bool = True

    # PPO 参数
    episodes: int = 5000
    rollout_steps: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 1e-4
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_start: float = 0.05
    entropy_end: float = 0.005
    update_epochs: int = 10
    minibatch_size: int = 256
    max_grad_norm: float = 1.0

    log_every: int = 100
    eval_episodes: int = 100

cfg = PPOConfig()
REFRESH_COST = 0.01

# =========================
# 随机种子（可复现）
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(cfg.seed)

# =========================
# 环境定义
# =========================
class KWalletEnv:
    def __init__(self, cfg: PPOConfig):
        self.C, self.k, self.F = cfg.C, cfg.k, cfg.F
        self.max_transaction = cfg.max_transaction
        self.max_steps = cfg.max_steps
        self.wallet_size = self.C / self.k
        self.enable_shaping = cfg.enable_shaping
        self.rng = np.random.default_rng(cfg.seed)
        self.reset()

    def reset(self):
        self.wallets = [self.wallet_size] * self.k
        self.freeze_until = [-1] * self.k
        self.pending_refill = [False] * self.k
        self.time = 0
        self.total_settled = 0
        self.drops = 0
        self.flushes = 0
        self.current_tx = self._gen_tx()
        return self._get_state()

    def _gen_tx(self):
        return int(self.rng.integers(1, self.max_transaction + 1))

    def _usable(self, i):
        return self.time > self.freeze_until[i]

    def _get_state(self):
        # 归一化状态：余额 / 冻结 / 冻结剩余时间 / 当前交易
        s = [w / self.wallet_size for w in self.wallets]
        s += [0.0 if self._usable(i) else 1.0 for i in range(self.k)]
        s += [max(0, self.freeze_until[i] - self.time) / max(1, self.F) for i in range(self.k)]
        s.append(self.current_tx / self.max_transaction)
        return np.array(s, dtype=np.float32)

    def step(self, settle_idx, flush_mask):
        reward = 0.0
        tx = self.current_tx
        refresh_targets = []

        # Flush 逻辑
        for i in range(self.k):
            if flush_mask[i] and self._usable(i):
                self.wallets[i] = 0.0
                self.freeze_until[i] = self.time + self.F - 1
                self.pending_refill[i] = True
                self.flushes += 1
                reward -= REFRESH_COST
                refresh_targets.append(i)

        # 支付逻辑
        if (
            settle_idx < self.k
            and self._usable(settle_idx)
            and settle_idx not in refresh_targets
            and self.wallets[settle_idx] >= tx
        ):
            self.wallets[settle_idx] -= tx
            self.total_settled += tx
            reward += tx / self.max_transaction
        else:
            self.drops += 1
            reward -= 0.02

        # Reward shaping（轻量）
        if self.enable_shaping:
            usable_bal = [self.wallets[i] for i in range(self.k) if self._usable(i)]
            if len(usable_bal) >= 2:
                reward -= 0.02 * (np.std(usable_bal) / self.wallet_size)

        # 时间推进 & refill
        self.time += 1
        for i in range(self.k):
            if self.pending_refill[i] and self._usable(i):
                self.wallets[i] = self.wallet_size
                self.pending_refill[i] = False

        self.current_tx = self._gen_tx()
        done = self.time >= self.max_steps
        return self._get_state(), reward, done, {}

    def utilization(self):
        return self.total_settled / (self.max_transaction * self.max_steps)

# =========================
# Actor-Critic 网络
# =========================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, k):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.settle_head = nn.Linear(128, k + 1)  # 选择哪个 wallet 或 drop
        self.flush_head = nn.Linear(128, k)       # k 个 Bernoulli
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.backbone(x)
        return (
            self.settle_head(x),
            self.flush_head(x),
            self.value_head(x).squeeze(-1),
        )

# =========================
# GAE 计算
# =========================
def compute_gae(rewards, values, dones, last_v):
    adv, gae = [], 0.0
    values = values + [last_v]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + cfg.gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + cfg.gamma * cfg.gae_lambda * (1 - dones[t]) * gae
        adv.insert(0, gae)
    returns = [a + v for a, v in zip(adv, values[:-1])]
    return torch.tensor(adv), torch.tensor(returns)

# =========================
# 训练主函数
# =========================
def train():
    env = KWalletEnv(cfg)
    device = torch.device(cfg.device)

    net = ActorCritic(len(env._get_state()), cfg.k).to(device)
    opt = optim.Adam(net.parameters(), lr=cfg.lr)

    returns_hist, loss_hist, ent_hist = [], [], []
    obs = env.reset()
    buffer = []

    for ep in range(cfg.episodes):
        frac = ep / cfg.episodes
        ent_coef = cfg.entropy_start + frac * (cfg.entropy_end - cfg.entropy_start)

        # Rollout
        for _ in range(cfg.rollout_steps):
            s = torch.tensor(obs).unsqueeze(0).to(device)
            sl, fl, v = net(s)

            sd = Categorical(logits=sl)
            fd = Bernoulli(logits=fl)

            settle = sd.sample()
            flush = fd.sample()

            logp = sd.log_prob(settle) + fd.log_prob(flush).sum(-1)
            entropy = sd.entropy() + fd.entropy().sum(-1)

            obs2, r, done, _ = env.step(settle.item(), flush.squeeze(0).cpu().numpy())
            buffer.append((obs, settle, flush, r, done, logp, v, entropy))
            obs = obs2

            if done:
                returns_hist.append(env.total_settled / 1000.0)
                obs = env.reset()

        # PPO Update
        obs_b, settle_b, flush_b, r_b, d_b, logp_b, v_b, ent_b = zip(*buffer)
        obs_b = torch.tensor(obs_b).to(device)
        settle_b = torch.cat(settle_b).to(device)
        flush_b = torch.cat(flush_b).to(device)
        logp_b = torch.cat(logp_b).detach().to(device)
        v_b = torch.cat(v_b).detach().to(device)

        with torch.no_grad():
            last_v = net(torch.tensor(obs).unsqueeze(0).to(device))[2].item()

        adv, ret = compute_gae(list(r_b), v_b.tolist(), list(d_b), last_v)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(cfg.update_epochs):
            idx = np.random.permutation(len(obs_b))
            for start in range(0, len(obs_b), cfg.minibatch_size):
                mb = idx[start:start + cfg.minibatch_size]

                sl, fl, val = net(obs_b[mb])
                sd = Categorical(logits=sl)
                fd = Bernoulli(logits=fl)

                logp = sd.log_prob(settle_b[mb]) + fd.log_prob(flush_b[mb]).sum(-1)
                ratio = torch.exp(logp - logp_b[mb])

                surr1 = ratio * adv[mb]
                surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv[mb]
                pi_loss = -torch.min(surr1, surr2).mean()

                v_loss = (val - ret[mb]).pow(2).mean()
                entropy = sd.entropy().mean() + fd.entropy().sum(-1).mean()

                loss = pi_loss + cfg.value_coef * v_loss - ent_coef * entropy

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                opt.step()

        loss_hist.append(loss.item())
        ent_hist.append(entropy.item())
        buffer.clear()

        # 日志输出（对齐你的示例风格）
        if (ep + 1) % cfg.log_every == 0:
            avg100 = np.mean(returns_hist[-100:])
            print(
                f"EP={ep+1:04d}/{cfg.episodes} | "
                f"G={returns_hist[-1]:7.2f} | avg100={avg100:7.2f} | "
                f"settled={env.total_settled} | drops={env.drops} | "
                f"flush={env.flushes} | util={env.utilization():.3f} | "
                f"loss={loss_hist[-1]:.5f} | ent={ent_hist[-1]:.4f}"
            )

    torch.save(net.state_dict(), "ppo_kwallet.pth")
    print("\nSaved model -> ppo_kwallet.pth")
    return net, returns_hist, loss_hist, ent_hist

# =========================
# 评估
# =========================
@torch.no_grad()
def evaluate(net):
    env = KWalletEnv(cfg)
    device = next(net.parameters()).device
    G, S, D, F = [], [], [], []

    for _ in range(cfg.eval_episodes):
        s = env.reset()
        done = False
        while not done:
            st = torch.tensor(s).unsqueeze(0).to(device)
            sl, fl, _ = net(st)
            settle = torch.argmax(sl, dim=1).item()
            flush = (torch.sigmoid(fl) > 0.5).int().squeeze(0).cpu().numpy()
            s, _, done, _ = env.step(settle, flush)
        G.append(env.total_settled / 1000.0)
        S.append(env.total_settled)
        D.append(env.drops)
        F.append(env.flushes)

    print(f"\n--- 评估结果 ({cfg.eval_episodes} 回合) ---")
    print(f"Return(G): {np.mean(G):.3f} ± {np.std(G):.3f}")
    print(f"Total Settled: {np.mean(S):.1f} ± {np.std(S):.1f}")
    print(f"Drops: {np.mean(D):.1f} ± {np.std(D):.1f}")
    print(f"Flushes: {np.mean(F):.1f} ± {np.std(F):.1f}")

# =========================
# 收敛图
# =========================
def plot_curves(returns, losses, ents):
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    ax[0].plot(returns, alpha=0.3)
    ax[0].plot(np.convolve(returns, np.ones(100)/100, mode="valid"), color="red")
    ax[0].set_title("Return Convergence")

    ax[1].plot(losses)
    ax[1].set_title("PPO Loss")

    ax[2].plot(ents)
    ax[2].set_title("Policy Entropy")

    plt.xlabel("Episodes")
    plt.tight_layout()
    plt.savefig("ppo_convergence.png")
    print("\nSaved plot -> ppo_convergence.png")

# =========================
# Main
# =========================
if __name__ == "__main__":
    net, rets, losses, ents = train()
    evaluate(net)
    plot_curves(rets, losses, ents)
