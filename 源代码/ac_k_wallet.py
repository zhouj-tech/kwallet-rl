

import os
import json
import random
from datetime import datetime
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


CONFIG = {
    "seed": 123,

    "env": {
        "C": 3000.0,
        "k": 6,
        "T": 100,
        "F": 1,
        "enable_shaping": True,
    },

    "train": {
        "episodes": 3000,
        "max_steps": 1000,
        "gamma": 0.98,
        "lr": 3e-4,
        "rollout_steps": 128,
        "value_coef": 0.5,
        "entropy_coef_start": 0.2,
        "entropy_coef_end": 0.00,
        "max_grad_norm": 1.0,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },

    "eval": {
        "num_episodes": 100,
        "test_start_idx": 4000,
        "tx_pool_path": "shared_tx_pool_t100.npy",
    },

    "output": {
        "model_path": "ac_aligned_t100.pth",
        "results_path": "ac_results_t100.json",
    }
}

LOG_EVERY_EP = 100   

# =========================================================
# Reward Params 
# =========================================================
REFRESH_COST = 0.01
IMBALANCE_PENALTY = 0.02
WASTEFUL_REFRESH_PENALTY = 0.02
WASTEFUL_REFRESH_THRESH = 0.6
ALPHA_DROP = 0.02
INVALID_ACTION_PENALTY = 0.05

# =========================================================
# Seed
# =========================================================
def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =========================================================
# Environment 
# =========================================================
class KWalletEnv:
    def __init__(self, C, k, F, max_steps, enable_shaping=False, **kwargs):
        self.C = float(C)
        self.k = int(k)
        self.F = int(F)
        self.max_steps = int(max_steps)
        self.T_val = CONFIG["env"].get("T", 1000)
        self.wallet_size = self.C / self.k
        self.enable_shaping = enable_shaping
        self.reset([0] * self.max_steps)

    def reset(self, tx_stream: List[int]):
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
        self.tx_stream = list(tx_stream)[:self.max_steps] # 确保不越界
        self.tx_norm = float(CONFIG["env"]["T"])
        self.current_tx = self.tx_stream[self.time]
        return self._get_state()

    def _usable(self, i):
        return self.time > self.freeze_until[i]

    def _get_state(self):
        s = []
        s += [w / self.wallet_size for w in self.wallets]
        s += [0.0 if self._usable(i) else 1.0 for i in range(self.k)]
        for i in range(self.k):
            rem = max(0, self.freeze_until[i] - self.time)
            s.append(rem / self.F if self.F > 0 else 0.0)
        s.append(self.current_tx / self.tx_norm)
        return np.array(s, dtype=np.float32)

    def _decode_action(self, a):
        num_refresh = 1 << self.k
        refresh_idx = a % num_refresh
        settle_idx = a // num_refresh
        flush_mask = [(refresh_idx >> i) & 1 for i in range(self.k)]
        settle_mask = [1 if i == settle_idx else 0 for i in range(self.k)]
        return settle_mask, flush_mask

    def step(self, action):
        reward = 0.0
        tx = self.current_tx

        settle_mask, flush_mask = self._decode_action(action)
        refresh_targets = []
        pre_bal = self.wallets.copy()

        for i in range(self.k):
            if flush_mask[i]:
                if self._usable(i):
                    self.wallets[i] = 0.0
                    self.pending_refill[i] = True
                    self.freeze_until[i] = self.time + self.F - 1
                    self.num_flushes += 1
                    refresh_targets.append(i)
                else:
                    reward -= INVALID_ACTION_PENALTY

        chosen = settle_mask.index(1) if 1 in settle_mask else -1

        if tx > self.wallet_size:
            self.drops += 1
            self.oversize_drops += 1
            reward -= ALPHA_DROP
        elif chosen != -1 and self._usable(chosen) and chosen not in refresh_targets and self.wallets[chosen] >= tx:
            self.wallets[chosen] -= tx
            self.total_settled += tx
            self.total_accepted += tx
            reward += float(tx) / self.tx_norm
        else:
            self.drops += 1
            self.insufficient_drops += 1
            reward -= ALPHA_DROP

        reward -= (len(refresh_targets) * self.F) / self.max_steps

        if self.enable_shaping:
            usable = [self.wallets[i] for i in range(self.k) if self._usable(i)]
            if len(usable) >= 2:
                reward -= IMBALANCE_PENALTY * (np.std(usable) / self.wallet_size)
            for i in refresh_targets:
                if pre_bal[i] / self.wallet_size >= WASTEFUL_REFRESH_THRESH:
                    reward -= WASTEFUL_REFRESH_PENALTY

        self.time += 1

        for i in range(self.k):
            if self.pending_refill[i] and self._usable(i):
                self.wallets[i] = self.wallet_size
                self.pending_refill[i] = False

        done = self.time >= self.max_steps
        if not done:
            self.current_tx = self.tx_stream[self.time]

        return self._get_state(), float(reward), done, {}

    def get_metrics(self):
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
# Actor-Critic Net
# =========================================================
class ActorCritic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(s_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        self.pi = nn.Linear(128, a_dim)
        self.v = nn.Linear(128, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.pi(h), self.v(h).squeeze(-1)

# =========================================================
# Training
# =========================================================
def train():
    set_seed(CONFIG["seed"])
    device = torch.device(CONFIG["train"]["device"])

    tx_pool = np.load(CONFIG["eval"]["tx_pool_path"])
    env = KWalletEnv(**CONFIG["env"], max_steps=CONFIG["train"]["max_steps"])

    s_dim = len(env._get_state())
    a_dim = (env.k + 1) * (1 << env.k)

    net = ActorCritic(s_dim, a_dim).to(device)
    opt = optim.Adam(net.parameters(), lr=CONFIG["train"]["lr"])
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=0.5)

    ep_returns = []

    for ep in range(CONFIG["train"]["episodes"]):
        s = env.reset(tx_pool[ep])
        done = False
        G = 0.0

        buf = {"logp": [], "v": [], "r": [], "d": [], "ent": []}

        ent_coef = CONFIG["train"]["entropy_coef_start"] + \
            (CONFIG["train"]["entropy_coef_end"] - CONFIG["train"]["entropy_coef_start"]) * \
            min(ep / CONFIG["train"]["episodes"], 1.0)

        for t in range(CONFIG["train"]["max_steps"]):
            st = torch.tensor(s, device=device).unsqueeze(0)
            logits, v = net(st)
            dist = Categorical(logits=logits)

            a = torch.argmax(logits, dim=1)   
            a_int = a.item()                 

            s2, r, done, _ = env.step(a_int) 

            G += r

            buf["logp"].append(dist.log_prob(a))
            buf["v"].append(v)
            buf["r"].append(r)
            buf["d"].append(done)
            buf["ent"].append(dist.entropy())

            s = s2
            
            # 执行 Actor-Critic 更新 
            if (t + 1) % CONFIG["train"]["rollout_steps"] == 0 or done:
                with torch.no_grad():
                    _, next_v = net(torch.tensor(s2, device=device).unsqueeze(0))
                    last_v = 0.0 if done else next_v.item()

                # 计算回报 (Returns)
                R = last_v
                returns = []
                for r, d in zip(reversed(buf["r"]), reversed(buf["d"])):
                    R = r + CONFIG["train"]["gamma"] * R * (1.0 - d)
                    returns.append(R)
                returns = torch.tensor(returns[::-1], dtype=torch.float32, device=device)

                values = torch.cat(buf["v"])
                adv = returns - values
                # 优势函数归一化（有助于稳定）
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # 计算损失
                pi_loss = -(torch.cat(buf["logp"]) * adv.detach()).mean()
                v_loss = adv.pow(2).mean()
                ent = torch.cat(buf["ent"]).mean()
                loss = pi_loss + CONFIG["train"]["value_coef"] * v_loss - ent_coef * ent

                # 更新网络
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), CONFIG["train"]["max_grad_norm"])
                opt.step()

                # 清空缓冲区
                for key in buf: buf[key].clear()
            # ---------------------------------------

            if done:
                break

        ep_returns.append(G)
        scheduler.step()

        if (ep + 1) % LOG_EVERY_EP == 0:
            avg100 = np.mean(ep_returns[-100:])
            m = env.get_metrics()
            print(
                f"EP={ep+1:04d} | "
                f"avgR={avg100:8.3f} | "
                f"settled={m['settled']:8.0f} | "
                f"drops={m['drops']:4d} | "
                f"oversz={m['oversize_drops']:4d} | "
                f"insuff={m['insufficient_drops']:4d} | "
                f"flush={m['flushes']:4d} | "
                f"util={m['utilization']:.4f} | "
                f"dropR={m['drop_rate']:.4f}"
            )

    torch.save(net.state_dict(), CONFIG["output"]["model_path"])
    print(f"\n✅ AC model saved to {CONFIG['output']['model_path']}")
    return net

# =========================================================
# Evaluation 
# =========================================================
@torch.no_grad()
def evaluate(net):
    device = next(net.parameters()).device
    tx_pool = np.load(CONFIG["eval"]["tx_pool_path"])

    env = KWalletEnv(**CONFIG["env"], max_steps=CONFIG["env"]["T"])

    start = CONFIG["eval"]["test_start_idx"]
    num = CONFIG["eval"]["num_episodes"]

    results = []

    for i in range(num):
        s = env.reset(tx_pool[start + i])
        done = False
        while not done:
            st = torch.tensor(s, device=device).unsqueeze(0)
            logits, _ = net(st)
            dist = Categorical(logits=logits)
            a = dist.sample().item()
            s, _, done, _ = env.step(a)
        results.append(env.get_metrics())

    summary = {}
    for k in results[0]:
        vals = [r[k] for r in results]
        summary[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "median": float(np.median(vals)),
        }

    print("\n========== AC Evaluation ==========")
    for k, v in summary.items():
        print(f"{k:20s}: mean={v['mean']:.4f}, std={v['std']:.4f}")

    with open(CONFIG["output"]["results_path"], "w") as f:
        json.dump({
            "config": CONFIG,
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
        }, f, indent=2)

# =========================================================
if __name__ == "__main__":
    try:
        net = train()
        print("Starting Evaluation...")
        evaluate(net)
        print(f"Successfully saved to {CONFIG['output']['results_path']}")
    except Exception as e:
        print(f"An error occurred: {e}")