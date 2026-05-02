#3/28 动作空间k+1（简单版）
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
benchmark_v1 最小 DQN 训练脚本（简化动作空间版）
======================================================
当前版本改动：
1. 动作空间简化为：
   - 0..k-1：选择某个钱包
   - k：主动放弃当前交易（drop）
2. 不再允许 agent 自由选择 flush mask
3. 当 agent 选中的钱包“可用但余额不足”时：
   - 环境自动 flush 该钱包
   - 当前交易仍然 drop
4. 目标：先判断在更干净的动作空间下，
   specialist / generalist 是否仍然表现为“万能通用策略”
"""

from __future__ import annotations

import csv
import json
import os
import random
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from benchmark_v1_generator import REGIME_CONFIGS, GLOBAL as BENCH_GLOBAL, generate_episode
    GENERATOR_SOURCE = "benchmark_v1_generator.py"
except ImportError:
    from idea1.regime_generator2 import REGIME_CONFIGS, GLOBAL as BENCH_GLOBAL, generate_episode
    GENERATOR_SOURCE = "phase1_regime_generator_compact.py"


CONFIG = {
    "seed": 123,
    "data": {
        "train_mode": "specialist",
        "train_regime": "SH",
        "eval_regimes": ["SL", "SH", "BL", "BH"],
    },
    "env": {
        "C": 3000.0,
        "k": 8,
        "T": 1000,
        "F": 3,
        "enable_shaping": True,
        "drop_penalty": 1.0,
        "flush_cost": 0.1,
        "imbalance_penalty": 0.0,
    },
    "train": {
        "episodes": 900,
        "max_steps": 300,
        "batch_size": 128,
        "gamma": 0.98,
        "lr": 1e-3,
        "buffer_size": 50000,
        "target_update_every": 200,
        "epsilon_start": 0.9,
        "epsilon_end": 0.05,
        "epsilon_decay": 0.995,
        "hidden_dim": 128,
        "greedy_eval_every": 30,
        "greedy_eval_episodes": 10
    },
    "eval": {
        "num_episodes": 50,
        "max_steps": 300,
    },
    "output": {
        "save_results": True,
        "results_dir": "dqn_benchmark_v1_outputs_simple_actions",
    },
}


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path_str: str) -> Path:
    p = Path(path_str)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return

    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            full_row = {key: row.get(key, "") for key in fieldnames}
            writer.writerow(full_row)


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b != 0 else 0.0


def build_run_name(config: Dict) -> str:
    mode = config["data"]["train_mode"]
    regime = config["data"]["train_regime"]
    if mode == "specialist":
        tag = f"specialist_{regime}"
    else:
        tag = "generalist_mixed"
    env = config["env"]
    return f"{tag}_simpleact_C{int(env['C'])}_k{env['k']}_T{env['T']}_F{env['F']}"


def choose_train_regime(config: Dict, episode_idx: int, rng: random.Random) -> str:
    mode = config["data"]["train_mode"]
    if mode == "specialist":
        return config["data"]["train_regime"]
    elif mode == "generalist":
        return rng.choice(config["data"]["eval_regimes"])
    else:
        raise ValueError(f"未知 train_mode: {mode}")


class KWalletEnv:
    """
    状态：
    - 每个钱包余额（归一化）
    - 每个钱包是否可用（0/1）
    - 每个钱包剩余冻结时间（归一化）
    - 当前交易大小（归一化）

    动作：
    - 0..k-1：尝试把当前交易放入某个钱包
    - k：主动放弃当前交易（drop）

    不再允许手动 flush mask。
    自动 flush 规则：
    - 如果选中的钱包当前可用但余额不足
    - 环境自动 flush 该钱包
    - 当前交易仍然 drop
    """

    def __init__(self, C: float, k: int, T: int, F: int, max_steps: int, reward_cfg: Dict):
        self.C = float(C)
        self.k = int(k)
        self.wallet_size = self.C / self.k
        self.T = int(T)
        self.F = int(F)
        self.max_steps = int(max_steps)
        self.reward_cfg = reward_cfg
        self.num_actions = self.k + 1
        self.reset_state()

    def reset_state(self):
        self.t = 0
        self.wallets = np.full(self.k, self.wallet_size, dtype=float)
        self.freeze_until = np.full(self.k, -1, dtype=int)

        self.total_offered = 0.0
        self.total_accepted = 0.0
        self.num_drops = 0
        self.num_flushes = 0
        self.oversize_drops = 0
        self.insufficient_drops = 0

        self.tx_stream = None
        self.current_tx = 0.0

    def _usable(self, i: int) -> bool:
        return self.t > self.freeze_until[i]

    def decode_action(self, action_idx: int) -> int:
        return action_idx

    def _get_state(self) -> np.ndarray:
        balances = self.wallets / self.wallet_size
        usable = np.array([1.0 if self._usable(i) else 0.0 for i in range(self.k)], dtype=float)

        remain = []
        for i in range(self.k):
            if self._usable(i):
                remain.append(0.0)
            else:
                remain_steps = max(0, self.freeze_until[i] - self.t + 1)
                remain.append(remain_steps / max(1, self.F))
        remain = np.array(remain, dtype=float)

        tx_feat = np.array([self.current_tx / self.wallet_size], dtype=float)
        state = np.concatenate([balances, usable, remain, tx_feat], axis=0)
        return state.astype(np.float32)

    def reset(self, tx_stream: np.ndarray | List[float]) -> np.ndarray:
        self.reset_state()
        tx_stream = list(tx_stream[:self.max_steps])
        if len(tx_stream) < self.max_steps:
            raise ValueError(f"交易流长度不足：需要 {self.max_steps}，实际 {len(tx_stream)}")
        self.tx_stream = tx_stream
        self.current_tx = float(self.tx_stream[self.t])
        return self._get_state()

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict]:
        settle_choice = self.decode_action(action_idx)
        tx = float(self.current_tx)
        self.total_offered += tx

        info = {
            "accepted": 0.0,
            "dropped": False,
            "oversize": False,
            "insufficient": False,
            "flushes_this_step": 0,
            "action_type": "none",
        }

        accepted = 0.0
        dropped = False

        if tx > self.wallet_size:
            dropped = True
            self.num_drops += 1
            self.oversize_drops += 1
            info["oversize"] = True
            info["action_type"] = "oversize_drop"
        else:
            if settle_choice == self.k:
                dropped = True
                self.num_drops += 1
                self.insufficient_drops += 1
                info["insufficient"] = True
                info["action_type"] = "manual_drop"
            else:
                i = settle_choice
                if not self._usable(i):
                    dropped = True
                    self.num_drops += 1
                    self.insufficient_drops += 1
                    info["insufficient"] = True
                    info["action_type"] = "chosen_wallet_frozen"
                else:
                    if self.wallets[i] >= tx:
                        self.wallets[i] -= tx
                        accepted = tx
                        self.total_accepted += accepted
                        info["action_type"] = "accept"
                    else:
                        self.wallets[i] = self.wallet_size
                        self.freeze_until[i] = self.t + self.F
                        self.num_flushes += 1

                        dropped = True
                        self.num_drops += 1
                        self.insufficient_drops += 1

                        info["flushes_this_step"] = 1
                        info["insufficient"] = True
                        info["action_type"] = "auto_flush_then_drop"

        info["accepted"] = accepted
        info["dropped"] = dropped

        reward = 20.0 * (accepted / self.wallet_size)
        if self.reward_cfg["enable_shaping"]:
            reward -= self.reward_cfg["flush_cost"] * info["flushes_this_step"]
            if dropped:
                reward -= self.reward_cfg["drop_penalty"]

        self.t += 1
        done = self.t >= self.max_steps
        if not done:
            self.current_tx = float(self.tx_stream[self.t])

        next_state = self._get_state() if not done else np.zeros_like(self._get_state(), dtype=np.float32)
        return next_state, float(reward), done, info

    def get_metrics(self) -> Dict[str, float]:
        val_acc = safe_div(self.total_accepted, self.total_offered)
        drop_rate = safe_div(self.num_drops, self.max_steps)
        return {
            "settled": self.total_accepted,
            "offered": self.total_offered,
            "drops": self.num_drops,
            "oversize_drops": self.oversize_drops,
            "insufficient_drops": self.insufficient_drops,
            "flushes": self.num_flushes,
            "value_acceptance": val_acc,
            "drop_rate": drop_rate,
        }


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(ns, dtype=np.float32),
            np.array(d, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class QNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, cfg: Dict):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = cfg["gamma"]
        self.batch_size = cfg["batch_size"]

        self.model = QNet(state_dim, action_dim, cfg["hidden_dim"]).to(self.device)
        self.target_model = QNet(state_dim, action_dim, cfg["hidden_dim"]).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg["lr"])
        self.buffer = ReplayBuffer(cfg["buffer_size"])
        self.loss_fn = nn.MSELoss()
        self.learn_steps = 0

    def act(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.model.net[-1].out_features)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.model(s)
            return int(torch.argmax(q, dim=1).item())

    def update(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None

        s, a, r, ns, d = self.buffer.sample(self.batch_size)

        s = torch.tensor(s, dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.long, device=self.device).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        ns = torch.tensor(ns, dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device).unsqueeze(1)

        q = self.model(s).gather(1, a)
        with torch.no_grad():
            next_q = self.target_model(ns).max(dim=1, keepdim=True)[0]
            target = r + (1.0 - d) * self.gamma * next_q

        loss = self.loss_fn(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.learn_steps += 1
        return float(loss.item())

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())


def evaluate_greedy_on_single_regime(agent: DQNAgent, regime_name: str, config: Dict, num_episodes: int = 10) -> Dict:
    env = KWalletEnv(
        C=config["env"]["C"],
        k=config["env"]["k"],
        T=config["env"]["T"],
        F=config["env"]["F"],
        max_steps=config["train"]["max_steps"],
        reward_cfg={
            "enable_shaping": config["env"]["enable_shaping"],
            "drop_penalty": config["env"]["drop_penalty"],
            "flush_cost": config["env"]["flush_cost"],
            "imbalance_penalty": config["env"]["imbalance_penalty"],
        }
    )

    valacc_list = []
    droprate_list = []
    settled_list = []
    flushes_list = []

    for ep in range(num_episodes):
        seed = 200000 + ep
        tx_stream, _ = generate_episode(
            config=REGIME_CONFIGS[regime_name],
            T=config["train"]["max_steps"],
            seed=seed,
        )
        state = env.reset(tx_stream=tx_stream)

        while True:
            action = agent.act(state, epsilon=0.0)
            next_state, reward, done, info = env.step(action)
            state = next_state
            if done:
                break

        metrics = env.get_metrics()
        valacc_list.append(metrics["value_acceptance"])
        droprate_list.append(metrics["drop_rate"])
        settled_list.append(metrics["settled"])
        flushes_list.append(metrics["flushes"])

    return {
        "regime": regime_name,
        "value_acceptance": float(np.mean(valacc_list)),
        "drop_rate": float(np.mean(droprate_list)),
        "settled": float(np.mean(settled_list)),
        "flushes": float(np.mean(flushes_list)),
    }


def train_agent(config: Dict) -> Tuple[DQNAgent, List[Dict]]:
    set_seed(config["seed"])
    py_rng = random.Random(config["seed"])

    env = KWalletEnv(
        C=config["env"]["C"],
        k=config["env"]["k"],
        T=config["env"]["T"],
        F=config["env"]["F"],
        max_steps=config["train"]["max_steps"],
        reward_cfg={
            "enable_shaping": config["env"]["enable_shaping"],
            "drop_penalty": config["env"]["drop_penalty"],
            "flush_cost": config["env"]["flush_cost"],
            "imbalance_penalty": config["env"]["imbalance_penalty"],
        }
    )

    state_dim = env._get_state().shape[0]
    action_dim = env.num_actions
    agent = DQNAgent(state_dim, action_dim, config["train"])

    epsilon = config["train"]["epsilon_start"]
    train_logs = []

    print("\n[开始训练]")
    print("-" * 72)
    print(f"generator_source = {GENERATOR_SOURCE}")
    print(f"train_mode = {config['data']['train_mode']}")
    print(f"train_regime = {config['data']['train_regime']}")
    print(f"episodes = {config['train']['episodes']}")
    print(f"max_steps = {config['train']['max_steps']}")
    print(f"action_space = {env.num_actions} （0..{env.k-1} 选钱包，{env.k} 主动 drop）")
    print("-" * 72)

    for ep in range(config["train"]["episodes"]):
        regime_name = choose_train_regime(config, ep, py_rng)
        seed = config["seed"] + ep

        tx_stream, _ = generate_episode(
            config=REGIME_CONFIGS[regime_name],
            T=config["train"]["max_steps"],
            seed=seed,
        )

        state = env.reset(tx_stream=tx_stream)
        episode_reward = 0.0
        losses = []

        for step in range(config["train"]["max_steps"]):
            action = agent.act(state, epsilon)
            next_state, reward, done, info = env.step(action)

            agent.buffer.push(state, action, reward, next_state, done)
            loss = agent.update()
            if loss is not None:
                losses.append(loss)

            if agent.learn_steps > 0 and agent.learn_steps % config["train"]["target_update_every"] == 0:
                agent.update_target()

            state = next_state
            episode_reward += reward

            if done:
                break

        epsilon = max(config["train"]["epsilon_end"], epsilon * config["train"]["epsilon_decay"])

        metrics = env.get_metrics()
        log_row = {
            "episode": ep,
            "train_regime_used": regime_name,
            "epsilon": float(epsilon),
            "episode_reward": float(episode_reward),
            "avg_loss": float(np.mean(losses)) if losses else None,
            **metrics,
        }
        train_logs.append(log_row)

        if (ep + 1) % max(1, config["train"]["episodes"] // 10) == 0:
            print(
                f"ep={ep+1:>4d} | regime={regime_name} | "
                f"reward={episode_reward:>8.2f} | "
                f"ValAcc={metrics['value_acceptance']:.4f} | "
                f"DropRate={metrics['drop_rate']:.4f} | "
                f"Flushes={metrics['flushes']:.2f} | "
                f"eps={epsilon:.3f}"
            )

        if (ep + 1) % config["train"]["greedy_eval_every"] == 0:
            greedy_result = evaluate_greedy_on_single_regime(
                agent=agent,
                regime_name="SL",
                config=config,
                num_episodes=config["train"]["greedy_eval_episodes"],
            )

            print(
                f"    [greedy eval on SL] "
                f"ValAcc={greedy_result['value_acceptance']:.4f} | "
                f"DropRate={greedy_result['drop_rate']:.4f} | "
                f"Settled={greedy_result['settled']:.2f} | "
                f"Flushes={greedy_result['flushes']:.2f}"
            )

            train_logs[-1]["greedy_eval_regime"] = "SL"
            train_logs[-1]["greedy_eval_valacc"] = greedy_result["value_acceptance"]
            train_logs[-1]["greedy_eval_droprate"] = greedy_result["drop_rate"]
            train_logs[-1]["greedy_eval_settled"] = greedy_result["settled"]
            train_logs[-1]["greedy_eval_flushes"] = greedy_result["flushes"]

    return agent, train_logs


def evaluate_agent_on_regime(agent: DQNAgent, regime_name: str, config: Dict) -> Dict:
    env = KWalletEnv(
        C=config["env"]["C"],
        k=config["env"]["k"],
        T=config["env"]["T"],
        F=config["env"]["F"],
        max_steps=config["eval"]["max_steps"],
        reward_cfg={
            "enable_shaping": config["env"]["enable_shaping"],
            "drop_penalty": config["env"]["drop_penalty"],
            "flush_cost": config["env"]["flush_cost"],
            "imbalance_penalty": config["env"]["imbalance_penalty"],
        }
    )

    per_episode_rows = []
    regime_index = config["data"]["eval_regimes"].index(regime_name)

    for ep in range(config["eval"]["num_episodes"]):
        seed = 100000 + regime_index * 10000 + ep

        tx_stream, state_labels = generate_episode(
            config=REGIME_CONFIGS[regime_name],
            T=config["eval"]["max_steps"],
            seed=seed,
        )

        state = env.reset(tx_stream=tx_stream)
        total_reward = 0.0

        while True:
            action = agent.act(state, epsilon=0.0)
            next_state, reward, done, info = env.step(action)
            state = next_state
            total_reward += reward
            if done:
                break

        metrics = env.get_metrics()
        row = {
            "test_regime": regime_name,
            "episode_index": ep,
            "episode_reward": float(total_reward),
            "state_burst_fraction": float(sum(1 for s in state_labels if s == "burst") / len(state_labels)),
            **metrics,
        }
        per_episode_rows.append(row)

    summary = {}
    numeric_keys = [k for k, v in per_episode_rows[0].items() if isinstance(v, (int, float, np.integer, np.floating))]
    for key in numeric_keys:
        vals = [float(r[key]) for r in per_episode_rows]
        summary[key] = float(np.mean(vals))
        summary[f"{key}_std"] = float(np.std(vals))

    return {
        "regime_name": regime_name,
        "summary": summary,
        "per_episode_rows": per_episode_rows,
    }


def evaluate_agent_cross_regime(agent: DQNAgent, config: Dict) -> Tuple[List[Dict], List[Dict]]:
    summary_rows = []
    all_per_episode_rows = []

    print("\n[开始 cross-regime 评估]")
    print("-" * 72)
    for regime_name in config["data"]["eval_regimes"]:
        result = evaluate_agent_on_regime(agent, regime_name, config)
        summary = result["summary"]

        row = {
            "regime": regime_name,
            "settled": summary["settled"],
            "drops": summary["drops"],
            "flushes": summary["flushes"],
            "value_acceptance": summary["value_acceptance"],
            "drop_rate": summary["drop_rate"],
            "episode_reward": summary["episode_reward"],
            "state_burst_fraction": summary["state_burst_fraction"],
        }
        summary_rows.append(row)
        all_per_episode_rows.extend(result["per_episode_rows"])

        print(
            f"{regime_name:<4s} | "
            f"ValAcc={row['value_acceptance']:.4f} | "
            f"DropRate={row['drop_rate']:.4f} | "
            f"Settled={row['settled']:.2f} | "
            f"Flushes={row['flushes']:.2f}"
        )

    return summary_rows, all_per_episode_rows


def save_outputs(config: Dict, train_logs: List[Dict], eval_summary_rows: List[Dict], eval_per_episode_rows: List[Dict], agent: DQNAgent):
    out_dir = ensure_dir(config["output"]["results_dir"])
    run_name = build_run_name(config)
    run_dir = ensure_dir(str(out_dir / run_name))

    save_json(
        {
            "generator_source": GENERATOR_SOURCE,
            "config": config,
        },
        run_dir / "run_config.json"
    )

    save_csv(train_logs, run_dir / "train_logs.csv")
    save_csv(eval_summary_rows, run_dir / "eval_summary.csv")
    save_csv(eval_per_episode_rows, run_dir / "eval_per_episode.csv")
    torch.save(agent.model.state_dict(), run_dir / "model.pth")

    print(f"\n结果已保存到：{run_dir.resolve()}")
    print("生成文件：")
    print("  - run_config.json")
    print("  - train_logs.csv")
    print("  - eval_summary.csv")
    print("  - eval_per_episode.csv")
    print("  - model.pth")


def print_eval_table(summary_rows: List[Dict]) -> None:
    print("\n[Cross-Regime Summary]")
    print("-" * 110)
    print(
        f"{'Regime':<8}"
        f"{'Settled':>12}"
        f"{'Drops':>10}"
        f"{'Flushes':>10}"
        f"{'ValAcc':>10}"
        f"{'DropRate':>11}"
        f"{'Reward':>10}"
    )
    print("-" * 110)
    for row in summary_rows:
        print(
            f"{row['regime']:<8}"
            f"{row['settled']:>12.2f}"
            f"{row['drops']:>10.2f}"
            f"{row['flushes']:>10.2f}"
            f"{row['value_acceptance']:>10.4f}"
            f"{row['drop_rate']:>11.4f}"
            f"{row['episode_reward']:>10.2f}"
        )


if __name__ == "__main__":
    set_seed(CONFIG["seed"])
    agent, train_logs = train_agent(CONFIG)
    eval_summary_rows, eval_per_episode_rows = evaluate_agent_cross_regime(agent, CONFIG)
    print_eval_table(eval_summary_rows)

    if CONFIG["output"]["save_results"]:
        save_outputs(CONFIG, train_logs, eval_summary_rows, eval_per_episode_rows, agent)
