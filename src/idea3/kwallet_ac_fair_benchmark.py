# =========================================================
# 5/3
# Phase-AC Fair Benchmark
# Factorized Actor-Critic / PPO for scalable K-Wallet RL
#
# This script is intentionally separate from the CTX DQN benchmark.
# It reuses the same environment, pools, metrics, and output protocol
# so AC results can be compared against the DQN/CTX line.
# =========================================================

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/kwallet_cache")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/kwallet_matplotlib_cache")
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

try:
    from kwallet_ctx_attn_fair_benchmark import (
        DATA_POOL_DIR,
        IDEA3_DIR,
        PROJECT_ROOT,
        REGIME_ORDER,
        DEFAULT_MIX_EQ_MASTER,
        DEFAULT_MIX_EQ_VAL,
        DEFAULT_STATIC_EVAL_FILES,
        KWalletEnv,
        RecentTxAttentionEncoder,
        build_pool_fingerprints,
        compute_cross_regime_aggregate,
        count_parameters,
        load_tx_pool,
        run_leakage_sanity_checks,
        train_pool_file_for_regime,
        verify_data_integrity,
    )
except ImportError:
    # Allows importing this file from outside src/idea3.
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from kwallet_ctx_attn_fair_benchmark import (  # type: ignore
        DATA_POOL_DIR,
        IDEA3_DIR,
        PROJECT_ROOT,
        REGIME_ORDER,
        DEFAULT_MIX_EQ_MASTER,
        DEFAULT_MIX_EQ_VAL,
        DEFAULT_STATIC_EVAL_FILES,
        KWalletEnv,
        RecentTxAttentionEncoder,
        build_pool_fingerprints,
        compute_cross_regime_aggregate,
        count_parameters,
        load_tx_pool,
        run_leakage_sanity_checks,
        train_pool_file_for_regime,
        verify_data_integrity,
    )


RESULT_ROOT = IDEA3_DIR / "ac_fair_benchmark_results"
LOG_EVERY_N = 25


CONFIG: Dict[str, Any] = {
    "seed": 123,
    "model_mode": "factorized_ac",
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
        "train_regime": "MIX12_EQ",
        "train_pool_file": DEFAULT_MIX_EQ_MASTER,
        "val_pool_file": DEFAULT_MIX_EQ_VAL,
        "test_pool_files": DEFAULT_STATIC_EVAL_FILES,
    },
    "train": {
        "episodes": 1000,
        "train_use_episodes": 3000,
        "max_steps": 1000,
        "device": "cpu",
        "learning_rate": 3e-4,
        "gamma": 0.98,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "update_epochs": 4,
        "minibatch_size": 256,
        "value_coef": 0.5,
        "entropy_coef_start": 0.03,
        "entropy_coef_end": 0.003,
        "max_grad_norm": 1.0,
        "val_every": 50,
        "val_num_episodes": 200,
        "val_metric": "value_accept_ratio",
        "use_best_model_for_final_eval": True,
        "hidden_size": 128,
    },
    "eval": {
        "num_episodes": 200,
        "max_steps": 1000,
    },
    "attention_context": {
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
    "plot": {
        "window": 50,
    },
    "output": {
        "output_dir": str(RESULT_ROOT),
    },
}


def set_seed(seed: int) -> None:
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


def ac_env_mode(model_mode: str) -> str:
    if model_mode == "factorized_ac":
        return "baseline"
    if model_mode == "attn_factorized_ac":
        return "attn_context"
    raise ValueError(f"Unsupported model_mode={model_mode}")


def build_run_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def build_scenario_name(config: Dict[str, Any]) -> str:
    env = config["env"]
    c_value = int(env["C"]) if float(env["C"]).is_integer() else env["C"]
    return (
        f"{config['model_mode']}_train{config['data']['train_regime']}_"
        f"C{c_value}_k{env['k']}_T{env['T']}_F{env['F']}_seed{config['seed']}"
    )


def build_title_tag(config: Dict[str, Any], run_stamp: str) -> str:
    env = config["env"]
    c_value = int(env["C"]) if float(env["C"]).is_integer() else env["C"]
    return (
        f"{run_stamp} | mode={config['model_mode']} | train={config['data']['train_regime']} | "
        f"C={c_value} k={env['k']} T={env['T']} F={env['F']}"
    )


def build_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    run_stamp = build_run_stamp()
    scenario = build_scenario_name(config)
    result_root = Path(config["output"]["output_dir"]).expanduser().resolve()
    result_run_dir = result_root / "runs" / scenario / run_stamp
    checkpoint_run_dir = result_root / "checkpoints" / scenario / run_stamp
    aggregate_dir = result_root / "aggregates"
    return {
        "project_root": str(PROJECT_ROOT),
        "data_pool_dir": str(DATA_POOL_DIR),
        "result_root": str(result_root),
        "scenario": scenario,
        "run_stamp": run_stamp,
        "title_tag": build_title_tag(config, run_stamp),
        "train_regime": config["data"]["train_regime"],
        "train_pool_path": str(DATA_POOL_DIR / config["data"]["train_pool_file"]),
        "val_pool_path": str(DATA_POOL_DIR / config["data"]["val_pool_file"]),
        "test_pool_paths": {
            regime: str(DATA_POOL_DIR / file_name)
            for regime, file_name in config["data"]["test_pool_files"].items()
        },
        "result_run_dir": str(result_run_dir),
        "checkpoint_run_dir": str(checkpoint_run_dir),
        "aggregate_dir": str(aggregate_dir),
        "run_info_path": str(result_run_dir / "run_info.json"),
        "results_json_path": str(result_run_dir / "cross_regime_results.json"),
        "summary_txt_path": str(result_run_dir / "summary_table.txt"),
        "training_history_path": str(result_run_dir / "training_history.json"),
        "validation_history_path": str(result_run_dir / "validation_history.json"),
        "eval_plot_path": str(result_run_dir / "cross_regime_plot.png"),
        "training_plot_path": str(result_run_dir / "training_curve.png"),
        "best_model_path": str(checkpoint_run_dir / "best_model.pth"),
        "last_model_path": str(checkpoint_run_dir / "last_model.pth"),
    }


def ensure_dirs(paths: Dict[str, Any], config: Dict[str, Any]) -> None:
    if config["debug_mode"] or config["save_mode"] == "none":
        return
    for key in ["result_run_dir", "checkpoint_run_dir", "aggregate_dir"]:
        os.makedirs(paths[key], exist_ok=True)


def make_env(config: Dict[str, Any], max_steps: int) -> KWalletEnv:
    env_cfg = config["env"]
    attn_cfg = config["attention_context"]
    reward_cfg = config["reward"]
    return KWalletEnv(
        C=env_cfg["C"],
        k=env_cfg["k"],
        F=env_cfg["F"],
        max_transaction=env_cfg["T"],
        max_steps=max_steps,
        seed=config["seed"],
        model_mode=ac_env_mode(config["model_mode"]),
        attention_window_size=attn_cfg["window_size"],
        enable_shaping=env_cfg["enable_shaping"],
        alpha_drop=reward_cfg["alpha_drop"],
        beta_flush=reward_cfg["beta_flush"],
    )


class FactorizedActorCritic(nn.Module):
    def __init__(
        self,
        model_mode: str,
        state_size: int,
        base_state_size: int,
        window_size: int,
        k: int,
        hidden_size: int = 128,
        d_model: int = 32,
        n_heads: int = 2,
        context_dim: int = 32,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.model_mode = model_mode
        self.base_state_size = int(base_state_size)
        self.window_size = int(window_size)
        self.k = int(k)
        self.uses_context = model_mode == "attn_factorized_ac"

        if self.uses_context:
            self.context_encoder = RecentTxAttentionEncoder(
                window_size=window_size,
                d_model=d_model,
                n_heads=n_heads,
                context_dim=context_dim,
                dropout=dropout,
            )
            encoder_in = base_state_size + context_dim
        else:
            self.context_encoder = None
            encoder_in = state_size

        self.shared = nn.Sequential(
            nn.Linear(encoder_in, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.settle_head = nn.Linear(hidden_size, k + 1)
        self.flush_head = nn.Linear(hidden_size, k + 1)
        self.value_head = nn.Linear(hidden_size, 1)

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        if not self.uses_context:
            return state
        base_state = state[:, :self.base_state_size]
        tx_start = self.base_state_size
        tx_end = tx_start + self.window_size
        recent_tx = state[:, tx_start:tx_end]
        recent_mask = state[:, tx_end:tx_end + self.window_size]
        context = self.context_encoder(recent_tx, recent_mask)
        return torch.cat([base_state, context], dim=1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.shared(self.encode(state))
        settle_logits = self.settle_head(h)
        flush_logits = self.flush_head(h)
        value = self.value_head(h).squeeze(-1)
        return settle_logits, flush_logits, value


class FactorizedPPOAgent:
    def __init__(
        self,
        config: Dict[str, Any],
        state_size: int,
        base_state_size: int,
        window_size: int,
        k: int,
    ):
        train_cfg = config["train"]
        attn_cfg = config["attention_context"]
        self.config = config
        self.k = int(k)
        self.device = torch.device(train_cfg["device"])
        self.model = FactorizedActorCritic(
            model_mode=config["model_mode"],
            state_size=state_size,
            base_state_size=base_state_size,
            window_size=window_size,
            k=k,
            hidden_size=int(train_cfg["hidden_size"]),
            d_model=int(attn_cfg["d_model"]),
            n_heads=int(attn_cfg["n_heads"]),
            context_dim=int(attn_cfg["context_dim"]),
            dropout=float(attn_cfg["dropout"]),
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=float(train_cfg["learning_rate"]))

    def distributions(self, states: torch.Tensor) -> Tuple[Categorical, Categorical, torch.Tensor]:
        settle_logits, flush_logits, values = self.model(states)
        return Categorical(logits=settle_logits), Categorical(logits=flush_logits), values

    @torch.no_grad()
    def act(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, int, float, float]:
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        settle_dist, flush_dist, value = self.distributions(state_t)
        if deterministic:
            settle = torch.argmax(settle_dist.logits, dim=-1)
            flush = torch.argmax(flush_dist.logits, dim=-1)
        else:
            settle = settle_dist.sample()
            flush = flush_dist.sample()
        logp = settle_dist.log_prob(settle) + flush_dist.log_prob(flush)
        action_int = int(settle.item()) * (self.k + 1) + int(flush.item())
        return action_int, int(settle.item()), float(logp.item()), float(value.item())

    def update(self, batch: Dict[str, List[Any]]) -> Dict[str, float]:
        train_cfg = self.config["train"]
        states = torch.tensor(np.array(batch["states"]), dtype=torch.float32, device=self.device)
        settle_actions = torch.tensor(batch["settle_actions"], dtype=torch.int64, device=self.device)
        flush_actions = torch.tensor(batch["flush_actions"], dtype=torch.int64, device=self.device)
        old_logp = torch.tensor(batch["logp"], dtype=torch.float32, device=self.device)
        returns = torch.tensor(batch["returns"], dtype=torch.float32, device=self.device)
        advantages = torch.tensor(batch["advantages"], dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = states.shape[0]
        mb_size = min(int(train_cfg["minibatch_size"]), n)
        last_metrics = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        for _ in range(int(train_cfg["update_epochs"])):
            order = torch.randperm(n, device=self.device)
            for start in range(0, n, mb_size):
                idx = order[start:start + mb_size]
                settle_dist, flush_dist, values = self.distributions(states[idx])
                logp = settle_dist.log_prob(settle_actions[idx]) + flush_dist.log_prob(flush_actions[idx])
                ratio = torch.exp(logp - old_logp[idx])
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(
                    ratio,
                    1.0 - float(train_cfg["clip_eps"]),
                    1.0 + float(train_cfg["clip_eps"]),
                ) * advantages[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (returns[idx] - values).pow(2).mean()
                entropy = settle_dist.entropy().mean() + flush_dist.entropy().mean()

                loss = (
                    policy_loss
                    + float(train_cfg["value_coef"]) * value_loss
                    - float(batch["entropy_coef"]) * entropy
                )
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), float(train_cfg["max_grad_norm"]))
                self.optimizer.step()

                last_metrics = {
                    "loss": float(loss.item()),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.item()),
                }
        return last_metrics


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[List[float], List[float]]:
    advantages = []
    gae = 0.0
    next_value = last_value
    for t in reversed(range(len(rewards))):
        nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages.insert(0, gae)
        next_value = values[t]
    returns = [adv + val for adv, val in zip(advantages, values)]
    return returns, advantages


@torch.no_grad()
def evaluate_agent_on_array(
    agent: FactorizedPPOAgent,
    config: Dict[str, Any],
    tx_pool: np.ndarray,
    label: str,
    num_eval_episodes: int,
    max_steps: int,
) -> Dict[str, Any]:
    num_eval_episodes = min(num_eval_episodes, tx_pool.shape[0])
    env = make_env(config, max_steps=max_steps)
    all_results = []

    for ep in range(num_eval_episodes):
        state = env.reset(tx_stream=tx_pool[ep])
        total_requested_value = 0.0
        total_tx_count = 0
        accepted_count = 0
        for _ in range(max_steps):
            total_requested_value += float(env.current_tx)
            total_tx_count += 1
            action, _, _, _ = agent.act(state, deterministic=True)
            state, _, done, info = env.step(action)
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

    return {
        "label": label,
        "num_episodes": num_eval_episodes,
        "summary": summarize_episode_metrics(all_results),
        "raw_results": all_results,
    }


def summarize_episode_metrics(all_results: List[Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
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


def evaluate_agent_on_pool(
    agent: FactorizedPPOAgent,
    config: Dict[str, Any],
    test_pool_path: str,
    test_regime: str,
) -> Dict[str, Any]:
    eval_cfg = config["eval"]
    if not verify_data_integrity(test_pool_path, expected_steps=eval_cfg["max_steps"], label=test_regime):
        raise RuntimeError(f"Data verification failed for {test_regime}")
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
    agent: FactorizedPPOAgent,
    config: Dict[str, Any],
    paths: Dict[str, Any],
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("AC Cross-Regime Evaluation")
    print("=" * 70)
    cross_results = {}
    for regime_name, pool_path in paths["test_pool_paths"].items():
        cross_results[regime_name] = evaluate_agent_on_pool(agent, config, pool_path, regime_name)
    return {
        "config": config,
        "scenario": paths["scenario"],
        "train_regime": paths["train_regime"],
        "model_mode": config["model_mode"],
        "seed": config["seed"],
        "timestamp": datetime.now().isoformat(),
        "test_results": cross_results,
        "aggregate": compute_cross_regime_aggregate(cross_results),
    }


def train_agent(
    config: Dict[str, Any],
    paths: Dict[str, Any],
) -> Tuple[FactorizedPPOAgent, List[float], List[float], List[Dict[str, float]], List[Dict[str, float]]]:
    train_cfg = config["train"]
    tx_pool_full = load_tx_pool(paths["train_pool_path"], expected_steps=train_cfg["max_steps"])
    tx_pool_val = load_tx_pool(paths["val_pool_path"], expected_steps=train_cfg["max_steps"])
    train_use_episodes = int(train_cfg["train_use_episodes"])
    if train_use_episodes > tx_pool_full.shape[0]:
        raise ValueError(f"train_use_episodes={train_use_episodes} exceeds pool size={tx_pool_full.shape[0]}")
    if int(train_cfg["episodes"]) > train_use_episodes:
        raise ValueError("Training episodes exceed selected train pool rows.")
    tx_pool_train = tx_pool_full[:train_use_episodes]

    env = make_env(config, max_steps=train_cfg["max_steps"])
    agent = FactorizedPPOAgent(
        config=config,
        state_size=env.state_size,
        base_state_size=env.base_state_size,
        window_size=config["attention_context"]["window_size"],
        k=env.k,
    )
    param_counts = count_parameters(agent.model)
    print("\n" + "=" * 70)
    print("Train Factorized Actor-Critic / PPO")
    print("=" * 70)
    print(f"mode={config['model_mode']} train_regime={paths['train_regime']} seed={config['seed']}")
    print(f"env C={env.C} k={env.k} F={env.F} T={env.max_transaction}")
    print(f"state={env.state_size} base_state={env.base_state_size} action={(env.k + 1) ** 2}")
    print(f"train_pool={tx_pool_train.shape} val_pool={tx_pool_val.shape}")
    print(f"model parameters: total={param_counts['total']} trainable={param_counts['trainable']}")

    returns_history: List[float] = []
    loss_history: List[float] = []
    validation_history: List[Dict[str, float]] = []
    best_val_score = -1e18
    best_state_dict = None
    best_val_snapshot = None
    t_start = time.perf_counter()

    for ep in range(int(train_cfg["episodes"])):
        state = env.reset(tx_stream=tx_pool_train[ep])
        episode_return = 0.0
        states: List[np.ndarray] = []
        settle_actions: List[int] = []
        flush_actions: List[int] = []
        logps: List[float] = []
        values: List[float] = []
        rewards: List[float] = []
        dones: List[bool] = []

        entropy_coef = float(train_cfg["entropy_coef_start"]) + (
            float(train_cfg["entropy_coef_end"]) - float(train_cfg["entropy_coef_start"])
        ) * min(ep / max(1, int(train_cfg["episodes"])), 1.0)

        for _ in range(int(train_cfg["max_steps"])):
            action, settle, logp, value = agent.act(state, deterministic=False)
            flush = action % (env.k + 1)
            next_state, reward, done, _ = env.step(action)
            states.append(state)
            settle_actions.append(settle)
            flush_actions.append(flush)
            logps.append(logp)
            values.append(value)
            rewards.append(float(reward))
            dones.append(bool(done))
            episode_return += float(reward)
            state = next_state
            if done:
                break

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
            _, _, last_value_t = agent.distributions(state_t)
            last_value = 0.0 if dones[-1] else float(last_value_t.item())

        returns, advantages = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            last_value=last_value,
            gamma=float(train_cfg["gamma"]),
            gae_lambda=float(train_cfg["gae_lambda"]),
        )
        metrics = agent.update({
            "states": states,
            "settle_actions": settle_actions,
            "flush_actions": flush_actions,
            "logp": logps,
            "returns": returns,
            "advantages": advantages,
            "entropy_coef": entropy_coef,
        })
        returns_history.append(float(episode_return))
        loss_history.append(float(metrics["loss"]))

        if (ep + 1) % LOG_EVERY_N == 0 or ep == 0:
            recent_mean = float(np.mean(returns_history[-LOG_EVERY_N:]))
            elapsed = time.perf_counter() - t_start
            print(
                f"[Train] ep={ep + 1:4d}/{train_cfg['episodes']} "
                f"return={episode_return:10.2f} recent={recent_mean:10.2f} "
                f"loss={metrics['loss']:.4f} ent={metrics['entropy']:.4f} elapsed={elapsed:.1f}s"
            )

        val_every = int(train_cfg["val_every"])
        if val_every > 0 and ((ep + 1) % val_every == 0 or (ep + 1) == int(train_cfg["episodes"])):
            val_result = evaluate_agent_on_array(
                agent=agent,
                config=config,
                tx_pool=tx_pool_val,
                label="VAL",
                num_eval_episodes=int(train_cfg["val_num_episodes"]),
                max_steps=int(train_cfg["max_steps"]),
            )
            metric_name = train_cfg["val_metric"]
            val_score = float(val_result["summary"][metric_name]["mean"])
            val_row = {
                "episode": ep + 1,
                "metric": metric_name,
                "value_accept_ratio": float(val_result["summary"]["value_accept_ratio"]["mean"]),
                "drop_rate": float(val_result["summary"]["drop_rate"]["mean"]),
                "drops": float(val_result["summary"]["drops"]["mean"]),
                "flushes": float(val_result["summary"]["flushes"]["mean"]),
                "score": val_score,
            }
            validation_history.append(val_row)
            print(f"[Val  ] ep={ep + 1:4d} {metric_name}={val_score:.4f}")
            if val_score > best_val_score:
                best_val_score = val_score
                best_val_snapshot = val_row
                best_state_dict = {k: v.detach().cpu().clone() for k, v in agent.model.state_dict().items()}
                if not config["debug_mode"] and config["save_mode"] == "full":
                    torch.save(best_state_dict, paths["best_model_path"])
                print(f"[Val  ] best checkpoint updated: ep={ep + 1}, score={val_score:.4f}")

    if train_cfg["use_best_model_for_final_eval"] and best_state_dict is not None:
        agent.model.load_state_dict(best_state_dict)
        print(f"Loaded best checkpoint: ep={best_val_snapshot['episode']} score={best_val_snapshot['score']:.4f}")

    if not config["debug_mode"] and config["save_mode"] == "full":
        torch.save(agent.model.state_dict(), paths["last_model_path"])
        print(f"Last model saved to: {paths['last_model_path']}")

    return agent, returns_history, loss_history, validation_history, []


def build_cross_regime_report_text(results: Dict[str, Any]) -> str:
    lines = []
    lines.append("=" * 96)
    lines.append("Phase-AC Fair Benchmark Cross-Regime Evaluation")
    lines.append("=" * 96)
    lines.append(f"timestamp    : {results['timestamp']}")
    lines.append(f"scenario     : {results['scenario']}")
    lines.append(f"model_mode   : {results['model_mode']}")
    lines.append(f"train_regime : {results['train_regime']}")
    lines.append(f"seed         : {results['seed']}")
    lines.append("-" * 96)
    lines.append(
        f"{'Test':<8}{'ValAcc(%)':>14}{'Drops':>12}{'Flushes':>12}"
        f"{'DropRate(%)':>14}{'CntAcc(%)':>14}"
    )
    lines.append("-" * 96)
    for regime_name, regime_result in results["test_results"].items():
        s = regime_result["summary"]
        lines.append(
            f"{regime_name:<8}"
            f"{100 * s['value_accept_ratio']['mean']:>14.2f}"
            f"{s['drops']['mean']:>12.2f}"
            f"{s['flushes']['mean']:>12.2f}"
            f"{100 * s['drop_rate']['mean']:>14.2f}"
            f"{100 * s['count_accept_ratio']['mean']:>14.2f}"
        )
    agg = results["aggregate"]
    lines.append("-" * 96)
    lines.append(f"Mean ValAcc (%)        : {100 * agg['mean_value_accept_ratio']:.4f}")
    lines.append(f"Worst-Regime ValAcc (%): {100 * agg['worst_regime_value_accept_ratio']:.4f}")
    lines.append(f"Std Across Regimes     : {agg['std_value_accept_ratio_across_regimes']:.6f}")
    lines.append(f"Mean Drops             : {agg['mean_drops']:.4f}")
    lines.append(f"Mean Flushes           : {agg['mean_flushes']:.4f}")
    lines.append("=" * 96)
    return "\n".join(lines)


def save_json(payload: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_results(results: Dict[str, Any], save_path: str) -> None:
    compact = {
        "config": results["config"],
        "scenario": results["scenario"],
        "train_regime": results["train_regime"],
        "model_mode": results["model_mode"],
        "seed": results["seed"],
        "timestamp": results["timestamp"],
        "aggregate": results["aggregate"],
        "test_results": {},
    }
    for regime_name, regime_result in results["test_results"].items():
        compact["test_results"][regime_name] = {
            "num_episodes": regime_result["num_episodes"],
            "summary": {},
        }
        for metric, data in regime_result["summary"].items():
            compact["test_results"][regime_name]["summary"][metric] = {
                "mean": data["mean"],
                "std": data["std"],
                "min": data["min"],
                "max": data["max"],
                "median": data["median"],
            }
    save_json(compact, save_path)
    print(f"Results saved to: {save_path}")


def moving_average(values: List[float], window: int) -> np.ndarray:
    if not values:
        return np.array([])
    arr = np.array(values, dtype=float)
    out = np.zeros_like(arr)
    for i in range(len(arr)):
        left = max(0, i - window + 1)
        out[i] = np.mean(arr[left:i + 1])
    return out


def plot_training_curves(returns: List[float], losses: List[float], save_path: str, title_tag: str, window: int) -> None:
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(returns, alpha=0.35, label="Return")
    plt.plot(moving_average(returns, window), linewidth=2, label=f"MA({window})")
    plt.title(f"AC Training Curves\n{title_tag}")
    plt.ylabel("Episode Return")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(losses, alpha=0.8)
    plt.xlabel("Episode")
    plt.ylabel("PPO Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Training plot saved to: {save_path}")


def plot_evaluation_results(results: Dict[str, Any], save_path: str, title_tag: str) -> None:
    regimes = list(results["test_results"].keys())
    val_acc = [100 * results["test_results"][r]["summary"]["value_accept_ratio"]["mean"] for r in regimes]
    drops = [results["test_results"][r]["summary"]["drops"]["mean"] for r in regimes]
    flushes = [results["test_results"][r]["summary"]["flushes"]["mean"] for r in regimes]
    drop_rate = [100 * results["test_results"][r]["summary"]["drop_rate"]["mean"] for r in regimes]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"AC Cross-Regime Evaluation\n{title_tag}", fontsize=14, fontweight="bold")
    for ax, values, label in [
        (axes[0, 0], val_acc, "Value Accept Ratio (%)"),
        (axes[0, 1], drops, "Drops"),
        (axes[1, 0], flushes, "Flushes"),
        (axes[1, 1], drop_rate, "Drop Rate (%)"),
    ]:
        ax.bar(regimes, values)
        ax.set_title(label)
        ax.grid(True, alpha=0.3, axis="y")
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Evaluation plot saved to: {save_path}")


def write_run_outputs(
    config: Dict[str, Any],
    paths: Dict[str, Any],
    results: Dict[str, Any],
    returns: List[float],
    losses: List[float],
    validation_history: List[Dict[str, float]],
) -> None:
    if config["debug_mode"] or config["save_mode"] == "none":
        print("Save skipped because debug_mode=True or save_mode=none.")
        return
    save_json(
        {
            "config": config,
            "paths": paths,
            "pool_fingerprints": build_pool_fingerprints(config, paths),
            "timestamp": datetime.now().isoformat(),
        },
        paths["run_info_path"],
    )
    save_results(results, paths["results_json_path"])
    with open(paths["summary_txt_path"], "w", encoding="utf-8") as f:
        f.write(build_cross_regime_report_text(results))
    save_json({"returns": returns, "loss_history": losses}, paths["training_history_path"])
    save_json({"validation_history": validation_history}, paths["validation_history_path"])
    plot_training_curves(returns, losses, paths["training_plot_path"], paths["title_tag"], config["plot"]["window"])
    plot_evaluation_results(results, paths["eval_plot_path"], paths["title_tag"])


def flatten_result_for_csv(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    config = payload["config"]
    aggregate = payload.get("aggregate", {})
    rows = []
    for regime, regime_result in payload["test_results"].items():
        s = regime_result["summary"]
        rows.append({
            "scenario": payload["scenario"],
            "model_mode": payload["model_mode"],
            "train_regime": payload["train_regime"],
            "seed": payload["seed"],
            "test_regime": regime,
            "value_accept_ratio": s["value_accept_ratio"]["mean"],
            "drops": s["drops"]["mean"],
            "flushes": s["flushes"]["mean"],
            "drop_rate": s["drop_rate"]["mean"],
            "count_accept_ratio": s["count_accept_ratio"]["mean"],
            "mean_value_accept_ratio": aggregate.get("mean_value_accept_ratio"),
            "worst_regime_value_accept_ratio": aggregate.get("worst_regime_value_accept_ratio"),
            "std_value_accept_ratio_across_regimes": aggregate.get("std_value_accept_ratio_across_regimes"),
            "C": config["env"]["C"],
            "k": config["env"]["k"],
            "F": config["env"]["F"],
            "T": config["env"]["T"],
        })
    return rows


def aggregate_results(result_root: Path = RESULT_ROOT) -> Path:
    result_files = sorted((result_root / "runs").glob("**/cross_regime_results.json"))
    rows: List[Dict[str, Any]] = []
    for path in result_files:
        rows.extend(flatten_result_for_csv(path))
    aggregate_dir = result_root / "aggregates"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    out_path = aggregate_dir / "ac_fair_benchmark_aggregated.csv"
    fieldnames = [
        "scenario", "model_mode", "train_regime", "seed", "test_regime",
        "value_accept_ratio", "drops", "flushes", "drop_rate", "count_accept_ratio",
        "mean_value_accept_ratio", "worst_regime_value_accept_ratio",
        "std_value_accept_ratio_across_regimes", "C", "k", "F", "T",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Aggregated CSV saved to: {out_path} ({len(rows)} rows)")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-AC fair benchmark for K-wallet actor-critic.")
    parser.add_argument("--model_mode", choices=["factorized_ac", "attn_factorized_ac"], default=CONFIG["model_mode"])
    parser.add_argument("--train_regime", type=str, default=CONFIG["data"]["train_regime"])
    parser.add_argument("--seed", type=int, default=CONFIG["seed"])
    parser.add_argument("--C", type=float, default=CONFIG["env"]["C"])
    parser.add_argument("--k", type=int, default=CONFIG["env"]["k"])
    parser.add_argument("--F", type=int, default=CONFIG["env"]["F"])
    parser.add_argument("--T", type=int, default=CONFIG["env"]["T"])
    parser.add_argument("--window_size", type=int, default=CONFIG["attention_context"]["window_size"])
    parser.add_argument("--output_dir", type=str, default=CONFIG["output"]["output_dir"])
    parser.add_argument("--device", type=str, default=CONFIG["train"]["device"])
    parser.add_argument("--episodes", type=int, default=CONFIG["train"]["episodes"])
    parser.add_argument("--eval_episodes", type=int, default=CONFIG["eval"]["num_episodes"])
    parser.add_argument("--save_mode", choices=["none", "full"], default=CONFIG["save_mode"])
    parser.add_argument("--debug_mode", action="store_true")
    parser.add_argument("--skip_leakage_check", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")
    return parser.parse_args()


def apply_args_to_config(args: argparse.Namespace) -> Dict[str, Any]:
    config = json.loads(json.dumps(CONFIG))
    config["model_mode"] = args.model_mode
    config["seed"] = int(args.seed)
    config["debug_mode"] = bool(args.debug_mode)
    config["save_mode"] = args.save_mode
    config["env"]["C"] = float(args.C)
    config["env"]["k"] = int(args.k)
    config["env"]["F"] = int(args.F)
    config["env"]["T"] = int(args.T)
    config["attention_context"]["window_size"] = int(args.window_size)
    config["plot"]["window"] = int(args.window_size)
    config["output"]["output_dir"] = args.output_dir
    config["data"]["train_regime"] = args.train_regime
    config["data"]["train_pool_file"] = train_pool_file_for_regime(args.train_regime)
    config["train"]["device"] = args.device
    config["train"]["episodes"] = int(args.episodes)
    config["eval"]["num_episodes"] = int(args.eval_episodes)
    return config


def main() -> None:
    args = parse_args()
    if args.aggregate_only:
        aggregate_results(Path(args.output_dir).expanduser().resolve())
        return

    config = apply_args_to_config(args)
    set_seed(config["seed"])
    if config["model_mode"] == "attn_factorized_ac" and not args.skip_leakage_check:
        run_leakage_sanity_checks({**config, "model_mode": "attn_context"})

    paths = build_paths(config)
    ensure_dirs(paths, config)
    print("\n" + "=" * 70)
    print("Phase-AC Fair Benchmark")
    print("=" * 70)
    print(f"scenario: {paths['scenario']}")
    print(f"results : {paths['result_run_dir']}")
    print(f"train   : {paths['train_pool_path']}")
    print(f"val     : {paths['val_pool_path']}")

    agent, returns, losses, validation_history, _ = train_agent(config, paths)
    results = evaluate_agent_cross_regime(agent, config, paths)
    print("\n" + build_cross_regime_report_text(results))
    write_run_outputs(config, paths, results, returns, losses, validation_history)
    if not config["debug_mode"] and config["save_mode"] != "none":
        aggregate_results(Path(paths["result_root"]))


if __name__ == "__main__":
    main()
