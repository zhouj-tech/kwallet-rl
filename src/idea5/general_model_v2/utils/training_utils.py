from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from general_model_v2.envs.two_pool_collateral_env import POOL_A, POOL_B, TwoPoolCollateralEnv
from general_model_v2.utils.metrics import compact_summary, compute_cross_regime_aggregate, summarize_episode_metrics
from general_model_v2.utils.threshold_expert import select_best_eta, threshold_decision


def count_parameters(model: nn.Module) -> Dict[str, int]:
    return {
        "total": int(sum(p.numel() for p in model.parameters())),
        "trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> Tuple[List[float], List[float]]:
    advantages: List[float] = []
    gae = 0.0
    next_value = float(last_value)
    for t in reversed(range(len(rewards))):
        nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages.insert(0, gae)
        next_value = values[t]
    returns = [adv + val for adv, val in zip(advantages, values)]
    return returns, advantages


def make_env_from_config(config: Dict[str, Any], max_steps: int | None = None) -> TwoPoolCollateralEnv:
    env_cfg = config["env"]
    reward_cfg = config["reward"]
    return TwoPoolCollateralEnv(
        C=float(env_cfg["C"]),
        F=int(env_cfg["F"]),
        max_transaction=float(env_cfg["T_max"]),
        max_steps=int(max_steps if max_steps is not None else env_cfg["T"]),
        seed=int(config["seed"]),
        money_p=float(reward_cfg["money_p"]),
        money_tau=float(reward_cfg["money_tau"]),
        drop_penalty=float(reward_cfg["drop_penalty"]),
        flush_levels=int(env_cfg.get("flush_levels", 17)),
        flush_grid=str(env_cfg.get("flush_grid", "uniform")),
        state_feature_mode=str(env_cfg.get("state_feature_mode", "base")),
        mask_mode=str(env_cfg.get("mask_mode", "none")),
    )


def print_env_summary(env: TwoPoolCollateralEnv) -> None:
    print(f"num_flush_choices: {env.num_flush_choices}")
    print(f"action_size: {env.action_size}")
    print(f"factorized_policy_output_size: {env.factorized_policy_output_size}")
    print(f"state_feature_mode: {env.state_feature_mode}")
    print(f"state_size: {env.state_size}")
    print(f"flush_action_decoding: {env.flush_action_decoding_summary()}")


def collect_rollout(
    agent: Any,
    config: Dict[str, Any],
    tx_values: np.ndarray,
    tx_types: np.ndarray,
    entropy_coef: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    train_cfg = config["train"]
    reward_scale = float(train_cfg.get("reward_scale", 100.0))
    env = make_env_from_config(config, max_steps=int(train_cfg["max_steps"]))
    state = env.reset(tx_values=tx_values, tx_types=tx_types)

    states: List[np.ndarray] = []
    actions: List[int] = []
    settle_actions: List[int] = []
    flush_actions: List[int] = []
    logps: List[float] = []
    values: List[float] = []
    rewards: List[float] = []
    dones: List[bool] = []
    settle_masks: List[np.ndarray] = []
    flush_masks: List[np.ndarray] = []
    use_masks = str(config["env"].get("mask_mode", "none")) != "none"

    settle_counts = [0, 0]
    flush_counts = [0 for _ in range(env.num_flush_choices)]
    episode_return = 0.0
    training_reward_return = 0.0
    t0 = time.perf_counter()

    for _ in range(int(train_cfg["max_steps"])):
        mask_info = env.get_action_mask()
        action_info = agent.act(state, deterministic=False, mask_info=mask_info)
        next_state, reward, done, _ = env.step(int(action_info["action_id"]))
        raw_reward = float(reward)
        scaled_reward = raw_reward / reward_scale

        states.append(state)
        actions.append(int(action_info["action_id"]))
        settle_actions.append(int(action_info["settle_action"]))
        flush_actions.append(int(action_info["flush_action"]))
        logps.append(float(action_info["logp"]))
        values.append(float(action_info["value"]))
        rewards.append(scaled_reward)
        dones.append(bool(done))
        if use_masks:
            if mask_info is None:
                settle_masks.append(np.ones(2, dtype=np.float32))
                flush_masks.append(np.ones(env.num_flush_choices, dtype=np.float32))
            else:
                settle_masks.append(np.asarray(mask_info["settle_mask"], dtype=np.float32))
                flush_masks.append(np.asarray(mask_info["flush_mask"], dtype=np.float32))
        settle_counts[int(action_info["settle_action"])] += 1
        flush_counts[int(action_info["flush_action"])] += 1
        episode_return += raw_reward
        training_reward_return += scaled_reward
        state = next_state
        if done:
            break

    last_value = 0.0 if dones[-1] else float(agent.value(state))
    returns, advantages = compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        last_value=last_value,
        gamma=float(train_cfg["gamma"]),
        gae_lambda=float(train_cfg["gae_lambda"]),
    )

    batch = {
        "states": states,
        "actions": actions,
        "settle_actions": settle_actions,
        "flush_actions": flush_actions,
        "logp": logps,
        "returns": returns,
        "advantages": advantages,
        "entropy_coef": float(entropy_coef),
    }
    if use_masks:
        batch["settle_masks"] = settle_masks
        batch["flush_masks"] = flush_masks

    metrics = env.get_metrics()
    metrics.update({
        "episode_return": float(episode_return),
        "training_reward_return": float(training_reward_return),
        "reward_scale": float(reward_scale),
        "steps": len(rewards),
        "settle_action_0": settle_counts[0],
        "settle_action_1": settle_counts[1],
        "elapsed_sec": float(time.perf_counter() - t0),
    })
    for idx, count in enumerate(flush_counts):
        metrics[f"flush_action_{idx}"] = count
    return batch, metrics


def imitation_regularization_loss(
    agent: Any,
    imitation_dataset: Dict[str, torch.Tensor],
    batch_size: int,
) -> torch.Tensor:
    n = int(imitation_dataset["states"].shape[0])
    if n <= 0:
        raise RuntimeError("Imitation regularization dataset is empty.")
    sample_size = min(int(batch_size), n)
    idx = torch.randint(0, n, (sample_size,), device=agent.device)
    states = imitation_dataset["states"][idx]
    settle_labels = imitation_dataset["settle_labels"][idx]
    flush_labels = imitation_dataset["flush_labels"][idx]
    settle_masks = imitation_dataset.get("settle_masks")
    flush_masks = imitation_dataset.get("flush_masks")
    settle_mask = settle_masks[idx] if settle_masks is not None else None
    flush_mask = flush_masks[idx] if flush_masks is not None else None
    settle_logits, flush_logits = agent.get_logits(
        states=states,
        settle_actions_for_flush=settle_labels,
        settle_mask=settle_mask,
        flush_mask=flush_mask,
    )
    settle_loss = nn.functional.cross_entropy(settle_logits, settle_labels)
    flush_loss = nn.functional.cross_entropy(flush_logits, flush_labels)
    return settle_loss + flush_loss


def ppo_minibatch_update(
    agent: Any,
    batch: Dict[str, Any],
    train_cfg: Dict[str, Any],
    imitation_dataset: Dict[str, torch.Tensor] | None = None,
    imitation_reg_coef: float = 0.0,
    imitation_reg_batch_size: int = 512,
) -> Dict[str, float]:
    states = torch.tensor(np.asarray(batch["states"]), dtype=torch.float32, device=agent.device)
    old_logp = torch.tensor(batch["logp"], dtype=torch.float32, device=agent.device)
    returns = torch.tensor(batch["returns"], dtype=torch.float32, device=agent.device)
    advantages = torch.tensor(batch["advantages"], dtype=torch.float32, device=agent.device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    action_batch = {
        "actions": torch.tensor(batch["actions"], dtype=torch.int64, device=agent.device),
        "settle_actions": torch.tensor(batch["settle_actions"], dtype=torch.int64, device=agent.device),
        "flush_actions": torch.tensor(batch["flush_actions"], dtype=torch.int64, device=agent.device),
    }
    if "settle_masks" in batch:
        action_batch["settle_masks"] = torch.tensor(
            np.asarray(batch["settle_masks"]),
            dtype=torch.float32,
            device=agent.device,
        )
    if "flush_masks" in batch:
        action_batch["flush_masks"] = torch.tensor(
            np.asarray(batch["flush_masks"]),
            dtype=torch.float32,
            device=agent.device,
        )

    n = states.shape[0]
    mb_size = min(int(train_cfg["minibatch_size"]), n)
    use_imitation_reg = float(imitation_reg_coef) > 0.0
    if use_imitation_reg and imitation_dataset is None:
        raise RuntimeError("imitation regularization requires an imitation dataset.")
    last_metrics = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    for _ in range(int(train_cfg["update_epochs"])):
        order = torch.randperm(n, device=agent.device)
        for start in range(0, n, mb_size):
            idx = order[start:start + mb_size]
            logp, values, entropy = agent.evaluate_actions(states[idx], action_batch, idx)
            ratio = torch.exp(logp - old_logp[idx])
            surr1 = ratio * advantages[idx]
            surr2 = torch.clamp(
                ratio,
                1.0 - float(train_cfg["clip_eps"]),
                1.0 + float(train_cfg["clip_eps"]),
            ) * advantages[idx]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (returns[idx] - values).pow(2).mean()
            entropy_mean = entropy.mean()
            loss = (
                policy_loss
                + float(train_cfg["value_coef"]) * value_loss
                - float(batch["entropy_coef"]) * entropy_mean
            )
            imitation_reg_value = None
            if use_imitation_reg:
                imitation_reg_value = imitation_regularization_loss(
                    agent=agent,
                    imitation_dataset=imitation_dataset,
                    batch_size=int(imitation_reg_batch_size),
                )
                loss = loss + float(imitation_reg_coef) * imitation_reg_value
            agent.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.model.parameters(), float(train_cfg["max_grad_norm"]))
            agent.optimizer.step()
            last_metrics = {
                "loss": float(loss.item()),
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "entropy": float(entropy_mean.item()),
            }
            if imitation_reg_value is not None:
                last_metrics["imitation_reg_loss"] = float(imitation_reg_value.item())
                last_metrics["imitation_reg_coef"] = float(imitation_reg_coef)
    return last_metrics


@torch.no_grad()
def deterministic_evaluation_loop(
    agent: Any,
    config: Dict[str, Any],
    pool: Dict[str, np.ndarray],
    num_episodes: int,
    label: str,
) -> Dict[str, Any]:
    values = pool["values"]
    types = pool["types"]
    eval_episodes = min(int(num_episodes), int(values.shape[0]))
    env = make_env_from_config(config, max_steps=int(config["eval"]["max_steps"]))
    all_results: List[Dict[str, Any]] = []
    pool_a_full_action = env.flush_levels - 1
    pool_b_full_action = env.num_flush_choices - 1
    for ep in range(eval_episodes):
        state = env.reset(tx_values=values[ep], tx_types=types[ep])
        total_steps = 0
        settle_accept_count = 0
        flush_hist = [0 for _ in range(env.num_flush_choices)]
        for _ in range(int(config["eval"]["max_steps"])):
            mask_info = env.get_action_mask()
            action_info = agent.act(state, deterministic=True, mask_info=mask_info)
            settle_action = int(action_info["settle_action"])
            flush_action = int(action_info["flush_action"])
            total_steps += 1
            if settle_action == 1:
                settle_accept_count += 1
            if 0 <= flush_action < env.num_flush_choices:
                flush_hist[flush_action] += 1
            state, _, done, _ = env.step(int(action_info["action_id"]))
            if done:
                break
        denom = max(1, total_steps)
        metrics = env.get_metrics()
        metrics.update({
            "total_steps": int(total_steps),
            "settle_accept_count": int(settle_accept_count),
            "settle_reject_count": int(total_steps - settle_accept_count),
            "settle_accept_rate": float(settle_accept_count / denom),
            "settle_reject_rate": float((total_steps - settle_accept_count) / denom),
            "flush_action_histogram": flush_hist,
            "flush_action_prob": [float(count / denom) for count in flush_hist],
            "zero_flush_rate": float(flush_hist[0] / denom),
            "no_flush_rate": float(flush_hist[0] / denom),
            "full_flush_A_rate": float(flush_hist[pool_a_full_action] / denom),
            "full_flush_B_rate": float(flush_hist[pool_b_full_action] / denom),
        })
        all_results.append(metrics)
    return {
        "label": label,
        "num_episodes": eval_episodes,
        "summary": summarize_episode_metrics(all_results),
    }


def validation_loop(agent: Any, config: Dict[str, Any], val_pool: Dict[str, np.ndarray], episode: int) -> Dict[str, Any]:
    result = deterministic_evaluation_loop(
        agent=agent,
        config=config,
        pool=val_pool,
        num_episodes=int(config["train"]["val_num_episodes"]),
        label="VAL",
    )
    summary = result["summary"]
    return {
        "episode": int(episode),
        "score_metric": "money",
        "score": float(summary["money"]["mean"]),
        "money": float(summary["money"]["mean"]),
        "value_accept_ratio": float(summary["value_accept_ratio"]["mean"]),
        "drops": float(summary["drops"]["mean"]),
        "flushes": float(summary["flushes"]["mean"]),
        "settled_value": float(summary["settled_value"]["mean"]),
    }


def threshold_imitation_pretrain(
    agent: Any,
    config: Dict[str, Any],
    train_pool: Dict[str, np.ndarray],
    val_pool: Dict[str, np.ndarray],
) -> Dict[str, torch.Tensor] | None:
    imitation_cfg = config.get("imitation", {})
    mode = str(imitation_cfg.get("imitation_mode", "none"))
    if mode == "none":
        return None
    if mode != "threshold_pretrain":
        raise ValueError(f"Unsupported imitation_mode={mode!r}.")
    if str(config.get("model_mode", "")) != "conditional_factorized_ac":
        raise NotImplementedError(
            "threshold_pretrain imitation is implemented only for "
            "Conditional Factorized AC in general_model_v2."
        )

    imitation_episodes = int(imitation_cfg.get("imitation_episodes", 0))
    if imitation_episodes <= 0:
        raise ValueError(
            "--imitation_episodes must be positive when "
            "--imitation_mode threshold_pretrain."
        )
    if imitation_episodes > train_pool["values"].shape[0]:
        raise ValueError(
            f"imitation_episodes={imitation_episodes} exceeds train pool rows="
            f"{train_pool['values'].shape[0]}."
        )

    # In v2 first version, validation and best_grid are aliases: both select
    # the eta with best validation money. Test regimes are never used here.
    eta_mode = str(imitation_cfg.get("imitation_eta_mode", "validation"))
    if eta_mode not in {"validation", "best_grid"}:
        raise ValueError("imitation_eta_mode must be 'validation' or 'best_grid'.")
    selected_eta, _ = select_best_eta(
        config=config,
        val_pool=val_pool,
        num_episodes=int(config["train"]["val_num_episodes"]),
    )

    train_cfg = config["train"]
    use_masks = str(config["env"].get("mask_mode", "none")) != "none"
    states: List[np.ndarray] = []
    settle_labels: List[int] = []
    flush_labels: List[int] = []
    settle_masks: List[np.ndarray] = []
    flush_masks: List[np.ndarray] = []
    episode_money: List[float] = []
    accept_count = 0
    no_flush_count = 0
    full_flush_A_count = 0
    full_flush_B_count = 0
    total_steps = 0

    env_for_actions = make_env_from_config(config, max_steps=int(train_cfg["max_steps"]))
    pool_A_full_action = env_for_actions.flush_levels - 1
    pool_B_full_action = env_for_actions.num_flush_choices - 1

    for ep in range(imitation_episodes):
        env = make_env_from_config(config, max_steps=int(train_cfg["max_steps"]))
        state = env.reset(
            tx_values=train_pool["values"][ep],
            tx_types=train_pool["types"][ep],
        )
        for _ in range(int(train_cfg["max_steps"])):
            mask_info = env.get_action_mask()
            settle_action, flush_action = threshold_decision(env, selected_eta)
            if use_masks:
                if mask_info is None:
                    raise RuntimeError("mask_mode is enabled but env returned no action mask.")
                if mask_info["settle_mask"][int(settle_action)] <= 0.0:
                    raise RuntimeError(
                        f"Expert settle label {settle_action} is invalid under the settle mask."
                    )
                if mask_info["flush_mask"][int(flush_action)] <= 0.0:
                    raise RuntimeError(
                        f"Expert flush label {flush_action} is invalid under the flush mask."
                    )

            states.append(state)
            settle_labels.append(int(settle_action))
            flush_labels.append(int(flush_action))
            if use_masks:
                settle_masks.append(np.asarray(mask_info["settle_mask"], dtype=np.float32))
                flush_masks.append(np.asarray(mask_info["flush_mask"], dtype=np.float32))

            total_steps += 1
            accept_count += int(settle_action == 1)
            no_flush_count += int(flush_action == 0)
            full_flush_A_count += int(flush_action == pool_A_full_action)
            full_flush_B_count += int(flush_action == pool_B_full_action)

            state, _, done, _ = env.step_decision(
                settle_choice=int(settle_action),
                flush_action=int(flush_action),
            )
            if done:
                break
        episode_money.append(float(env.get_metrics()["money"]))

    if not states:
        raise RuntimeError("Threshold imitation dataset is empty.")

    config["imitation"]["selected_eta"] = float(selected_eta)
    config["imitation"]["num_samples"] = int(len(states))
    config["imitation"]["expert_train_money_mean"] = float(np.mean(episode_money))

    denom = max(1, total_steps)
    print(
        "[Imitation] "
        f"selected_eta={float(selected_eta):.6f} "
        f"imitation samples = {len(states)} "
        f"expert_train_money_mean={float(np.mean(episode_money)):.2f}"
    )
    print(
        "[Imitation] "
        f"expert_settle_accept_rate={accept_count / denom:.6f} "
        f"expert_no_flush_rate={no_flush_count / denom:.6f} "
        f"expert_full_flush_A_rate={full_flush_A_count / denom:.6f} "
        f"expert_full_flush_B_rate={full_flush_B_count / denom:.6f}"
    )

    states_t = torch.tensor(np.asarray(states), dtype=torch.float32, device=agent.device)
    settle_t = torch.tensor(settle_labels, dtype=torch.int64, device=agent.device)
    flush_t = torch.tensor(flush_labels, dtype=torch.int64, device=agent.device)
    settle_masks_t = (
        torch.tensor(np.asarray(settle_masks), dtype=torch.float32, device=agent.device)
        if use_masks
        else None
    )
    flush_masks_t = (
        torch.tensor(np.asarray(flush_masks), dtype=torch.float32, device=agent.device)
        if use_masks
        else None
    )

    optimizer = optim.Adam(
        agent.model.parameters(),
        lr=float(imitation_cfg.get("imitation_lr", 3e-4)),
    )
    batch_size = min(int(imitation_cfg.get("imitation_batch_size", 512)), states_t.shape[0])
    epochs = int(imitation_cfg.get("imitation_epochs", 1))

    for epoch in range(epochs):
        order = torch.randperm(states_t.shape[0], device=agent.device)
        losses: List[float] = []
        settle_losses: List[float] = []
        flush_losses: List[float] = []
        for start in range(0, states_t.shape[0], batch_size):
            idx = order[start:start + batch_size]
            settle_mask = settle_masks_t[idx] if settle_masks_t is not None else None
            flush_mask = flush_masks_t[idx] if flush_masks_t is not None else None
            settle_logits, flush_logits = agent.get_logits(
                states=states_t[idx],
                settle_actions_for_flush=settle_t[idx],
                settle_mask=settle_mask,
                flush_mask=flush_mask,
            )
            settle_loss = nn.functional.cross_entropy(settle_logits, settle_t[idx])
            flush_loss = nn.functional.cross_entropy(flush_logits, flush_t[idx])
            loss = settle_loss + flush_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                agent.model.parameters(),
                float(train_cfg["max_grad_norm"]),
            )
            optimizer.step()
            losses.append(float(loss.item()))
            settle_losses.append(float(settle_loss.item()))
            flush_losses.append(float(flush_loss.item()))
        print(
            f"[Imitation] epoch={epoch + 1}/{epochs} "
            f"loss={np.mean(losses):.6f} "
            f"settle_loss={np.mean(settle_losses):.6f} "
            f"flush_loss={np.mean(flush_losses):.6f}"
        )
    dataset = {
        "states": states_t.detach(),
        "settle_labels": settle_t.detach(),
        "flush_labels": flush_t.detach(),
    }
    if settle_masks_t is not None:
        dataset["settle_masks"] = settle_masks_t.detach()
    if flush_masks_t is not None:
        dataset["flush_masks"] = flush_masks_t.detach()
    return dataset


def train_agent(
    agent: Any,
    config: Dict[str, Any],
    paths: Dict[str, Any],
    train_pool: Dict[str, np.ndarray],
    val_pool: Dict[str, np.ndarray],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_cfg = config["train"]
    episodes = int(train_cfg["episodes"])
    if episodes > train_pool["values"].shape[0]:
        raise ValueError(f"episodes={episodes} exceeds train pool rows={train_pool['values'].shape[0]}")

    training_log: List[Dict[str, Any]] = []
    validation_log: List[Dict[str, Any]] = []
    best_score = -1e30
    best_state_dict: Dict[str, torch.Tensor] | None = None
    t0 = time.perf_counter()

    env_for_shape = make_env_from_config(config)
    print("\n" + "=" * 70)
    print(f"Train {config['model_name']}")
    print("=" * 70)
    print(f"train_pool={train_pool['values'].shape} val_pool={val_pool['values'].shape}")
    print_env_summary(env_for_shape)
    print(f"parameters={count_parameters(agent.model)}")

    imitation_dataset = threshold_imitation_pretrain(
        agent=agent,
        config=config,
        train_pool=train_pool,
        val_pool=val_pool,
    )
    imitation_cfg = config.get("imitation", {})
    imitation_reg_coef = float(imitation_cfg.get("imitation_reg_coef", 0.0) or 0.0)
    imitation_reg_batch_size = int(imitation_cfg.get("imitation_reg_batch_size", 512))
    if imitation_reg_coef > 0.0 and imitation_dataset is None:
        raise RuntimeError("imitation regularization requires threshold_pretrain imitation data.")

    for ep in range(episodes):
        entropy_coef = float(train_cfg["entropy_coef_start"]) + (
            float(train_cfg["entropy_coef_end"]) - float(train_cfg["entropy_coef_start"])
        ) * min(ep / max(1, episodes), 1.0)
        batch, rollout_metrics = collect_rollout(
            agent=agent,
            config=config,
            tx_values=train_pool["values"][ep],
            tx_types=train_pool["types"][ep],
            entropy_coef=entropy_coef,
        )
        update_metrics = ppo_minibatch_update(
            agent,
            batch,
            train_cfg,
            imitation_dataset=imitation_dataset,
            imitation_reg_coef=imitation_reg_coef,
            imitation_reg_batch_size=imitation_reg_batch_size,
        )
        row = {"episode": ep + 1, **rollout_metrics, **update_metrics, "entropy_coef": float(entropy_coef)}
        training_log.append(row)
        if ep == 0 or (ep + 1) % int(train_cfg["log_every"]) == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"[Train] ep={ep + 1:4d}/{episodes} "
                f"return={row['episode_return']:10.2f} money={row['money']:10.2f} "
                f"loss={row['loss']:.4f} ent={row['entropy']:.4f} elapsed={elapsed:.1f}s"
            )
        val_every = int(train_cfg["val_every"])
        if val_every > 0 and ((ep + 1) % val_every == 0 or (ep + 1) == episodes):
            val_row = validation_loop(agent, config, val_pool, ep + 1)
            validation_log.append(val_row)
            print(
                f"[Val  ] ep={ep + 1:4d} money={val_row['money']:.2f} "
                f"val_acc={val_row['value_accept_ratio']:.4f}"
            )
            if val_row["score"] > best_score:
                best_score = float(val_row["score"])
                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in agent.model.state_dict().items()
                }
                if config["save_mode"] == "full":
                    torch.save(best_state_dict, paths["best_model_path"])
                print(f"[Val  ] best checkpoint updated: ep={ep + 1}, money={best_score:.2f}")

    if config["save_mode"] == "full":
        torch.save(agent.model.state_dict(), paths["last_model_path"])
    if best_state_dict is not None:
        agent.model.load_state_dict(best_state_dict)
    if config["save_mode"] == "full":
        best_path = Path(paths["best_model_path"])
        if best_path.exists():
            agent.model.load_state_dict(torch.load(best_path, map_location=agent.device))
        print(f"Saved best model to: {paths['best_model_path']}")
        print(f"Saved last model to: {paths['last_model_path']}")
    return training_log, validation_log


@torch.no_grad()
def evaluate_cross_regime(
    agent: Any,
    config: Dict[str, Any],
    test_pools: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, Any]:
    cross_results: Dict[str, Any] = {}
    for regime, pool in test_pools.items():
        print(f"[Eval ] {regime}")
        result = deterministic_evaluation_loop(
            agent=agent,
            config=config,
            pool=pool,
            num_episodes=int(config["eval"]["num_episodes"]),
            label=regime,
        )
        result["test_regime"] = regime
        result["summary"] = compact_summary(result["summary"])
        cross_results[regime] = result
    return {
        "config": config,
        "scenario": config.get("scenario"),
        "model_name": config["model_name"],
        "model_mode": config["model_mode"],
        "train_regime": config["data"]["train_regime"],
        "seed": config["seed"],
        "timestamp": datetime.now().isoformat(),
        "test_results": cross_results,
        "aggregate": compute_cross_regime_aggregate(cross_results),
    }


def print_action_diagnostics(results: Dict[str, Any]) -> None:
    aggregate = results.get("aggregate", {})
    for key in [
        "mean_settle_accept_rate",
        "mean_zero_flush_rate",
        "mean_full_flush_A_rate",
        "mean_full_flush_B_rate",
        "mean_no_flush_rate",
    ]:
        if key in aggregate and aggregate[key] != "":
            print(f"{key}: {float(aggregate[key]):.6f}")
