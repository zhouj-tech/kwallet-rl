from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from general_model.envs.general_collateral_env import GeneralCollateralEnv
from general_model.utils.metrics import compact_summary, compute_cross_regime_aggregate, summarize_episode_metrics


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


def make_env_from_config(config: Dict[str, Any], max_steps: int | None = None) -> GeneralCollateralEnv:
    env_cfg = config["env"]
    reward_cfg = config["reward"]
    return GeneralCollateralEnv(
        C=float(env_cfg["C"]),
        F=int(env_cfg["F"]),
        max_transaction=int(env_cfg["T_max"]),
        max_steps=int(max_steps if max_steps is not None else env_cfg["T"]),
        seed=int(config["seed"]),
        money_p=float(reward_cfg["money_p"]),
        money_tau=float(reward_cfg["money_tau"]),
        drop_penalty=float(reward_cfg["drop_penalty"]),
        flush_levels=int(env_cfg.get("flush_levels", 5)),
        flush_grid=str(env_cfg.get("flush_grid", "uniform")),
        state_feature_mode=str(env_cfg.get("state_feature_mode", "base")),
        mask_mode=str(env_cfg.get("mask_mode", "none")),
    )


def collect_rollout(
    agent: Any,
    config: Dict[str, Any],
    tx_stream: np.ndarray,
    entropy_coef: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    train_cfg = config["train"]
    reward_scale = float(train_cfg.get("reward_scale", 1.0))
    env = make_env_from_config(config, max_steps=int(train_cfg["max_steps"]))
    state = env.reset(tx_stream=tx_stream)

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
        next_state, reward, done, info = env.step(int(action_info["action_id"]))
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


def ppo_minibatch_update(agent: Any, batch: Dict[str, Any], train_cfg: Dict[str, Any]) -> Dict[str, float]:
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
    return last_metrics


@torch.no_grad()
def deterministic_evaluation_loop(
    agent: Any,
    config: Dict[str, Any],
    tx_pool: np.ndarray,
    num_episodes: int,
    label: str,
) -> Dict[str, Any]:
    eval_episodes = min(int(num_episodes), int(tx_pool.shape[0]))
    env = make_env_from_config(config, max_steps=int(config["eval"]["max_steps"]))
    all_results: List[Dict[str, Any]] = []
    for ep in range(eval_episodes):
        state = env.reset(tx_stream=tx_pool[ep])
        total_steps = 0
        settle_accept_count = 0
        settle_reject_count = 0
        flush_action_histogram = [0 for _ in range(env.num_flush_choices)]
        flush_action_sum = 0.0
        flush_fraction_sum = 0.0

        for _ in range(int(config["eval"]["max_steps"])):
            mask_info = env.get_action_mask()
            action_info = agent.act(state, deterministic=True, mask_info=mask_info)
            settle_action = int(action_info.get("settle_action", 0))
            flush_action = int(action_info.get("flush_action", 0))
            flush_fraction = float(
                action_info.get(
                    "flush_fraction",
                    env.flush_fractions[flush_action],
                )
            )

            total_steps += 1
            if settle_action == 1:
                settle_accept_count += 1
            else:
                settle_reject_count += 1
            if 0 <= flush_action < env.num_flush_choices:
                flush_action_histogram[flush_action] += 1
            flush_action_sum += float(flush_action)
            flush_fraction_sum += flush_fraction

            state, _, done, _ = env.step(int(action_info["action_id"]))
            if done:
                break

        denom = max(1, total_steps)
        flush_fractions = np.asarray(env.flush_fractions, dtype=float)
        max_flush_fraction = float(np.max(flush_fractions)) if flush_fractions.size else 1.0
        zero_flush_count = sum(
            count
            for idx, count in enumerate(flush_action_histogram)
            if np.isclose(flush_fractions[idx], 0.0)
        )
        full_flush_count = sum(
            count
            for idx, count in enumerate(flush_action_histogram)
            if np.isclose(flush_fractions[idx], max_flush_fraction)
        )
        middle_flush_count = sum(
            count
            for idx, count in enumerate(flush_action_histogram)
            if flush_fractions[idx] > 0.0 and flush_fractions[idx] < 1.0
        )
        metrics = env.get_metrics()
        metrics.update({
            "total_steps": int(total_steps),
            "settle_accept_count": int(settle_accept_count),
            "settle_reject_count": int(settle_reject_count),
            "settle_accept_rate": float(settle_accept_count / denom),
            "settle_reject_rate": float(settle_reject_count / denom),
            "flush_action_histogram": flush_action_histogram,
            "flush_action_prob": [
                float(count / denom)
                for count in flush_action_histogram
            ],
            "mean_flush_action": float(flush_action_sum / denom),
            "mean_flush_fraction": float(flush_fraction_sum / denom),
            "zero_flush_rate": float(zero_flush_count / denom),
            "full_flush_rate": float(full_flush_count / denom),
            "middle_flush_rate": float(middle_flush_count / denom),
        })
        all_results.append(metrics)
    return {
        "label": label,
        "num_episodes": eval_episodes,
        "summary": summarize_episode_metrics(all_results),
    }


def validation_loop(
    agent: Any,
    config: Dict[str, Any],
    val_pool: np.ndarray,
    episode: int,
) -> Dict[str, Any]:
    result = deterministic_evaluation_loop(
        agent=agent,
        config=config,
        tx_pool=val_pool,
        num_episodes=int(config["train"]["val_num_episodes"]),
        label="VAL",
    )
    summary = result["summary"]
    row = {
        "episode": int(episode),
        "score_metric": "money",
        "score": float(summary["money"]["mean"]),
        "money": float(summary["money"]["mean"]),
        "value_accept_ratio": float(summary["value_accept_ratio"]["mean"]),
        "drops": float(summary["drops"]["mean"]),
        "drop_rate": float(summary["drop_rate"]["mean"]),
        "flushes": float(summary["flushes"]["mean"]),
        "settled_value": float(summary["settled_value"]["mean"]),
    }
    return row


def _nearest_flush_label(
    flush_amount: float,
    committed_after_settle: float,
    flush_fractions: List[float],
) -> int:
    if committed_after_settle <= 0.0 or flush_amount <= 0.0:
        return 0
    target_fraction = float(flush_amount) / float(committed_after_settle)
    best_idx = 0
    best_dist = float("inf")
    if np.isclose(float(flush_amount), float(committed_after_settle)):
        return int(len(flush_fractions) - 1)
    for idx, fraction in enumerate(flush_fractions):
        dist = abs(float(fraction) - target_fraction)
        if dist < best_dist - 1e-12:
            best_dist = dist
            best_idx = idx
    return int(best_idx)


def threshold_imitation_pretrain(
    agent: Any,
    config: Dict[str, Any],
    train_pool: np.ndarray,
    val_pool: np.ndarray,
) -> None:
    imitation_cfg = config.get("imitation", {})
    imitation_mode = str(
        imitation_cfg.get("imitation_mode", imitation_cfg.get("mode", "none"))
    )
    if imitation_mode != "threshold_pretrain":
        return
    if str(config.get("model_mode", "")) != "conditional_factorized_ac":
        raise NotImplementedError(
            "threshold_pretrain imitation is implemented only for "
            "Conditional Factorized AC. Use --imitation_mode none for flat PPO "
            "or independent Factorized AC."
        )

    from general_model.baselines.threshold_baselines import (
        ETA_GRID,
        select_best_eta,
    )

    train_cfg = config["train"]
    env_cfg = config["env"]
    num_val = min(int(train_cfg["val_num_episodes"]), int(val_pool.shape[0]))
    selected_eta, eta_rows = select_best_eta(
        eta_grid=ETA_GRID,
        config=config,
        val_pool=val_pool,
        num_episodes=num_val,
    )
    config["imitation"]["selected_eta"] = float(selected_eta)
    config["imitation"]["selection_method"] = "grid_validation_money"

    num_episodes = min(
        int(imitation_cfg.get("imitation_episodes", imitation_cfg.get("episodes", 100))),
        int(train_pool.shape[0]),
    )
    states: List[np.ndarray] = []
    settle_labels: List[int] = []
    flush_labels: List[int] = []
    settle_masks: List[np.ndarray] = []
    flush_masks: List[np.ndarray] = []
    use_masks = str(env_cfg.get("mask_mode", "none")) != "none"
    episode_money: List[float] = []
    accept_count = 0
    no_flush_count = 0
    full_flush_count = 0
    total_steps = 0

    for ep in range(num_episodes):
        env = make_env_from_config(config, max_steps=int(train_cfg["max_steps"]))
        state = env.reset(tx_stream=train_pool[ep])
        for _ in range(int(train_cfg["max_steps"])):
            mask_info = env.get_action_mask()
            current_tx = float(env.current_tx)
            can_settle = float(env.available_collateral) >= current_tx
            settle_choice = 1 if can_settle else 0
            projected_committed = float(env.committed_unflushed)
            if can_settle:
                projected_committed += current_tx
            pressure = projected_committed / max(1e-12, float(env_cfg["C"]))
            # One-pool imitation expert: validation-selected threshold trigger,
            # then full flush. This keeps labels aligned with the discrete
            # full-flush action while eta still comes from the tuned baseline.
            flush_label = (
                int(env.num_flush_choices - 1)
                if pressure >= float(selected_eta)
                else 0
            )
            if use_masks:
                if mask_info is None:
                    settle_mask = np.ones(2, dtype=np.float32)
                    flush_mask = np.ones(env.num_flush_choices, dtype=np.float32)
                else:
                    settle_mask = np.asarray(mask_info["settle_mask"], dtype=np.float32)
                    flush_mask = np.asarray(mask_info["flush_mask"], dtype=np.float32)
                if settle_mask[int(settle_choice)] <= 0.0:
                    raise RuntimeError(
                        f"Expert settle label {settle_choice} is invalid under the settle mask."
                    )
                if flush_mask[int(flush_label)] <= 0.0:
                    raise RuntimeError(
                        f"Expert flush label {flush_label} is invalid under the flush mask."
                    )

            states.append(state)
            settle_labels.append(int(settle_choice))
            flush_labels.append(int(flush_label))
            if use_masks:
                settle_masks.append(settle_mask)
                flush_masks.append(flush_mask)

            total_steps += 1
            accept_count += int(settle_choice == 1)
            no_flush_count += int(flush_label == 0)
            full_flush_count += int(flush_label == env.num_flush_choices - 1)

            state, _, done, _ = env.step_decision(
                settle_choice=int(settle_choice),
                flush_choice=int(flush_label),
                flush_amount=None,
            )
            if done:
                break
        episode_money.append(float(env.get_metrics()["money"]))

    if not states:
        raise RuntimeError("Threshold imitation dataset is empty.")
    config["imitation"]["num_samples"] = int(len(states))
    config["imitation"]["expert_train_money_mean"] = float(np.mean(episode_money))

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
        lr=float(imitation_cfg.get("imitation_lr", imitation_cfg.get("lr", 3e-4))),
    )
    batch_size = min(
        int(imitation_cfg.get("imitation_batch_size", imitation_cfg.get("batch_size", 512))),
        states_t.shape[0],
    )
    epochs = int(imitation_cfg.get("imitation_epochs", imitation_cfg.get("epochs", 3)))

    denom = max(1, total_steps)
    print(
        "[Imitation] "
        f"selected_eta={float(selected_eta):.6f} "
        f"imitation samples = {states_t.shape[0]} "
        f"expert_train_money_mean={float(np.mean(episode_money)):.2f}"
    )
    print(
        "[Imitation] "
        f"expert_settle_accept_rate={accept_count / denom:.6f} "
        f"expert_no_flush_rate={no_flush_count / denom:.6f} "
        f"expert_full_flush_rate={full_flush_count / denom:.6f}"
    )

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


def train_agent(
    agent: Any,
    config: Dict[str, Any],
    paths: Dict[str, Any],
    train_pool: np.ndarray,
    val_pool: np.ndarray,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_cfg = config["train"]
    episodes = int(train_cfg["episodes"])
    if episodes > train_pool.shape[0]:
        raise ValueError(f"episodes={episodes} exceeds train pool rows={train_pool.shape[0]}")

    training_log: List[Dict[str, Any]] = []
    validation_log: List[Dict[str, Any]] = []
    best_score = -1e30
    best_state_dict: Dict[str, torch.Tensor] | None = None
    t0 = time.perf_counter()

    print("\n" + "=" * 70)
    print(f"Train {config['model_name']}")
    print("=" * 70)
    print(f"train_pool={train_pool.shape} val_pool={val_pool.shape}")
    env_for_shape = make_env_from_config(config)
    print(
        f"state_size={env_for_shape.state_size} "
        f"action_size={env_for_shape.action_size} "
        f"flush_levels={env_for_shape.flush_levels} "
        f"num_flush_choices={env_for_shape.num_flush_choices}"
    )
    print(f"flush_grid={env_for_shape.flush_grid}")
    print(f"flush_fractions={env_for_shape.flush_fractions}")
    print(f"state_feature_mode={env_for_shape.state_feature_mode}")
    print(f"mask_mode={env_for_shape.mask_mode}")
    print(f"imitation_mode={config.get('imitation', {}).get('mode', 'none')}")
    print(f"parameters={count_parameters(agent.model)}")

    threshold_imitation_pretrain(
        agent=agent,
        config=config,
        train_pool=train_pool,
        val_pool=val_pool,
    )

    for ep in range(episodes):
        entropy_coef = float(train_cfg["entropy_coef_start"]) + (
            float(train_cfg["entropy_coef_end"]) - float(train_cfg["entropy_coef_start"])
        ) * min(ep / max(1, episodes), 1.0)
        batch, rollout_metrics = collect_rollout(agent, config, train_pool[ep], entropy_coef)
        update_metrics = ppo_minibatch_update(agent, batch, train_cfg)
        row = {
            "episode": ep + 1,
            **rollout_metrics,
            **update_metrics,
            "entropy_coef": float(entropy_coef),
        }
        training_log.append(row)

        if ep == 0 or (ep + 1) % int(train_cfg["log_every"]) == 0:
            elapsed = time.perf_counter() - t0
            recent = np.mean([x["episode_return"] for x in training_log[-int(train_cfg["log_every"]):]])
            print(
                f"[Train] ep={ep + 1:4d}/{episodes} "
                f"return={row['episode_return']:10.2f} recent={recent:10.2f} "
                f"money={row['money']:10.2f} loss={row['loss']:.4f} "
                f"ent={row['entropy']:.4f} elapsed={elapsed:.1f}s"
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
    test_pools: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    cross_results: Dict[str, Any] = {}
    for regime, tx_pool in test_pools.items():
        print(f"[Eval ] {regime}")
        result = deterministic_evaluation_loop(
            agent=agent,
            config=config,
            tx_pool=tx_pool,
            num_episodes=int(config["eval"]["num_episodes"]),
            label=regime,
        )
        result["test_regime"] = regime
        result["summary"] = compact_summary(result["summary"])
        cross_results[regime] = result
    aggregate = compute_cross_regime_aggregate(cross_results)
    return {
        "config": config,
        "scenario": config.get("scenario"),
        "model_name": config["model_name"],
        "model_mode": config["model_mode"],
        "train_regime": config["data"]["train_regime"],
        "seed": config["seed"],
        "timestamp": datetime.now().isoformat(),
        "test_results": cross_results,
        "aggregate": aggregate,
    }


def print_action_diagnostics(results: Dict[str, Any]) -> None:
    aggregate = results.get("aggregate", {})
    if not isinstance(aggregate, dict):
        return

    fields = [
        ("mean_settle_accept_rate", "mean settle_accept_rate"),
        ("mean_zero_flush_rate", "mean zero_flush_rate"),
        ("mean_full_flush_rate", "mean full_flush_rate"),
        ("mean_middle_flush_rate", "mean middle_flush_rate"),
        ("mean_flush_fraction", "mean flush_fraction"),
    ]
    for key, label in fields:
        if key in aggregate and aggregate[key] != "":
            print(f"{label}: {float(aggregate[key]):.6f}")
