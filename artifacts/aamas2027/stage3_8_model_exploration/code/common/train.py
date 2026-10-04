"""Shared training driver for SC-M / IA-SC / TP-SC (frozen SC-FAC protocol).

Protocol invariants (MODEL_DESIGN.md):
  - Exact historical MIX12_EQ training pool, first 1000 training episodes out of
    its declared first 3000 rows, in the historical fixed order; SHA256-verified.
  - Entropy schedule: linear 0.03 -> 0.003 over exactly 1000 episodes.
  - PPO: lr 3e-4 Adam, gamma 0.99, GAE lambda 0.95, clip 0.2, 4 update epochs,
    minibatch 256, grad-norm clip 1.0, advantages normalized per episode.
  - CPU-only, torch threads pinned to 4.
  - Validation (VAL-v3) every 50 training episodes on a separate environment,
    deterministic argmax, consuming no RNG state (asserted in tests).
  - Checkpoint bank retains all 20 validation checkpoints + designations.
  - TP-SC: teacher labels computed at pre-action states during rollout, ONLY
    while lambda_BF(e) > 0 (episodes 0..249); lambda = 0.01*max(1 - e/250, 0).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .bank import CheckpointBank
from .config import (
    ALPHA_DROP,
    BETA_FLUSH,
    C,
    CHECKPOINT_EPISODES,
    F,
    FROZEN_TRAINING_POOL,
    K,
    MAX_STEPS,
    MAX_TRANSACTION,
    TRAIN_EPISODES,
    TRAIN_POOL_ROWS,
    CommonRunConfig,
    runtime_snapshot,
    set_global_seed,
    sha256_file,
)
from .evaluate import evaluate_panel, money_identity_ok
from .features import augment_state, extract_c_features
from .frozen_env import KWalletEnv
from .models import (
    build_ia_sc_model,
    build_paired_ia_model,
    build_sc_m_model,
    build_tp_sc_model,
    count_parameters,
)
from .ppo import (
    ConditionalFactorizedPPOAgent,
    compute_gae,
    entropy_coef_at,
)
from .pools import load_training_pool
from .teacher import lambda_bf, teacher_labels

TORCH_NUM_THREADS = 4

CANDIDATE_BUILDERS = {
    "SC-M": build_sc_m_model,
    "IA-SC": build_ia_sc_model,
    "TP-SC": build_tp_sc_model,
}


def _expected_pool_sha256() -> str:
    cache = Path(__file__).resolve().parent.parent / "configs" / "frozen_pool_sha256.txt"
    if cache.exists():
        return cache.read_text().strip()
    digest = sha256_file(FROZEN_TRAINING_POOL)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(digest + "\n")
    return digest


def load_val_panel(release_root: Path, split: str = "VAL-v3") -> Dict[str, np.ndarray]:
    from .evaluate import load_panel_from_release

    return load_panel_from_release(release_root, split)


def validate_on_panel(agent: ConditionalFactorizedPPOAgent,
                      panel: Dict[str, np.ndarray], capacity: Optional[float] = None) -> Dict:
    return evaluate_panel(agent, panel, episodes_per_regime=None, capacity=capacity)


def train_candidate(
    config: CommonRunConfig,
    output_dir: Path,
    release_root: Optional[Path] = None,
    val_panel: Optional[Dict[str, np.ndarray]] = None,
    log_every: int = 50,
) -> Dict:
    assert config.candidate in CANDIDATE_BUILDERS, f"unknown candidate {config.candidate}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(TORCH_NUM_THREADS)
    set_global_seed(config.seed)

    pool_sha = _expected_pool_sha256()
    tx_pool_full = load_training_pool(expected_sha256=pool_sha)
    tx_pool_train = tx_pool_full[:TRAIN_POOL_ROWS]  # declared first 3000 rows

    # ---- candidate model construction (frozen per candidate) ----
    # IA-SC uses the paired-initialization sequence: global seed already set above
    # exactly as for the matched SC-M run; overlapping blocks copied verbatim from a
    # same-seed SC-M reference init, added columns zeroed, torch RNG restored to the
    # post-SC-M-init state before training (MODEL_DESIGN C1-F, PI-approved repair).
    wallet_size = float(config.C) / float(config.k)
    if config.candidate == "IA-SC":
        model = build_paired_ia_model()
        state_fn = lambda s: augment_state(s, wallet_size=wallet_size)  # noqa: E731
    else:
        model = CANDIDATE_BUILDERS[config.candidate]()
        state_fn = None
    agent = ConditionalFactorizedPPOAgent(
        model, state_fn=state_fn, lr=config.ppo_lr, wallet_size=wallet_size
    )

    param_count = count_parameters(model)

    if val_panel is None:
        assert release_root is not None, "release_root required to load VAL-v3 panel"
        val_panel = load_val_panel(release_root, config.val_panel)

    bank = CheckpointBank(output_dir, CHECKPOINT_EPISODES)

    run_log: List[Dict] = []
    rng_snapshot_at_validation: List[bool] = []
    train_start = time.time()

    env = KWalletEnv(
        C=config.C, k=config.k, F=config.F,
        max_transaction=config.max_transaction, max_steps=config.max_steps,
        seed=config.seed, enable_shaping=config.enable_shaping,
        alpha_drop=config.alpha_drop, beta_flush=config.beta_flush,
    )

    for ep in range(config.train_episodes):
        ep_start = time.time()
        entropy_coef = entropy_coef_at(ep, config.train_episodes)
        teacher_lambda = 0.0
        if config.candidate == "TP-SC":
            teacher_lambda = lambda_bf(
                ep, config.tp_teacher_lambda0, config.tp_teacher_decay_episodes
            )

        raw_state = env.reset(tx_stream=tx_pool_train[ep])

        needs_cond = config.candidate == "IA-SC"
        states: List[np.ndarray] = []
        cond_extras: List[np.ndarray] = []
        settle_actions: List[int] = []
        flush_actions: List[int] = []
        logps: List[float] = []
        values: List[float] = []
        rewards: List[float] = []
        dones: List[bool] = []
        t_settle: List[int] = []
        t_flush: List[int] = []

        episode_original_reward = 0.0
        done = False
        while not done:
            model_state = agent.state_fn(np.asarray(raw_state, dtype=np.float32)) \
                if state_fn is not None else np.asarray(raw_state, dtype=np.float32)
            settle_action, flush_action, logp, value = agent.act(raw_state)
            if teacher_lambda > 0.0:
                b, q = teacher_labels(env, settle_action)
                t_settle.append(int(b))
                t_flush.append(int(q))

            states.append(np.asarray(model_state, dtype=np.float32))
            if needs_cond:
                cond_extras.append(
                    extract_c_features(np.asarray(raw_state, dtype=np.float64)[:74], settle_action)
                )
            settle_actions.append(settle_action)
            flush_actions.append(flush_action)
            logps.append(logp)
            values.append(value)

            action = settle_action * (config.k + 1) + flush_action
            next_state, reward, done, _info = env.step(action)
            rewards.append(float(reward))
            dones.append(bool(done))
            episode_original_reward += float(reward)
            raw_state = next_state

        # ---- GAE (frozen semantics) ----
        with torch.no_grad():
            state_t = agent.prepare_state(raw_state)
            _, _, last_value_t = agent.distributions(state_t)
        last_value = 0.0 if dones[-1] else float(last_value_t.item())
        returns, advantages = compute_gae(
            rewards, values, dones, last_value,
            config.gamma, config.gae_lambda,
        )

        batch = {
            "states": states,
            "settle_actions": settle_actions,
            "flush_actions": flush_actions,
            "logp": logps,
            "returns": returns,
            "advantages": advantages,
        }
        if needs_cond:
            batch["cond_extras"] = cond_extras
        update_info = agent.update(
            batch,
            entropy_coef,
            teacher_settle=t_settle if teacher_lambda > 0.0 else None,
            teacher_flush=t_flush if teacher_lambda > 0.0 else None,
            teacher_lambda=teacher_lambda,
        )

        episode_1based = ep + 1
        m = env.get_metrics()
        settled = float(m["settled"])
        flushes = int(m["flushes"])
        train_money = settled - 10.0 * flushes
        assert money_identity_ok(settled, flushes, train_money)
        bank.record_training_return(episode_1based, episode_original_reward)

        run_log.append({
            "episode": episode_1based,
            "original_reward": episode_original_reward,
            "settled": settled,
            "flushes": flushes,
            "drops": int(m["drops"]),
            "train_money": train_money,
            "entropy_coef": entropy_coef,
            "teacher_lambda": teacher_lambda,
            "update": update_info,
            "seconds": time.time() - ep_start,
        })

        if episode_1based in CHECKPOINT_EPISODES:
            torch_rng_before = torch.get_rng_state().clone()
            bank.save_checkpoint(episode_1based, model, agent.optimizer)
            t_val = time.time()
            val_metrics = validate_on_panel(agent, val_panel, capacity=config.C)
            val_metrics["validation_seconds"] = time.time() - t_val
            bank.record_validation(episode_1based, val_metrics)
            rng_unchanged = bool(torch.equal(torch_rng_before, torch.get_rng_state()))
            rng_snapshot_at_validation.append(rng_unchanged)
            if log_every and episode_1based % log_every == 0:
                print(
                    f"[{config.candidate} s{config.seed}] ep {episode_1based:4d} "
                    f"orig_ret {episode_original_reward:8.2f} "
                    f"val_macro12_money {val_metrics['macro12_mean_money']:8.2f} "
                    f"({val_metrics['validation_seconds']:.1f}s)",
                    flush=True,
                )

    train_seconds = time.time() - train_start

    extra_designations: Dict[str, int] = {}
    if config.candidate == "TP-SC":
        late = [r for r in bank.validation_history if r["episode"] >= 300]
        best_late = max(late, key=lambda r: r["macro12_mean_money"])["episode"] if late else None
        extra_designations = {
            "teacher_boundary_episode": 250,
            "best_money_from_episode_300": best_late,
        }

    manifest = bank.finalize(extra_designations=extra_designations)

    # training metrics json
    with open(output_dir / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "config": config.to_json(),
            "runtime": runtime_snapshot(),
            "param_count": param_count,
            "training_pool_sha256": pool_sha,
            "training_pool_rows_loaded": TRAIN_POOL_ROWS,
            "training_pool_first_n_used": config.train_episodes,
            "training_seconds": train_seconds,
            "rng_unchanged_at_every_validation": all(rng_snapshot_at_validation)
            and len(rng_snapshot_at_validation) == len(CHECKPOINT_EPISODES),
            "episode_log": run_log,
        }, f, indent=2, sort_keys=True)

    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        f.write(config.to_json())

    summary = {
        "candidate": config.candidate,
        "seed": config.seed,
        "param_count": param_count,
        "training_seconds": train_seconds,
        "designations": manifest["designations"],
        "final_val_macro12_money": bank.validation_history[-1]["macro12_mean_money"],
        "rng_unchanged_at_every_validation": all(rng_snapshot_at_validation),
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary
