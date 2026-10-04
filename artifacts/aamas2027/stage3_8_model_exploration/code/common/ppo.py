"""Historical SC-FAC PPO (conditional factorized actor-critic), exact frozen semantics.

Provenance: /Users/zhouzhou/Desktop/kwallet-rl/src/idea4/ac/code/run_conditional_factorized_ac_benchmark.py
(frozen worktree HEAD 281e02920a32e9878177593d62c7f2d841ed6078), class
ConditionalFactorizedPPOAgent + compute_gae. The only extensions are:
  - `state_fn`: identity for SC-M / TP-SC, base-state -> 83-dim augmentation for IA-SC.
  - optional teacher-prior loss (TP-SC): loss += lambda_BF(e) * L_teacher, where
    L_teacher = mean over the minibatch of
        -log pi_s(b(x)|x) - log pi_f(q_BF(x, a_s)|x, a_s),
    computed from rollout-time teacher labels. When lambda == 0 the teacher loss
    is exactly zero (teacher labels are not even consulted).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .config import (
    BASE_STATE_SIZE,
    CRITIC_COEF,
    ENTROPY_COEF_END,
    ENTROPY_COEF_START,
    GAMMA,
    GAE_LAMBDA,
    GRAD_CLIP_NORM,
    MINIBATCH_SIZE,
    PPO_EPS_CLIP,
    PPO_LR,
    TRAIN_EPISODES,
    UPDATE_EPOCHS,
)


def entropy_coef_at(episode_zero_based: int, episodes: int = TRAIN_EPISODES) -> float:
    """Historical linear schedule: start + (end-start) * min(ep / episodes, 1)."""
    return ENTROPY_COEF_START + (ENTROPY_COEF_END - ENTROPY_COEF_START) * min(
        episode_zero_based / max(1, episodes), 1.0
    )


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    last_value: float,
    gamma: float = GAMMA,
    gae_lambda: float = GAE_LAMBDA,
) -> Tuple[List[float], List[float]]:
    """Verbatim frozen GAE."""
    advantages: List[float] = []
    gae = 0.0
    next_value = float(last_value)
    for t in reversed(range(len(rewards))):
        nonterminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages.insert(0, float(gae))
        next_value = values[t]
    returns = [float(adv + val) for adv, val in zip(advantages, values)]
    return returns, advantages


class ConditionalFactorizedPPOAgent:
    def __init__(
        self,
        model: nn.Module,
        state_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        lr: float = PPO_LR,
        gamma: float = GAMMA,
        gae_lambda: float = GAE_LAMBDA,
        eps_clip: float = PPO_EPS_CLIP,
        critic_coef: float = CRITIC_COEF,
        update_epochs: int = UPDATE_EPOCHS,
        minibatch_size: int = MINIBATCH_SIZE,
        grad_clip_norm: float = GRAD_CLIP_NORM,
        wallet_size: Optional[float] = None,
    ):
        self.model = model
        self.state_fn = state_fn if state_fn is not None else (lambda s: s)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.critic_coef = critic_coef
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.grad_clip_norm = grad_clip_norm
        # IA-SC c(x, a_s) features are computed in terms of the wallet capacity
        # w = C/k; None -> frozen config default (C=800). Threaded from the run's
        # capacity so C1200 cells use w = 1200/24 = 50.
        self.wallet_size = wallet_size

    def _wallet_size(self) -> float:
        from .config import WALLET_SIZE

        return WALLET_SIZE if self.wallet_size is None else float(self.wallet_size)

    # ------------------------------------------------------------------ policy
    def _needs_cond_extra(self) -> bool:
        return getattr(self.model, "flush_cond_size", 0) == 164

    def cond_extra_from(self, raw_state: np.ndarray, settle_action: int) -> Optional[torch.Tensor]:
        """c(x, a_s) as a (1, 4) tensor for IA-SC; None for SC-M / TP-SC."""
        if not self._needs_cond_extra():
            return None
        from .features import extract_c_features

        c = extract_c_features(
            np.asarray(raw_state, dtype=np.float64)[:74], settle_action,
            wallet_size=self._wallet_size(),
        )
        return torch.tensor(c, dtype=torch.float32).unsqueeze(0)

    def distributions(self, state: torch.Tensor,
                      cond_extra: Optional[torch.Tensor] = None):
        """(settle_dist, flush_dist, value) exactly as in the frozen trainer.

        The flush distribution is conditioned on the argmax settlement (frozen
        diagnostic path semantics); for IA-SC, c(x, argmax_settle) is appended.
        """
        settle_logits = self.model.settle_logits(state)
        settle_dist = Categorical(logits=settle_logits)
        settle_argmax = torch.argmax(settle_logits, dim=-1)
        if cond_extra is None and self._needs_cond_extra():
            # Derive c(x, argmax_settle) from the state's base 74 dims (frozen
            # diagnostic-path semantics; identical to act()'s conditioning).
            from .features import extract_c_features

            base = state.detach().cpu().numpy().reshape(-1)[:BASE_STATE_SIZE].astype(np.float64)
            c = extract_c_features(base, int(settle_argmax.item()),
                                   wallet_size=self._wallet_size())
            cond_extra = torch.tensor(c, dtype=torch.float32).unsqueeze(0)
        flush_logits = self.model.flush_logits(state, settle_argmax, cond_extra)
        flush_dist = Categorical(logits=flush_logits)
        value = self.model.value(state)
        return settle_dist, flush_dist, value

    def prepare_state(self, raw_state: np.ndarray) -> torch.Tensor:
        s = self.state_fn(np.asarray(raw_state, dtype=np.float32))
        return torch.tensor(np.asarray(s, dtype=np.float32), dtype=torch.float32).unsqueeze(0)

    def act(self, raw_state: np.ndarray) -> Tuple[int, int, float, float]:
        state = self.prepare_state(raw_state)
        settle_logits = self.model.settle_logits(state)
        settle_dist = Categorical(logits=settle_logits)
        settle_argmax = torch.argmax(settle_logits, dim=-1)
        cond_extra = self.cond_extra_from(raw_state, int(settle_argmax.item()))
        flush_logits = self.model.flush_logits(state, settle_argmax, cond_extra)
        flush_dist = Categorical(logits=flush_logits)
        value = self.model.value(state)
        settle_action = settle_dist.sample()
        flush_action = flush_dist.sample()
        logp = settle_dist.log_prob(settle_action) + flush_dist.log_prob(flush_action)
        return (
            int(settle_action.item()),
            int(flush_action.item()),
            float(logp.item()),
            float(value.item()),
        )

    def argmax_action(self, raw_state: np.ndarray) -> Tuple[int, int]:
        """Deterministic evaluation: settlement argmax, then conditional flush argmax."""
        state = self.prepare_state(raw_state)
        with torch.no_grad():
            settle_logits = self.model.settle_logits(state)
            settle_action = int(torch.argmax(settle_logits, dim=-1).item())
            cond_extra = self.cond_extra_from(raw_state, settle_action)
            flush_logits = self.model.flush_logits(
                state, torch.tensor([settle_action], dtype=torch.int64), cond_extra
            )
            flush_action = int(torch.argmax(flush_logits, dim=-1).item())
        return settle_action, flush_action

    # ------------------------------------------------------------------ update
    def update(
        self,
        batch: Dict[str, list],
        entropy_coef: float,
        teacher_settle: Optional[List[int]] = None,
        teacher_flush: Optional[List[int]] = None,
        teacher_lambda: float = 0.0,
    ) -> Dict[str, float]:
        states = torch.tensor(np.array(batch["states"]), dtype=torch.float32)
        settle_actions = torch.tensor(batch["settle_actions"], dtype=torch.int64)
        flush_actions = torch.tensor(batch["flush_actions"], dtype=torch.int64)
        old_logp = torch.tensor(batch["logp"], dtype=torch.float32)
        returns = torch.tensor(batch["returns"], dtype=torch.float32)
        advantages = torch.tensor(batch["advantages"], dtype=torch.float32)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        cond_extras = None
        if self._needs_cond_extra():
            assert "cond_extras" in batch, "IA-SC batches must include per-step c(x, a_s)"
            cond_extras = torch.tensor(np.array(batch["cond_extras"]), dtype=torch.float32)

        n = states.shape[0]
        mb_size = min(self.minibatch_size, n)

        if teacher_settle is not None and teacher_lambda > 0.0:
            t_settle = torch.tensor(teacher_settle, dtype=torch.int64)
            t_flush = torch.tensor(teacher_flush, dtype=torch.int64)
        else:
            t_settle = None
            t_flush = None

        last = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0,
                "entropy": 0.0, "teacher_loss": 0.0}

        for _ in range(self.update_epochs):
            order = torch.randperm(n)
            for start in range(0, n, mb_size):
                idx = order[start:start + mb_size]
                settle_logits, values = (
                    self.model.settle_logits(states[idx]),
                    self.model.value(states[idx]),
                )
                settle_dist = Categorical(logits=settle_logits)
                cond_mb = cond_extras[idx] if cond_extras is not None else None
                flush_logits = self.model.flush_logits(
                    states[idx], settle_actions[idx], cond_extra=cond_mb
                )
                flush_dist = Categorical(logits=flush_logits)

                logp = (
                    settle_dist.log_prob(settle_actions[idx])
                    + flush_dist.log_prob(flush_actions[idx])
                )
                ratio = torch.exp(logp - old_logp[idx])
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages[idx]

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (returns[idx] - values).pow(2).mean()
                entropy = settle_dist.entropy().mean() + flush_dist.entropy().mean()

                loss = (
                    policy_loss
                    + self.critic_coef * value_loss
                    - entropy_coef * entropy
                )

                teacher_loss_val = 0.0
                if t_settle is not None:
                    t_s_logp = settle_dist.log_prob(t_settle[idx])
                    t_f_logp = flush_dist.log_prob(t_flush[idx])
                    teacher_loss = -(t_s_logp + t_f_logp).mean()
                    loss = loss + teacher_lambda * teacher_loss
                    teacher_loss_val = float(teacher_loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

                last = {
                    "loss": float(loss.item()),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.item()),
                    "teacher_loss": teacher_loss_val,
                }
        return last
