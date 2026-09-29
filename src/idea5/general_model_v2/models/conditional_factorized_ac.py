from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


def _apply_mask(logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return logits
    mask = mask.to(device=logits.device, dtype=torch.bool)
    return logits.masked_fill(~mask, -1e9)


class ConditionalFactorizedActorCritic(nn.Module):
    def __init__(
        self,
        state_size: int,
        hidden_size: int = 128,
        settle_embed_dim: int = 32,
        conditional_hidden_size: int = 128,
        num_flush_choices: int = 5,
    ) -> None:
        super().__init__()
        self.num_flush_choices = int(num_flush_choices)
        self.encoder = nn.Sequential(
            nn.Linear(int(state_size), int(hidden_size)),
            nn.ReLU(),
            nn.Linear(int(hidden_size), int(hidden_size)),
            nn.ReLU(),
        )
        self.settle_head = nn.Linear(int(hidden_size), 2)
        self.settle_embedding = nn.Embedding(2, int(settle_embed_dim))
        self.flush_head = nn.Sequential(
            nn.Linear(int(hidden_size) + int(settle_embed_dim), int(conditional_hidden_size)),
            nn.ReLU(),
            nn.Linear(int(conditional_hidden_size), self.num_flush_choices),
        )
        self.value_head = nn.Linear(int(hidden_size), 1)

    def encode(self, states: torch.Tensor) -> torch.Tensor:
        return self.encoder(states)

    def settle_logits_and_value(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encode(states)
        settle_logits = self.settle_head(h)
        values = self.value_head(h).squeeze(-1)
        return h, settle_logits, values

    def flush_logits(self, h: torch.Tensor, settle_actions: torch.Tensor) -> torch.Tensor:
        settle_emb = self.settle_embedding(settle_actions)
        return self.flush_head(torch.cat([h, settle_emb], dim=-1))


class ConditionalFactorizedACAgent:
    def __init__(
        self,
        state_size: int,
        config: Dict[str, Any],
        num_flush_choices: int = 5,
        flush_fractions: Sequence[float] | None = None,
    ) -> None:
        train_cfg = config["train"]
        cond_cfg = config["conditional"]
        self.device = torch.device(train_cfg["device"])
        self.num_flush_choices = int(num_flush_choices)
        if flush_fractions is None:
            flush_fractions = [i / max(1, self.num_flush_choices - 1) for i in range(self.num_flush_choices)]
        self.flush_fractions = [float(x) for x in flush_fractions]
        self.model = ConditionalFactorizedActorCritic(
            state_size=state_size,
            hidden_size=int(train_cfg["hidden_size"]),
            settle_embed_dim=int(cond_cfg["settle_embed_dim"]),
            conditional_hidden_size=int(cond_cfg["conditional_hidden_size"]),
            num_flush_choices=self.num_flush_choices,
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=float(train_cfg["learning_rate"]))

    @torch.no_grad()
    def act(
        self,
        state: np.ndarray,
        deterministic: bool = False,
        mask_info: Dict[str, np.ndarray] | None = None,
    ) -> Dict[str, float | int]:
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        h, settle_logits, value = self.model.settle_logits_and_value(state_t)
        settle_mask = None
        flush_mask = None
        if mask_info is not None:
            settle_mask = torch.tensor(mask_info["settle_mask"], dtype=torch.float32, device=self.device).unsqueeze(0)
            flush_mask = torch.tensor(mask_info["flush_mask"], dtype=torch.float32, device=self.device).unsqueeze(0)
        settle_logits = _apply_mask(settle_logits, settle_mask)
        settle_dist = Categorical(logits=settle_logits)
        if deterministic:
            settle = torch.argmax(settle_dist.logits, dim=-1)
            flush_logits = self.model.flush_logits(h, settle)
            flush_logits = _apply_mask(flush_logits, flush_mask)
            flush_dist = Categorical(logits=flush_logits)
            flush = torch.argmax(flush_dist.logits, dim=-1)
        else:
            settle = settle_dist.sample()
            flush_logits = self.model.flush_logits(h, settle)
            flush_logits = _apply_mask(flush_logits, flush_mask)
            flush_dist = Categorical(logits=flush_logits)
            flush = flush_dist.sample()

        logp = settle_dist.log_prob(settle) + flush_dist.log_prob(flush)
        settle_int = int(settle.item())
        flush_int = int(flush.item())
        flush_fraction = self.flush_fractions[flush_int]
        return {
            "action_id": settle_int * self.num_flush_choices + flush_int,
            "settle_action": settle_int,
            "flush_action": flush_int,
            "flush_fraction": float(flush_fraction),
            "logp": float(logp.item()),
            "value": float(value.item()),
        }

    @torch.no_grad()
    def value(self, state: np.ndarray) -> float:
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        _, _, value = self.model.settle_logits_and_value(state_t)
        return float(value.item())

    def get_logits(
        self,
        states: torch.Tensor,
        settle_actions_for_flush: torch.Tensor,
        settle_mask: torch.Tensor | None = None,
        flush_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h, settle_logits, _ = self.model.settle_logits_and_value(states)
        settle_logits = _apply_mask(settle_logits, settle_mask)
        flush_logits = self.model.flush_logits(h, settle_actions_for_flush)
        flush_logits = _apply_mask(flush_logits, flush_mask)
        return settle_logits, flush_logits

    def evaluate_actions(
        self,
        states: torch.Tensor,
        action_batch: Dict[str, torch.Tensor],
        idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, settle_logits, values = self.model.settle_logits_and_value(states)
        settle_actions = action_batch["settle_actions"][idx]
        flush_actions = action_batch["flush_actions"][idx]
        settle_mask = action_batch.get("settle_masks")
        flush_mask = action_batch.get("flush_masks")
        if settle_mask is not None:
            settle_mask = settle_mask[idx]
        if flush_mask is not None:
            flush_mask = flush_mask[idx]
        settle_logits = _apply_mask(settle_logits, settle_mask)
        settle_dist = Categorical(logits=settle_logits)
        flush_logits = self.model.flush_logits(h, settle_actions)
        flush_logits = _apply_mask(flush_logits, flush_mask)
        flush_dist = Categorical(logits=flush_logits)
        logp = settle_dist.log_prob(settle_actions) + flush_dist.log_prob(flush_actions)
        entropy = settle_dist.entropy() + flush_dist.entropy()
        return logp, values, entropy
