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


class FactorizedActorCritic(nn.Module):
    def __init__(
        self,
        state_size: int,
        hidden_size: int = 128,
        num_flush_choices: int = 5,
    ) -> None:
        super().__init__()
        self.num_flush_choices = int(num_flush_choices)
        self.shared = nn.Sequential(
            nn.Linear(int(state_size), int(hidden_size)),
            nn.ReLU(),
            nn.Linear(int(hidden_size), int(hidden_size)),
            nn.ReLU(),
        )
        self.settle_head = nn.Linear(int(hidden_size), 2)
        self.flush_head = nn.Linear(int(hidden_size), self.num_flush_choices)
        self.value_head = nn.Linear(int(hidden_size), 1)

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.shared(states)
        settle_logits = self.settle_head(h)
        flush_logits = self.flush_head(h)
        values = self.value_head(h).squeeze(-1)
        return settle_logits, flush_logits, values


class FactorizedACAgent:
    def __init__(
        self,
        state_size: int,
        config: Dict[str, Any],
        num_flush_choices: int = 5,
        flush_fractions: Sequence[float] | None = None,
    ) -> None:
        train_cfg = config["train"]
        self.device = torch.device(train_cfg["device"])
        self.num_flush_choices = int(num_flush_choices)
        if flush_fractions is None:
            flush_fractions = [i / max(1, self.num_flush_choices - 1) for i in range(self.num_flush_choices)]
        self.flush_fractions = [float(x) for x in flush_fractions]
        self.model = FactorizedActorCritic(
            state_size=state_size,
            hidden_size=int(train_cfg["hidden_size"]),
            num_flush_choices=self.num_flush_choices,
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=float(train_cfg["learning_rate"]))

    def distributions(
        self,
        states: torch.Tensor,
        settle_mask: torch.Tensor | None = None,
        flush_mask: torch.Tensor | None = None,
    ) -> Tuple[Categorical, Categorical, torch.Tensor]:
        settle_logits, flush_logits, values = self.model(states)
        settle_logits = _apply_mask(settle_logits, settle_mask)
        flush_logits = _apply_mask(flush_logits, flush_mask)
        return Categorical(logits=settle_logits), Categorical(logits=flush_logits), values

    @torch.no_grad()
    def act(
        self,
        state: np.ndarray,
        deterministic: bool = False,
        mask_info: Dict[str, np.ndarray] | None = None,
    ) -> Dict[str, float | int]:
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        settle_mask = None
        flush_mask = None
        if mask_info is not None:
            settle_mask = torch.tensor(mask_info["settle_mask"], dtype=torch.float32, device=self.device).unsqueeze(0)
            flush_mask = torch.tensor(mask_info["flush_mask"], dtype=torch.float32, device=self.device).unsqueeze(0)
        settle_dist, flush_dist, value = self.distributions(
            state_t,
            settle_mask=settle_mask,
            flush_mask=flush_mask,
        )
        settle = torch.argmax(settle_dist.logits, dim=-1) if deterministic else settle_dist.sample()
        flush = torch.argmax(flush_dist.logits, dim=-1) if deterministic else flush_dist.sample()
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
        _, _, value = self.model(state_t)
        return float(value.item())

    def evaluate_actions(
        self,
        states: torch.Tensor,
        action_batch: Dict[str, torch.Tensor],
        idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        settle_mask = action_batch.get("settle_masks")
        flush_mask = action_batch.get("flush_masks")
        if settle_mask is not None:
            settle_mask = settle_mask[idx]
        if flush_mask is not None:
            flush_mask = flush_mask[idx]
        settle_dist, flush_dist, values = self.distributions(
            states,
            settle_mask=settle_mask,
            flush_mask=flush_mask,
        )
        settle_actions = action_batch["settle_actions"][idx]
        flush_actions = action_batch["flush_actions"][idx]
        logp = settle_dist.log_prob(settle_actions) + flush_dist.log_prob(flush_actions)
        entropy = settle_dist.entropy() + flush_dist.entropy()
        return logp, values, entropy
