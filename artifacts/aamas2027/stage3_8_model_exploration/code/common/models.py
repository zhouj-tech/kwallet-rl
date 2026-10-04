"""SCFAC candidate models: SC-M, IA-SC, TP-SC.

Architecture (frozen SC-FAC "conditional factorized" actor-critic, condH256):
  shared encoder : Linear(state_size, 128) ReLU -> Linear(128, 128) ReLU
  settle head    : Linear(128, 25)                     (25 = (k+1) settlement choices)
  settle embed   : Embedding(25, 32)
  flush head     : Linear(160, 256) ReLU -> Linear(256, 25)
                   (160 = 128 encoder + 32 embedding; IA adds 4 -> 164)
  value head     : Linear(128, 1)

Parameter counts (verified in tests):
  SC-M  / TP-SC : state 74, flush cond 160 -> 77907
  IA-SC         : state 83, flush cond 164 -> 80083

Paired initialization (IA-SC): construct SC-M first, then build IA-SC by copying
all overlapping weight blocks and zero-initializing the columns that consume the
9 added state features (encoder layer 1, columns 74..82) and the 4 added flush
condition features (flush head layer 1, columns 160..163). This makes the IA
policy functionally identical to SC-M at initialization.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .config import BASE_STATE_SIZE, IA_COND_EXTRA_FEATURES, IA_EXTRA_STATE_FEATURES

HIDDEN = 128
SETTLE_CHOICES = 25
EMBED_DIM = 32
FLUSH_HIDDEN = 256
SC_FLUSH_COND_SIZE = HIDDEN + EMBED_DIM            # 160
IA_FLUSH_COND_SIZE = HIDDEN + EMBED_DIM + IA_COND_EXTRA_FEATURES  # 164
SC_STATE_SIZE = BASE_STATE_SIZE                    # 74
IA_STATE_SIZE = BASE_STATE_SIZE + IA_EXTRA_STATE_FEATURES         # 83


class ConditionalFactorizedActorCritic(nn.Module):
    """Frozen SC-FAC network. state_size/flush_cond_size parametrized to cover SC-M and IA-SC."""

    def __init__(self, state_size: int, flush_cond_size: int):
        super().__init__()
        self.state_size = state_size
        self.flush_cond_size = flush_cond_size
        self.shared_encoder = nn.Sequential(
            nn.Linear(state_size, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
        )
        self.settle_head = nn.Linear(HIDDEN, SETTLE_CHOICES)
        self.settle_embedding = nn.Embedding(SETTLE_CHOICES, EMBED_DIM)
        self.conditional_flush_head = nn.Sequential(
            nn.Linear(flush_cond_size, FLUSH_HIDDEN),
            nn.ReLU(),
            nn.Linear(FLUSH_HIDDEN, SETTLE_CHOICES),
        )
        self.value_head = nn.Linear(HIDDEN, 1)

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.shared_encoder(state)

    def settle_logits(self, state: torch.Tensor) -> torch.Tensor:
        return self.settle_head(self.encode(state))

    def flush_logits(self, state: torch.Tensor, settle_action: torch.Tensor,
                     cond_extra: torch.Tensor = None) -> torch.Tensor:
        """Conditional flush logits: [h, emb] (+ [c(x, a_s)] for IA-SC, 164-dim input)."""
        h = self.encode(state)
        emb = self.settle_embedding(settle_action)
        parts = [h, emb]
        if self.flush_cond_size == SC_FLUSH_COND_SIZE + IA_COND_EXTRA_FEATURES:
            assert cond_extra is not None, "IA flush head requires c(x, a_s) extra features"
            parts.append(cond_extra)
        cond = torch.cat(parts, dim=-1)
        return self.conditional_flush_head(cond)

    def value(self, state: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.encode(state)).squeeze(-1)

    def forward(self, state: torch.Tensor, settle_action: torch.Tensor):
        return self.settle_logits(state), self.flush_logits(state, settle_action), self.value(state)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_sc_m_model() -> ConditionalFactorizedActorCritic:
    return ConditionalFactorizedActorCritic(SC_STATE_SIZE, SC_FLUSH_COND_SIZE)


def build_tp_sc_model() -> ConditionalFactorizedActorCritic:
    return ConditionalFactorizedActorCritic(SC_STATE_SIZE, SC_FLUSH_COND_SIZE)


def build_ia_sc_model() -> ConditionalFactorizedActorCritic:
    return ConditionalFactorizedActorCritic(IA_STATE_SIZE, IA_FLUSH_COND_SIZE)


def init_ia_from_sc(ia_model: ConditionalFactorizedActorCritic,
                    sc_model: ConditionalFactorizedActorCritic) -> None:
    """Paired initialization: copy overlapping blocks; zero the new input columns.

    - encoder[0].weight: shape (128, 83); columns 0..73 copied from SC-M, columns 74..82 zero.
    - conditional_flush_head[0].weight: shape (256, 164); columns 0..159 copied, 160..163 zero.
    - every other parameter copied verbatim.
    After this, for any valid base state x: pi_IA([x, q(x)]) == pi_SC(x) and values match
    (the c(x, a_s) columns are zero, so any cond_extra contributes nothing at init).
    """
    ia_sd: Dict[str, torch.Tensor] = ia_model.state_dict()
    sc_sd: Dict[str, torch.Tensor] = sc_model.state_dict()

    enc_w_key = "shared_encoder.0.weight"
    flush_w_key = "conditional_flush_head.0.weight"

    for key, sc_t in sc_sd.items():
        if key == enc_w_key:
            ia_sd[key] = sc_t.clone()                       # overlapping columns
            ia_sd[key] = torch.cat([sc_t.clone(),
                                    torch.zeros(sc_t.shape[0], IA_STATE_SIZE - SC_STATE_SIZE)],
                                   dim=1)
        elif key == flush_w_key:
            ia_sd[key] = torch.cat([sc_t.clone(),
                                    torch.zeros(sc_t.shape[0], IA_FLUSH_COND_SIZE - SC_FLUSH_COND_SIZE)],
                                   dim=1)
        else:
            assert ia_sd[key].shape == sc_t.shape, \
                f"shape mismatch for {key}: {ia_sd[key].shape} vs {sc_t.shape}"
            ia_sd[key] = sc_t.clone()

    ia_model.load_state_dict(ia_sd)


def paired_models():
    """(sc_model, ia_model) with IA paired-initialized from SC."""
    sc = build_sc_m_model()
    ia = build_ia_sc_model()
    init_ia_from_sc(ia, sc)
    return sc, ia


def build_paired_ia_model() -> ConditionalFactorizedActorCritic:
    """Production paired-init sequence used by the training driver (MODEL_DESIGN C1-F).

    Caller contract: the global seed must already be set exactly as for the matched
    SC-M run (torch/np/random). Sequence:
      1. build the SC-M reference initialization (consumes the identical torch RNG
         draws a matched SC-M run consumes for its model init);
      2. capture the torch RNG state immediately after that SC-M initialization;
      3. build IA-SC;
      4. init_ia_from_sc so all original weight blocks are copied exactly and all
         added IA feature columns are zero;
      5. restore the captured post-SC-M torch RNG state before training begins.
    Result: (a) initial policy/value function identical to matched SC-M before
    learning; (b) downstream torch stochastic stream matches the matched SC-M run
    as closely as possible.
    """
    sc_ref = build_sc_m_model()
    rng_after_sc = torch.get_rng_state().clone()
    ia = build_ia_sc_model()
    init_ia_from_sc(ia, sc_ref)
    torch.set_rng_state(rng_after_sc)
    return ia
