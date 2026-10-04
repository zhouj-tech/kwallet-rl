"""Frozen configuration and provenance for the SCFAC candidate experiments.

All values below are transcribed from:
  - MODEL_DESIGN.md (this directory) - authoritative task design
  - /Users/zhouzhou/Desktop/kwallet-rl/src/idea4/ac/code/run_conditional_factorized_ac_benchmark.py
    (historical SC-FAC trainer; frozen worktree, HEAD 281e02920a32e9878177593d62c7f2d841ed6078)
  - /Users/zhouzhou/Desktop/kwallet-aamas-b0/tools/aamas_stage2/adapter.py (BF-T0.5 + eval identities)
"""
from __future__ import annotations

import hashlib
import json
import platform
import random
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch

# ----------------------------------------------------------------------------
# Paths (read-only frozen evidence)
# ----------------------------------------------------------------------------
SCFAC_ROOT = Path(__file__).resolve().parent.parent          # .../scfac_candidates
SANDBOX_ROOT = SCFAC_ROOT.parent                             # .../aamas2027_explore
REPO_ROOT = SANDBOX_ROOT.parent.parent                       # kwallet-aamas-explore worktree

FROZEN_TRAINING_POOL = Path(
    "/Users/zhouzhou/Desktop/kwallet-aamas-artifacts/pools/"
    "MIX12_EQ_US_TLS_LNS_TLNS_TPLS_PLS_UB_TLB_LNB_TLNB_TPLB_PLB_master_T1000.npy"
)
FROZEN_TRAINING_POOL_SUMMARY = Path(
    "/Users/zhouzhou/Desktop/kwallet-aamas-artifacts/metadata/pools/"
    "MIX12_EQ_US_TLS_LNS_TLNS_TPLS_PLS_UB_TLB_LNB_TLNB_TPLB_PLB_master_T1000_summary.json"
)
FROZEN_VAL_POOL = Path(
    "/Users/zhouzhou/Desktop/kwallet-aamas-artifacts/pools/"
    "MIX12_EQ_US_TLS_LNS_TLNS_TPLS_PLS_UB_TLB_LNB_TLNB_TPLB_PLB_val_T1000.npy"
)
FROZEN_STATIC_EVAL_POOL_DIR = Path("/Users/zhouzhou/Desktop/kwallet-aamas-artifacts/pools")
NEW12_ROOT = Path("/Users/zhouzhou/benchmarks/A-EVAL-JA-C800-S123/data/streams/NEW12-v1")
D1_SEED_CAPACITY_CSV = Path(
    "/Users/zhouzhou/Desktop/kwallet-aamas-stage4-d1/artifacts/aamas2027/stage4/d1_h1_safe/"
    "D1_SEED_CAPACITY.csv"
)
D1_RECONCILED_CELLS_DIR = Path(
    "/Users/zhouzhou/Desktop/kwallet-aamas-stage4-d1/artifacts/aamas2027/stage4/d1_h1_safe/outputs"
)

# Provenance hashes of frozen sources we vendor or depend on
FROZEN_ENV_SOURCE = Path(
    "/Users/zhouzhou/Desktop/kwallet-rl/src/idea3/context_attention/kwallet_ctx_attn_fair_benchmark.py"
)
FROZEN_TRAINER_SOURCE = Path(
    "/Users/zhouzhou/Desktop/kwallet-rl/src/idea4/ac/code/run_conditional_factorized_ac_benchmark.py"
)
FROZEN_GENERATOR_SHA256 = "d702061f6b9556979a453a7014dfa06659deb7b7e23904e26133fd0af1f623b8"
FROZEN_WORKTREE_HEAD = "281e02920a32e9878177593d62c7f2d841ed6078"

DESIGN_DOC = SCFAC_ROOT / "MODEL_DESIGN.md"
DATA_V3_ROOT = SCFAC_ROOT / "data" / "scfac_design_v3"

# ----------------------------------------------------------------------------
# E0 environment (frozen; C=800 per 2026-10 AAMAS setting)
# ----------------------------------------------------------------------------
K = 24
F = 3
MAX_TRANSACTION = 1000
MAX_STEPS = 1000
C = 800.0
ALPHA_DROP = 0.02
BETA_FLUSH = 0.01
NUM_ACTIONS = (K + 1) ** 2
BASE_STATE_SIZE = 3 * K + 2          # 74
WALLET_SIZE = C / K                  # w
MONEY_TAU = 10.0                     # Money = settled - 10 * executed_flushes
VALUE_ACCEPT_DENOM = MAX_TRANSACTION * MAX_STEPS

# ----------------------------------------------------------------------------
# PPO protocol (historical SC-FAC defaults; frozen)
# ----------------------------------------------------------------------------
ENTROPY_COEF_START = 0.03
ENTROPY_COEF_END = 0.003
PPO_LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
PPO_EPS_CLIP = 0.2
CRITIC_COEF = 0.5
UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 256
GRAD_CLIP_NORM = 1.0
TRAIN_EPISODES = 1000
TRAIN_POOL_ROWS = 3000  # master pool rows loaded; first 1000 used in fixed order

# Checkpoint bank / validation
CHECKPOINT_EVERY = 50
CHECKPOINT_EPISODES = tuple(range(CHECKPOINT_EVERY, TRAIN_EPISODES + 1, CHECKPOINT_EVERY))  # 20

# ----------------------------------------------------------------------------
# Candidate-specific
# ----------------------------------------------------------------------------
IA_EXTRA_STATE_FEATURES = 9          # q(x)
IA_COND_EXTRA_FEATURES = 4           # c(x, a_s)
TP_TEACHER_LAMBDA_0 = 0.01
TP_TEACHER_DECAY_EPISODES = 250
TP_FLUSH_THRESHOLD = 0.5             # BF-T0.5 flush threshold on balance < 0.5 * C/k

# ----------------------------------------------------------------------------
# SCFAC-DESIGN-v3 release
# ----------------------------------------------------------------------------
SEED_RULE_PREFIX = "SCFAC-DESIGN-v3"
SEED_RULE_DATE = "20261003"
SEED_RULE_NUM_BYTES = 8
V3_SPLITS: Dict[str, int] = {        # split -> episodes per regime
    "VAL-v3": 20,
    "SCREEN-v3": 40,
    "CHECK-v3": 60,
    "FULL-DEV-v3": 200,
}

# ----------------------------------------------------------------------------
# Runtime pinning (matches frozen D1/B0 runtime snapshot)
# ----------------------------------------------------------------------------
TORCH_NUM_THREADS = 4


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    """SHA256 over canonical little-endian bytes (established pool fingerprint convention)."""
    return hashlib.sha256(np.ascontiguousarray(arr.astype("<i4", copy=False)).tobytes(order="C")).hexdigest()


def episode_fingerprint(row: np.ndarray) -> str:
    return sha256_array(row)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def runtime_snapshot() -> Dict:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


@dataclass
class CommonRunConfig:
    """Effective config for one training run (serialized into run_config.json)."""
    candidate: str                       # "SC-M" | "IA-SC" | "TP-SC"
    seed: int
    C: float = C
    k: int = K
    F: int = F
    max_transaction: int = MAX_TRANSACTION
    max_steps: int = MAX_STEPS
    enable_shaping: bool = False
    alpha_drop: float = ALPHA_DROP
    beta_flush: float = BETA_FLUSH
    train_episodes: int = TRAIN_EPISODES
    checkpoint_every: int = CHECKPOINT_EVERY
    entropy_coef_start: float = ENTROPY_COEF_START
    entropy_coef_end: float = ENTROPY_COEF_END
    ppo_lr: float = PPO_LR
    gamma: float = GAMMA
    gae_lambda: float = GAE_LAMBDA
    ppo_eps_clip: float = PPO_EPS_CLIP
    critic_coef: float = CRITIC_COEF
    update_epochs: int = UPDATE_EPOCHS
    minibatch_size: int = MINIBATCH_SIZE
    grad_clip_norm: float = GRAD_CLIP_NORM
    train_pool_sha256: str = ""
    val_panel: str = "VAL-v3"
    torch_num_threads: int = TORCH_NUM_THREADS
    # candidate specifics
    tp_teacher_lambda0: float = TP_TEACHER_LAMBDA_0
    tp_teacher_decay_episodes: int = TP_TEACHER_DECAY_EPISODES
    tp_flush_threshold: float = TP_FLUSH_THRESHOLD
    ia_extra_state_features: int = IA_EXTRA_STATE_FEATURES
    ia_cond_extra_features: int = IA_COND_EXTRA_FEATURES
    # bookkeeping (filled by driver)
    training_pool_first_n: int = TRAIN_EPISODES

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)
