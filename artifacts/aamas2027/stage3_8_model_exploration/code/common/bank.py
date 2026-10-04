"""Checkpoint bank, VAL-v3 validation, and selection rules (frozen tie-breaks).

- A checkpoint bank retains every validation checkpoint (episodes 50..1000 step 50)
  plus the final episode checkpoint (=1000, already in the set); designations
  (best_by_money / best_by_value_accept_ratio / best_by_training_return, and the
  TP-SC diagnostics) are recorded in bank_manifest.json and may coincide with a
  numbered checkpoint (single file, multiple designations).
- best_by_money: highest equal-weight mean of per-regime mean Money on VAL-v3.
- best_by_value_accept_ratio: highest mean value acceptance ratio on VAL-v3.
- best_by_training_return: highest mean undiscounted ORIGINAL on-policy episode
  return over the preceding 50 training episodes, including the checkpoint
  episode (window [max(1, e-49), e] in 1-based episode numbers).
- All tie-breaks: earliest checkpoint episode.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .config import CHECKPOINT_EPISODES


class CheckpointBank:
    def __init__(self, run_dir: Path, checkpoint_episodes=CHECKPOINT_EPISODES):
        self.bank_dir = run_dir / "bank"
        self.bank_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_episodes = list(checkpoint_episodes)
        self.validation_history: List[Dict] = []       # one record per validation episode
        self.training_returns: Dict[int, float] = {}   # 1-based episode -> original reward

    # ---------------------------------------------------------------- storage
    def save_checkpoint(self, episode: int, model, optimizer=None) -> Path:
        path = self.bank_dir / f"ckpt_ep{episode:04d}.pth"
        torch.save(
            {
                "episode": episode,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            },
            path,
        )
        return path

    def record_training_return(self, episode_1based: int, original_reward: float) -> None:
        self.training_returns[episode_1based] = float(original_reward)

    def record_validation(self, episode: int, val_metrics: Dict) -> None:
        self.validation_history.append({"episode": episode, **val_metrics})

    # ---------------------------------------------------------------- selection
    def _argmax_episode(self, key_fn) -> int:
        best_ep, best_val = None, None
        for rec in self.validation_history:
            v = key_fn(rec)
            if best_val is None or v > best_val:   # strict > => earliest wins ties
                best_val, best_ep = v, rec["episode"]
        return int(best_ep)

    def best_by_money(self) -> int:
        return self._argmax_episode(lambda r: r["macro12_mean_money"])

    def best_by_value_accept_ratio(self) -> int:
        return self._argmax_episode(lambda r: r["macro12_value_accept_ratio"])

    def best_by_training_return(self) -> int:
        def window_mean(rec: Dict) -> float:
            e = rec["episode"]
            lo = max(1, e - 49)
            vals = [self.training_returns[i] for i in range(lo, e + 1)]
            return sum(vals) / len(vals)

        return self._argmax_episode(window_mean)

    def finalize(self, extra_designations: Optional[Dict[str, int]] = None) -> Dict:
        designations = {
            "best_by_money": self.best_by_money(),
            "best_by_value_accept_ratio": self.best_by_value_accept_ratio(),
            "best_by_training_return": self.best_by_training_return(),
        }
        if extra_designations:
            designations.update(extra_designations)
        manifest = {
            "checkpoint_episodes": self.checkpoint_episodes,
            "num_validation_checkpoints": len(self.checkpoint_episodes),
            "designations": designations,
            "validation_history": self.validation_history,
        }
        with open(self.bank_dir / "bank_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        return manifest

    def checkpoint_path(self, episode: int) -> Path:
        return self.bank_dir / f"ckpt_ep{episode:04d}.pth"
