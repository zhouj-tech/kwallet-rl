"""Fixed BF-T0.5 reference controller evaluation (overnight tooling).

Usage:
  python runtime/eval_bf.py --capacity 800 --split SCREEN-v3 --out runtime/eval_out/BF_C800__SCREEN-v3.json

Implements the frozen BF-T0.5 controller from the 74-dim benchmark state
(no model, no RNG):
  settlement = min over usable wallets with balance >= tx by (balance, index);
               no-op k when none feasible
  flush      = min over usable wallets, excluding the settlement choice, with
               balance < 0.5 * C/k by (balance, index); no-op k when none
Frozen-rule parity is enforced by reusing common.teacher label functions
(unit-tested line-by-line against the frozen bf_action adapter). This is a
standalone reference tool; the learned-policy deployment path (common/evaluate)
never imports the teacher module.

Run exactly once per capacity per panel, per the overnight spec.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SCFAC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCFAC_ROOT))

from common.config import (  # noqa: E402
    K,
    MAX_STEPS,
    MAX_TRANSACTION,
    SCFAC_ROOT as ROOT,
)
from common.evaluate import evaluate_panel, load_panel_from_release  # noqa: E402
from common.frozen_env import KWalletEnv  # noqa: E402
from common.teacher import bf_flush_label, bf_settlement_label, reset_counters  # noqa: E402


class BFController:
    """Deterministic BF-T0.5 policy shim exposing the agent argmax interface."""

    def __init__(self, capacity: float):
        self.wallet_size = float(capacity) / K

    def argmax_action(self, raw_state):
        x = np.asarray(raw_state, dtype=np.float64)
        balances = (x[:K] * self.wallet_size).tolist()
        frozen = (x[K:2 * K] > 0.5).tolist()
        tx = float(x[3 * K] * MAX_TRANSACTION)
        settle = int(bf_settlement_label(balances, [bool(f) for f in frozen], tx))
        flush = int(bf_flush_label(balances, [bool(f) for f in frozen], settle,
                                   wallet_size=self.wallet_size))
        return settle, flush


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--capacity", type=float, required=True)
    p.add_argument("--split", default="SCREEN-v3")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    reset_counters()
    panel = load_panel_from_release(ROOT / "data" / "scfac_design_v3", args.split)
    controller = BFController(args.capacity)

    t0 = time.time()
    res = evaluate_panel(controller, panel, capacity=args.capacity)
    seconds = time.time() - t0

    n_episodes = sum(int(res[r]["episodes"]) for r in res if r != "macro12_mean_money"
                     and r != "macro12_value_accept_ratio")
    out = {
        "controller": "BF-T0.5 (frozen fixed rule)",
        "capacity": args.capacity,
        "wallet_size": controller.wallet_size,
        "split": args.split,
        "episodes": n_episodes,
        "eval_seconds": seconds,
        "result": res,
    }
    out_path = Path(args.out) if args.out else (
        SCFAC_ROOT / "runtime" / "eval_out" /
        f"BF_C{int(args.capacity)}__{args.split}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"BF C{int(args.capacity)} {args.split}: "
          f"macro12_money={res['macro12_mean_money']:.4f} ({seconds:.1f}s) -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())