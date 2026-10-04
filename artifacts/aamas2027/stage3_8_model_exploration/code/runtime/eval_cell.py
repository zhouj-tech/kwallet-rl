"""Learned-policy panel evaluation for completed matrix cells (overnight tooling).

Usage:
  python runtime/eval_cell.py --cell outputs/SCM_C800_s123 --split SCREEN-v3 \
      [--designation best_by_money|best_by_value_accept_ratio|best_by_training_return|all]

Loads the cell's run_config (candidate, capacity), reads the bank manifest
designations, reloads the selected checkpoint(s), and evaluates the shared
deployment pipeline (deterministic argmax; zero BF calls) on the requested
SCFAC-DESIGN-v3 split at the cell's training capacity. Money identity is
asserted inside evaluate_episode for every episode.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SCFAC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCFAC_ROOT))

from common.config import SCFAC_ROOT as ROOT  # noqa: E402
from common.evaluate import (  # noqa: E402
    evaluate_panel,
    load_checkpoint_into_agent,
    load_panel_from_release,
)
from common.models import (  # noqa: E402
    build_ia_sc_model,
    build_sc_m_model,
    build_tp_sc_model,
)
from common.ppo import ConditionalFactorizedPPOAgent  # noqa: E402

DESIGNATIONS = {
    "best_by_money": "best_by_money",
    "best_by_value_accept_ratio": "best_by_value_accept_ratio",
    "best_by_training_return": "best_by_training_return",
}


def build_agent(candidate: str, capacity: float, ckpt: Path):
    w = float(capacity) / 24.0
    if candidate == "IA-SC":
        from common.features import augment_state

        model = build_ia_sc_model()
        load_checkpoint_into_agent(ckpt, model)
        return ConditionalFactorizedPPOAgent(
            model, state_fn=lambda s: augment_state(s, wallet_size=w), wallet_size=w)
    model = build_tp_sc_model() if candidate == "TP-SC" else build_sc_m_model()
    load_checkpoint_into_agent(ckpt, model)
    return ConditionalFactorizedPPOAgent(model, wallet_size=w)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cell", required=True)
    p.add_argument("--split", default="SCREEN-v3")
    p.add_argument("--designation", default="best_by_money")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cell = Path(args.cell)
    config = json.loads((cell / "run_config.json").read_text())
    candidate, capacity = config["candidate"], float(config["C"])
    manifest = json.loads((cell / "bank" / "bank_manifest.json").read_text())

    wanted = list(DESIGNATIONS) if args.designation == "all" else [args.designation]
    results = {}
    for designation in wanted:
        ep = int(manifest["designations"][DESIGNATIONS[designation]])
        agent = build_agent(candidate, capacity,
                            cell / "bank" / f"ckpt_ep{ep:04d}.pth")
        panel = load_panel_from_release(ROOT / "data" / "scfac_design_v3", args.split)
        t0 = time.time()
        res = evaluate_panel(agent, panel, capacity=capacity)
        seconds = time.time() - t0
        results[designation] = {
            "cell": cell.name, "candidate": candidate, "capacity": capacity,
            "split": args.split, "checkpoint_episode": ep,
            "eval_seconds": seconds, "result": res,
        }

    out_path = Path(args.out) if args.out else (
        SCFAC_ROOT / "runtime" / "eval_out" /
        f"{cell.name}__{args.split}__{args.designation}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    for designation, r in results.items():
        print(f"{designation}: ep{r['checkpoint_episode']} "
              f"macro12_money={r['result']['macro12_mean_money']:.4f} "
              f"({r['eval_seconds']:.1f}s) -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())