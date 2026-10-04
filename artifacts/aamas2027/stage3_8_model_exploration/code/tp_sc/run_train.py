"""TP-SC training entry (candidate TP-SC, teacher prior with decaying lambda)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import SCFAC_ROOT, CommonRunConfig
from common.train import train_candidate


def main() -> None:
    p = argparse.ArgumentParser(description="Train TP-SC (SCFAC candidate)")
    p.add_argument("--seed", type=int, default=323)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--capacity", type=float, default=800.0,
                   help="run capacity C (800 or 1200); threads through training, validation and naming")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--release-root", type=str, default=str(SCFAC_ROOT / "data" / "scfac_design_v3"))
    args = p.parse_args()

    out_dir = Path(args.output) if args.output else (
        SCFAC_ROOT / "outputs" / f"TPSC_C{int(args.capacity)}_s{args.seed}"
    )
    config = CommonRunConfig(candidate="TP-SC", seed=args.seed, C=args.capacity,
                             train_episodes=args.episodes)
    summary = train_candidate(config, out_dir, release_root=Path(args.release_root))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
