"""Generate the SCFAC-DESIGN-v3 release (VAL-v3 / SCREEN-v3 / CHECK-v3 / FULL-DEV-v3)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.config import DATA_V3_ROOT, FROZEN_GENERATOR_SHA256, sha256_file
from common.release import RELEASE_BUILDER  # noqa: F401  (imports provenance constants)
from common.release import build_release


def main() -> None:
    p = argparse.ArgumentParser(description="Build SCFAC-DESIGN-v3 release")
    p.add_argument("--force", action="store_true", help="rebuild even if MANIFEST exists")
    p.add_argument("--output", type=str, default=str(DATA_V3_ROOT))
    args = p.parse_args()

    gen_path = Path(__file__).resolve().parent / "common" / "frozen_kwallet_ideaextra_generator.py"
    actual_gen_sha = sha256_file(gen_path)
    if actual_gen_sha != FROZEN_GENERATOR_SHA256:
        raise RuntimeError(
            f"Vendored generator hash mismatch: {actual_gen_sha} != {FROZEN_GENERATOR_SHA256}"
        )
    manifest = build_release(Path(args.output), force=args.force)
    print(json.dumps({k: v for k, v in manifest.items() if k != "splits"}, indent=2, sort_keys=True))
    print("split sha256:")
    for split, meta in manifest["splits"].items():
        print(f"  {split}: {meta['sha256']} ({meta['total_episodes']} episodes)")


if __name__ == "__main__":
    main()
