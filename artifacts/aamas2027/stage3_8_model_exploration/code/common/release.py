"""SCFAC-DESIGN-v3 release builder (VAL-v3 / SCREEN-v3 / CHECK-v3 / FULL-DEV-v3).

Mandatory design points (MODEL_DESIGN.md section 8):
  - Twelve frozen regime specifications and frozen generator semantics
    (vendored byte-identical copy of the frozen generator, SHA256-verified).
  - Seed rule: first 8 bytes big-endian of
      SHA256("SCFAC-DESIGN-v3|<split>|<regime>|<zero-based-episode>|20261003|<collision-counter>")
    Counter starts at 0 and increments ONLY on a verified collision; every
    increment is logged.
  - Four separate, mutually nonoverlapping pools:
      VAL-v3 20 eps/regime, SCREEN-v3 40, CHECK-v3 60, FULL-DEV-v3 200.
  - Seed AND episode-fingerprint disjointness against the historical training
    pool, historical validation pool, historical evaluation pools, NEW12, and
    every other SCFAC-DESIGN-v3 split.
  - Immutable release manifests + SHA256 sums; immutable once written.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .config import (
    DATA_V3_ROOT,
    FROZEN_GENERATOR_SHA256,
    SEED_RULE_DATE,
    V3_SPLITS,
    sha256_file,
)
from .frozen_kwallet_ideaextra_generator import (
    DEFAULT_CONFIG,
    REGIME_ORDER,
    REGIME_SPECS,
    generate_regime_episode,
)
from .pools import all_historical_pools
from .seedrule import (
    MASTER_BASE_SEED,
    MASTER_SHUFFLE_SEED,
    NEW12_BASE_SEED,
    STATIC_EVAL_BASE_SEED,
    VAL_BASE_SEED,
    CALIBRATION_SEED,
    design_v3_seed,
    historical_seed_namespace,
)

CALIBRATION_SAMPLE_SIZE = DEFAULT_CONFIG["calibration_sample_size"]  # 200000
CALIBRATION_SEED_CONST = 1234567

RELEASE_BUILDER = Path(__file__).resolve()


def episode_fingerprint(row: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(row.astype("<i4", copy=False)).tobytes(order="C")
    ).hexdigest()


def build_release(release_root: Path = DATA_V3_ROOT, force: bool = False,
                  log=print) -> Dict:
    release_root = Path(release_root)
    done_marker = release_root / "MANIFEST.json"
    if done_marker.exists() and not force:
        log(f"[release] already exists at {release_root} (use force=True to rebuild)")
        return json.loads(done_marker.read_text())

    # ---------------- historical reference sets ----------------
    log("[release] loading historical pools for overlap audit ...")
    historical_fp: Dict[str, str] = {}
    historical_pools_meta: Dict[str, Dict] = {}
    for name, arr in all_historical_pools():
        meta = {"episodes": int(arr.shape[0]), "steps": int(arr.shape[1])}
        pool_path = {
            "historical_train_master": "/Users/zhouzhou/Desktop/kwallet-aamas-artifacts/pools/MIX12_EQ_US_TLS_LNS_TLNS_TPLS_PLS_UB_TLB_LNB_TLNB_TPLB_PLB_master_T1000.npy",
            "historical_val": "/Users/zhouzhou/Desktop/kwallet-aamas-artifacts/pools/MIX12_EQ_US_TLS_LNS_TLNS_TPLS_PLS_UB_TLB_LNB_TLNB_TPLB_PLB_val_T1000.npy",
        }.get(name)
        if pool_path:
            meta["sha256"] = sha256_file(Path(pool_path))
            meta["path"] = pool_path
        historical_pools_meta[name] = meta
        for i in range(arr.shape[0]):
            historical_fp[episode_fingerprint(arr[i])] = f"{name}[{i}]"

    seed_ns = historical_seed_namespace()
    historical_seeds: set = set(seed_ns["seeds"]) | set(seed_ns["reserved_constants"])

    log(f"[release] historical fingerprints: {len(historical_fp)}, "
        f"historical seeds: {len(seed_ns['seeds'])}")

    # ---------------- generate v3 ----------------
    counter_increments: List[Dict] = []
    seed_collisions: List[Dict] = []
    fp_collisions: List[Dict] = []

    v3_seeds: Dict[str, set] = {s: set() for s in V3_SPLITS}
    v3_fp: Dict[str, Dict[str, str]] = {s: {} for s in V3_SPLITS}   # fp -> "split/regime/ep"
    all_v3_seeds: set = set()
    all_v3_fp: Dict[str, str] = {}

    split_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    split_masks: Dict[str, Dict[str, np.ndarray]] = {}
    split_seed_meta: Dict[str, Dict[str, List[Dict]]] = {}

    for split, per_regime in V3_SPLITS.items():
        streams: Dict[str, np.ndarray] = {}
        masks: Dict[str, np.ndarray] = {}
        seed_meta: Dict[str, List[Dict]] = {}
        for regime in REGIME_ORDER:
            spec = REGIME_SPECS[regime]
            eps_streams = np.zeros((per_regime, 1000), dtype=np.int32)
            eps_masks = np.zeros((per_regime, 1000), dtype=np.int8)
            metas: List[Dict] = []
            for ep in range(per_regime):
                counter = 0
                while True:
                    seed = design_v3_seed(split, regime, ep, counter)
                    fp = None  # computed after generation
                    # seed-level collision checks (against historical + all v3)
                    seed_conflicts = []
                    if seed in historical_seeds:
                        seed_conflicts.append("historical_seed_namespace")
                    if seed in all_v3_seeds:
                        seed_conflicts.append("v3_cross_split_seed")
                    if seed_conflicts:
                        seed_collisions.append({
                            "split": split, "regime": regime, "episode": ep,
                            "counter": counter, "seed": seed,
                            "conflicts": seed_conflicts,
                        })
                        counter += 1
                        counter_increments.append({
                            "split": split, "regime": regime, "episode": ep,
                            "new_counter": counter, "reason": "seed_collision",
                        })
                        continue
                    rng = np.random.default_rng(seed)
                    tx, mask = generate_regime_episode(
                        spec, 1000, rng, CALIBRATION_SAMPLE_SIZE
                    )
                    fp = episode_fingerprint(tx)
                    fp_conflicts = []
                    if fp in historical_fp:
                        fp_conflicts.append(f"historical:{historical_fp[fp]}")
                    if fp in all_v3_fp:
                        fp_conflicts.append(f"v3:{all_v3_fp[fp]}")
                    if fp_conflicts:
                        fp_collisions.append({
                            "split": split, "regime": regime, "episode": ep,
                            "counter": counter, "seed": seed, "fingerprint": fp,
                            "conflicts": fp_conflicts,
                        })
                        counter += 1
                        counter_increments.append({
                            "split": split, "regime": regime, "episode": ep,
                            "new_counter": counter, "reason": "fingerprint_collision",
                        })
                        continue
                    break
                eps_streams[ep] = tx
                eps_masks[ep] = mask
                all_v3_seeds.add(seed)
                v3_seeds[split].add(seed)
                all_v3_fp[fp] = f"{split}/{regime}/{ep}"
                v3_fp[split][fp] = f"{split}/{regime}/{ep}"
                metas.append({
                    "episode": ep,
                    "seed": seed,
                    "collision_counter": counter,
                    "fingerprint": fp,
                })
            streams[regime] = eps_streams
            masks[regime] = eps_masks
            seed_meta[regime] = metas
            log(f"[release] {split} {regime}: {per_regime} episodes generated")
        split_arrays[split] = streams
        split_masks[split] = masks
        split_seed_meta[split] = seed_meta

    # ---------------- write release ----------------
    release_root.mkdir(parents=True, exist_ok=True)
    split_sha: Dict[str, str] = {}
    for split in V3_SPLITS:
        npz_path = release_root / f"{split}.npz"
        arrays: Dict[str, np.ndarray] = {}
        for regime in REGIME_ORDER:
            arrays[f"{regime}__streams"] = split_arrays[split][regime]
            arrays[f"{regime}__burstmasks"] = split_masks[split][regime]
        np.savez_compressed(npz_path, **arrays)
        split_sha[split] = sha256_file(npz_path)

    audit = {
        "result": "PASS" if not seed_collisions and not fp_collisions else "PASS_WITH_LOGGED_COLLISIONS",
        "seed_collisions": seed_collisions,
        "fingerprint_collisions": fp_collisions,
        "collision_counter_increments": counter_increments,
        "historical_pools": historical_pools_meta,
        "historical_seed_namespace": {
            "seed_count": seed_ns["seed_count"],
            "formulas": {
                "MIX12_EQ_master": "seed_i_j = 20000532 + 100000*i + j (counts 417x8+416x4)",
                "MIX12_EQ_val": "seed_i_j = 220000532 + 100000*i + j (25/regime)",
                "static_eval": "seed_i_j = 100000532 + 1000000*i + j (200/regime)",
                "NEW12_v1": "seed_i_j = 710000001 + 100000*i + j (200/regime)",
            },
            "reserved_constants": seed_ns["reserved_constants"],
        },
        "v3_seed_counts": {s: len(v3_seeds[s]) for s in V3_SPLITS},
        "v3_fingerprint_counts": {s: len(v3_fp[s]) for s in V3_SPLITS},
        "checked_pairs": {
            "v3_vs_historical_seeds": True,
            "v3_vs_historical_fingerprints": True,
            "v3_cross_split_seeds": True,
            "v3_cross_split_fingerprints": True,
        },
    }
    with open(release_root / "OVERLAP_CHECK.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, sort_keys=True)

    manifest = {
        "release": "SCFAC-DESIGN-v3",
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "design_doc": "MODEL_DESIGN.md",
        "design_doc_sha256": sha256_file(RELEASE_BUILDER.parent.parent / "MODEL_DESIGN.md"),
        "builder_script": str(RELEASE_BUILDER),
        "builder_script_sha256": sha256_file(RELEASE_BUILDER),
        "generator": {
            "vendored_path": str(RELEASE_BUILDER.parent / "frozen_kwallet_ideaextra_generator.py"),
            "sha256": sha256_file(RELEASE_BUILDER.parent / "frozen_kwallet_ideaextra_generator.py"),
            "expected_sha256": FROZEN_GENERATOR_SHA256,
        },
        "seed_rule": {
            "formula": "SHA256(\"SCFAC-DESIGN-v3|<split>|<regime>|<zero-based-episode>|20261003|<collision-counter>\")[:8] big-endian",
            "prefix": "SCFAC-DESIGN-v3",
            "date": SEED_RULE_DATE,
            "collision_counter_policy": "increment ONLY on verified collision; logged in OVERLAP_CHECK.json",
        },
        "regimes": list(REGIME_ORDER),
        "regime_specs": {k: {kk: vv for kk, vv in vars(v).items()} for k, v in REGIME_SPECS.items()},
        "calibration": {
            "seed": CALIBRATION_SEED_CONST,
            "sample_size": CALIBRATION_SAMPLE_SIZE,
        },
        "splits": {
            split: {
                "episodes_per_regime": per_regime,
                "total_episodes": per_regime * 12,
                "file": f"{split}.npz",
                "sha256": split_sha[split],
                "seeds": {
                    regime: [
                        {"episode": m["episode"], "seed": m["seed"],
                         "collision_counter": m["collision_counter"],
                         "fingerprint": m["fingerprint"]}
                        for m in split_seed_meta[split][regime]
                    ]
                    for regime in REGIME_ORDER
                },
            }
            for split, per_regime in V3_SPLITS.items()
        },
        "overlap_audit_result": audit["result"],
    }
    with open(release_root / "MANIFEST.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    lines = []
    for split in V3_SPLITS:
        lines.append(f"{split_sha[split]}  {split}.npz")
    for aux in ["MANIFEST.json", "OVERLAP_CHECK.json"]:
        lines.append(f"{sha256_file(release_root / aux)}  {aux}")
    (release_root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")

    log(f"[release] DONE: {release_root} (result={audit['result']}, "
        f"seed_collisions={len(seed_collisions)}, fp_collisions={len(fp_collisions)})")
    return manifest
