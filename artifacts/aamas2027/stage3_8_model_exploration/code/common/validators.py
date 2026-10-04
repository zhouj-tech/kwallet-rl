"""Input/artifact validators (MODEL_DESIGN.md section 9).

- D1_SEED_CAPACITY.csv must be rejected as a unique-key input: the frozen file
  contains 20 rows but only 10 unique (C, seed) primary keys, so the remaining
  4 cells must come from the reconciled per-cell experiment outputs, never from
  the CSV. The frozen file itself is NEVER modified or rewritten.
- Money identity: eval_money == settled - 10 * executed_flushes (frozen adapter
  convention, money_p = 1.0, money_tau = 10.0).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

from .config import D1_RECONCILED_CELLS_DIR, D1_SEED_CAPACITY_CSV, MONEY_TAU


class ValidationError(RuntimeError):
    pass


def load_d1_seed_capacity_rows(path: Path = D1_SEED_CAPACITY_CSV) -> List[Dict[str, str]]:
    if not path.exists():
        raise ValidationError(f"D1_SEED_CAPACITY.csv missing: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def reject_d1_seed_capacity_as_unique_key_table(path: Path = D1_SEED_CAPACITY_CSV) -> Dict:
    """Attempt to treat the CSV as a (C, seed) unique-key table; MUST fail.

    Returns a structured rejection record; raises ValidationError semantics via
    the returned status (the frozen file is never modified).
    """
    rows = load_d1_seed_capacity_rows(path)
    key_fields = [k for k in rows[0].keys() if k.strip().lower() in ("c", "capacity", "seed")]
    keys: List[Tuple] = []
    for r in rows:
        key = tuple((f, r[f]) for f in key_fields)
        keys.append(key)
    unique = set(keys)
    seen: Dict[Tuple, int] = {}
    duplicates: List[Dict] = []
    for key in keys:
        seen[key] = seen.get(key, 0) + 1
    for key, n in seen.items():
        if n > 1:
            duplicates.append({"key": dict(key), "rows": n})
    result = {
        "file": str(path),
        "file_untouched": True,
        "rows": len(rows),
        "unique_primary_keys": len(unique),
        "duplicate_keys": sorted(duplicates, key=lambda d: json.dumps(d["key"], sort_keys=True)),
        "acceptable_as_unique_key_table": len(duplicates) == 0,
        "verdict": (
            "REJECTED as unique-key input: duplicate (C, seed) keys present. "
            "Use the reconciled per-cell experiment outputs instead."
            if duplicates else "ACCEPTED"
        ),
    }
    return result


def reconciled_d1_cells(dir_path: Path = D1_RECONCILED_CELLS_DIR) -> Dict:
    """Locate the reconciled per-cell experiment outputs (C, seed) -> artifacts."""
    cells: Dict[str, Dict] = {}
    if not dir_path.exists():
        raise ValidationError(f"reconciled D1 cells dir missing: {dir_path}")
    for result_json in sorted(dir_path.rglob("result.json")):
        try:
            data = json.loads(result_json.read_text())
        except Exception:
            continue
        c_val = data.get("config", {}).get("C")
        seed = data.get("config", {}).get("seed")
        if c_val is None or seed is None:
            continue
        cells[f"C{c_val}_seed{seed}"] = {
            "path": str(result_json),
            "C": c_val,
            "seed": seed,
        }
    return {"dir": str(dir_path), "cells": cells, "num_cells": len(cells)}


def money_identity(settled: float, flushes: int, eval_money: float,
                   money_tau: float = MONEY_TAU, tol: float = 1e-6) -> bool:
    return abs(eval_money - (settled - money_tau * flushes)) < tol


def assert_money_identity_or_raise(settled: float, flushes: int, eval_money: float,
                                   context: str = "") -> None:
    if not money_identity(settled, flushes, eval_money):
        raise ValidationError(
            f"Money identity violated{(' in ' + context) if context else ''}: "
            f"eval_money={eval_money} != settled={settled} - {MONEY_TAU}*flushes={flushes}"
        )
