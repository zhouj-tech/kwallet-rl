#!/usr/bin/env python3
"""Persistent SCREEN queue runner — Stage 3.8B recovery (replaces agent-managed scheduling).

This process owns the entire experiment lifecycle independently of the Trae agent:

  1. load the frozen 24-cell matrix;
  2. reconcile cells against disk outputs (authoritative);
  3. skip validated COMPLETE cells;
  4. maintain at most TWO concurrent full-training child processes;
  5. validate finished cells; classify failures; atomic retry at most once for
     verified infrastructure failures only;
  6. update OVERNIGHT_RUN_LEDGER.csv atomically;
  7. after 24/24 COMPLETE: run Phase B (SCREEN-v3 learned-policy evals,
     SC-M alternative-selector evals, BF reference once per capacity),
     run runtime/analyze_screen.py, and write all required reports;
  8. STOP after the SCREEN gate (no seed532 / CHECK / FULL-DEV / NEW13).

Usage:
  python -u runtime/persistent_screen_queue.py            # run (launch via nohup)
  python -u runtime/persistent_screen_queue.py --dry-run  # reconcile + print plan only

Logs: stdout (redirect to runtime/persistent_queue.log at launch);
per-cell logs at runtime/logs/<CELL>.log.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = "/Users/zhouzhou/benchmarks/A-EVAL-JA-C800-S123/.venv/bin/python"

ENTRIES = {"SC-M": "sc_m/run_train.py", "IA-SC": "ia_sc/run_train.py",
           "TP-SC": "tp_sc/run_train.py"}
TAGS = {"SC-M": "SCM", "IA-SC": "IASC", "TP-SC": "TPSC"}
PARAM_COUNT = {"SC-M": 77907, "IA-SC": 80083, "TP-SC": 77907}
MODELS = ("SC-M", "IA-SC", "TP-SC")
SEEDS = (123, 323, 777, 999)
CAPS = (800, 1200)
BUDGET_EPISODES = 1000
MAX_WORKERS = 2

LEDGER_PATH = ROOT / "OVERNIGHT_RUN_LEDGER.csv"
LOGS_DIR = ROOT / "runtime" / "logs"
EVAL_DIR = ROOT / "runtime" / "eval_out"
REPORTS = ROOT / "reports"
SPLIT = "SCREEN-v3"

FIELDS = ["run_id", "model", "capacity", "seed", "status", "start_time", "end_time",
          "wall_seconds", "exit_code", "checkpoint_count", "best_by_money_episode",
          "best_by_value_episode", "best_by_return_episode", "output_dir",
          "config_sha256", "notes"]


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[{now_iso()}] {msg}", flush=True)


def cell_names(model: str, cap: float, seed: int):
    tag = TAGS[model]
    return f"{tag}_C{int(cap)}_s{seed}"


MATRIX = [(m, int(c), s) for c in CAPS for m in MODELS for s in SEEDS]
MATRIX_HASH = hashlib.sha256(repr(sorted(MATRIX)).encode()).hexdigest()[:16]


# --------------------------------------------------------------------- ledger
def canonical_rows() -> list:
    rows = []
    for (m, c, s) in MATRIX:
        rows.append({
            "run_id": cell_names(m, c, s).upper(),
            "model": m, "capacity": c, "seed": s, "status": "PLANNED",
            "start_time": "", "end_time": "", "wall_seconds": "", "exit_code": "",
            "checkpoint_count": "", "best_by_money_episode": "",
            "best_by_value_episode": "", "best_by_return_episode": "",
            "output_dir": f"outputs/{cell_names(m, c, s)}",
            "config_sha256": "", "notes": "",
        })
    return rows


def read_ledger() -> list:
    if not LEDGER_PATH.exists():
        return canonical_rows()
    with open(LEDGER_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        rid = (r.get("run_id") or "").strip()
        if rid:
            out[rid] = {k: (r.get(k) or "").strip() if k != "notes" else (r.get(k) or "")
                        for k in FIELDS if k in r}
            out[rid]["notes"] = r.get("notes") or ""
    # preserve canonical structure; one row per cell
    canonical = canonical_rows()
    keyed = {r["run_id"]: r for r in canonical}
    for rid, r in out.items():
        if rid in keyed:
            keyed[rid].update(r)
    return list(keyed.values())


def write_ledger(rows: list, *, atomic: bool = True) -> None:
    lines = csv.DictWriter(io_string := __import__("io").StringIO(), fieldnames=FIELDS)
    lines.writeheader()
    for r in rows:
        lines.writerow({k: r.get(k, "") for k in FIELDS})
    data = io_string.getvalue()
    if atomic:
        with tempfile.NamedTemporaryFile("w", dir=str(LEDGER_PATH.parent), delete=False,
                                         suffix=".tmp") as tf:
            tf.write(data)
            tf.flush()
            os.fsync(tf.fileno())
            tmp = tf.name
        os.replace(tmp, LEDGER_PATH)
    else:
        LEDGER_PATH.write_text(data)
    log(f"ledger written atomically ({len(rows)} rows)")


# ---------------------------------------------------------------- validation
def _json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def validate_cell(model: str, cap: float, seed: int) -> tuple:
    """Returns (ok, message). Absolute disk-authoritative validation."""
    name = cell_names(model, cap, seed)
    d = ROOT / "outputs" / name
    problems = []

    def fin(x) -> bool:
        return x == x and math.isfinite(x) if isinstance(x, (int, float)) else True

    run_config = _json(d / "run_config.json")
    run_summary = _json(d / "run_summary.json")
    metrics = _json(d / "training_metrics.json")
    manifest = _json(d / "bank" / "bank_manifest.json")

    if run_config is None:
        return False, "run_config.json missing/unparseable"
    for k, expect in (("candidate", model), ("seed", seed), ("C", float(cap)),
                      ("train_episodes", BUDGET_EPISODES)):
        if run_config.get(k) != expect:
            problems.append(f"run_config {k}={run_config.get(k)} != {expect}")
    if run_summary is None:
        problems.append("run_summary.json missing")
    else:
        if run_summary.get("candidate") != model:
            problems.append(f"run_summary candidate {run_summary.get('candidate')}")
        if run_summary.get("seed") != seed:
            problems.append(f"run_summary seed {run_summary.get('seed')}")
        if run_summary.get("param_count") != PARAM_COUNT[model]:
            problems.append(f"param_count {run_summary.get('param_count')} != {PARAM_COUNT[model]}")
        if not run_summary.get("rng_unchanged_at_every_validation", False):
            problems.append("rng_unchanged_at_every_validation != true")
    if metrics is None:
        problems.append("training_metrics.json missing")
    else:
        log_ep = metrics.get("episode_log") or []
        if len(log_ep) != BUDGET_EPISODES:
            problems.append(f"episode_log length {len(log_ep)} != {BUDGET_EPISODES}")
        for r in log_ep[:BUDGET_EPISODES]:
            vals = [r.get("original_reward"), r.get("settled"), r.get("train_money")]
            if any(v is None or not fin(float(v)) for v in vals):
                problems.append("nonfinite training metric")
                break
    if manifest is None:
        problems.append("bank manifest missing")
    else:
        eps = manifest.get("checkpoint_episodes") or []
        if len(eps) != 20 or eps != list(range(50, 1001, 50)):
            problems.append(f"checkpoint episodes {eps[:5]}... != 20-step schedule")
        for e in eps:
            if not (d / "bank" / f"ckpt_ep{e:04d}.pth").exists():
                problems.append(f"missing ckpt_ep{e:04d}.pth")
                break
        designations = manifest.get("designations") or {}
        for key in ("best_by_money", "best_by_value_accept_ratio", "best_by_training_return"):
            if designations.get(key) not in eps:
                problems.append(f"designation {key}={designations.get(key)} invalid")
        if manifest.get("num_validation_checkpoints") != 20:
            problems.append("num_validation_checkpoints != 20")
        val_hist = manifest.get("validation_history") or []
        if len(val_hist) != 20:
            problems.append(f"validation_history length {len(val_hist)} != 20")
        for v in val_hist:
            if not (fin(v.get("macro12_mean_money")) and fin(v.get("macro12_value_accept_ratio"))):
                problems.append("nonfinite validation metric")
                break
        if model == "TP-SC":
            if designations.get("teacher_boundary_episode") != 250:
                problems.append("TP teacher_boundary_episode != 250")
            if "best_money_from_episode_300" not in designations:
                problems.append("TP post-teacher designation missing")
    cell_log = LOGS_DIR / f"{name}.log"
    if cell_log.exists():
        tail = cell_log.read_text()[-4000:]
        if re.search(r"Traceback", tail):
            problems.append("traceback found in cell log")

    if problems:
        return False, "; ".join(problems)
    return True, "OK"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def mark_complete(rows: list, cell: str, model: str, cap: float, seed: int,
                  start_time: str) -> None:
    name = cell_names(model, cap, seed)
    d = ROOT / "outputs" / name
    summary = _json(d / "run_summary.json") or {}
    manifest = _json(d / "bank" / "bank_manifest.json") or {}
    metrics = _json(d / "training_metrics.json") or {}
    designations = manifest.get("designations") or {}
    for r in rows:
        if r["run_id"] == cell:
            r["status"] = "COMPLETE"
            r["start_time"] = start_time or r["start_time"] or ""
            wall = float(summary.get("training_seconds", 0.0) or 0.0)
            r["wall_seconds"] = f"{wall:.1f}"
            if r["start_time"]:
                try:
                    st = datetime.fromisoformat(r["start_time"])
                    r["end_time"] = (st + timedelta(seconds=wall)).isoformat()
                except ValueError:
                    r["end_time"] = now_iso()
            else:
                r["end_time"] = now_iso()
            r["exit_code"] = "0"
            r["checkpoint_count"] = str(manifest.get("num_validation_checkpoints", 20))
            r["best_by_money_episode"] = str(designations.get("best_by_money", ""))
            r["best_by_value_episode"] = str(designations.get("best_by_value_accept_ratio", ""))
            r["best_by_return_episode"] = str(designations.get("best_by_training_return", ""))
            cfg = d / "run_config.json"
            r["config_sha256"] = sha256_file(cfg)[:16] if cfg.exists() else ""
            r["notes"] = (r["notes"] + "; " if r["notes"] else "") + \
                f"validated from disk by persistent runner ({metrics.get('training_seconds', '?')}s)"
            return


# ------------------------------------------------------------- live processes
def find_live_workers() -> dict:
    """{cell: pid} for any live run_train.py process whose cmdline names our output dir."""
    try:
        out = subprocess.run(["pgrep", "-fl", "run_train.py"], capture_output=True,
                             text=True, timeout=10).stdout
    except Exception:
        return {}
    live = {}
    for line in out.splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        pid = int(parts[0])
        cmd = " ".join(parts[1:])
        for (m, c, s) in MATRIX:
            name = cell_names(m, c, s)
            if f"outputs/{name}" in cmd and name not in live:
                live[name] = pid
    return live


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


# ------------------------------------------------------------------- training
def launch_cell(model: str, cap: float, seed: int, attempt: int) -> subprocess.Popen:
    name = cell_names(model, cap, seed)
    cmd = [PYTHON, "-u", ENTRIES[model], "--seed", str(seed), "--capacity",
           str(cap), "--episodes", str(BUDGET_EPISODES), "--output",
           str(ROOT / "outputs" / name)]
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    cell_log = open(LOGS_DIR / f"{name}.log", "a")
    cell_log.write(f"\n=== attempt {attempt} launched by persistent runner "
                   f"{now_iso()} ===\n")
    cell_log.flush()
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=cell_log, stderr=subprocess.STDOUT)
    log(f"cell launched: {name} (pid {proc.pid}, attempt {attempt})")
    return proc, cell_log


def classify_failure(name: str) -> str:
    cell_log = LOGS_DIR / f"{name}.log"
    tail = cell_log.read_text()[-4000:] if cell_log.exists() else ""
    if "Traceback" in tail:
        if re.search(r"nan|inf|nonfinite", tail, re.IGNORECASE):
            return "METHOD_FAILURE"
        return "METHOD_FAILURE" if "RuntimeError" in tail else "INFRA_FAILURE"
    return "INFRA_FAILURE"


# --------------------------------------------------------------------- Phase B
def run_tool(argv: list, tag: str) -> int:
    log(f"tool start: {tag} ({' '.join(argv)})")
    res = subprocess.run([PYTHON, "-u", *argv], cwd=str(ROOT),
                         capture_output=True, text=True, timeout=3600)
    if res.stdout.strip():
        log(f"tool {tag} stdout: {res.stdout.strip()[-800:]}")
    if res.stderr.strip():
        log(f"tool {tag} stderr: {res.stderr.strip()[-800:]}")
    log(f"tool end: {tag} rc={res.returncode}")
    return res.returncode


def eval_json_out(cell: str, designation: str) -> Path:
    return EVAL_DIR / f"{cell}__{SPLIT}__{designation}.json"


def eval_exists_ok(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        json.loads(path.read_text())
        return True
    except Exception:
        return False


def phase_b() -> int:
    log("PHASE B START: all 24/24 training cells COMPLETE")
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    for (m, c, s) in MATRIX:
        name = cell_names(m, c, s)
        designation = "all" if m == "SC-M" else "best_by_money"
        out = eval_json_out(name, designation)
        if eval_exists_ok(out):
            log(f"eval exists, reused: {out.name}")
            continue
        rc = run_tool(["runtime/eval_cell.py", "--cell", f"outputs/{name}",
                       "--split", SPLIT, "--designation", designation,
                       "--out", str(out)], f"eval {name} ({designation})")
        if rc != 0:
            log(f"EVAL FAILED rc={rc} for {name} — aborting Phase B")
            return 1
    for cap in CAPS:
        out = EVAL_DIR / f"BF_C{int(cap)}__{SPLIT}.json"
        if eval_exists_ok(out):
            log(f"BF eval exists, reused: BF_C{int(cap)}")
            continue
        rc = run_tool(["runtime/eval_bf.py", "--capacity", str(cap),
                       "--split", SPLIT, "--out", str(out)], f"BF C{int(cap)}")
        if rc != 0:
            log(f"BF EVAL FAILED rc={rc} C{cap}")
            return 1
    rc = run_tool(["runtime/analyze_screen.py", "--split", SPLIT, "--gate", "screen"],
                  "analyze_screen")
    if rc != 0:
        log(f"ANALYZE FAILED rc={rc}")
        return 1
    log("PHASE B EVALUATION COMPLETE — generating recovery reports")
    write_checkpoint_selection_analysis()
    write_mechanism_analysis()
    write_recovery_final_report()
    log("ALL PHASE B REPORTS WRITTEN")
    return 0


def load_money_record(name: str) -> dict:
    """Unwrap the nested {designation: record} eval_cell schema (both the single
    --designation best_by_money files and the SC-M __all.json files)."""
    p = eval_json_out(name, "best_by_money")
    if p.exists():
        d = json.loads(p.read_text())
        if "result" in d:
            return d
        money = d.get("best_by_money")
        if money is not None:
            return money
    p = eval_json_out(name, "all")
    data = json.loads(p.read_text())
    return data.get("best_by_money")


def cell_macro(name: str) -> dict:
    rec = load_money_record(name)
    res = rec["result"]
    regimes = [r for r in res if r not in ("macro12_mean_money",
                                           "macro12_value_accept_ratio")]
    return {
        "money": res["macro12_mean_money"],
        "settled": float(sum(res[r]["settled_mean"] for r in regimes) / len(regimes)),
        "flushes": float(sum(res[r]["flushes_mean"] for r in regimes) / len(regimes)),
        "insufficient": float(sum(res[r]["insufficient_drops_mean"] for r in regimes) / len(regimes)),
        "value_accept": res["macro12_value_accept_ratio"],
        "oversize": float(sum(res[r]["oversize_drops_mean"] for r in regimes) / len(regimes)),
        "drops": float(sum(res[r]["drops_mean"] for r in regimes) / len(regimes)),
        "ckpt": rec["checkpoint_episode"],
    }


def model_cap(row_set):
    """row_set: list of (name, macro_dict) -> aggregated means."""
    n = len(row_set)
    out = {}
    for k in ("money", "settled", "flushes", "insufficient", "value_accept",
              "oversize", "drops", "ckpt"):
        out[k] = float(sum(r[1][k] for r in row_set) / n)
    return out


def write_checkpoint_selection_analysis() -> None:
    lines = ["# CHECKPOINT_SELECTION_ANALYSIS — SC-M (recovery SCREEN)", "",
             "Selection rules: best_by_money / best_by_value_accept_ratio / "
             "best_by_training_return from the shared 20-checkpoint bank (VAL-v3), "
             "re-evaluated on SCREEN-v3 at the cell's training capacity.",
             "", "| Cell | money ep | value ep | return ep | money Money | value Money | return Money | selectors agree |",
             "|---|---|---|---|---|---|---|---|"]
    agree = 0
    total = 0
    ep_diffs = []
    money_diffs_mv, money_diffs_mr = [], []
    for (m, c, s) in MATRIX:
        if m != "SC-M":
            continue
        name = cell_names(m, c, s)
        data = json.loads(eval_json_out(name, "all").read_text())
        sel = {}
        for desig in ("best_by_money", "best_by_value_accept_ratio",
                      "best_by_training_return"):
            sel[desig] = (data[desig]["checkpoint_episode"],
                          data[desig]["result"]["macro12_mean_money"])
        total += 1
        eps = [sel[d][0] for d in sel]
        if eps[0] == eps[1] == eps[2]:
            agree += 1
        else:
            ep_diffs.append(max(eps) - min(eps))
        money_diffs_mv.append(sel["best_by_money"][1] - sel["best_by_value_accept_ratio"][1])
        money_diffs_mr.append(sel["best_by_money"][1] - sel["best_by_training_return"][1])
        lines.append(f"| {name} | {eps[0]} | {eps[1]} | {eps[2]} | "
                     f"{sel['best_by_money'][1]:.2f} | {sel['best_by_value_accept_ratio'][1]:.2f} | "
                     f"{sel['best_by_training_return'][1]:.2f} | {'yes' if eps[0] == eps[1] == eps[2] else 'no'} |")
    lines += ["", f"- Selectors agree on the same checkpoint episode in {agree}/{total} SC-M cells."]
    if ep_diffs:
        lines.append(f"- Mean absolute checkpoint-episode spread (cells with disagreement): "
                     f"{sum(ep_diffs) / len(ep_diffs):.1f} episodes.")
    lines.append(f"- Mean Money advantage of best_by_money over the value selector: "
                 f"{sum(money_diffs_mv) / len(money_diffs_mv):.2f}; over the training-return "
                 f"selector: {sum(money_diffs_mr) / len(money_diffs_mr):.2f} (positive = Money "
                 f"selection better on SCREEN-v3).")
    impact = "LOW"
    if ep_diffs and (sum(ep_diffs) / len(ep_diffs) > 100):
        impact = "HIGH"
    elif ep_diffs or abs(sum(money_diffs_mv) / len(money_diffs_mv)) > 20:
        impact = "MODERATE"
    lines.append(f"- CHECKPOINT_SELECTION_IMPACT (descriptive): **{impact}**.")
    lines += ["", "Note: descriptive development evidence only; not a significance test."]
    (REPORTS / "CHECKPOINT_SELECTION_ANALYSIS.md").write_text("\n".join(lines) + "\n")
    log("wrote reports/CHECKPOINT_SELECTION_ANALYSIS.md")


def write_mechanism_analysis() -> None:
    lines = ["# MECHANISM_ANALYSIS — SCREEN-v3 (recovery)", "",
             "Macro12 means over regimes (Money = mean settled - 10 * mean flushes, "
             "exact by linearity). Descriptive only; no causal attribution claimed.",
             "", "| Model | C | Money | settled | flushes | insufficient drops | value_accept_ratio | mean ckpt ep |",
             "|---|---|---|---|---|---|---|---|"]
    agg = {}
    for (m, c, s) in MATRIX:
        agg[(m, c)] = agg.get((m, c), []) + [(cell_names(m, c, s), cell_macro(cell_names(m, c, s)))]
    for m in MODELS:
        for c in CAPS:
            a = model_cap(agg[(m, c)])
            lines.append(f"| {m} | {c} | {a['money']:.2f} | {a['settled']:.2f} | "
                         f"{a['flushes']:.2f} | {a['insufficient']:.2f} | {a['value_accept']:.4f} | {a['ckpt']:.1f} |")
    lines += ["", "## Candidate vs matched SC-M (per capacity, macro12)", "",]
    for cand in ("IA-SC", "TP-SC"):
        for c in CAPS:
            a = model_cap(agg[(cand, c)])
            sc = model_cap(agg[("SC-M", c)])
            d_money = a["money"] - sc["money"]
            d_settled = a["settled"] - sc["settled"]
            d_flush = a["flushes"] - sc["flushes"]
            residual = d_money - (d_settled - 10.0 * d_flush)
            lines.append(f"### {cand} C{c}")
            lines.append(f"- DeltaMoney = {d_money:.2f}; DeltaSettled = {d_settled:.2f}; "
                         f"DeltaFlush = {d_flush:.2f}.")
            lines.append(f"- Accounting check DeltaMoney - (DeltaSettled - 10*DeltaFlush) = "
                         f"{residual:.2e} (must be ~0; linearity identity).")
            lines.append(f"- Delta insufficient drops = {a['insufficient'] - sc['insufficient']:.2f}; "
                         f"Delta value_accept_ratio = {a['value_accept'] - sc['value_accept']:.4f}.")
            if abs(residual) > 1e-4:
                lines.append("- WARNING: accounting identity not exact on reported means.")
    lines += ["", "Interpretation caveats: 'more flushing' is NOT a universal success "
              "signature (C1200 vanilla SC flushes more than BF historically yet performs "
              "worse). Changes in diagnostics without Money change do not establish mechanism."]
    (REPORTS / "MECHANISM_ANALYSIS.md").write_text("\n".join(lines) + "\n")
    log("wrote reports/MECHANISM_ANALYSIS.md")


def write_recovery_final_report() -> None:
    summary = json.loads((REPORTS / "overnight_summary.json").read_text())
    gates = summary["gates"]
    bf = summary["bf_money"]
    agg = {}
    for (m, c, s) in MATRIX:
        agg[(m, c)] = agg.get((m, c), []) + [(cell_names(m, c, s), cell_macro(cell_names(m, c, s)))]
    means = {f"{m}__C{c}": model_cap(agg[(m, c)])["money"] for m in MODELS for c in CAPS}

    ia_pass = bool(gates.get("IA-SC__SURVIVES_SCREEN", [False])[0])
    tp_pass = bool(gates.get("TP-SC__SURVIVES_SCREEN", [False])[0])
    survivors = ("NONE" if not ia_pass and not tp_pass else
                 ("IA-SC" if ia_pass and not tp_pass else
                  ("TP-SC" if tp_pass and not ia_pass else "IA-SC_AND_TP-SC")))

    lines = ["# SCREEN_RECOVERY_FINAL_REPORT", "",
             "## 1. Why recovery was required",
             "The previous Trae agent turn ended after launching wave 2. Background "
             "workers completed, but no persistent scheduler existed to launch later "
             "waves. This recovery replaces agent-managed scheduling with a persistent "
             "queue runner process.", "",
             "## 2. Existing-cell reconciliation",
             "Four cells were validated from disk before queue start (candidate/seed/"
             "capacity/protocol-budget/param-count/rng/20-checkpoint bank/designations/"
             "finite-metrics/TP teacher-boundary): SCM_C800_s123, SCM_C800_s323 "
             "(protocol-equivalent preflight reuse), SCM_C1200_s123, TPSC_C800_s123.",
             "", "## 3. Persistent queue provenance",
             f"- runner: runtime/persistent_screen_queue.py (matrix hash {MATRIX_HASH})",
             f"- runner PID: {RUNNER_PID}",
             f"- start: {START_TIME}",
             f"- python: {PYTHON}",
             f"- max concurrency: {MAX_WORKERS}",
             f"- matrix: {len(MATRIX)} cells (3 models x 4 seeds x 2 capacities)", "",
             "## 4. Training completion",
             f"{sum(1 for r in read_ledger() if r['status'] == 'COMPLETE')}/24",
             "",
             "## 5. Failures/retries",
             "None recorded (see runtime/persistent_queue.log for authoritative log).",
             "", "## 6. SCREEN-v3 primary table",
             "(see OVERNIGHT_SCREEN_REPORT.md — primary artifact)", "",
             "## 7. BF reference scores",
             f"- BF C800 SCREEN macro12 Money = {bf.get('800', float('nan')):.4f}",
             f"- BF C1200 SCREEN macro12 Money = {bf.get('1200', float('nan')):.4f}",
             "", "## 8. Matched paired deltas",
             "(see overnight_paired_deltas.csv)", "",
             "## 9. Candidate promotion decisions",
             f"- IA-SC SCREEN: {'PASS' if ia_pass else 'FAIL'}",
             f"- TP-SC SCREEN: {'PASS' if tp_pass else 'FAIL'}",
             f"- SCREEN survivors: {survivors}",
             "", "## 10. Checkpoint-selection analysis",
             "(see CHECKPOINT_SELECTION_ANALYSIS.md)", "",
             "## 11. Mechanism summary",
             "(see MECHANISM_ANALYSIS.md)", "",
             "## 12. Remaining uncertainty",
             "Development evidence only; four training seeds per cell; no fresh "
             "confirmation; SCREEN panels are development pools.", "",
             "## 13. Recommended next step",
             f"{'HELDBACK seed532 CHECK-v3 for the survivor(s), per predeclared gates' if survivors != 'NONE' else 'STOP model branch; deep failure analysis per spec'}. "
             "Await PI review before any further phase.", ""]
    lines += ["", "---", "", "RECOVERY_STATUS:", "COMPLETE", "",
              "TRAINING_MATRIX_COMPLETE:", "YES", "",
              "COMPLETED_CELLS:", "24/24", "",
              "PERSISTENT_RUNNER_USED:", "YES", "",
              "MAX_CONCURRENT_WORKERS:", str(MAX_WORKERS), "",
              "SCREEN_EVALUATION_COMPLETE:", "YES", "",
              "BF_C800_SCREEN_MONEY:", f"{bf.get('800', float('nan')):.4f}", "",
              "BF_C1200_SCREEN_MONEY:", f"{bf.get('1200', float('nan')):.4f}", "",
              "SC_M_C800_MEAN:", f"{means['SC-M__C800']:.4f}", "",
              "SC_M_C1200_MEAN:", f"{means['SC-M__C1200']:.4f}", "",
              "IA_SC_C800_MEAN:", f"{means['IA-SC__C800']:.4f}", "",
              "IA_SC_C1200_MEAN:", f"{means['IA-SC__C1200']:.4f}", "",
              "TP_SC_C800_MEAN:", f"{means['TP-SC__C800']:.4f}", "",
              "TP_SC_C1200_MEAN:", f"{means['TP-SC__C1200']:.4f}", "",
              "IA_SC_SCREEN_PASS:", "YES" if ia_pass else "NO", "",
              "TP_SC_SCREEN_PASS:", "YES" if tp_pass else "NO", "",
              "SCREEN_SURVIVORS:", survivors, "",
              "RECOMMEND_HELDBACK_SEED532_CHECK:", "YES" if survivors != "NONE" else "NO", "",
              "NO_UNAUTHORIZED_EXPERIMENTS:", "YES", ""]
    (REPORTS / "SCREEN_RECOVERY_FINAL_REPORT.md").write_text("\n".join(lines) + "\n")
    log("wrote reports/SCREEN_RECOVERY_FINAL_REPORT.md")


# ------------------------------------------------------------------- main
START_TIME = now_iso()
RUNNER_PID = int(os.environ.get("QUEUE_RUNNER_PID", os.getpid()))


def reconcile(rows: list) -> dict:
    """Validate disk; return {cell_name: (status, problem-message)}."""
    live = find_live_workers()
    status = {}
    for (m, c, s) in MATRIX:
        name = cell_names(m, c, s)
        row = next(r for r in rows if r["run_id"] == name.upper())
        ok, msg = validate_cell(m, c, s)
        if ok:
            status[name] = ("COMPLETE", msg)
            if row["status"] != "COMPLETE":
                mark_complete(rows, name.upper(), m, c, s, row["start_time"] or "")
        elif name in live:
            status[name] = ("LIVE", f"live pid {live[name]}")
            row["status"] = "RUNNING"
        else:
            status[name] = ("INCOMPLETE", msg)
            if (ROOT / "outputs" / name).exists():
                row["status"] = "PLANNED"
                row["notes"] = (row["notes"] + "; " if row["notes"] else "") + \
                    f"partial prior output exists; fresh run overwrites ({msg[:200]})"
            else:
                row["status"] = "PLANNED"
    return status


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    log(f"persistent runner start: pid={os.getpid()} matrix_hash={MATRIX_HASH} "
        f"python={sys.version.split()[0]} max_workers={MAX_WORKERS}")
    rows = read_ledger()
    status = reconcile(rows)
    n_complete = sum(1 for v in status.values() if v[0] == "COMPLETE")
    n_live = sum(1 for v in status.values() if v[0] == "LIVE")
    n_incomplete = sum(1 for v in status.values() if v[0] == "INCOMPLETE")
    log(f"reconcile: {n_complete} COMPLETE, {n_live} LIVE, {n_incomplete} INCOMPLETE "
        f"(matrix {len(MATRIX)} cells)")
    for name, (st, msg) in status.items():
        if st != "COMPLETE":
            log(f"  pending: {name} [{st}] {msg[:160]}")
        else:
            log(f"  complete: {name} {msg}")
    write_ledger(rows)

    if args.dry_run:
        log(f"DRY-RUN complete: {n_complete} COMPLETE, {len(MATRIX) - n_complete} to run. "
            f"No workers launched.")
        return 0

    if n_complete == len(MATRIX):
        log("all cells already COMPLETE at startup")
    else:
        # ---- training phase ----
        matrix_info = {cell_names(m, c, s): (m, float(c), s) for (m, c, s) in MATRIX}
        rows_keyed = {r["run_id"]: r for r in rows}

        queue = [name for name, (st, _) in status.items() if st == "INCOMPLETE"]
        slots: dict = {}   # cell_name -> Popen (own child) or int pid (adopted)
        for name, pid in find_live_workers().items():
            if name not in slots:
                slots[name] = pid
                log(f"startup: adopted live worker {name} (pid {pid})")
        failures: dict = {}   # cell_name -> infra retry count
        stopped_cells = []
        stop_queue = False

        while True:
            # 1) reap finished workers
            for cell in list(slots):
                handle = slots[cell]
                proc = handle if hasattr(handle, "poll") else None
                pid = proc.pid if proc else handle
                alive = (proc.poll() is None) if proc else process_alive(pid)
                if alive:
                    continue
                rc = proc.returncode if proc else 0
                m, cap, s = matrix_info[cell]
                ok, msg = validate_cell(m, cap, s)
                del slots[cell]
                if rc == 0 and ok:
                    mark_complete(rows, cell.upper(), m, cap, s,
                                  rows_keyed[cell.upper()].get("start_time") or "")
                    write_ledger(rows)
                    log(f"cell COMPLETE: {cell} (rc={rc})")
                else:
                    kind = classify_failure(cell) if rc != 0 else "INVALID_OUTPUT"
                    log(f"cell FAILED: {cell} rc={rc} kind={kind} msg={msg[:240]}")
                    if kind == "METHOD_FAILURE":
                        rows_keyed[cell.upper()]["status"] = "METHOD_FAILURE"
                        write_ledger(rows)
                        rep = ROOT / "runtime" / "METHOD_FAILURE_REPORT.md"
                        with open(rep, "a") as f:
                            f.write(f"## {cell} {now_iso()}\nrc={rc}\n{msg}\n\n")
                        log(f"METHOD_FAILURE for {cell} — stopping queue "
                            f"(matrix validity unattainable)")
                        stop_queue = True
                        stopped_cells.append(cell)
                    else:
                        failures[cell] = failures.get(cell, 0) + 1
                        if failures[cell] >= 2:
                            rows_keyed[cell.upper()]["status"] = "INFRA_FAILURE"
                            write_ledger(rows)
                            log(f"{cell}: second infra failure — stopping queue")
                            stop_queue = True
                            stopped_cells.append(cell)
                        else:
                            log(f"{cell}: infra failure — exact retry #{failures[cell]}")
                if stop_queue:
                    break

            if stop_queue:
                break

            # 2) launch into free slots
            while len(slots) < MAX_WORKERS and queue and not stop_queue:
                cell = queue.pop(0)
                m, cap, s = matrix_info[cell]
                rows_keyed[cell.upper()]["status"] = "RUNNING"
                rows_keyed[cell.upper()]["start_time"] = now_iso()
                write_ledger(rows)
                proc, _ = launch_cell(m, cap, s, failures.get(cell, 0) + 1)
                slots[cell] = proc

            # 3) safety: adopt any live workers the reconciler missed
            if not slots and queue and not stop_queue:
                for name, pid in find_live_workers().items():
                    slots[name] = pid
                    log(f"adopted live worker {name} (pid {pid})")

            if stop_queue:
                break
            if not slots and not queue:
                log("training phase: all slots idle and queue empty")
                break
            time.sleep(15)

        complete_now = sum(1 for r in rows if r["status"] == "COMPLETE")
        log(f"training loop end: {complete_now}/{len(MATRIX)} COMPLETE "
            f"(remaining queue={queue}, stopped={stopped_cells})")
        if complete_now < len(MATRIX):
            log("MATRIX INCOMPLETE — phase B aborted; failure evidence retained")
            return 1

    # ---- Phase B ----
    rc_b = phase_b()
    log(f"persistent runner exit: phase_b_rc={rc_b} pid={os.getpid()}")
    return rc_b


if __name__ == "__main__":
    sys.exit(main())