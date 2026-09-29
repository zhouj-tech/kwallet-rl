from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


TARGET_K = [3, 6, 12]
TARGET_SEEDS = [123, 323, 532]
TARGET_C = 1200
TARGET_F = 3
TARGET_T = 1000
TARGET_TRAIN_REGIME = "MIX12_EQ"

MAIN_MODEL_FAMILIES = ["DQN_baseline", "Basic_AC_or_PPO", "Factorized_AC"]
APPENDIX_MODEL_FAMILIES = [
    "Dual_Branch_AC",
    "Gate_Balanced_Dual_AC",
    "Gate_Regularized_Dual_AC",
    "Aux_Risk_Dual_AC",
]

OUT_DIR = Path(__file__).resolve().parent
REPO_ROOT = OUT_DIR.parents[3]

SOURCE_CONFIDENCE = {
    "raw_cross_regime_json": "high",
    "aggregate_csv": "medium",
    "final_comparison_table": "low",
    "log_inferred": "verify",
    "unclear": "verify",
}

REGIME_NAMES = {
    "US",
    "TLS",
    "LNS",
    "TLNS",
    "TPLS",
    "PLS",
    "UB",
    "TLB",
    "LNB",
    "TLNB",
    "TPLB",
    "PLB",
}


@dataclass
class ParsedResult:
    k: Optional[int] = None
    C: Optional[float] = None
    F: Optional[int] = None
    T: Optional[int] = None
    model_family: str = "Unknown"
    model_mode: str = "unknown"
    seed: Optional[int] = None
    train_regime: str = "unknown"
    eval_protocol: str = "unknown_eval_protocol"
    scenario: str = "unknown"
    mean_value_accept_ratio: Optional[float] = None
    worst_regime_value_accept_ratio: Optional[float] = None
    std_value_accept_ratio_across_regimes: Optional[float] = None
    avg_drops: Optional[float] = None
    avg_flushes: Optional[float] = None
    avg_drop_rate: Optional[float] = None
    avg_count_accept_ratio: Optional[float] = None
    num_test_regimes: int = 0
    source_type: str = "unclear"
    main_result_confidence: str = "verify"
    from_archive: bool = False
    source_file: str = ""
    notes: str = ""
    timestamp: str = ""
    regime_rows: List[Dict[str, Any]] = field(default_factory=list)

    def dedup_key(self) -> Tuple[Any, ...]:
        return (
            self.model_family,
            self.model_mode,
            norm_float(self.C),
            self.k,
            self.F,
            self.T,
            self.seed,
            self.train_regime,
            self.eval_protocol,
            self.scenario,
        )


def norm_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 8)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v):
        return None
    return v


def to_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    return float(sum(clean) / len(clean)) if clean else None


def std(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    m = sum(clean) / len(clean)
    return float((sum((v - m) ** 2 for v in clean) / len(clean)) ** 0.5)


def extract_pattern_int(text: str, pattern: str) -> Optional[int]:
    m = re.search(pattern, text)
    return int(m.group(1)) if m else None


def extract_pattern_float(text: str, pattern: str) -> Optional[float]:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def infer_eval_protocol(path: Path, scenario: str, test_regimes: Iterable[str]) -> str:
    regimes = {str(r) for r in test_regimes if str(r)}
    text = f"{path} {scenario}".lower()
    if len(regimes & REGIME_NAMES) >= 12 or "cross" in text or "static_eval" in text:
        return "cross_regime_12_static_eval" if len(regimes & REGIME_NAMES) >= 12 else "unknown_eval_protocol"
    if "mix12" in text or "mixed" in text:
        return "mixed_eval"
    if len(regimes) == 1:
        only = next(iter(regimes))
        if only and only.lower() in text:
            return "own_regime_eval"
    return "unknown_eval_protocol"


def classify_model(model_mode: str, scenario: str, path: Path) -> Tuple[str, str]:
    raw = f"{model_mode} {scenario} {path}".lower()
    mode = model_mode or "unknown"

    if "gate_regularized" in raw:
        return "Gate_Regularized_Dual_AC", mode
    if "gate_balanced" in raw:
        return "Gate_Balanced_Dual_AC", mode
    if "auxrisk" in raw:
        return "Aux_Risk_Dual_AC", mode
    if "dual_branch_factorized_ac" in raw or "dual_ac" in raw:
        return "Dual_Branch_AC", mode
    if "basic_ppo" in raw:
        return "Basic_AC_or_PPO", mode if mode != "unknown" else "basic_ppo"
    if "factorized_ac" in raw and "dual" not in raw:
        return "Factorized_AC", mode

    if model_mode in {"baseline", "attn_context"}:
        return "DQN_baseline", mode

    if "dqn" in raw or "baseline" in raw or "ideaextra/results" in raw:
        return "DQN_baseline", mode if mode != "unknown" else "dqn_baseline"

    if ("ppo" in raw or "actor_critic" in raw or re.search(r"(^|[_/-])ac([_/-]|$)", raw)) and "factorized" not in raw and "dual" not in raw:
        return "Basic_AC_or_PPO", mode

    return "Unknown", mode


def action_space_size(row: ParsedResult) -> str:
    if row.k is None:
        return "unknown"
    if row.model_family == "Factorized_AC":
        return "2*(k+1) heads"
    if row.model_family in {"DQN_baseline", "Basic_AC_or_PPO"}:
        return str((row.k + 1) ** 2)
    return "appendix"


def policy_output_size(row: ParsedResult) -> str:
    if row.k is None:
        return "unknown"
    if row.model_family == "Factorized_AC":
        return str(2 * (row.k + 1))
    if row.model_family in {"DQN_baseline", "Basic_AC_or_PPO"}:
        return str((row.k + 1) ** 2)
    return "appendix"


def parse_json_result(path: Path) -> Optional[ParsedResult]:
    payload = read_json(path)
    if not payload or "test_results" not in payload:
        return None

    config = payload.get("config", {}) if isinstance(payload.get("config"), dict) else {}
    env = config.get("env", {}) if isinstance(config.get("env"), dict) else {}
    scenario = str(payload.get("scenario") or path.parent.parent.name)
    model_mode = str(payload.get("model_mode") or config.get("model_mode") or "")
    family, normalized_mode = classify_model(model_mode, scenario, path)

    result = ParsedResult(
        k=to_int(env.get("k")) or extract_pattern_int(f"{scenario} {path}", r"_k(\d+)"),
        C=to_float(env.get("C")) or extract_pattern_float(f"{scenario} {path}", r"_C([0-9.]+)"),
        F=to_int(env.get("F")) or extract_pattern_int(f"{scenario} {path}", r"_F(\d+)"),
        T=to_int(env.get("T")) or extract_pattern_int(f"{scenario} {path}", r"_T(\d+)"),
        model_family=family,
        model_mode=normalized_mode,
        seed=to_int(payload.get("seed")) or to_int(config.get("seed")) or extract_pattern_int(f"{scenario} {path}", r"seed[_-]?(\d+)"),
        train_regime=str(payload.get("train_regime") or config.get("data", {}).get("train_regime") or extract_train_regime(scenario) or "unknown"),
        scenario=scenario,
        source_type="raw_cross_regime_json",
        main_result_confidence=SOURCE_CONFIDENCE["raw_cross_regime_json"],
        from_archive="archive" in path.parts,
        source_file=rel(path),
        timestamp=str(payload.get("timestamp") or path.parent.name),
    )

    test_results = payload.get("test_results", {})
    regimes = list(test_results.keys())
    result.eval_protocol = infer_eval_protocol(path, scenario, regimes)
    result.num_test_regimes = len(regimes)

    regime_rows: List[Dict[str, Any]] = []
    for regime, regime_result in test_results.items():
        summary = regime_result.get("summary", {}) if isinstance(regime_result, dict) else {}
        row = {
            "test_regime": regime,
            "value_accept_ratio": summary_mean(summary, "value_accept_ratio"),
            "drops": summary_mean(summary, "drops"),
            "flushes": summary_mean(summary, "flushes"),
            "drop_rate": summary_mean(summary, "drop_rate"),
            "count_accept_ratio": summary_mean(summary, "count_accept_ratio"),
        }
        regime_rows.append(row)
    result.regime_rows = regime_rows

    aggregate = payload.get("aggregate", {}) if isinstance(payload.get("aggregate"), dict) else {}
    values = [r["value_accept_ratio"] for r in regime_rows]
    result.mean_value_accept_ratio = to_float(aggregate.get("mean_value_accept_ratio")) or mean(values)
    result.worst_regime_value_accept_ratio = to_float(aggregate.get("worst_regime_value_accept_ratio")) or (min(v for v in values if v is not None) if any(v is not None for v in values) else None)
    result.std_value_accept_ratio_across_regimes = to_float(aggregate.get("std_value_accept_ratio_across_regimes")) or std(values)
    result.avg_drops = to_float(aggregate.get("mean_drops")) or mean(r["drops"] for r in regime_rows)
    result.avg_flushes = to_float(aggregate.get("mean_flushes")) or mean(r["flushes"] for r in regime_rows)
    result.avg_drop_rate = mean(r["drop_rate"] for r in regime_rows)
    result.avg_count_accept_ratio = mean(r["count_accept_ratio"] for r in regime_rows)

    if result.model_family == "Unknown":
        result.notes = "unclear model family"
        result.main_result_confidence = "verify"
    if result.num_test_regimes == 0 or result.mean_value_accept_ratio is None:
        result.notes = add_note(result.notes, "missing metrics")

    return result


def summary_mean(summary: Dict[str, Any], key: str) -> Optional[float]:
    value = summary.get(key)
    if isinstance(value, dict):
        return to_float(value.get("mean"))
    return to_float(value)


def extract_train_regime(text: str) -> Optional[str]:
    m = re.search(r"train([A-Za-z0-9_]+?)_cross", text)
    if m:
        return m.group(1)
    m = re.search(r"train([A-Za-z0-9_]+?)_C", text)
    if m:
        return m.group(1)
    m = re.search(r"train([A-Za-z0-9_]+?)_C?\d", text)
    return m.group(1) if m else None


def add_note(existing: str, note: str) -> str:
    if not existing:
        return note
    if note in existing:
        return existing
    return f"{existing}; {note}"


def parse_csv_results(path: Path) -> List[ParsedResult]:
    rows: List[ParsedResult] = []
    source_type = csv_source_type(path)
    confidence = SOURCE_CONFIDENCE[source_type]

    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            data = list(reader)
    except Exception:
        return rows

    if not data:
        return rows

    fields = set(data[0].keys())
    has_regime_metrics = {
        "value_accept_ratio",
        "drops",
        "flushes",
    }.issubset(fields)
    if not has_regime_metrics:
        return rows

    grouped: Dict[Tuple[Any, ...], List[Dict[str, str]]] = defaultdict(list)
    for row in data:
        model_mode = row.get("model_mode") or row.get("model") or row.get("model_label") or ""
        scenario = row.get("scenario") or f"{model_mode}_seed{row.get('seed','unknown')}_k{row.get('k','unknown')}"
        family, normalized_mode = classify_model(model_mode, scenario, path)
        key = (
            family,
            normalized_mode,
            row.get("C"),
            row.get("k"),
            row.get("F"),
            row.get("T"),
            row.get("seed"),
            row.get("train_regime") or TARGET_TRAIN_REGIME,
            scenario,
        )
        grouped[key].append(row)

    for group_rows in grouped.values():
        first = group_rows[0]
        model_mode = first.get("model_mode") or first.get("model") or first.get("model_label") or ""
        scenario = first.get("scenario") or f"{model_mode}_seed{first.get('seed','unknown')}_k{first.get('k','unknown')}"
        family, normalized_mode = classify_model(model_mode, scenario, path)
        regimes = [r.get("test_regime", "") for r in group_rows if r.get("test_regime")]
        result = ParsedResult(
            k=to_int(first.get("k")),
            C=to_float(first.get("C")),
            F=to_int(first.get("F")),
            T=to_int(first.get("T")),
            model_family=family,
            model_mode=normalized_mode,
            seed=to_int(first.get("seed")),
            train_regime=str(first.get("train_regime") or TARGET_TRAIN_REGIME),
            eval_protocol=infer_eval_protocol(path, scenario, regimes),
            scenario=scenario,
            source_type=source_type,
            main_result_confidence=confidence,
            from_archive="archive" in path.parts,
            source_file=rel(path),
            num_test_regimes=len(regimes),
        )
        result.regime_rows = [
            {
                "test_regime": r.get("test_regime"),
                "value_accept_ratio": to_float(r.get("value_accept_ratio")),
                "drops": to_float(r.get("drops")),
                "flushes": to_float(r.get("flushes")),
                "drop_rate": to_float(r.get("drop_rate")),
                "count_accept_ratio": to_float(r.get("count_accept_ratio")),
            }
            for r in group_rows
        ]
        result.mean_value_accept_ratio = to_float(first.get("mean_value_accept_ratio")) or to_float(first.get("mean_value_accept_ratio_percent"))
        if result.mean_value_accept_ratio and result.mean_value_accept_ratio > 1.0:
            result.mean_value_accept_ratio /= 100.0
        result.mean_value_accept_ratio = result.mean_value_accept_ratio or mean(r["value_accept_ratio"] for r in result.regime_rows)
        result.worst_regime_value_accept_ratio = to_float(first.get("worst_regime_value_accept_ratio")) or mean([to_float(first.get("avg_worst_regime_value_accept_ratio"))])
        result.worst_regime_value_accept_ratio = result.worst_regime_value_accept_ratio or (
            min(v for v in (r["value_accept_ratio"] for r in result.regime_rows) if v is not None)
            if any(r["value_accept_ratio"] is not None for r in result.regime_rows)
            else None
        )
        result.std_value_accept_ratio_across_regimes = (
            to_float(first.get("std_value_accept_ratio_across_regimes"))
            or to_float(first.get("std_value_accept_ratio"))
            or std(r["value_accept_ratio"] for r in result.regime_rows)
        )
        result.avg_drops = to_float(first.get("mean_drops")) or mean(r["drops"] for r in result.regime_rows)
        result.avg_flushes = to_float(first.get("mean_flushes")) or mean(r["flushes"] for r in result.regime_rows)
        result.avg_drop_rate = mean(r["drop_rate"] for r in result.regime_rows)
        result.avg_count_accept_ratio = mean(r["count_accept_ratio"] for r in result.regime_rows)
        if result.model_family == "Unknown":
            result.main_result_confidence = "verify"
            result.notes = "unclear model family"
        rows.append(result)

    return rows


def csv_source_type(path: Path) -> str:
    text = str(path).lower()
    if "final_comparison_tables" in text or "summary_tables" in text:
        return "final_comparison_table"
    if "aggregate" in text or "aggregated" in text:
        return "aggregate_csv"
    return "final_comparison_table"


def parse_log_results(path: Path) -> Optional[ParsedResult]:
    text = str(path)
    family, mode = classify_model("", text, path)
    if family == "Unknown":
        return None
    return ParsedResult(
        k=extract_pattern_int(text, r"_k(\d+)") or extract_pattern_int(text, r"k(\d+)"),
        C=extract_pattern_float(text, r"_C([0-9.]+)"),
        F=extract_pattern_int(text, r"_F(\d+)"),
        T=extract_pattern_int(text, r"_T(\d+)"),
        model_family=family,
        model_mode=mode,
        seed=extract_pattern_int(text, r"seed[_-]?(\d+)"),
        train_regime=extract_train_regime(text) or TARGET_TRAIN_REGIME,
        eval_protocol="unknown_eval_protocol",
        scenario=Path(path).stem,
        source_type="log_inferred",
        main_result_confidence="verify",
        from_archive="archive" in path.parts,
        source_file=rel(path),
        notes="log filename only; metrics not parsed",
    )


def collect_results() -> Tuple[List[ParsedResult], List[ParsedResult]]:
    parsed: List[ParsedResult] = []
    needs_verification: List[ParsedResult] = []

    for path in sorted(REPO_ROOT.glob("**/cross_regime_results.json")):
        if skip_path(path):
            continue
        result = parse_json_result(path)
        if result is None:
            continue
        parsed.append(result)

    for path in sorted(REPO_ROOT.glob("**/*.csv")):
        if skip_path(path):
            continue
        for result in parse_csv_results(path):
            parsed.append(result)

    for path in sorted(REPO_ROOT.glob("**/*.log")):
        if skip_path(path):
            continue
        result = parse_log_results(path)
        if result is not None:
            needs_verification.append(result)

    return parsed, needs_verification


def skip_path(path: Path) -> bool:
    parts = set(path.parts)
    if ".git" in parts or ".venv" in parts or "__pycache__" in parts:
        return True
    if OUT_DIR in path.parents:
        return True
    return False


def source_rank(row: ParsedResult) -> Tuple[int, int, str]:
    current_rank = 0 if not row.from_archive else 1
    confidence_rank = {"high": 0, "medium": 1, "low": 2, "verify": 3}.get(row.main_result_confidence, 4)
    return current_rank, confidence_rank, row.timestamp


def deduplicate(rows: List[ParsedResult]) -> Tuple[List[ParsedResult], List[ParsedResult]]:
    grouped: Dict[Tuple[Any, ...], List[ParsedResult]] = defaultdict(list)
    for row in rows:
        grouped[row.dedup_key()].append(row)

    kept: List[ParsedResult] = []
    duplicates: List[ParsedResult] = []
    for group in grouped.values():
        group_sorted = sorted(group, key=source_rank)
        winner = group_sorted[0]
        kept.append(winner)
        for dup in group_sorted[1:]:
            dup.notes = add_note(dup.notes, "archive duplicate" if dup.from_archive else "duplicate lower-priority source")
            duplicates.append(dup)
    return kept, duplicates


def is_target_scope(row: ParsedResult) -> Tuple[bool, str]:
    reasons = []
    if row.C is None or abs(float(row.C) - TARGET_C) > 1e-8:
        reasons.append("C not target")
    if row.k not in TARGET_K:
        reasons.append("k not target")
    if row.F != TARGET_F:
        reasons.append("F not target")
    if row.T != TARGET_T:
        reasons.append("T not target")
    if row.seed not in TARGET_SEEDS:
        reasons.append("seed not target")
    if row.train_regime != TARGET_TRAIN_REGIME:
        reasons.append("train_regime not target")
    if row.mean_value_accept_ratio is None:
        reasons.append("missing metrics")
    if row.model_family == "Unknown":
        reasons.append("unclear source")
    return not reasons, "; ".join(reasons)


def split_rows(rows: List[ParsedResult], duplicate_rows: List[ParsedResult], log_rows: List[ParsedResult]) -> Tuple[List[ParsedResult], List[ParsedResult], List[Dict[str, Any]], List[ParsedResult]]:
    main_candidates: List[ParsedResult] = []
    appendix: List[ParsedResult] = []
    excluded: List[Dict[str, Any]] = []
    needs_verification: List[ParsedResult] = list(log_rows)

    for row in duplicate_rows:
        excluded.append(row_to_dict(row, reason_excluded=row.notes or "duplicate lower-priority source"))

    for row in rows:
        in_scope, reason = is_target_scope(row)
        if row.main_result_confidence == "verify":
            needs_verification.append(row)
            continue
        if row.model_family in APPENDIX_MODEL_FAMILIES:
            appendix.append(row)
            excluded.append(row_to_dict(row, reason_excluded="dual appendix only"))
            continue
        if row.model_family not in MAIN_MODEL_FAMILIES:
            excluded.append(row_to_dict(row, reason_excluded=reason or "unclear source"))
            continue
        if row.eval_protocol != "cross_regime_12_static_eval":
            excluded.append(row_to_dict(row, reason_excluded=reason or "eval_protocol not preferred"))
            continue
        if not in_scope:
            excluded.append(row_to_dict(row, reason_excluded=reason))
            continue
        main_candidates.append(row)

    main_rows = choose_best_main_rows(main_candidates)
    main_keys = {r.dedup_key() for r in main_rows}
    for row in main_candidates:
        if row.dedup_key() not in main_keys:
            excluded.append(row_to_dict(row, reason_excluded="lower-priority main candidate"))

    return main_rows, appendix, excluded, needs_verification


def choose_best_main_rows(rows: List[ParsedResult]) -> List[ParsedResult]:
    grouped: Dict[Tuple[Any, ...], List[ParsedResult]] = defaultdict(list)
    for row in rows:
        key = (
            row.model_family,
            row.k,
            row.seed,
            row.C,
            row.F,
            row.T,
            row.train_regime,
            row.eval_protocol,
        )
        grouped[key].append(row)
    return [sorted(group, key=source_rank)[0] for group in grouped.values()]


def row_to_dict(row: ParsedResult, reason_excluded: str = "") -> Dict[str, Any]:
    return {
        "k": row.k,
        "C": row.C,
        "F": row.F,
        "T": row.T,
        "model_family": row.model_family,
        "model_mode": row.model_mode,
        "seed": row.seed,
        "train_regime": row.train_regime,
        "eval_protocol": row.eval_protocol,
        "scenario": row.scenario,
        "mean_value_accept_ratio": row.mean_value_accept_ratio,
        "worst_regime_value_accept_ratio": row.worst_regime_value_accept_ratio,
        "std_value_accept_ratio_across_regimes": row.std_value_accept_ratio_across_regimes,
        "avg_drops": row.avg_drops,
        "avg_flushes": row.avg_flushes,
        "avg_drop_rate": row.avg_drop_rate,
        "avg_count_accept_ratio": row.avg_count_accept_ratio,
        "num_test_regimes": row.num_test_regimes,
        "action_space_size": action_space_size(row),
        "policy_output_size": policy_output_size(row),
        "source_type": row.source_type,
        "main_result_confidence": row.main_result_confidence,
        "from_archive": row.from_archive,
        "source_file": row.source_file,
        "notes": row.notes,
        "reason_excluded": reason_excluded,
    }


def main_table_rows(main_rows: List[ParsedResult]) -> List[Dict[str, Any]]:
    rows = []
    for row in sorted(main_rows, key=lambda r: (r.k or 0, r.model_family, r.seed or 0, r.source_file)):
        d = row_to_dict(row)
        d.pop("reason_excluded", None)
        rows.append(d)
    return rows


def aggregate_by_model(main_rows: List[ParsedResult]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, str], List[ParsedResult]] = defaultdict(list)
    for row in main_rows:
        if row.k is not None:
            grouped[(row.k, row.model_family)].append(row)

    out: List[Dict[str, Any]] = []
    for (k, family), group in sorted(grouped.items()):
        means = [r.mean_value_accept_ratio for r in group]
        out.append(
            {
                "k": k,
                "model_family": family,
                "num_seeds": len({r.seed for r in group if r.seed is not None}),
                "mean_of_mean_value_accept_ratio": mean(means),
                "sd_of_mean_value_accept_ratio": std(means),
                "mean_worst_regime_value_accept_ratio": mean(r.worst_regime_value_accept_ratio for r in group),
                "mean_avg_drops": mean(r.avg_drops for r in group),
                "mean_avg_drop_rate": mean(r.avg_drop_rate for r in group),
                "action_space_size": action_space_size(group[0]),
                "short_interpretation": interpretation(k, family, group),
                "source_file": "; ".join(sorted({r.source_file for r in group})),
            }
        )

    return out


def interpretation(k: int, family: str, group: List[ParsedResult]) -> str:
    if family == "DQN_baseline":
        return f"DQN uses flat joint actions; action space is {(k + 1) ** 2} at k={k}."
    if family == "Factorized_AC":
        return "Factorized AC decomposes settle and flush heads for structural scaling."
    if family == "Basic_AC_or_PPO":
        return "Basic non-factorized PPO/AC result is present for this k." if group else "Missing basic PPO result for this k."
    return ""


def regime_level_rows(main_rows: List[ParsedResult]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in main_rows:
        for rr in row.regime_rows:
            out.append(
                {
                    "k": row.k,
                    "C": row.C,
                    "F": row.F,
                    "T": row.T,
                    "model_family": row.model_family,
                    "model_mode": row.model_mode,
                    "seed": row.seed,
                    "train_regime": row.train_regime,
                    "eval_protocol": row.eval_protocol,
                    "test_regime": rr.get("test_regime"),
                    "value_accept_ratio": rr.get("value_accept_ratio"),
                    "drops": rr.get("drops"),
                    "flushes": rr.get("flushes"),
                    "drop_rate": rr.get("drop_rate"),
                    "count_accept_ratio": rr.get("count_accept_ratio"),
                    "scenario": row.scenario,
                    "source_type": row.source_type,
                    "main_result_confidence": row.main_result_confidence,
                    "from_archive": row.from_archive,
                    "source_file": row.source_file,
                }
            )
    return sorted(out, key=lambda r: (r["k"] or 0, r["model_family"], r["seed"] or 0, str(r["test_regime"])))


def expected_vs_found(main_rows: List[ParsedResult], needs_verification: List[ParsedResult]) -> List[Dict[str, Any]]:
    lookup: Dict[Tuple[int, int, str], ParsedResult] = {}
    for row in main_rows:
        if row.k is not None and row.seed is not None:
            lookup[(row.k, row.seed, row.model_family)] = row
    verify_lookup = {(r.k, r.seed, r.model_family): r for r in needs_verification if r.k is not None and r.seed is not None}

    out: List[Dict[str, Any]] = []
    for k in TARGET_K:
        for seed in TARGET_SEEDS:
            for family in MAIN_MODEL_FAMILIES:
                found = lookup.get((k, seed, family))
                verify = verify_lookup.get((k, seed, family))
                status = "found" if found else ("needs_verification" if verify else "missing")
                source = found or verify
                out.append(
                    {
                        "k": k,
                        "seed": seed,
                        "model_family": family,
                        "expected": True,
                        "found": bool(found),
                        "status": status,
                        "source_type": source.source_type if source else "",
                        "source_file": source.source_file if source else "",
                        "notes": source.notes if source else missing_note(k, seed, family),
                    }
                )
    return out


def missing_note(k: int, seed: int, family: str) -> str:
    if family == "Basic_AC_or_PPO":
        return "No clear non-factorized PPO/AC result found; do not infer from factorized or dual AC."
    return f"No target-scope {family} result found for k={k}, seed={seed}."


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_readme(
    path: Path,
    main_rows: List[Dict[str, Any]],
    appendix_rows: List[Dict[str, Any]],
    excluded_rows: List[Dict[str, Any]],
    needs_rows: List[Dict[str, Any]],
    matrix_rows: List[Dict[str, Any]],
    generated_files: List[Path],
) -> None:
    included_counts = Counter((r["model_family"], r["k"]) for r in main_rows)
    appendix_counts = Counter((r["model_family"], r["k"]) for r in appendix_rows)
    excluded_counts = Counter(r.get("reason_excluded", "") for r in excluded_rows)
    missing = [r for r in matrix_rows if r["status"] != "found"]
    safe_to_cite = [r for r in main_rows if r["main_result_confidence"] in {"high", "medium"}]

    dqn_ks = {r["k"] for r in main_rows if r["model_family"] == "DQN_baseline"}
    fac_ks = {r["k"] for r in main_rows if r["model_family"] == "Factorized_AC"}
    basic_present = any(r["model_family"] == "Basic_AC_or_PPO" for r in main_rows)
    dqn_vs_fac_supported = set(TARGET_K).issubset(dqn_ks) and set(TARGET_K).issubset(fac_ks)

    lines = [
        "# K-Wallet Main Results Summary",
        "",
        "This directory was generated by `build_kwallet_main_results_summary.py`.",
        "",
        "## Analysis Scope",
        "",
        f"- TARGET_K: {TARGET_K}",
        f"- TARGET_SEEDS: {TARGET_SEEDS}",
        f"- TARGET_C: {TARGET_C}",
        f"- TARGET_F: {TARGET_F}",
        f"- TARGET_T: {TARGET_T}",
        f"- TARGET_TRAIN_REGIME: `{TARGET_TRAIN_REGIME}`",
        "",
        "## Generated Files",
        "",
    ]
    lines.extend(f"- `{rel(p)}`" for p in generated_files)
    lines.extend(
        [
            "",
            "## Results Included In Main Comparison",
            "",
        ]
    )
    if included_counts:
        lines.extend(f"- {family}, k={k}: {count} row(s)" for (family, k), count in sorted(included_counts.items()))
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Results Used Only For Appendix",
            "",
        ]
    )
    if appendix_counts:
        lines.extend(f"- {family}, k={k}: {count} row(s)" for (family, k), count in sorted(appendix_counts.items()))
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Results Excluded",
            "",
        ]
    )
    if excluded_counts:
        lines.extend(f"- {reason or 'unspecified'}: {count}" for reason, count in sorted(excluded_counts.items()))
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Results Needing Verification",
            "",
            f"- {len(needs_rows)} row(s) need verification. These rows are not included in the main comparison.",
            "",
            "## Missing Expected Experiments",
            "",
        ]
    )
    if missing:
        lines.extend(f"- k={r['k']} seed={r['seed']} {r['model_family']}: {r['status']} ({r['notes']})" for r in missing)
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Research Story",
            "",
            "DQN treats each settle-flush combination as a flat joint action. As k increases, the joint action space grows quadratically. Factorized AC decomposes the policy into settle and flush heads, matching the K-Wallet action structure more directly. The key comparison is whether Factorized AC remains more stable than DQN across k=3,6,12 and seeds 123,323,532.",
            "",
            "## Clean Main Story Status",
            "",
            f"- Is DQN vs Factorized AC currently supported across k=3,6,12? {'Yes' if dqn_vs_fac_supported else 'Not fully yet.'}",
            f"- Is Basic_AC_or_PPO actually available? {'Yes' if basic_present else 'No clear non-factorized PPO/AC result was found.'}",
            "- Which missing runs are most important to run next?",
        ]
    )
    priority_missing = [
        r for r in missing
        if r["model_family"] in {"DQN_baseline", "Factorized_AC", "Basic_AC_or_PPO"}
    ][:12]
    if priority_missing:
        lines.extend(f"  - k={r['k']} seed={r['seed']} {r['model_family']}" for r in priority_missing)
    else:
        lines.append("  - None")
    lines.extend(
        [
            "- Which results are safe to cite in the report?",
        ]
    )
    if safe_to_cite:
        for r in safe_to_cite:
            lines.append(
                f"  - {r['model_family']} k={r['k']} seed={r['seed']} from `{r['source_file']}` ({r['main_result_confidence']})"
            )
    else:
        lines.append("  - None")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parsed_rows, log_rows = collect_results()
    deduped_rows, duplicate_rows = deduplicate(parsed_rows)
    main_rows, appendix_rows, excluded_rows, needs_verification = split_rows(deduped_rows, duplicate_rows, log_rows)

    main_dicts = main_table_rows(main_rows)
    aggregate_dicts = aggregate_by_model(main_rows)
    regime_dicts = regime_level_rows(main_rows)
    appendix_dicts = [row_to_dict(r) for r in sorted(appendix_rows, key=lambda r: (r.k or 0, r.model_family, r.seed or 0, r.source_file))]
    needs_dicts = [row_to_dict(r) for r in needs_verification]
    matrix_dicts = expected_vs_found(main_rows, needs_verification)
    missing_dicts = [r for r in matrix_dicts if r["status"] != "found"]

    files = {
        "main": OUT_DIR / "kwallet_main_k_scaling_summary.csv",
        "aggregate": OUT_DIR / "kwallet_k_scaling_aggregate_by_model.csv",
        "regime": OUT_DIR / "kwallet_regime_level_results.csv",
        "excluded": OUT_DIR / "kwallet_excluded_results.csv",
        "matrix": OUT_DIR / "kwallet_expected_vs_found_matrix.csv",
        "readme": OUT_DIR / "kwallet_results_summary_README.md",
        "excel": OUT_DIR / "kwallet_main_results_summary.xlsx",
    }

    write_csv(files["main"], main_dicts)
    write_csv(files["aggregate"], aggregate_dicts)
    write_csv(files["regime"], regime_dicts)
    write_csv(files["excluded"], excluded_rows)
    write_csv(files["matrix"], matrix_dicts)

    with pd.ExcelWriter(files["excel"], engine="openpyxl") as writer:
        pd.DataFrame(main_dicts).to_excel(writer, index=False, sheet_name="k_scaling_summary")
        pd.DataFrame(aggregate_dicts).to_excel(writer, index=False, sheet_name="aggregate_by_model")
        pd.DataFrame(regime_dicts).to_excel(writer, index=False, sheet_name="regime_level_results")
        pd.DataFrame(missing_dicts).to_excel(writer, index=False, sheet_name="missing_results")
        pd.DataFrame(appendix_dicts).to_excel(writer, index=False, sheet_name="appendix_dual_models")
        pd.DataFrame(excluded_rows).to_excel(writer, index=False, sheet_name="excluded_results")
        pd.DataFrame(needs_dicts).to_excel(writer, index=False, sheet_name="needs_verification")
        pd.DataFrame(matrix_dicts).to_excel(writer, index=False, sheet_name="expected_vs_found_matrix")

    write_readme(
        files["readme"],
        main_dicts,
        appendix_dicts,
        excluded_rows,
        needs_dicts,
        matrix_dicts,
        list(files.values()),
    )

    print("Generated files:")
    for path in files.values():
        print(f"- {path}")

    print("\nMain table coverage:")
    for key, count in sorted(Counter((r["model_family"], r["k"], r["seed"]) for r in main_dicts).items()):
        print(f"- model_family={key[0]} k={key[1]} seed={key[2]} rows={count}")

    print("\nAppendix dual coverage:")
    for key, count in sorted(Counter((r["model_family"], r["k"], r["seed"]) for r in appendix_dicts).items()):
        print(f"- model_family={key[0]} k={key[1]} seed={key[2]} rows={count}")

    print("\nExcluded count by reason:")
    for reason, count in sorted(Counter(r.get("reason_excluded", "") for r in excluded_rows).items()):
        print(f"- {reason or 'unspecified'}: {count}")

    print(f"\nNeeds verification count: {len(needs_dicts)}")

    print("\nMissing expected combinations:")
    for row in missing_dicts:
        print(f"- k={row['k']} seed={row['seed']} model_family={row['model_family']} status={row['status']}")

    print("\nRecommended next runs:")
    for row in missing_dicts[:12]:
        print(f"- Run {row['model_family']} for k={row['k']} seed={row['seed']} under MIX12_EQ C1200 F3 T1000.")


if __name__ == "__main__":
    main()
