#!/usr/bin/env python3
"""Evaluate standard online K-wallet policies on the final shared eval pools.

This is an evaluation-only bridge between the standard-paper rule policies and
the generated pools used by the final K-Wallet RL comparisons. It does not load
trained models or reuse RL trajectories.

The policy names and rule structure follow:
  Ghada Almashaqbeh, Sixia Chen, and Alexander Russell,
  "Competitive Policies for Online Collateral Maintenance," AFT 2024.
  https://doi.org/10.4230/LIPIcs.AFT.2024.26

Simulator adaptation:
  - The standard paper gives rule-level behavior but does not prescribe a
    tie-break when multiple wallets can settle the same transaction. This
    evaluator uses the lowest-index eligible wallet.
  - The final K-Wallet environment treats C as total collateral and partitions
    it across k wallets, so each wallet has capacity C / k.
  - Each refreshed wallet counts as one flush, matching the existing K-Wallet
    metric convention. FlushAll can therefore add multiple flushes in one step.
  - interface_mode=native preserves that standard-paper rule behavior.
    interface_mode=one_flush limits every policy to at most one refreshed wallet
    per step so the refresh interface matches the learned K-Wallet policies.
  - A flush makes a wallet unavailable for the current step and the next F - 1
    decisions, matching the shared KWalletEnv timing convention.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[4]
IDEA4_AC_CODE_DIR = THIS_FILE.parent
IDEA3_CONTEXT_DIR = PROJECT_ROOT / "src" / "idea3" / "context_attention"
if str(IDEA4_AC_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(IDEA4_AC_CODE_DIR))
if str(IDEA3_CONTEXT_DIR) not in sys.path:
    sys.path.insert(0, str(IDEA3_CONTEXT_DIR))

from kwallet_ctx_attn_fair_benchmark import (  # noqa: E402
    DATA_POOL_DIR,
    DEFAULT_STATIC_EVAL_FILES,
    REGIME_ORDER,
    compute_cross_regime_aggregate,
    load_tx_pool,
    verify_data_integrity,
)
from kwallet_basic_ppo_fair_benchmark import (  # noqa: E402
    add_eval_money_metrics,
    compute_eval_money_aggregate,
    summarize_episode_metrics,
)


POLICY_NAMES = ("FlushAll", "FlushWhenFull", "FlushTwoWhenFull")
INTERFACE_MODES = ("native", "one_flush")
DEFAULT_NATIVE_OUTPUT_DIR = PROJECT_ROOT / "src" / "idea4" / "ac" / "results" / "standard_policies"
DEFAULT_CONSTRAINED_OUTPUT_DIR = (
    PROJECT_ROOT / "src" / "idea4" / "ac" / "results" / "standard_policies_constrained"
)
AGGREGATE_FIELDS = [
    "timestamp",
    "policy",
    "interface_mode",
    "C",
    "k",
    "F",
    "T",
    "eval_episodes",
    "money_p",
    "money_tau",
    "mean_value_accept_ratio",
    "worst_regime_value_accept_ratio",
    "std_value_accept_ratio_across_regimes",
    "mean_drops",
    "mean_flushes",
    "mean_eval_money",
    "worst_regime_eval_money",
    "std_eval_money_across_regimes",
    "mean_settled",
    "run_json",
]


@dataclass
class RuleBaselineEnv:
    """Minimal rule-policy simulator with the shared KWalletEnv timing semantics."""

    C: float
    k: int
    F: int
    max_steps: int
    policy: str
    interface_mode: str

    def __post_init__(self) -> None:
        if self.policy not in POLICY_NAMES:
            raise ValueError(f"Unsupported policy={self.policy}")
        if self.interface_mode not in INTERFACE_MODES:
            raise ValueError(f"Unsupported interface_mode={self.interface_mode}")
        if self.k <= 0:
            raise ValueError("k must be positive")
        if self.F <= 0:
            raise ValueError("F must be positive")
        if self.policy == "FlushTwoWhenFull" and self.k % 2 != 0:
            raise ValueError("FlushTwoWhenFull requires an even k so wallets can be paired")
        self.wallet_size = float(self.C) / int(self.k)
        self.reset()

    def reset(self) -> None:
        self.wallets = [self.wallet_size] * self.k
        self.freeze_until = [-1] * self.k
        self.pending_refill = [False] * self.k
        self.total_settled = 0.0
        self.num_flushes = 0
        self.drops = 0
        self.oversize_drops = 0
        self.insufficient_drops = 0
        self.time = 0
        self.current_wallet = 0
        self.current_pair = 0

    def _usable(self, wallet_idx: int) -> bool:
        return self.time > self.freeze_until[wallet_idx]

    def _settle(self, wallet_idx: int, tx: float) -> None:
        self.wallets[wallet_idx] -= tx
        self.total_settled += tx

    def _drop(self, oversize: bool = False) -> None:
        self.drops += 1
        if oversize:
            self.oversize_drops += 1
        else:
            self.insufficient_drops += 1

    def _flush(self, wallet_indices: Iterable[int]) -> None:
        for wallet_idx in wallet_indices:
            if not self._usable(wallet_idx):
                continue
            self.pending_refill[wallet_idx] = True
            self.wallets[wallet_idx] = 0.0
            self.freeze_until[wallet_idx] = self.time + self.F - 1
            self.num_flushes += 1
            if self.interface_mode == "one_flush":
                break

    def _advance_time(self) -> None:
        self.time += 1
        for wallet_idx in range(self.k):
            if self.pending_refill[wallet_idx] and self._usable(wallet_idx):
                self.wallets[wallet_idx] = self.wallet_size
                self.pending_refill[wallet_idx] = False

    def _first_fit(self, wallet_indices: Iterable[int], tx: float) -> int | None:
        for wallet_idx in wallet_indices:
            if self._usable(wallet_idx) and self.wallets[wallet_idx] >= tx:
                return wallet_idx
        return None

    def _next_usable_wallet(self, start: int) -> int | None:
        for offset in range(self.k):
            wallet_idx = (start + offset) % self.k
            if self._usable(wallet_idx):
                return wallet_idx
        return None

    def _next_available_pair(self, start: int) -> int | None:
        num_pairs = self.k // 2
        for offset in range(num_pairs):
            pair_idx = (start + offset) % num_pairs
            pair_start = 2 * pair_idx
            if self._usable(pair_start) or self._usable(pair_start + 1):
                return pair_idx
        return None

    def _step_flush_all(self, tx: float) -> None:
        fit_idx = self._first_fit(range(self.k), tx)
        if fit_idx is not None:
            self._settle(fit_idx, tx)
            return
        self._flush(wallet_idx for wallet_idx in range(self.k) if self._usable(wallet_idx))
        self._drop()

    def _step_flush_when_full(self, tx: float) -> None:
        wallet_idx = self._next_usable_wallet(self.current_wallet)
        if wallet_idx is None:
            self._drop()
            return
        self.current_wallet = wallet_idx
        if self.wallets[wallet_idx] >= tx:
            self._settle(wallet_idx, tx)
            return
        self._flush([wallet_idx])
        self.current_wallet = (self.current_wallet + 1) % self.k
        self._drop()

    def _step_flush_two_when_full(self, tx: float) -> None:
        pair_idx = self._next_available_pair(self.current_pair)
        if pair_idx is None:
            self._drop()
            return
        self.current_pair = pair_idx
        pair_start = 2 * pair_idx
        pair = (pair_start, pair_start + 1)
        fit_idx = self._first_fit(pair, tx)
        if fit_idx is not None:
            self._settle(fit_idx, tx)
            return
        self._flush(wallet_idx for wallet_idx in pair if self._usable(wallet_idx))
        self.current_pair = (self.current_pair + 1) % (self.k // 2)
        self._drop()

    def step(self, tx: float) -> None:
        if tx > self.wallet_size:
            self._drop(oversize=True)
            self._advance_time()
            return
        if self.policy == "FlushAll":
            self._step_flush_all(tx)
        elif self.policy == "FlushWhenFull":
            self._step_flush_when_full(tx)
        else:
            self._step_flush_two_when_full(tx)
        self._advance_time()

    def get_metrics(self) -> Dict[str, float]:
        return {
            "settled": self.total_settled,
            "drops": float(self.drops),
            "oversize_drops": float(self.oversize_drops),
            "insufficient_drops": float(self.insufficient_drops),
            "flushes": float(self.num_flushes),
            "utilization": self.total_settled / (self.C * self.max_steps),
            "avg_tx_value": self.total_settled / max(1, self.max_steps - self.drops),
            "drop_rate": self.drops / self.max_steps,
        }


def evaluate_policy_on_array(
    policy: str,
    tx_pool: np.ndarray,
    C: float,
    k: int,
    F: int,
    eval_episodes: int,
    money_p: float,
    money_tau: float,
    interface_mode: str,
) -> Dict[str, Any]:
    eval_episodes = min(int(eval_episodes), int(tx_pool.shape[0]))
    max_steps = int(tx_pool.shape[1])
    all_results: List[Dict[str, float]] = []
    for episode_idx in range(eval_episodes):
        env = RuleBaselineEnv(
            C=C,
            k=k,
            F=F,
            max_steps=max_steps,
            policy=policy,
            interface_mode=interface_mode,
        )
        tx_stream = tx_pool[episode_idx]
        total_requested_value = float(np.sum(tx_stream))
        for tx in tx_stream:
            env.step(float(tx))
        metrics = env.get_metrics()
        metrics["value_accept_ratio"] = (
            metrics["settled"] / total_requested_value if total_requested_value > 0 else 0.0
        )
        metrics["count_accept_ratio"] = (
            (max_steps - metrics["drops"]) / max_steps if max_steps > 0 else 0.0
        )
        metrics["total_requested_value"] = total_requested_value
        metrics["total_tx_count"] = float(max_steps)
        metrics["accepted_count"] = float(max_steps - metrics["drops"])
        add_eval_money_metrics(
            metrics,
            {"reward": {"money_p": float(money_p), "money_tau": float(money_tau)}},
        )
        all_results.append(metrics)
    return {
        "num_episodes": eval_episodes,
        "summary": summarize_episode_metrics(all_results),
        "raw_results": all_results,
    }


def evaluate_cross_regime(args: argparse.Namespace) -> Dict[str, Any]:
    data_root = Path(args.data_root).expanduser().resolve()
    test_results: Dict[str, Any] = {}
    for regime in REGIME_ORDER:
        pool_path = data_root / DEFAULT_STATIC_EVAL_FILES[regime]
        if not verify_data_integrity(str(pool_path), expected_steps=args.T, label=regime):
            raise RuntimeError(f"Data verification failed for {regime}")
        tx_pool = load_tx_pool(str(pool_path), expected_steps=args.T)
        result = evaluate_policy_on_array(
            policy=args.policy,
            tx_pool=tx_pool,
            C=args.C,
            k=args.k,
            F=args.F,
            eval_episodes=args.eval_episodes,
            money_p=args.money_p,
            money_tau=args.money_tau,
            interface_mode=args.interface_mode,
        )
        result["test_regime"] = regime
        result["test_pool_path"] = str(pool_path)
        test_results[regime] = result

    aggregate = compute_cross_regime_aggregate(test_results)
    aggregate.update(compute_eval_money_aggregate(test_results))
    return {
        "timestamp": datetime.now().isoformat(),
        "policy": args.policy,
        "interface_mode": args.interface_mode,
        "model_mode": "standard_online_policy",
        "config": {
            "C": float(args.C),
            "k": int(args.k),
            "F": int(args.F),
            "T": int(args.T),
            "eval_episodes": int(args.eval_episodes),
            "data_root": str(data_root),
            "test_pool_files": dict(DEFAULT_STATIC_EVAL_FILES),
            "money_p": float(args.money_p),
            "money_tau": float(args.money_tau),
            "money_formula": "eval_money = money_p * settled - money_tau * flushes",
            "tie_break": "lowest-index eligible wallet",
            "flush_counting": "each refreshed wallet counts as one flush",
            "interface_mode": args.interface_mode,
            "max_flush_choices_per_step": 1 if args.interface_mode == "one_flush" else None,
        },
        "test_results": test_results,
        "aggregate": aggregate,
    }


def write_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def aggregate_row(payload: Dict[str, Any], run_json_path: Path) -> Dict[str, Any]:
    config = payload["config"]
    return {
        "timestamp": payload["timestamp"],
        "policy": payload["policy"],
        "interface_mode": payload["interface_mode"],
        "C": config["C"],
        "k": config["k"],
        "F": config["F"],
        "T": config["T"],
        "eval_episodes": config["eval_episodes"],
        "money_p": config["money_p"],
        "money_tau": config["money_tau"],
        **payload["aggregate"],
        "run_json": str(run_json_path),
    }


def append_aggregate_csv(row: Dict[str, Any], aggregate_path: Path) -> None:
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    exists = aggregate_path.exists()
    fieldnames = AGGREGATE_FIELDS
    if exists:
        with aggregate_path.open("r", newline="", encoding="utf-8") as handle:
            existing_fields = next(csv.reader(handle), [])
        if existing_fields:
            fieldnames = existing_fields
    with aggregate_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one standard online K-wallet policy on the final shared eval pools."
    )
    parser.add_argument("--policy", choices=POLICY_NAMES, required=True)
    parser.add_argument("--interface_mode", choices=INTERFACE_MODES, default="native")
    parser.add_argument("--C", type=float, required=True)
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--F", type=int, default=3)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=200)
    parser.add_argument("--money_p", type=float, default=1.0)
    parser.add_argument("--money_tau", type=float, default=10.0)
    parser.add_argument("--data_root", default=str(DATA_POOL_DIR))
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Override the mode-specific output root.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the exact pools and settings without loading pools or writing results.",
    )
    return parser


def print_plan(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root).expanduser().resolve()
    print("Standard-policy same-pool evaluation plan")
    print(
        f"policy={args.policy} interface_mode={args.interface_mode} "
        f"C={args.C:g} k={args.k} F={args.F} T={args.T}"
    )
    print(f"eval_episodes={args.eval_episodes} money_p={args.money_p:g} money_tau={args.money_tau:g}")
    print(f"data_root={data_root}")
    for regime in REGIME_ORDER:
        print(f"  {regime}: {data_root / DEFAULT_STATIC_EVAL_FILES[regime]}")


def output_dir_for_args(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    if args.interface_mode == "one_flush":
        return DEFAULT_CONSTRAINED_OUTPUT_DIR.resolve()
    return DEFAULT_NATIVE_OUTPUT_DIR.resolve()


def main() -> None:
    args = build_parser().parse_args()
    print_plan(args)
    if args.dry_run:
        print("Dry run only: no pools loaded and no results written.")
        return

    payload = evaluate_cross_regime(args)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    c_value = int(args.C) if float(args.C).is_integer() else args.C
    output_dir = output_dir_for_args(args)
    run_json_path = (
        output_dir
        / "runs"
        / f"{args.policy}_{args.interface_mode}_C{c_value}_k{args.k}_T{args.T}_F{args.F}"
        / stamp
        / "cross_regime_results.json"
    )
    aggregate_path = output_dir / "aggregates" / "standard_policy_same_pool_results.csv"
    write_json(payload, run_json_path)
    append_aggregate_csv(aggregate_row(payload, run_json_path), aggregate_path)
    print(f"Saved run JSON: {run_json_path}")
    print(f"Updated aggregate CSV: {aggregate_path}")


if __name__ == "__main__":
    main()
