#threshold baseline 的总控程序：grid 模式负责在 validation pool 上找最好的 η，
# fixed 模式负责用指定或理论 η，然后统一拿这个 η 去 12 个 test regimes 上测试。
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

THIS_FILE = Path(__file__).resolve()
IDEA5_ROOT = THIS_FILE.parents[2]
if str(IDEA5_ROOT) not in sys.path:
    sys.path.insert(0, str(IDEA5_ROOT))

from general_model.baselines.threshold_baselines import (
    ETA_GRID,
    evaluate_threshold_cross_regime,
    evaluate_threshold_on_pool,
    select_best_eta,
    theoretical_eta,
)
from general_model.utils.io_utils import (
    DEFAULT_LOG_ROOT,
    DEFAULT_RESULT_ROOT,
    build_run_paths,
    ensure_run_dirs,
    write_required_run_outputs,
)
from general_model.utils.pool_loader import (
    build_pool_fingerprints,
    infer_t_max,
    load_train_val_pools,
    load_tx_pool,
    resolve_pool_paths,
)
from general_model.utils.seed_utils import set_seed
from general_model.utils.training_utils import make_env_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Threshold baselines for the general collateral model."
    )

    parser.add_argument("--mode", choices=["fixed", "grid"], default="grid")
    parser.add_argument("--eta", type=float, default=None)

    parser.add_argument("--train_regime", type=str, default="MIX12_EQ")
    parser.add_argument("--C", type=float, default=1200.0)
    parser.add_argument("--F", type=int, default=3)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--T_max", type=float, default=None)
    parser.add_argument("--flush_levels", type=int, default=5)
    parser.add_argument(
        "--flush_grid",
        choices=["uniform", "nonuniform_v1"],
        default="uniform",
    )
    parser.add_argument(
        "--state_feature_mode",
        choices=["base", "pressure"],
        default="base",
    )
    parser.add_argument(
        "--mask_mode",
        choices=["none", "safe"],
        default="none",
    )
    parser.add_argument(
        "--threshold_advantage_mode",
        choices=["none", "residual"],
        default="none",
    )

    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--money_p", type=float, default=1.0)
    parser.add_argument("--money_tau", type=float, default=100.0)
    parser.add_argument("--drop_penalty", type=float, default=0.0)

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_mode", choices=["none", "full"], default="full")

    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--result_root", type=str, default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--log_root", type=str, default=str(DEFAULT_LOG_ROOT))

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.C <= 0:
        raise ValueError("--C must be positive.")

    if args.F < 0:
        raise ValueError("--F must be non-negative.")

    if args.T <= 0:
        raise ValueError("--T must be positive.")

    if args.flush_levels < 2:
        raise ValueError("--flush_levels must be at least 2.")

    if args.episodes <= 0:
        raise ValueError("--episodes must be positive.")

    if args.eval_episodes <= 0:
        raise ValueError("--eval_episodes must be positive.")

    if args.money_p <= 0:
        raise ValueError("--money_p must be positive.")

    if args.money_tau < 0:
        raise ValueError("--money_tau must be non-negative.")

    if args.drop_penalty < 0:
        raise ValueError("--drop_penalty must be non-negative.")

    if args.eta is not None and not (0.0 < float(args.eta) <= 1.0):
        raise ValueError("--eta must be in (0, 1].")

    if args.mode == "grid" and args.eta is not None:
        print(
            "[Warning] --eta is ignored when --mode grid. "
            "Grid mode selects eta by validation money."
        )

    if args.threshold_advantage_mode == "residual":
        raise NotImplementedError(
            "--threshold_advantage_mode residual reserves the interface for "
            "future experiments; residual threshold advantage logic is not "
            "implemented yet."
        )


def build_config(
    args: argparse.Namespace,
    t_max: float,
    pool_paths: Dict[str, Any],
    selected_eta: float | None = None,
) -> Dict[str, Any]:
    model_name = "grid_threshold" if args.mode == "grid" else "fixed_threshold"

    config: Dict[str, Any] = {
        "model_name": model_name,
        "model_mode": model_name,
        "threshold_mode": args.mode,
        "seed": int(args.seed),
        "save_mode": args.save_mode,

        "env": {
            "C": float(args.C),
            "F": int(args.F),
            "T": int(args.T),
            "T_max": float(t_max),
            "flush_levels": int(args.flush_levels),
            "flush_grid": str(args.flush_grid),
            "state_feature_mode": str(args.state_feature_mode),
            "mask_mode": str(args.mask_mode),
        },

        "reward": {
            "money_p": float(args.money_p),
            "money_tau": float(args.money_tau),
            "drop_penalty": float(args.drop_penalty),
        },

        "data": {
            "train_regime": args.train_regime,
            **pool_paths,
        },

        "train": {
            "episodes": int(args.episodes),
            "max_steps": int(args.T),
            "device": args.device,
            "val_num_episodes": int(args.eval_episodes),
        },

        "eval": {
            "num_episodes": int(args.eval_episodes),
            "max_steps": int(args.T),
        },

        "output": {
            "result_root": args.result_root,
            "log_root": args.log_root,
        },

        "threshold_advantage_mode": str(args.threshold_advantage_mode),
    }

    if selected_eta is not None:
        config["eta"] = float(selected_eta)

    return config


def build_fixed_validation_log(
    selected_eta: float,
    provisional_config: Dict[str, Any],
    val_pool,
    eval_episodes: int,
) -> list[dict[str, float]]:
    val_result = evaluate_threshold_on_pool(
        eta=float(selected_eta),
        config=provisional_config,
        tx_pool=val_pool,
        num_episodes=int(eval_episodes),
        label="VAL",
    )

    s = val_result["summary"]

    row: Dict[str, float] = {
        "eta": float(selected_eta),
        "money": float(s["money"]["mean"]),
        "value_accept_ratio": float(s["value_accept_ratio"]["mean"]),
        "drops": float(s["drops"]["mean"]),
        "drop_rate": float(s["drop_rate"]["mean"]),
        "flushes": float(s["flushes"]["mean"]),
        "settled_value": float(s["settled_value"]["mean"]),
    }

    if "policy_discards" in s:
        row["policy_discards"] = float(s["policy_discards"]["mean"])
    if "capacity_drops" in s:
        row["capacity_drops"] = float(s["capacity_drops"]["mean"])
    if "policy_discard_rate" in s:
        row["policy_discard_rate"] = float(s["policy_discard_rate"]["mean"])
    if "capacity_drop_rate" in s:
        row["capacity_drop_rate"] = float(s["capacity_drop_rate"]["mean"])

    return [row]


def main() -> None:
    args = parse_args()
    validate_args(args)

    set_seed(args.seed)

    pool_paths = resolve_pool_paths(args.train_regime, args.data_root)

    train_pool, val_pool = load_train_val_pools(
        pool_paths["train_pool_path"],
        pool_paths["val_pool_path"],
        args.T,
    )

    t_max = infer_t_max(train_pool, args.T_max)

    provisional_config = build_config(
        args=args,
        t_max=t_max,
        pool_paths=pool_paths,
        selected_eta=None,
    )

    if args.mode == "grid":
        selected_eta, validation_log = select_best_eta(
            eta_grid=ETA_GRID,
            config=provisional_config,
            val_pool=val_pool,
            num_episodes=int(args.eval_episodes),
        )
        selection_method = "grid_validation_money"

    else:
        if args.eta is not None:
            selected_eta = float(args.eta)
            selection_method = "cli_eta"
        else:
            selected_eta = theoretical_eta(
                money_tau=float(args.money_tau),
                money_p=float(args.money_p),
                C=float(args.C),
                T_max=float(t_max),
            )
            selection_method = "theoretical_eta"

        validation_log = build_fixed_validation_log(
            selected_eta=selected_eta,
            provisional_config=provisional_config,
            val_pool=val_pool,
            eval_episodes=int(args.eval_episodes),
        )

    config = build_config(
        args=args,
        t_max=t_max,
        pool_paths=pool_paths,
        selected_eta=selected_eta,
    )

    config["selection_method"] = selection_method
    config["pool_fingerprints"] = build_pool_fingerprints(pool_paths, args.T)
    env_metadata = make_env_from_config(config)
    config["env"]["num_flush_choices"] = int(env_metadata.num_flush_choices)
    config["env"]["flush_fractions"] = list(env_metadata.flush_fractions)
    config["env"]["action_size"] = int(env_metadata.action_size)
    config["env"]["state_size"] = int(env_metadata.state_size)

    paths = build_run_paths(config)
    config["scenario"] = paths["scenario"]

    ensure_run_dirs(paths, args.save_mode)

    print(f"scenario: {paths['scenario']}")
    print(f"results : {paths['run_dir']}")
    print(f"logs    : {paths['log_dir']}")
    print(f"T_max   : {float(t_max):.6f}")
    print(f"flush_levels: {int(args.flush_levels)}")
    print(f"flush_grid: {args.flush_grid}")
    print(f"state_feature_mode: {args.state_feature_mode}")
    print(f"mask_mode: {args.mask_mode}")
    print(f"eta     : {float(selected_eta):.6f} ({selection_method})")
    env_for_shape = make_env_from_config(config)
    print(f"flush_fractions: {env_for_shape.flush_fractions}")
    print(f"num_flush_choices: {env_for_shape.num_flush_choices}")
    print(f"action_size: {env_for_shape.action_size}")
    print(f"state_size: {env_for_shape.state_size}")

    test_pools = {
        regime: load_tx_pool(pool_path, args.T)
        for regime, pool_path in pool_paths["test_pool_paths"].items()
    }

    cross = evaluate_threshold_cross_regime(
        eta=float(selected_eta),
        config=config,
        test_pools=test_pools,
    )

    results = {
        "config": config,
        "scenario": paths["scenario"],
        "model_name": config["model_name"],
        "model_mode": config["model_mode"],
        "train_regime": args.train_regime,
        "seed": int(args.seed),
        "timestamp": datetime.now().isoformat(),
        "selected_eta": float(selected_eta),
        "selection_method": selection_method,
        **cross,
    }

    training_log = [
        {
            "selection_method": selection_method,
            "selected_eta": float(selected_eta),
            "T_max": float(t_max),
            "C": float(args.C),
            "F": int(args.F),
            "T": int(args.T),
            "flush_levels": int(args.flush_levels),
            "flush_grid": str(args.flush_grid),
            "state_feature_mode": str(args.state_feature_mode),
            "mask_mode": str(args.mask_mode),
            "money_p": float(args.money_p),
            "money_tau": float(args.money_tau),
            "drop_penalty": float(args.drop_penalty),
            "train_regime": args.train_regime,
            "seed": int(args.seed),
        }
    ]

    write_required_run_outputs(
        config,
        paths,
        training_log,
        validation_log,
        results,
    )

    print(f"mean money: {results['aggregate']['mean_money']:.2f}")
    print(f"mean value_accept_ratio: {results['aggregate']['mean_value_accept_ratio']:.6f}")
    print(f"worst regime value_accept_ratio: {results['aggregate']['worst_regime_value_accept_ratio']:.6f}")
    print(f"mean drops: {results['aggregate']['mean_drops']:.2f}")
    print(f"mean flushes: {results['aggregate']['mean_flushes']:.2f}")


if __name__ == "__main__":
    main()
