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

from general_model_v2.scripts.run_flat_ppo import add_env_metadata, print_startup
from general_model_v2.utils.io_utils import (
    DEFAULT_LOG_ROOT,
    DEFAULT_RESULT_ROOT,
    build_run_paths,
    ensure_run_dirs,
    write_csv,
    write_required_run_outputs,
)
from general_model_v2.utils.pool_loader import (
    build_pool_fingerprints,
    infer_t_max,
    load_train_val_pools,
    load_two_pool,
    resolve_pool_paths,
)
from general_model_v2.utils.seed_utils import set_seed
from general_model_v2.utils.threshold_expert import (
    evaluate_cross_regime_threshold,
    select_best_eta,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-pool threshold benchmark.")
    parser.add_argument("--train_regime", type=str, default="MIX12_EQ")
    parser.add_argument("--C", type=float, default=1000.0)
    parser.add_argument("--F", type=int, default=3)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--T_max", type=float, default=None)
    parser.add_argument("--flush_levels", type=int, default=17)
    parser.add_argument("--flush_grid", choices=["uniform"], default="uniform")
    parser.add_argument("--threshold_mode", choices=["pool_specific", "global"], default="pool_specific")
    parser.add_argument(
        "--state_feature_mode",
        choices=["base", "pressure", "pressure_release"],
        default="base",
    )
    parser.add_argument("--mask_mode", choices=["none", "safe"], default="none")
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
    if args.C <= 0 or args.T <= 0 or args.flush_levels < 2:
        raise ValueError("--C and --T must be positive; --flush_levels must be at least 2.")
    if args.F < 0:
        raise ValueError("--F must be non-negative.")
    if args.episodes <= 0 or args.eval_episodes <= 0:
        raise ValueError("--episodes and --eval_episodes must be positive.")
    if args.money_p <= 0 or args.money_tau < 0 or args.drop_penalty < 0:
        raise ValueError("Invalid reward parameters.")


def build_config(args: argparse.Namespace, t_max: float, pool_paths: Dict[str, Any]) -> Dict[str, Any]:
    model_name = "global_threshold" if args.threshold_mode == "global" else "grid_threshold"
    return {
        "model_name": model_name,
        "model_mode": model_name,
        "threshold_mode": str(args.threshold_mode),
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
            "reward_scale": 1.0,
            "clip_eps": 0.0,
        },
        "eval": {
            "num_episodes": int(args.eval_episodes),
            "max_steps": int(args.T),
        },
        "output": {
            "result_root": args.result_root,
            "log_root": args.log_root,
        },
    }


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
    config = build_config(args, t_max, pool_paths)
    add_env_metadata(config)
    config["pool_fingerprints"] = build_pool_fingerprints(pool_paths, args.T)
    selected_eta, eta_rows = select_best_eta(config, val_pool)
    config["eta"] = float(selected_eta)
    config["selection_method"] = "grid_validation_money"
    paths = build_run_paths(config)
    config["scenario"] = paths["scenario"]
    ensure_run_dirs(paths, args.save_mode)
    print_startup(config, paths, t_max)
    print(f"threshold_mode: {config['threshold_mode']}")
    print(f"eta: {float(selected_eta):.6f} (grid_validation_money)")

    test_pools = {
        regime: load_two_pool(pool_path, args.T)
        for regime, pool_path in pool_paths["test_pool_paths"].items()
    }
    cross = evaluate_cross_regime_threshold(float(selected_eta), config, test_pools)
    results = {
        "config": config,
        "scenario": paths["scenario"],
        "model_name": config["model_name"],
        "model_mode": config["model_mode"],
        "train_regime": args.train_regime,
        "seed": int(args.seed),
        "timestamp": datetime.now().isoformat(),
        "threshold_mode": config["threshold_mode"],
        "selected_eta": float(selected_eta),
        "selection_method": "grid_validation_money",
        **cross,
    }
    training_log = [{
        "selection_method": "grid_validation_money",
        "threshold_mode": config["threshold_mode"],
        "selected_eta": float(selected_eta),
        "C": float(args.C),
        "F": int(args.F),
        "T": int(args.T),
        "flush_levels": int(args.flush_levels),
        "seed": int(args.seed),
    }]
    validation_log = eta_rows
    if args.save_mode == "full":
        write_csv(eta_rows, paths["eta_grid_log_path"])
    write_required_run_outputs(config, paths, training_log, validation_log, results)
    print(f"mean money: {results['aggregate']['mean_money']:.2f}")
    print(f"mean value_accept_ratio: {results['aggregate']['mean_value_accept_ratio']:.6f}")
    print(f"worst regime value_accept_ratio: {results['aggregate']['worst_regime_value_accept_ratio']:.6f}")
    print(f"mean drops: {results['aggregate']['mean_drops']:.2f}")
    print(f"mean flushes: {results['aggregate']['mean_flushes']:.2f}")


if __name__ == "__main__":
    main()
