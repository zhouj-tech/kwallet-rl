from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

THIS_FILE = Path(__file__).resolve()
IDEA5_ROOT = THIS_FILE.parents[2]
if str(IDEA5_ROOT) not in sys.path:
    sys.path.insert(0, str(IDEA5_ROOT))

from general_model_v2.models.flat_ppo import FlatPPOAgent
from general_model_v2.utils.io_utils import (
    DEFAULT_LOG_ROOT,
    DEFAULT_RESULT_ROOT,
    build_run_paths,
    ensure_run_dirs,
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
from general_model_v2.utils.training_utils import (
    evaluate_cross_regime,
    make_env_from_config,
    print_action_diagnostics,
    print_env_summary,
    train_agent,
)


def parse_args(
    description: str = "Flat joint-action PPO for the two-pool general collateral model.",
    extra_args_fn: Optional[Callable[[argparse.ArgumentParser], None]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--train_regime", type=str, default="MIX12_EQ")
    parser.add_argument("--C", type=float, default=1000.0)
    parser.add_argument("--F", type=int, default=3)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--T_max", type=float, default=None)
    parser.add_argument("--flush_levels", type=int, default=17)
    parser.add_argument("--flush_grid", choices=["uniform"], default="uniform")
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
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--clip_eps", type=float, default=0.1)
    parser.add_argument("--reward_scale", type=float, default=100.0)
    if extra_args_fn is not None:
        extra_args_fn(parser)
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
    if args.hidden_size <= 0:
        raise ValueError("--hidden_size must be positive.")
    if args.learning_rate <= 0:
        raise ValueError("--learning_rate must be positive.")
    if args.clip_eps <= 0:
        raise ValueError("--clip_eps must be positive.")
    if args.reward_scale <= 0:
        raise ValueError("--reward_scale must be positive.")


def build_config(args: argparse.Namespace, t_max: float, pool_paths: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model_name": "flat_ppo",
        "model_mode": "flat_ppo",
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
            "learning_rate": float(args.learning_rate),
            "gamma": 0.98,
            "gae_lambda": 0.95,
            "clip_eps": float(args.clip_eps),
            "reward_scale": float(args.reward_scale),
            "update_epochs": 4,
            "minibatch_size": 256,
            "value_coef": 0.5,
            "entropy_coef_start": 0.03,
            "entropy_coef_end": 0.003,
            "max_grad_norm": 1.0,
            "val_every": 50,
            "val_num_episodes": int(args.eval_episodes),
            "hidden_size": int(args.hidden_size),
            "log_every": 25,
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


def add_env_metadata(config: Dict[str, Any]) -> None:
    env = make_env_from_config(config)
    config["env"]["num_flush_choices"] = int(env.num_flush_choices)
    config["env"]["action_size"] = int(env.action_size)
    config["env"]["factorized_policy_output_size"] = int(env.factorized_policy_output_size)
    config["env"]["state_size"] = int(env.state_size)
    config["env"]["flush_fractions"] = list(env.flush_fractions)
    config["env"]["flush_action_decoding"] = env.flush_action_decoding_summary()


def print_startup(config: Dict[str, Any], paths: Dict[str, Any], t_max: float) -> None:
    print(f"scenario: {paths['scenario']}")
    print(f"results : {paths['run_dir']}")
    print(f"logs    : {paths['log_dir']}")
    print(f"T_max   : {float(t_max):.6f}")
    env = make_env_from_config(config)
    print_env_summary(env)
    print(f"reward_scale: {float(config['train']['reward_scale']):g}")
    print(f"clip_eps: {float(config['train']['clip_eps']):g}")


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
    paths = build_run_paths(config)
    config["scenario"] = paths["scenario"]
    ensure_run_dirs(paths, args.save_mode)
    print_startup(config, paths, t_max)

    env = make_env_from_config(config)
    agent = FlatPPOAgent(
        state_size=env.state_size,
        config=config,
        num_flush_choices=env.num_flush_choices,
        flush_fractions=[0.0 for _ in range(env.num_flush_choices)],
    )
    training_log, validation_log = train_agent(agent, config, paths, train_pool, val_pool)
    test_pools = {
        regime: load_two_pool(pool_path, args.T)
        for regime, pool_path in pool_paths["test_pool_paths"].items()
    }
    results = evaluate_cross_regime(agent, config, test_pools)
    write_required_run_outputs(config, paths, training_log, validation_log, results)
    print(f"mean money: {results['aggregate']['mean_money']:.2f}")
    print(f"mean value_accept_ratio: {results['aggregate']['mean_value_accept_ratio']:.6f}")
    print(f"worst regime value_accept_ratio: {results['aggregate']['worst_regime_value_accept_ratio']:.6f}")
    print(f"mean drops: {results['aggregate']['mean_drops']:.2f}")
    print(f"mean flushes: {results['aggregate']['mean_flushes']:.2f}")
    print_action_diagnostics(results)


if __name__ == "__main__":
    main()
