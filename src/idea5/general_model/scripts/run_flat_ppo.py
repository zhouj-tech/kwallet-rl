from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Callable, Optional

THIS_FILE = Path(__file__).resolve()
IDEA5_ROOT = THIS_FILE.parents[2]
if str(IDEA5_ROOT) not in sys.path:
    sys.path.insert(0, str(IDEA5_ROOT))

from general_model.models.flat_ppo import FlatPPOAgent
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
from general_model.utils.training_utils import (
    evaluate_cross_regime,
    make_env_from_config,
    print_action_diagnostics,
    train_agent,
)


def parse_args(
    description: str = "Flat joint-action PPO for the general collateral model.",
    extra_args_fn: Optional[Callable[[argparse.ArgumentParser], None]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)

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
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument(
        "--reward_scale",
        type=float,
        default=1.0,
        help=(
            "Scale factor for PPO training rewards. Training uses reward / "
            "reward_scale, while evaluation metrics keep original money."
        ),
    )
    parser.add_argument(
        "--imitation_mode",
        choices=["none", "threshold_pretrain"],
        default="none",
    )
    parser.add_argument("--imitation_episodes", type=int, default=100)
    parser.add_argument("--imitation_epochs", type=int, default=3)
    parser.add_argument("--imitation_batch_size", type=int, default=512)
    parser.add_argument("--imitation_lr", type=float, default=3e-4)
    parser.add_argument(
        "--threshold_advantage_mode",
        choices=["none", "residual"],
        default="none",
    )

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

    if args.imitation_episodes <= 0:
        raise ValueError("--imitation_episodes must be positive.")

    if args.imitation_epochs <= 0:
        raise ValueError("--imitation_epochs must be positive.")

    if args.imitation_batch_size <= 0:
        raise ValueError("--imitation_batch_size must be positive.")

    if args.imitation_lr <= 0:
        raise ValueError("--imitation_lr must be positive.")

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
) -> Dict[str, Any]:
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

        "imitation": {
            "mode": str(args.imitation_mode),
            "episodes": int(args.imitation_episodes),
            "epochs": int(args.imitation_epochs),
            "batch_size": int(args.imitation_batch_size),
            "lr": float(args.imitation_lr),
        },

        "threshold_advantage_mode": str(args.threshold_advantage_mode),

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

    pool_paths = resolve_pool_paths(
        args.train_regime,
        args.data_root,
    )

    train_pool, val_pool = load_train_val_pools(
        pool_paths["train_pool_path"],
        pool_paths["val_pool_path"],
        args.T,
    )

    t_max = infer_t_max(
        train_pool,
        args.T_max,
    )

    config = build_config(
        args=args,
        t_max=t_max,
        pool_paths=pool_paths,
    )

    config["pool_fingerprints"] = build_pool_fingerprints(
        pool_paths,
        args.T,
    )
    env_metadata = make_env_from_config(config)
    config["env"]["num_flush_choices"] = int(env_metadata.num_flush_choices)
    config["env"]["flush_fractions"] = list(env_metadata.flush_fractions)
    config["env"]["action_size"] = int(env_metadata.action_size)
    config["env"]["state_size"] = int(env_metadata.state_size)

    paths = build_run_paths(config)
    config["scenario"] = paths["scenario"]

    ensure_run_dirs(
        paths,
        args.save_mode,
    )

    print(f"scenario: {paths['scenario']}")
    print(f"results : {paths['run_dir']}")
    print(f"logs    : {paths['log_dir']}")
    print(f"T_max   : {float(t_max):.6f}")
    print(f"flush_levels: {int(args.flush_levels)}")
    print(f"flush_grid: {args.flush_grid}")
    print(f"reward_scale: {float(args.reward_scale):g}")
    print(f"clip_eps: {float(args.clip_eps):g}")
    print(f"state_feature_mode: {args.state_feature_mode}")
    print(f"mask_mode: {args.mask_mode}")
    print(f"imitation_mode: {args.imitation_mode}")

    env = make_env_from_config(config)
    print(f"flush_fractions: {env.flush_fractions}")
    print(f"num_flush_choices: {env.num_flush_choices}")
    print(f"action_size: {env.action_size}")
    print(f"state_size: {env.state_size}")

    agent = FlatPPOAgent(
        state_size=env.state_size,
        config=config,
        num_flush_choices=env.num_flush_choices,
        flush_fractions=env.flush_fractions,
    )

    training_log, validation_log = train_agent(
        agent=agent,
        config=config,
        paths=paths,
        train_pool=train_pool,
        val_pool=val_pool,
    )

    test_pools = {
        regime: load_tx_pool(pool_path, args.T)
        for regime, pool_path in pool_paths["test_pool_paths"].items()
    }

    cross = evaluate_cross_regime(
        agent=agent,
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
        **cross,
    }

    write_required_run_outputs(
        config,
        paths,
        training_log,
        validation_log,
        results,
    )

    print(f"mean money: {results['aggregate']['mean_money']:.2f}")
    print(
        f"mean value_accept_ratio: "
        f"{results['aggregate']['mean_value_accept_ratio']:.6f}"
    )
    print(
        f"worst regime value_accept_ratio: "
        f"{results['aggregate']['worst_regime_value_accept_ratio']:.6f}"
    )
    print(f"mean drops: {results['aggregate']['mean_drops']:.2f}")
    print(f"mean flushes: {results['aggregate']['mean_flushes']:.2f}")
    print_action_diagnostics(results)


if __name__ == "__main__":
    main()
