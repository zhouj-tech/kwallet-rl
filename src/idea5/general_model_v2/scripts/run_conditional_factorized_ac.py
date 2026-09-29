from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
IDEA5_ROOT = THIS_FILE.parents[2]
if str(IDEA5_ROOT) not in sys.path:
    sys.path.insert(0, str(IDEA5_ROOT))

from run_flat_ppo import (
    add_env_metadata,
    build_config as build_base_config,
    parse_args as parse_base_args,
    print_startup,
    validate_args as validate_base_args,
)

from general_model_v2.models.conditional_factorized_ac import ConditionalFactorizedACAgent
from general_model_v2.utils.io_utils import build_run_paths, ensure_run_dirs, write_required_run_outputs
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
    train_agent,
)


def add_conditional_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--settle_embed_dim", type=int, default=32)
    parser.add_argument("--conditional_hidden_size", type=int, default=256)
    parser.add_argument(
        "--imitation_mode",
        choices=["none", "threshold_pretrain"],
        default="none",
    )
    parser.add_argument("--imitation_episodes", type=int, default=0)
    parser.add_argument("--imitation_epochs", type=int, default=1)
    parser.add_argument("--imitation_batch_size", type=int, default=512)
    parser.add_argument("--imitation_lr", type=float, default=3e-4)
    parser.add_argument("--imitation_reg_coef", type=float, default=0.0)
    parser.add_argument("--imitation_reg_batch_size", type=int, default=512)
    parser.add_argument(
        "--imitation_eta_mode",
        choices=["validation", "best_grid"],
        default="validation",
    )


def parse_args() -> argparse.Namespace:
    return parse_base_args(
        description="Conditional factorized AC for the two-pool general collateral model.",
        extra_args_fn=add_conditional_args,
    )


def validate_args(args: argparse.Namespace) -> None:
    validate_base_args(args)
    if args.settle_embed_dim <= 0:
        raise ValueError("--settle_embed_dim must be positive.")
    if args.conditional_hidden_size <= 0:
        raise ValueError("--conditional_hidden_size must be positive.")
    if args.imitation_episodes < 0:
        raise ValueError("--imitation_episodes must be non-negative.")
    if args.imitation_epochs <= 0:
        raise ValueError("--imitation_epochs must be positive.")
    if args.imitation_batch_size <= 0:
        raise ValueError("--imitation_batch_size must be positive.")
    if args.imitation_lr <= 0:
        raise ValueError("--imitation_lr must be positive.")
    if args.imitation_reg_coef < 0:
        raise ValueError("--imitation_reg_coef must be non-negative.")
    if args.imitation_reg_batch_size <= 0:
        raise ValueError("--imitation_reg_batch_size must be positive.")
    if args.imitation_mode == "threshold_pretrain" and args.imitation_episodes <= 0:
        raise ValueError(
            "--imitation_episodes must be positive when "
            "--imitation_mode threshold_pretrain."
        )
    if args.imitation_reg_coef > 0 and args.imitation_mode != "threshold_pretrain":
        raise ValueError("imitation regularization requires imitation_mode=threshold_pretrain")
    if args.imitation_reg_coef > 0 and args.imitation_episodes <= 0:
        raise ValueError("imitation regularization requires imitation_episodes > 0")


def build_config(args: argparse.Namespace, t_max: float, pool_paths):
    config = build_base_config(args, t_max, pool_paths)
    config["model_name"] = "conditional_factorized_ac"
    config["model_mode"] = "conditional_factorized_ac"
    config["conditional"] = {
        "settle_embed_dim": int(args.settle_embed_dim),
        "conditional_hidden_size": int(args.conditional_hidden_size),
    }
    config["imitation"] = {
        "imitation_mode": str(args.imitation_mode),
        "imitation_episodes": int(args.imitation_episodes),
        "imitation_epochs": int(args.imitation_epochs),
        "imitation_batch_size": int(args.imitation_batch_size),
        "imitation_lr": float(args.imitation_lr),
        "imitation_reg_coef": float(args.imitation_reg_coef),
        "imitation_reg_batch_size": int(args.imitation_reg_batch_size),
        "imitation_eta_mode": str(args.imitation_eta_mode),
        "selected_eta": None,
        "num_samples": 0,
        "expert_train_money_mean": None,
    }
    return config


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
    print(f"settle_embed_dim: {config['conditional']['settle_embed_dim']}")
    print(f"conditional_hidden_size: {config['conditional']['conditional_hidden_size']}")
    print(f"imitation_mode: {config['imitation']['imitation_mode']}")
    print(f"imitation_episodes: {config['imitation']['imitation_episodes']}")
    print(f"imitation_epochs: {config['imitation']['imitation_epochs']}")
    print(f"imitation_reg_coef: {config['imitation']['imitation_reg_coef']:g}")
    print(f"imitation_reg_batch_size: {config['imitation']['imitation_reg_batch_size']}")

    env = make_env_from_config(config)
    agent = ConditionalFactorizedACAgent(
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
