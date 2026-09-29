from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
IDEA5_ROOT = THIS_FILE.parents[2]
if str(IDEA5_ROOT) not in sys.path:
    sys.path.insert(0, str(IDEA5_ROOT))

from run_flat_ppo import build_config as build_base_config
from run_flat_ppo import parse_args as parse_base_args
from run_flat_ppo import validate_args

from general_model.models.factorized_ac import FactorizedACAgent
from general_model.utils.io_utils import (
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


def main() -> None:
    args = parse_base_args(
        "Independent factorized AC for the general collateral model."
    )
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

    config = build_base_config(
        args=args,
        t_max=t_max,
        pool_paths=pool_paths,
    )

    config["model_name"] = "factorized_ac"
    config["model_mode"] = "factorized_ac"
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

    agent = FactorizedACAgent(
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
