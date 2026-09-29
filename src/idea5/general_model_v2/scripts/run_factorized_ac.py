from __future__ import annotations

import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
IDEA5_ROOT = THIS_FILE.parents[2]
if str(IDEA5_ROOT) not in sys.path:
    sys.path.insert(0, str(IDEA5_ROOT))

from run_flat_ppo import (
    add_env_metadata,
    build_config,
    parse_args,
    print_startup,
    validate_args,
)

from general_model_v2.models.factorized_ac import FactorizedACAgent
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


def main() -> None:
    args = parse_args("Independent factorized AC for the two-pool general collateral model.")
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
    config["model_name"] = "factorized_ac"
    config["model_mode"] = "factorized_ac"
    add_env_metadata(config)
    config["pool_fingerprints"] = build_pool_fingerprints(pool_paths, args.T)
    paths = build_run_paths(config)
    config["scenario"] = paths["scenario"]
    ensure_run_dirs(paths, args.save_mode)
    print_startup(config, paths, t_max)

    env = make_env_from_config(config)
    agent = FactorizedACAgent(
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
