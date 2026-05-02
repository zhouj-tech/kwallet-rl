from pathlib import Path
import sys
import json

# 把项目根目录加入 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.idea1.flow_generator import (
    generate_regular_flow,
    generate_heavytail_flow,
    generate_burst_flow,
    generate_switching_flow,
)


def save_pool_as_json(pool, save_path: Path) -> None:
    """Save one 2D pool to a json file."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(pool, f, indent=2)


def generate_episode_pool(generator_func, num_episodes: int, steps_per_episode: int, base_seed: int):
    """
    Generate a 2D pool:
    shape = [num_episodes, steps_per_episode]
    """
    pool = []
    for ep in range(num_episodes):
        flow = generator_func(length=steps_per_episode, seed=base_seed + ep)
        pool.append(flow)
    return pool


def main():
    # ===== basic setting =====
    num_episodes = 500
    steps_per_episode = 300
    base_seed = 42
    scale = "quick"

    # ===== save directory =====
    pool_dir = Path("data/pools")
    pool_dir.mkdir(parents=True, exist_ok=True)

    # ===== generate 2D pools =====
    pool_a = generate_episode_pool(
        generator_func=generate_regular_flow,
        num_episodes=num_episodes,
        steps_per_episode=steps_per_episode,
        base_seed=base_seed,
    )

    pool_b = generate_episode_pool(
        generator_func=generate_heavytail_flow,
        num_episodes=num_episodes,
        steps_per_episode=steps_per_episode,
        base_seed=base_seed,
    )

    pool_c = generate_episode_pool(
        generator_func=generate_burst_flow,
        num_episodes=num_episodes,
        steps_per_episode=steps_per_episode,
        base_seed=base_seed,
    )

    pool_d = generate_episode_pool(
        generator_func=generate_switching_flow,
        num_episodes=num_episodes,
        steps_per_episode=steps_per_episode,
        base_seed=base_seed,
    )

    # ===== file names =====
    path_a = pool_dir / f"A_regular_{scale}_seed{base_seed}.json"
    path_b = pool_dir / f"B_heavytail_{scale}_seed{base_seed}.json"
    path_c = pool_dir / f"C_burst_{scale}_seed{base_seed}.json"
    path_d = pool_dir / f"D_switching_{scale}_seed{base_seed}.json"

    save_pool_as_json(pool_a, path_a)
    save_pool_as_json(pool_b, path_b)
    save_pool_as_json(pool_c, path_c)
    save_pool_as_json(pool_d, path_d)

    print("2D episode pools generated successfully:")
    print(path_a)
    print(path_b)
    print(path_c)
    print(path_d)
    print()
    print(f"num_episodes = {num_episodes}")
    print(f"steps_per_episode = {steps_per_episode}")


if __name__ == "__main__":
    main()