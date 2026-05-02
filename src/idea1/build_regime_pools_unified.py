import json
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ============================================================
# 1) 定位目录：以后一般不用改
# ============================================================
THIS_FILE = Path(__file__).resolve()
IDEA3_DIR = THIS_FILE.parent
SRC_DIR = IDEA3_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

# ============================================================
# 2) 指定 generator 文件路径：以后如果换 generator 文件名，只改这里
# ============================================================
GENERATOR_FILE = IDEA3_DIR / "regime_generator2.py"

if not GENERATOR_FILE.exists():
    raise FileNotFoundError(f"找不到 generator 文件: {GENERATOR_FILE}")

# ============================================================
# 3) 按文件路径动态导入 generator
#    这样最稳，不依赖 sys.path 和模块搜索路径
# ============================================================
spec = importlib.util.spec_from_file_location("regime_generator2", GENERATOR_FILE)
if spec is None or spec.loader is None:
    raise ImportError(f"无法为 generator 文件创建 import spec: {GENERATOR_FILE}")

gen_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen_mod)

# ============================================================
# 4) 取出统一版脚本需要的两个对象
#    你的 generator 文件里这两个名字是存在的
# ============================================================
REGIME_CONFIGS = gen_mod.REGIME_CONFIGS
generate_episode = gen_mod.generate_episode
# ============================================================
# 1) 这里先改成你自己的 generator 文件名（不带 .py）
# ============================================================
GENERATOR_MODULE = "regime_generator2"


# ============================================================
# 2) 总配置：以后最常改这里
# ============================================================
POOL_CONFIG = {
    "output_dir": "data/pools",

    # =========================
    # train pools
    # =========================
    "static_num_episodes": 1000,
    "static_episode_length": 1000,
    "save_mixed_generalist_pool": True,

    "switching_num_episodes": 1000,
    "switching_total_length": 1000,

    # =========================
    # eval pools（独立评估池）
    # =========================
    "save_eval_pools": True,
    "eval_num_episodes": 200,
    "eval_episode_length": 1000,

    # 训练和评估的 seed 要彻底分开
    "base_seed": 20260401,
    "eval_seed_offset": 1000000,
}


# ============================================================
# 3) switching specs：以后想加新切换模板，就在这里加
#    比例不要求和为 1，代码里会自动归一化
# ============================================================
SWITCHING_SPECS = [
    {
        "name": "SL_BH_SH_20_50_30",
        "sequence": ["SL", "BH", "SH"],
        "weights": {"SL": 0.2, "BH": 0.5, "SH": 0.3},
        "num_episodes": 1000,
        "total_length": 1000,
        "seed_offset": 90000,
    },
    # 你以后想再加一个模板，就照着下面这种格式加：
    # {
    #     "name": "SL_BH_BL_SH_15_45_20_20",
    #     "sequence": ["SL", "BH", "BL", "SH"],
    #     "weights": {"SL": 0.15, "BH": 0.45, "BL": 0.20, "SH": 0.20},
    #     "num_episodes": 100,
    #     "total_length": 300,
    #     "seed_offset": 91000,
    # },
]


# ============================================================
# 4) regime 固定 seed 偏移：不要用 hash()，可复现性更稳
# ============================================================
REGIME_SEED_OFFSET = {
    "SL": 100,
    "SH": 200,
    "BL": 300,
    "BH": 400,
}


# ============================================================
# 5) 动态导入你的 generator
# ============================================================
try:
    gen_mod = importlib.import_module(GENERATOR_MODULE)
except Exception as e:
    raise ImportError(
        f"无法导入 generator 模块: {GENERATOR_MODULE}\n"
        f"请检查 GENERATOR_MODULE 是否写对（不要带 .py）\n"
        f"原始报错: {e}"
    )

try:
    REGIME_CONFIGS = gen_mod.REGIME_CONFIGS
    GLOBAL = gen_mod.GLOBAL
    generate_episode = gen_mod.generate_episode
except AttributeError as e:
    raise AttributeError(
        "导入成功，但你的 generator 文件里缺少下面这些名字之一：\n"
        "REGIME_CONFIGS, GLOBAL, generate_episode\n"
        f"原始报错: {e}"
    )


# ============================================================
# 6) 基础工具函数
# ============================================================
def ensure_dir(path_str: str) -> Path:
    path = Path(path_str)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_pool_npy(pool: np.ndarray, path: Path) -> None:
    np.save(path, pool)


def pretty_print_preview(pool: np.ndarray, title: str) -> None:
    """简单打印，方便肉眼检查 pool 大致正常"""
    print(f"\n=== {title} ===")
    print("shape =", pool.shape)
    print("第1条 episode 前10笔 =", pool[0][:10].tolist())
    if pool.shape[1] >= 110:
        print("第95~105笔 =", pool[0][95:105].tolist())
    if pool.shape[1] >= 210:
        print("第195~205笔 =", pool[0][195:205].tolist())
    print("最后10笔 =", pool[0][-10:].tolist())


# ============================================================
# 7) 长度分配函数：把比例转成具体步数
# ============================================================
def allocate_segment_lengths(total_length: int, regime_weights: Dict[str, float]) -> Dict[str, int]:
    """
    根据比例分配每个 regime 占多少步，并保证总和恰好等于 total_length。
    例如：
        total_length = 300
        weights = {"SL": 0.2, "BH": 0.5, "SH": 0.3}
    可能得到：
        {"SL": 60, "BH": 150, "SH": 90}
    """
    regime_names = list(regime_weights.keys())
    weights = np.array([regime_weights[r] for r in regime_names], dtype=float)

    if np.any(weights < 0):
        raise ValueError("regime_weights 里不能有负数")
    if weights.sum() <= 0:
        raise ValueError("regime_weights 总和必须大于 0")

    weights = weights / weights.sum()
    raw_lengths = weights * total_length
    floor_lengths = np.floor(raw_lengths).astype(int)

    remainder = total_length - floor_lengths.sum()
    fractional_parts = raw_lengths - floor_lengths
    order = np.argsort(-fractional_parts)

    for i in range(remainder):
        floor_lengths[order[i]] += 1

    return {regime_names[i]: int(floor_lengths[i]) for i in range(len(regime_names))}


# ============================================================
# 8) 生成 static pool
# ============================================================
def generate_static_pool(
    config: Dict,
    num_episodes: int,
    episode_length: int,
    base_seed: int,
) -> np.ndarray:
    """
    生成某一个 regime 的 static pool
    输出 shape = (num_episodes, episode_length)
    """
    episodes = []

    for ep in range(num_episodes):
        seed = base_seed + ep
        tx_sizes, _ = generate_episode(config=config, T=episode_length, seed=seed)
        episodes.append(tx_sizes)

    return np.array(episodes, dtype=float)


# ============================================================
# 9) 生成 mixed generalist static pool
#    作用：以后训练 generalist 时可以直接用
# ============================================================
def generate_mixed_generalist_pool(
    regime_names: List[str],
    regime_configs: Dict[str, Dict],
    num_episodes_per_regime: int,
    episode_length: int,
    base_seed: int,
) -> np.ndarray:
    """
    把多个 static regime 按“每类同样条数”拼成一个 mixed pool。
    输出 shape = (len(regime_names) * num_episodes_per_regime, episode_length)
    """
    all_rows = []

    for regime_name in regime_names:
        regime_seed_base = base_seed + REGIME_SEED_OFFSET[regime_name]
        pool = generate_static_pool(
            config=regime_configs[regime_name],
            num_episodes=num_episodes_per_regime,
            episode_length=episode_length,
            base_seed=regime_seed_base,
        )
        all_rows.append(pool)

    mixed_pool = np.concatenate(all_rows, axis=0)
    return mixed_pool

def generate_mixed_equal_pool(
    regime_names: List[str],
    regime_configs: Dict[str, Dict],
    total_episodes: int,
    episode_length: int,
    base_seed: int,
) -> np.ndarray:
    """
    生成一个总条数固定的 mixed pool。
    例如 total_episodes=1000，4 个 regime 时，每类取 250 条。
    输出 shape = (total_episodes, episode_length)
    """
    num_regimes = len(regime_names)
    if total_episodes % num_regimes != 0:
        raise ValueError(
            f"total_episodes={total_episodes} 不能被 regime 数量 {num_regimes} 整除"
        )

    episodes_per_regime = total_episodes // num_regimes
    all_rows = []

    for regime_name in regime_names:
        regime_seed_base = base_seed + REGIME_SEED_OFFSET[regime_name]
        pool = generate_static_pool(
            config=regime_configs[regime_name],
            num_episodes=episodes_per_regime,
            episode_length=episode_length,
            base_seed=regime_seed_base,
        )
        all_rows.append(pool)

    mixed_pool = np.concatenate(all_rows, axis=0)

    # 打乱顺序，让 mixed 更自然
    rng = np.random.default_rng(base_seed + 999)
    rng.shuffle(mixed_pool, axis=0)

    return mixed_pool

# ============================================================
# 10) 生成 switching pool（顺序固定，比例可调）
# ============================================================
def generate_switching_pool_by_ratio(
    regime_sequence: List[str],
    regime_weights: Dict[str, float],
    regime_configs: Dict[str, Dict],
    total_length: int,
    num_episodes: int,
    base_seed: int,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    生成 switching pool。

    参数：
    - regime_sequence: 切换顺序，例如 ["SL", "BH", "SH"]
    - regime_weights: 各段比例，例如 {"SL":0.2, "BH":0.5, "SH":0.3}
    - total_length: 每条 episode 总长度，例如 300

    输出：
    - pool: shape = (num_episodes, total_length)
    - allocated_lengths: 实际分配到每段的步数
    """
    # 只保留 sequence 里真正用到的 regime
    filtered_weights = {r: regime_weights[r] for r in regime_sequence}
    allocated_lengths = allocate_segment_lengths(total_length, filtered_weights)
    segment_lengths = [allocated_lengths[r] for r in regime_sequence]

    episodes = []

    for ep in range(num_episodes):
        full_episode = []

        for seg_idx, regime_name in enumerate(regime_sequence):
            cfg = regime_configs[regime_name]
            seg_len = segment_lengths[seg_idx]

            # 这里是每个 segment 的种子生成规则
            # 以后如果想改可复现方式，就改这里
            seed = base_seed + ep * 1000 + REGIME_SEED_OFFSET[regime_name] + seg_idx

            tx_sizes, _ = generate_episode(config=cfg, T=seg_len, seed=seed)
            full_episode.extend(tx_sizes.tolist())

        episodes.append(full_episode)

    return np.array(episodes, dtype=float), allocated_lengths


# ============================================================
# 11) 文件名工具：统一命名，后面训练时更好找
# ============================================================
def build_static_pool_filename(regime_name: str, num_episodes: int, episode_length: int) -> str:
    return f"{regime_name}_static_pool_E{num_episodes}_T{episode_length}.npy"


def build_static_meta_filename(regime_name: str, num_episodes: int, episode_length: int) -> str:
    return f"{regime_name}_static_pool_E{num_episodes}_T{episode_length}_meta.json"


def build_mixed_pool_filename(regime_names: List[str], num_total_episodes: int, episode_length: int) -> str:
    tag = "".join(regime_names)
    return f"MIXED_{tag}_pool_E{num_total_episodes}_T{episode_length}.npy"


def build_mixed_meta_filename(regime_names: List[str], num_total_episodes: int, episode_length: int) -> str:
    tag = "".join(regime_names)
    return f"MIXED_{tag}_pool_E{num_total_episodes}_T{episode_length}_meta.json"


def build_switch_pool_filename(spec_name: str, num_episodes: int, total_length: int) -> str:
    return f"{spec_name}_switch_pool_E{num_episodes}_T{total_length}.npy"


def build_switch_meta_filename(spec_name: str, num_episodes: int, total_length: int) -> str:
    return f"{spec_name}_switch_pool_E{num_episodes}_T{total_length}_meta.json"


# ============================================================
# 12) 主函数：统一生成 static + mixed + switching
# ============================================================
def build_and_save_all_pools() -> None:
    output_dir = ensure_dir(POOL_CONFIG["output_dir"])

    static_num_episodes = POOL_CONFIG["static_num_episodes"]
    static_episode_length = POOL_CONFIG["static_episode_length"]
    switching_num_episodes = POOL_CONFIG["switching_num_episodes"]
    switching_total_length = POOL_CONFIG["switching_total_length"]
    base_seed = POOL_CONFIG["base_seed"]

    eval_num_episodes = POOL_CONFIG["eval_num_episodes"]
    eval_episode_length = POOL_CONFIG["eval_episode_length"]
    eval_seed_offset = POOL_CONFIG["eval_seed_offset"]
    eval_base_seed = base_seed + eval_seed_offset

    # --------------------------------------------------------
    # A. 先生成 4 个 static pools
    #    以后 specialist 训练主要用这里
    # --------------------------------------------------------
    print("\n" + "=" * 72)
    print("开始生成 static pools")
    print("=" * 72)

    static_regime_names = list(REGIME_CONFIGS.keys())

    for regime_name in static_regime_names:
        cfg = REGIME_CONFIGS[regime_name]
        regime_seed_base = base_seed + REGIME_SEED_OFFSET[regime_name]

        pool = generate_static_pool(
            config=cfg,
            num_episodes=static_num_episodes,
            episode_length=static_episode_length,
            base_seed=regime_seed_base,
        )

        pool_path = output_dir / build_static_pool_filename(regime_name, static_num_episodes, static_episode_length)
        meta_path = output_dir / build_static_meta_filename(regime_name, static_num_episodes, static_episode_length)

        save_pool_npy(pool, pool_path)
        save_json(
            {
                "type": "static_pool",
                "generator_module": GENERATOR_MODULE,
                "regime": regime_name,
                "num_episodes": static_num_episodes,
                "episode_length": static_episode_length,
                "base_seed": regime_seed_base,
                "config": cfg,
            },
            meta_path,
        )

        print(f"[已保存] {pool_path}")
        pretty_print_preview(pool, f"static pool - {regime_name}")

    # --------------------------------------------------------
    # B. 可选：生成 mixed generalist static pool
    #    以后 generalist 训练可以直接用这里
    # --------------------------------------------------------
    if POOL_CONFIG["save_mixed_generalist_pool"]:
            # --------------------------------------------------------
    # B2. 可选：生成一个“总条数固定”的 mixed pool
    #     例如 E1000_T1000，而不是把四类全部拼成 E4000
    # --------------------------------------------------------
     if POOL_CONFIG.get("save_mixed_equal_pool", False):
        print("\n" + "=" * 72)
        print("开始生成 mixed equal pool")
        print("=" * 72)

        mixed_equal_total_episodes = POOL_CONFIG["mixed_equal_total_episodes"]

        mixed_equal_pool = generate_mixed_equal_pool(
            regime_names=static_regime_names,
            regime_configs=REGIME_CONFIGS,
            total_episodes=mixed_equal_total_episodes,
            episode_length=static_episode_length,
            base_seed=base_seed + 60000,
        )

        mixed_equal_pool_path = output_dir / f"MIXED_EQ_{''.join(static_regime_names)}_pool_E{mixed_equal_total_episodes}_T{static_episode_length}.npy"
        mixed_equal_meta_path = output_dir / f"MIXED_EQ_{''.join(static_regime_names)}_pool_E{mixed_equal_total_episodes}_T{static_episode_length}_meta.json"

        save_pool_npy(mixed_equal_pool, mixed_equal_pool_path)
        save_json(
            {
                "type": "mixed_equal_pool",
                "generator_module": GENERATOR_MODULE,
                "regime_names": static_regime_names,
                "total_episodes": mixed_equal_total_episodes,
                "episode_length": static_episode_length,
                "base_seed": base_seed + 60000,
            },
            mixed_equal_meta_path,
        )

        print(f"[已保存] {mixed_equal_pool_path}")
        pretty_print_preview(mixed_equal_pool, "mixed equal pool")
        print("\n" + "=" * 72)
        print("开始生成 mixed generalist static pool")
        print("=" * 72)

        mixed_pool = generate_mixed_generalist_pool(
            regime_names=static_regime_names,
            regime_configs=REGIME_CONFIGS,
            num_episodes_per_regime=static_num_episodes,
            episode_length=static_episode_length,
            base_seed=base_seed + 50000,
        )

        total_eps = mixed_pool.shape[0]
        mixed_pool_path = output_dir / build_mixed_pool_filename(static_regime_names, total_eps, static_episode_length)
        mixed_meta_path = output_dir / build_mixed_meta_filename(static_regime_names, total_eps, static_episode_length)

        save_pool_npy(mixed_pool, mixed_pool_path)
        save_json(
            {
                "type": "mixed_static_pool",
                "generator_module": GENERATOR_MODULE,
                "regime_names": static_regime_names,
                "num_episodes_per_regime": static_num_episodes,
                "total_episodes": total_eps,
                "episode_length": static_episode_length,
                "base_seed": base_seed + 50000,
            },
            mixed_meta_path,
        )

        print(f"[已保存] {mixed_pool_path}")
        pretty_print_preview(mixed_pool, "mixed generalist static pool")

    # --------------------------------------------------------
    # C. 生成 switching pools
    #    以后 hidden switching benchmark 主要用这里
    # --------------------------------------------------------
    print("\n" + "=" * 72)
    print("开始生成 switching pools")
    print("=" * 72)

    for spec in SWITCHING_SPECS:
        spec_name = spec["name"]
        sequence = spec["sequence"]
        weights = spec["weights"]
        num_episodes = spec.get("num_episodes", switching_num_episodes)
        total_length = spec.get("total_length", switching_total_length)
        seed_offset = spec.get("seed_offset", 90000)

        pool, allocated_lengths = generate_switching_pool_by_ratio(
            regime_sequence=sequence,
            regime_weights=weights,
            regime_configs=REGIME_CONFIGS,
            total_length=total_length,
            num_episodes=num_episodes,
            base_seed=base_seed + seed_offset,
        )

        pool_path = output_dir / build_switch_pool_filename(spec_name, num_episodes, total_length)
        meta_path = output_dir / build_switch_meta_filename(spec_name, num_episodes, total_length)

        save_pool_npy(pool, pool_path)
        save_json(
            {
                "type": "switching_pool",
                "generator_module": GENERATOR_MODULE,
                "sequence": sequence,
                "weights": weights,
                "allocated_lengths": allocated_lengths,
                "num_episodes": num_episodes,
                "total_length": total_length,
                "base_seed": base_seed + seed_offset,
            },
            meta_path,
        )

        print(f"[已保存] {pool_path}")
        print("allocated_lengths =", allocated_lengths)
        pretty_print_preview(pool, f"switching pool - {spec_name}")

    print("\n全部 pool 生成完成。")
    print(f"输出目录：{output_dir.resolve()}")


# ============================================================
# 13) 程序入口
# ============================================================
if __name__ == "__main__":
    build_and_save_all_pools()
