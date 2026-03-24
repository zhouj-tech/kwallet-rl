import json
import os
from datetime import datetime
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# =========================================================
# ✅ 固定项目根目录
# 假设当前脚本在 kwallet-rl/源代码/ 下
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG = {
    "seed": 123,
    "episodes": 5000,
    "steps_per_ep": 1000,

    # 最终交易金额范围（生成后会被 clip 到这个区间）
    "min_tx": 1,
    "max_tx": 1000,

    # 是否允许覆盖已存在文件
    "overwrite": False,

    # 输出目录
    "output_dir": "data",

    # 生成器配置
    "generator": {
        "kind": "mixture",
        "name": "mix_lognorm_small_mid_uniform_tail_v1",
        "params": {
            "components": [
                {
                    "weight": 0.70,
                    "kind": "lognormal",
                    "params": {"mean": 3.2, "sigma": 0.5}
                },
                {
                    "weight": 0.25,
                    "kind": "lognormal",
                    "params": {"mean": 5.0, "sigma": 0.35}
                },
                {
                    "weight": 0.05,
                    "kind": "uniform",
                    "params": {"low": 700, "high": 1000}
                }
            ]
        }
    }
}


def validate_config(config: Dict[str, Any]) -> None:
    episodes = config["episodes"]
    steps_per_ep = config["steps_per_ep"]
    min_tx = config["min_tx"]
    max_tx = config["max_tx"]
    generator = config["generator"]

    if not isinstance(episodes, int) or episodes <= 0:
        raise ValueError(f"episodes 必须是正整数，当前为: {episodes}")

    if not isinstance(steps_per_ep, int) or steps_per_ep <= 0:
        raise ValueError(f"steps_per_ep 必须是正整数，当前为: {steps_per_ep}")

    if not isinstance(min_tx, int) or not isinstance(max_tx, int):
        raise ValueError("min_tx 和 max_tx 必须是整数")

    if min_tx <= 0:
        raise ValueError(f"min_tx 必须 > 0，当前为: {min_tx}")

    if max_tx < min_tx:
        raise ValueError(f"max_tx 必须 >= min_tx，当前 min_tx={min_tx}, max_tx={max_tx}")

    if "kind" not in generator:
        raise ValueError("generator 配置缺少 kind")

    if "params" not in generator:
        raise ValueError("generator 配置缺少 params")

    kind = generator["kind"]
    params = generator["params"]

    supported_base = {"uniform", "lognormal", "exponential", "pareto"}
    supported_all = supported_base | {"mixture", "piecewise"}

    if kind not in supported_all:
        raise ValueError(f"Unsupported generator kind: {kind}")

    if kind == "mixture":
        if "components" not in params or not params["components"]:
            raise ValueError("mixture 生成器必须包含非空 components")

        weights = [c["weight"] for c in params["components"]]
        if any(w < 0 for w in weights):
            raise ValueError("mixture 的 weight 不能为负数")

        if sum(weights) <= 0:
            raise ValueError("mixture 的 weight 总和必须 > 0")

    if kind == "piecewise":
        if "segments" not in params or not params["segments"]:
            raise ValueError("piecewise 生成器必须包含非空 segments")

        ratios = [seg["length_ratio"] for seg in params["segments"]]
        if any(r < 0 for r in ratios):
            raise ValueError("piecewise 的 length_ratio 不能为负数")

        if sum(ratios) <= 0:
            raise ValueError("piecewise 的 length_ratio 总和必须 > 0")


def clip_and_cast(x: np.ndarray, min_tx: int, max_tx: int) -> np.ndarray:
    x = np.clip(x, min_tx, max_tx)
    x = np.rint(x).astype(int)
    return x


def sample_base(
    kind: str,
    params: Dict[str, Any],
    size: int,
    rng: np.random.Generator
) -> np.ndarray:
    if kind == "uniform":
        low = params.get("low", 1)
        high = params.get("high", 1000)
        if high < low:
            raise ValueError(f"uniform 参数非法: low={low}, high={high}")
        return rng.integers(low, high + 1, size=size)

    elif kind == "lognormal":
        mean = params["mean"]
        sigma = params["sigma"]
        if sigma < 0:
            raise ValueError(f"lognormal 的 sigma 必须 >= 0，当前为: {sigma}")
        return rng.lognormal(mean=mean, sigma=sigma, size=size)

    elif kind == "exponential":
        scale = params["scale"]
        if scale <= 0:
            raise ValueError(f"exponential 的 scale 必须 > 0，当前为: {scale}")
        return rng.exponential(scale=scale, size=size)

    elif kind == "pareto":
        alpha = params["alpha"]
        xm = params.get("xm", 1.0)
        if alpha <= 0:
            raise ValueError(f"pareto 的 alpha 必须 > 0，当前为: {alpha}")
        if xm <= 0:
            raise ValueError(f"pareto 的 xm 必须 > 0，当前为: {xm}")
        return xm * (1.0 + rng.pareto(alpha, size=size))

    else:
        raise ValueError(f"Unsupported base distribution kind: {kind}")


def sample_mixture(
    params: Dict[str, Any],
    size: int,
    rng: np.random.Generator
) -> np.ndarray:
    components = params["components"]
    weights = np.array([c["weight"] for c in components], dtype=float)
    weights = weights / weights.sum()

    choices = rng.choice(len(components), size=size, p=weights)
    out = np.zeros(size, dtype=float)

    for i, comp in enumerate(components):
        mask = (choices == i)
        n = int(mask.sum())
        if n == 0:
            continue

        out[mask] = sample_base(
            kind=comp["kind"],
            params=comp["params"],
            size=n,
            rng=rng
        )

    return out


def sample_piecewise(
    params: Dict[str, Any],
    size: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    适合做“不同时间段不同分布”
    params 例子:
    {
        "segments": [
            {"length_ratio": 0.5, "kind": "uniform", "params": {"low": 1, "high": 100}},
            {"length_ratio": 0.5, "kind": "uniform", "params": {"low": 200, "high": 1000}}
        ]
    }
    """
    segments = params["segments"]
    ratios = np.array([seg["length_ratio"] for seg in segments], dtype=float)
    ratios = ratios / ratios.sum()

    lengths = np.floor(ratios * size).astype(int)
    lengths[-1] += size - lengths.sum()

    parts = []
    for seg, n in zip(segments, lengths):
        kind = seg["kind"]
        seg_params = seg["params"]

        if kind in ["uniform", "lognormal", "exponential", "pareto"]:
            part = sample_base(kind, seg_params, n, rng)
        elif kind == "mixture":
            part = sample_mixture(seg_params, n, rng)
        else:
            raise ValueError(f"Unsupported piecewise segment kind: {kind}")

        parts.append(part)

    return np.concatenate(parts)


def sample_transactions(
    generator_cfg: Dict[str, Any],
    size: int,
    rng: np.random.Generator
) -> np.ndarray:
    kind = generator_cfg["kind"]
    params = generator_cfg["params"]

    if kind in ["uniform", "lognormal", "exponential", "pareto"]:
        return sample_base(kind, params, size, rng)
    elif kind == "mixture":
        return sample_mixture(params, size, rng)
    elif kind == "piecewise":
        return sample_piecewise(params, size, rng)
    else:
        raise ValueError(f"Unsupported generator kind: {kind}")


def build_base_filename(config: Dict[str, Any]) -> str:
    generator = config["generator"]
    name = generator.get("name", generator["kind"])
    max_tx = config["max_tx"]
    return f"tx_pool_{name}_T{max_tx}"


def summarize_distribution(
    raw_all: np.ndarray,
    clipped_all: np.ndarray,
    min_tx: int,
    max_tx: int
) -> Dict[str, Any]:
    raw_all = raw_all.astype(float)
    clipped_all = clipped_all.astype(float)

    summary = {
        "raw_mean": float(np.mean(raw_all)),
        "raw_std": float(np.std(raw_all)),
        "raw_min": float(np.min(raw_all)),
        "raw_max": float(np.max(raw_all)),
        "raw_p50": float(np.percentile(raw_all, 50)),
        "raw_p90": float(np.percentile(raw_all, 90)),
        "raw_p95": float(np.percentile(raw_all, 95)),
        "raw_p99": float(np.percentile(raw_all, 99)),

        "final_mean": float(np.mean(clipped_all)),
        "final_std": float(np.std(clipped_all)),
        "final_min": float(np.min(clipped_all)),
        "final_max": float(np.max(clipped_all)),
        "final_p50": float(np.percentile(clipped_all, 50)),
        "final_p90": float(np.percentile(clipped_all, 90)),
        "final_p95": float(np.percentile(clipped_all, 95)),
        "final_p99": float(np.percentile(clipped_all, 99)),

        "clip_low_ratio": float(np.mean(raw_all < min_tx)),
        "clip_high_ratio": float(np.mean(raw_all > max_tx)),
        "final_at_min_ratio": float(np.mean(clipped_all == min_tx)),
        "final_at_max_ratio": float(np.mean(clipped_all == max_tx)),
    }

    return summary


def save_json(obj: Dict[str, Any], save_path: str) -> None:
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_histogram(clipped_all: np.ndarray, save_path: str, title: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(clipped_all, bins=80, edgecolor="black", alpha=0.8)
    plt.title(title)
    plt.xlabel("Transaction Amount")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_log_histogram(clipped_all: np.ndarray, save_path: str, title: str) -> None:
    positive_vals = clipped_all[clipped_all > 0]

    plt.figure(figsize=(10, 6))
    plt.hist(positive_vals, bins=80, edgecolor="black", alpha=0.8, log=True)
    plt.title(title + " (log-frequency)")
    plt.xlabel("Transaction Amount")
    plt.ylabel("Log Frequency")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("📊 分布统计摘要")
    print("=" * 60)

    print("\n[Raw samples before clipping]")
    print(f"mean    : {summary['raw_mean']:.4f}")
    print(f"std     : {summary['raw_std']:.4f}")
    print(f"min     : {summary['raw_min']:.4f}")
    print(f"max     : {summary['raw_max']:.4f}")
    print(f"p50     : {summary['raw_p50']:.4f}")
    print(f"p90     : {summary['raw_p90']:.4f}")
    print(f"p95     : {summary['raw_p95']:.4f}")
    print(f"p99     : {summary['raw_p99']:.4f}")

    print("\n[Final samples after clipping + rounding]")
    print(f"mean    : {summary['final_mean']:.4f}")
    print(f"std     : {summary['final_std']:.4f}")
    print(f"min     : {summary['final_min']:.4f}")
    print(f"max     : {summary['final_max']:.4f}")
    print(f"p50     : {summary['final_p50']:.4f}")
    print(f"p90     : {summary['final_p90']:.4f}")
    print(f"p95     : {summary['final_p95']:.4f}")
    print(f"p99     : {summary['final_p99']:.4f}")

    print("\n[Clipping diagnostics]")
    print(f"clip_low_ratio    : {summary['clip_low_ratio']:.6f}")
    print(f"clip_high_ratio   : {summary['clip_high_ratio']:.6f}")
    print(f"final_at_min_ratio: {summary['final_at_min_ratio']:.6f}")
    print(f"final_at_max_ratio: {summary['final_at_max_ratio']:.6f}")
    print("=" * 60)


def validate_generated_arrays(
    raw_collector: np.ndarray,
    all_tx: np.ndarray,
    episodes: int,
    steps_per_ep: int,
    min_tx: int,
    max_tx: int
) -> None:
    expected_shape = (episodes, steps_per_ep)

    if raw_collector.shape != expected_shape:
        raise ValueError(
            f"raw_collector 形状错误: 期望 {expected_shape}, 实际 {raw_collector.shape}"
        )

    if all_tx.shape != expected_shape:
        raise ValueError(
            f"all_tx 形状错误: 期望 {expected_shape}, 实际 {all_tx.shape}"
        )

    if not np.isfinite(raw_collector).all():
        raise ValueError("raw_collector 中存在 NaN 或 Inf")

    if not np.isfinite(all_tx).all():
        raise ValueError("all_tx 中存在 NaN 或 Inf")

    actual_min = int(np.min(all_tx))
    actual_max = int(np.max(all_tx))

    if actual_min < min_tx or actual_max > max_tx:
        raise ValueError(
            f"生成后的交易金额超出范围: 实际范围 [{actual_min}, {actual_max}], "
            f"期望范围 [{min_tx}, {max_tx}]"
        )


def build_output_paths(config: Dict[str, Any]) -> Dict[str, str]:
    base_filename = build_base_filename(config)

    pool_dir = PROJECT_ROOT / "data" / "pools"
    report_dir = PROJECT_ROOT / "data" / "reports" / base_filename

    return {
        "pool_dir": str(pool_dir),
        "report_dir": str(report_dir),

        "npy_path": str(pool_dir / f"{base_filename}.npy"),
        "cfg_path": str(report_dir / "data_config.json"),
        "summary_path": str(report_dir / "data_summary.json"),
        "metadata_path": str(report_dir / "data_metadata.json"),
        "hist_path": str(report_dir / "hist.png"),
        "log_hist_path": str(report_dir / "hist_log.png"),
    }


def check_overwrite(paths: Dict[str, str], overwrite: bool) -> None:
    existing = [
        paths["npy_path"],
        paths["cfg_path"],
        paths["summary_path"],
        paths["metadata_path"],
        paths["hist_path"],
        paths["log_hist_path"],
    ]
    existing = [p for p in existing if os.path.exists(p)]

    if existing and not overwrite:
        msg = "以下文件已存在，且 overwrite=False，为避免覆盖，程序终止：\n"
        msg += "\n".join(existing)
        raise FileExistsError(msg)


def build_metadata(
    config: Dict[str, Any],
    paths: Dict[str, str],
    all_tx: np.ndarray
) -> Dict[str, Any]:
    return {
        "timestamp": datetime.now().isoformat(),
        "seed": config["seed"],
        "episodes": config["episodes"],
        "steps_per_ep": config["steps_per_ep"],
        "min_tx": config["min_tx"],
        "max_tx": config["max_tx"],
        "generator_kind": config["generator"]["kind"],
        "generator_name": config["generator"].get("name", config["generator"]["kind"]),
        "shape": list(all_tx.shape),
        "dtype": str(all_tx.dtype),
        "paths": {
            "npy_path": paths["npy_path"],
            "cfg_path": paths["cfg_path"],
            "summary_path": paths["summary_path"],
            "metadata_path": paths["metadata_path"],
            "hist_path": paths["hist_path"],
            "log_hist_path": paths["log_hist_path"],
        },
        "preview": {
            "EP0_first_5": all_tx[0, :5].tolist(),
            "EP1000_first_6": all_tx[min(1000, all_tx.shape[0] - 1), :6].tolist(),
            "last_episode_last_5": all_tx[-1, -5:].tolist(),
        }
    }


def generate_tx_pool() -> None:
    validate_config(CONFIG)

    rng = np.random.default_rng(CONFIG["seed"])

    episodes = CONFIG["episodes"]
    steps_per_ep = CONFIG["steps_per_ep"]
    min_tx = CONFIG["min_tx"]
    max_tx = CONFIG["max_tx"]
    overwrite = CONFIG.get("overwrite", False)

    paths = build_output_paths(CONFIG)
    os.makedirs(paths["pool_dir"], exist_ok=True)
    os.makedirs(paths["report_dir"], exist_ok=True)
    check_overwrite(paths, overwrite=overwrite)

    all_tx = np.zeros((episodes, steps_per_ep), dtype=int)
    raw_collector = np.zeros((episodes, steps_per_ep), dtype=float)

    for ep in range(episodes):
        raw = sample_transactions(
            generator_cfg=CONFIG["generator"],
            size=steps_per_ep,
            rng=rng
        )
        raw_collector[ep] = raw
        all_tx[ep] = clip_and_cast(raw, min_tx=min_tx, max_tx=max_tx)

    validate_generated_arrays(
        raw_collector=raw_collector,
        all_tx=all_tx,
        episodes=episodes,
        steps_per_ep=steps_per_ep,
        min_tx=min_tx,
        max_tx=max_tx
    )

    summary = summarize_distribution(
        raw_all=raw_collector.reshape(-1),
        clipped_all=all_tx.reshape(-1),
        min_tx=min_tx,
        max_tx=max_tx
    )

    metadata = build_metadata(CONFIG, paths, all_tx)

    np.save(paths["npy_path"], all_tx)
    save_json(CONFIG, paths["cfg_path"])
    save_json(summary, paths["summary_path"])
    save_json(metadata, paths["metadata_path"])

    title = f"{CONFIG['generator'].get('name', CONFIG['generator']['kind'])} | T={max_tx}"
    save_histogram(all_tx.reshape(-1), paths["hist_path"], title)
    save_log_histogram(all_tx.reshape(-1), paths["log_hist_path"], title)

    print("=" * 60)
    print("✅ 交易序列生成完成")
    print("=" * 60)
    print(f"generator kind : {CONFIG['generator']['kind']}")
    print(f"generator name : {CONFIG['generator'].get('name', CONFIG['generator']['kind'])}")
    print(f"pool path      : {paths['npy_path']}")
    print(f"report dir     : {paths['report_dir']}")
    print(f"config path    : {paths['cfg_path']}")
    print(f"summary path   : {paths['summary_path']}")
    print(f"metadata path  : {paths['metadata_path']}")
    print(f"hist path      : {paths['hist_path']}")
    print(f"log hist path  : {paths['log_hist_path']}")
    print(f"shape          : {all_tx.shape}")
    print(f"EP0 first 5    : {all_tx[0, :5].tolist()}")
    print(f"EP1000 first 6 : {all_tx[min(1000, all_tx.shape[0] - 1), :6].tolist()}")
    print(f"EP4999 last 5  : {all_tx[-1, -5:].tolist()}")
    print("=" * 60)

    print_summary(summary)


if __name__ == "__main__":
    generate_tx_pool()