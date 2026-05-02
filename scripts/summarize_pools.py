from pathlib import Path
import json
import math
import statistics
from typing import List, Dict


def load_flow(json_path: Path) -> List[int]:
    """Load one flow from a json file."""
    with open(json_path, "r", encoding="utf-8") as f:
        flow = json.load(f)

    if not isinstance(flow, list):
        raise ValueError(f"{json_path} does not contain a list.")
    if len(flow) == 0:
        raise ValueError(f"{json_path} is empty.")
    return flow


def percentile(values: List[int], p: float) -> float:
    """
    Compute percentile using linear interpolation.
    p should be in [0, 100].
    """
    if not values:
        raise ValueError("values is empty")
    if not (0 <= p <= 100):
        raise ValueError("p must be between 0 and 100")

    sorted_vals = sorted(values)
    n = len(sorted_vals)

    if n == 1:
        return float(sorted_vals[0])

    pos = (p / 100) * (n - 1)
    lower = math.floor(pos)
    upper = math.ceil(pos)

    if lower == upper:
        return float(sorted_vals[lower])

    weight = pos - lower
    return sorted_vals[lower] * (1 - weight) + sorted_vals[upper] * weight


def summarize_flow(flow: List[int]) -> Dict[str, float]:
    """Return summary statistics for one flow."""
    return {
        "count": len(flow),
        "mean": statistics.mean(flow),
        "std": statistics.stdev(flow) if len(flow) > 1 else 0.0,
        "min": min(flow),
        "p50": percentile(flow, 50),
        "p90": percentile(flow, 90),
        "p95": percentile(flow, 95),
        "p99": percentile(flow, 99),
        "max": max(flow),
    }


def print_summary(name: str, summary: Dict[str, float]) -> None:
    print(f"\n=== {name} ===")
    print(f"count: {summary['count']}")
    print(f"mean : {summary['mean']:.2f}")
    print(f"std  : {summary['std']:.2f}")
    print(f"min  : {summary['min']}")
    print(f"p50  : {summary['p50']:.2f}")
    print(f"p90  : {summary['p90']:.2f}")
    print(f"p95  : {summary['p95']:.2f}")
    print(f"p99  : {summary['p99']:.2f}")
    print(f"max  : {summary['max']}")


def main():
    pool_dir = Path("data/pools")

    files = [
        pool_dir / "A_regular_quick_seed42.json",
        pool_dir / "B_heavytail_quick_seed42.json",
        pool_dir / "C_burst_quick_seed42.json",
        pool_dir / "D_switching_quick_seed42.json",
    ]

    for file_path in files:
        if not file_path.exists():
            print(f"Missing file: {file_path}")
            continue

        flow = load_flow(file_path)
        summary = summarize_flow(flow)
        print_summary(file_path.stem, summary)


if __name__ == "__main__":
    main()
    