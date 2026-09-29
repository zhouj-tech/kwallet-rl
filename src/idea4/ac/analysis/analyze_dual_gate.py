import json
import csv
from pathlib import Path

ROOT = Path("/Users/qiubi/kwallet-rl")
DUAL_RUNS = ROOT / "src/idea4/ac/results/dual_branch_ac/runs"
OUT_DIR = ROOT / "src/idea4/ac/results/final_comparison_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REGIMES = ["US", "TLS", "LNS", "TLNS", "TPLS", "PLS", "UB", "TLB", "LNB", "TLNB", "TPLB", "PLB"]

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def get_nested_mean(d, metric):
    """
    Try to read:
    d[metric]
    d[metric]["mean"]
    d["summary"][metric]["mean"]
    """
    if not isinstance(d, dict):
        return None

    if metric in d:
        v = d[metric]
        if isinstance(v, dict) and "mean" in v:
            return safe_float(v["mean"])
        return safe_float(v)

    if "summary" in d and isinstance(d["summary"], dict):
        s = d["summary"]
        if metric in s:
            v = s[metric]
            if isinstance(v, dict) and "mean" in v:
                return safe_float(v["mean"])
            return safe_float(v)

    return None

def detect_k_from_scenario(scenario):
    text = str(scenario)
    for k in [3, 6, 12]:
        if f"_k{k}_" in text or f"k{k}" in text:
            return k
    return None

def read_cross_regime_json(path):
    data = json.loads(path.read_text(encoding="utf-8"))

    scenario = data.get("scenario", path.parent.parent.name)
    model_mode = data.get("model_mode", "dual_branch_ac")
    train_regime = data.get("train_regime", "")
    seed = data.get("seed", None)

    if seed is None:
        try:
            seed = data["config"]["seed"]
        except Exception:
            seed = None

    k_value = None
    try:
        k_value = int(data["config"]["env"]["k"])
    except Exception:
        k_value = detect_k_from_scenario(scenario)

    test_results = None
    for key in ["test_results", "cross_regime_results", "results"]:
        if key in data and isinstance(data[key], dict):
            test_results = data[key]
            break

    if test_results is None:
        possible = {r: data[r] for r in REGIMES if r in data and isinstance(data[r], dict)}
        if possible:
            test_results = possible

    rows = []

    if test_results is None:
        return rows

    for regime in REGIMES:
        if regime not in test_results:
            continue

        rr = test_results[regime]

        gate = None
        for key in [
            "gate", "gate_mean", "mean_gate", "avg_gate",
            "gate_value", "mean_gate_value", "risk_gate", "capacity_gate"
        ]:
            gate = get_nested_mean(rr, key)
            if gate is not None:
                break

        rows.append({
            "scenario": scenario,
            "model_mode": model_mode,
            "train_regime": train_regime,
            "seed": seed,
            "k": k_value,
            "test_regime": regime,
            "value_accept_ratio": get_nested_mean(rr, "value_accept_ratio"),
            "drops": get_nested_mean(rr, "drops"),
            "flushes": get_nested_mean(rr, "flushes"),
            "drop_rate": get_nested_mean(rr, "drop_rate"),
            "count_accept_ratio": get_nested_mean(rr, "count_accept_ratio"),
            "gate": gate,
            "source": str(path.relative_to(ROOT)),
        })

    return rows

def main():
    json_files = sorted(DUAL_RUNS.glob("**/cross_regime_results.json"))

    all_rows = []
    for p in json_files:
        rows = read_cross_regime_json(p)
        all_rows.extend(rows)

    out_path = OUT_DIR / "dual_gate_analysis.csv"

    fieldnames = [
        "scenario", "model_mode", "train_regime", "seed", "k", "test_regime",
        "value_accept_ratio", "drops", "flushes", "drop_rate", "count_accept_ratio",
        "gate", "source"
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print("Saved:", out_path)
    print("Rows:", len(all_rows))

    gate_rows = [r for r in all_rows if r["gate"] is not None]

    if not gate_rows:
        print()
        print("WARNING: No gate values were found in saved cross_regime_results.json files.")
        print("This means current evaluation results probably printed gate to terminal,")
        print("but did not save gate into JSON/CSV.")
        print()
        print("Next action:")
        print("We need to modify run_dual_branch_ac_benchmark.py so evaluation saves gate statistics.")
        return

    print()
    print("===== Gate Summary by k and seed =====")

    groups = {}
    for r in gate_rows:
        key = (r["k"], r["seed"])
        groups.setdefault(key, []).append(r)

    for key, rows in sorted(groups.items()):
        gates = [safe_float(r["gate"]) for r in rows if safe_float(r["gate"]) is not None]
        vals = [safe_float(r["value_accept_ratio"]) for r in rows if safe_float(r["value_accept_ratio"]) is not None]
        drops = [safe_float(r["drops"]) for r in rows if safe_float(r["drops"]) is not None]

        if not gates:
            continue

        gate_mean = sum(gates) / len(gates)
        gate_min = min(gates)
        gate_max = max(gates)
        gate_range = gate_max - gate_min

        val_mean = sum(vals) / len(vals) if vals else None
        drop_mean = sum(drops) / len(drops) if drops else None

        print(
            f"k={key[0]} seed={key[1]} | "
            f"gate_mean={gate_mean:.4f} | gate_min={gate_min:.4f} | "
            f"gate_max={gate_max:.4f} | gate_range={gate_range:.4f} | "
            f"mean_valacc={val_mean:.4f} | mean_drops={drop_mean:.2f}"
        )

    print()
    print("Interpretation guide:")
    print("- If gate_range is very small, gate is almost constant.")
    print("- If gate differs across regimes, gate is regime-sensitive.")
    print("- If hard regimes have different gate values, dual branch may be using context.")
    print("- If all gates are nearly identical, we need auxiliary risk loss or richer gate inputs.")

if __name__ == "__main__":
    main()
