from pathlib import Path
import json
import pandas as pd

root = Path("src/idea4/ac/results")
files = sorted(root.glob("**/cross_regime_results.json"))

rows = []
for p in files:
    try:
        data = json.loads(p.read_text())
        cfg = data.get("config", {})
        env = cfg.get("env", {})
        agg = data.get("aggregate", {})
        maskA = cfg.get("maskA", {})

        rows.append({
            "timestamp": data.get("timestamp"),
            "scenario": data.get("scenario"),
            "model_mode": data.get("model_mode"),
            "maskA_mode": maskA.get("mode"),
            "maskA_soft_penalty": maskA.get("soft_penalty"),
            "seed": data.get("seed"),
            "train_regime": data.get("train_regime"),
            "C": env.get("C"),
            "k": env.get("k"),
            "F": env.get("F"),
            "T": env.get("T"),
            "mean_value_accept_ratio": agg.get("mean_value_accept_ratio"),
            "worst_regime_value_accept_ratio": agg.get("worst_regime_value_accept_ratio"),
            "std_value_accept_ratio_across_regimes": agg.get("std_value_accept_ratio_across_regimes"),
            "mean_drops": agg.get("mean_drops"),
            "mean_flushes": agg.get("mean_flushes"),
            "mean_eval_money": agg.get("mean_eval_money"),
            "result_path": str(p),
        })
    except Exception as e:
        rows.append({"scenario": "READ_ERROR", "result_path": str(p), "error": str(e)})

df = pd.DataFrame(rows)
if not df.empty and "timestamp" in df.columns:
    df["_time"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("_time").drop(columns=["_time"])

out = root / "experiment_timeline.csv"
df.to_csv(out, index=False)

print(f"Saved: {out}")
print(f"Runs found: {len(df)}")

cols = [
    "timestamp", "model_mode", "maskA_mode", "maskA_soft_penalty",
    "seed", "C", "k", "F", "T",
    "mean_value_accept_ratio",
    "worst_regime_value_accept_ratio",
    "std_value_accept_ratio_across_regimes",
    "mean_drops", "mean_flushes", "mean_eval_money"
]
cols = [c for c in cols if c in df.columns]
print(df[cols].tail(10).to_string(index=False))
