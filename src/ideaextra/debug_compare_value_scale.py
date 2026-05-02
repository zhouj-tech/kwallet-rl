import json
from pathlib import Path

PROJECT_ROOT = Path("/Users/qiubi/kwallet-rl")

ROOTS = [
    PROJECT_ROOT / "src" / "ideaextra" / "results",
    PROJECT_ROOT / "results",
]

for root in ROOTS:
    if not root.exists():
        continue

    for p in root.rglob("cross_regime_results.json"):
        text = str(p)

        if "C1200_k3_T1000_F3" not in text:
            continue

        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)

            cfg = data.get("config", {})
            seed = cfg.get("seed", None)
            C = cfg.get("env", {}).get("C", None)

            if seed != 123 or float(C) != 1200.0:
                continue

            model = data.get("train_regime", "UNKNOWN")

            settled_values = []
            drops_values = []
            flush_values = []

            for test, r in data["test_results"].items():
                s = r["summary"]
                settled_values.append(s["settled"]["mean"])
                drops_values.append(s["drops"]["mean"])
                flush_values.append(s["flushes"]["mean"])

            print("=" * 90)
            print("model:", model)
            print("path :", p)
            print("settled range:", min(settled_values), "to", max(settled_values))
            print("avg settled  :", sum(settled_values) / len(settled_values))
            print("avg drops    :", sum(drops_values) / len(drops_values))
            print("avg flushes  :", sum(flush_values) / len(flush_values))

        except Exception as e:
            print("ERROR:", p, e)