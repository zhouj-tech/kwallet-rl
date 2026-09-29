from pathlib import Path
import re
import pandas as pd
import numpy as np

ROOT = Path(".")
OUT = Path("paper_planning/result_audit_source_candidates")
OUT.mkdir(parents=True, exist_ok=True)

EXPECTED_SEEDS = {123, 323, 532, 777, 999, 2027, 3407, 4501, 6101, 8888}
EXPECTED_C = {800, 900, 1000, 1200}

TARGETS = {
    ("JA-PPO", 800): 3368.57,
    ("IFAC", 800): 3607.05,
    ("SC-FAC", 800): 3999.25,
    ("JA-PPO", 900): 5636.16,
    ("IFAC", 900): 6347.35,
    ("SC-FAC", 900): 6627.98,
    ("JA-PPO", 1000): 8332.27,
    ("IFAC", 1000): 8786.44,
    ("SC-FAC", 1000): 9064.80,
    ("JA-PPO", 1200): 14443.01,
    ("IFAC", 1200): 14472.93,
    ("SC-FAC", 1200): 14687.65,
}

def norm_col(c):
    return str(c).strip().lower().replace(" ", "_").replace("-", "_")

def parse_num(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace("%", "").strip()
    s = s.split("±")[0].strip()
    m = re.search(r"-?\d+(\.\d+)?", s)
    return float(m.group(0)) if m else np.nan

def canon_model(x):
    s = str(x).strip()
    low = s.lower()
    if "dqn" in low:
        return "DQN baseline"
    if "basic ppo" in low or "ja-ppo" in low or "ja_ppo" in low or "joint" in low:
        return "JA-PPO"
    if "conditional factorized" in low or "conditional_factorized" in low or "sc-fac" in low or "sc_fac" in low:
        return "SC-FAC"
    if "factorized ac" in low or "factorized_actor" in low or "ifac" in low:
        return "IFAC"
    return s

def extract_from_path(path, key):
    text = str(path)
    if key == "seed":
        pats = [
            r"seed[_=\-]?(\d+)",
            r"[_/\-]s(\d{3,5})[_/\-]",
            r"[_/\-](123|323|532|777|999|2027|3407|4501|6101|8888)[_/\-]",
        ]
    elif key == "C":
        pats = [r"[_/\-]C[_=\-]?(\d+)", r"capacity[_=\-]?(\d+)"]
    elif key == "k":
        pats = [r"[_/\-]k[_=\-]?(\d+)"]
    else:
        pats = []
    for pat in pats:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    return np.nan

rows = []

for p in ROOT.rglob("*.csv"):
    if "result_audit" in str(p):
        continue

    try:
        df = pd.read_csv(p)
    except Exception:
        continue

    if df.empty:
        continue

    orig_cols = list(df.columns)
    df = df.rename(columns={c: norm_col(c) for c in df.columns})
    cols = set(df.columns)

    model_col = next((c for c in ["model", "method", "policy", "agent"] if c in cols), None)
    c_col = next((c for c in ["c", "capacity"] if c in cols), None)
    k_col = "k" if "k" in cols else None
    seed_col = "seed" if "seed" in cols else None

    money_col = next((c for c in ["money", "mean_money", "episode_money"] if c in cols), None)

    if not model_col or not money_col:
        continue

    tmp = []

    for _, r in df.iterrows():
        model = canon_model(r.get(model_col))
        C = parse_num(r.get(c_col)) if c_col else extract_from_path(p, "C")
        k = parse_num(r.get(k_col)) if k_col else extract_from_path(p, "k")
        seed = parse_num(r.get(seed_col)) if seed_col else extract_from_path(p, "seed")
        money = parse_num(r.get(money_col))

        if pd.isna(C) or pd.isna(k) or pd.isna(seed) or pd.isna(money):
            continue

        C, k, seed = int(C), int(k), int(seed)

        if C not in EXPECTED_C or k != 24:
            continue

        if seed not in EXPECTED_SEEDS:
            continue

        if model not in ["JA-PPO", "IFAC", "SC-FAC", "DQN baseline"]:
            continue

        tmp.append({
            "source_file": str(p),
            "model": model,
            "C": C,
            "k": k,
            "seed": seed,
            "money": money,
            "columns": ", ".join(orig_cols),
        })

    if tmp:
        rows.extend(tmp)

raw = pd.DataFrame(rows)
raw.to_csv(OUT / "all_candidate_seed_rows.csv", index=False)

if raw.empty:
    print("No candidate rows found.")
    raise SystemExit

summary = (
    raw.groupby(["source_file", "model", "C", "k"])
    .agg(
        n_rows=("money", "size"),
        n_seeds=("seed", lambda x: len(set(x))),
        seed_list=("seed", lambda x: ",".join(map(str, sorted(set(x))))),
        mean_money=("money", "mean"),
        sd_money=("money", "std"),
        columns=("columns", "first"),
    )
    .reset_index()
)

summary["target_money"] = summary.apply(
    lambda r: TARGETS.get((r["model"], int(r["C"])), np.nan), axis=1
)
summary["abs_diff_from_table2"] = (summary["mean_money"] - summary["target_money"]).abs()

summary = summary.sort_values(
    ["model", "C", "abs_diff_from_table2", "n_seeds", "source_file"],
    ascending=[True, True, True, False, True]
)

summary.to_csv(OUT / "candidate_source_summary.csv", index=False)

print("\nSaved:")
print(OUT / "all_candidate_seed_rows.csv")
print(OUT / "candidate_source_summary.csv")

print("\n=== Best candidates matching Table II ===")
best = summary[
    summary["model"].isin(["JA-PPO", "IFAC", "SC-FAC"])
].sort_values(["abs_diff_from_table2", "n_seeds"], ascending=[True, False])

print(best.head(80)[[
    "model", "C", "n_rows", "n_seeds", "seed_list",
    "mean_money", "target_money", "abs_diff_from_table2",
    "source_file"
]].to_string(index=False))

print("\n=== DQN candidates ===")
dqn = summary[summary["model"] == "DQN baseline"].sort_values(["C", "n_seeds", "source_file"])
if dqn.empty:
    print("No DQN candidate seed-level rows found.")
else:
    print(dqn[[
        "model", "C", "n_rows", "n_seeds", "seed_list",
        "mean_money", "source_file"
    ]].to_string(index=False))
