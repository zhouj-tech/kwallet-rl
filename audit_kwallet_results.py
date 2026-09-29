from pathlib import Path
import re
import pandas as pd
import numpy as np

ROOT = Path(".")
OUT = Path("paper_planning/result_audit_from_vscode")
OUT.mkdir(parents=True, exist_ok=True)

EXPECTED_SEEDS = {123, 323, 532, 777, 999, 2027, 3407, 4501, 6101, 8888}
EXPECTED_C = {800, 900, 1000, 1200}
EXPECTED_K = {24}

SKIP_PARTS = {".git", ".venv", "venv", "__pycache__", ".ipynb_checkpoints"}

def norm_col(c):
    return str(c).strip().lower().replace(" ", "_").replace("-", "_")

def parse_num(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace("%", "").strip()
    # handles "3999.25 ± 123.13"
    s = s.split("±")[0].strip()
    m = re.search(r"-?\d+(\.\d+)?", s)
    return float(m.group(0)) if m else np.nan

def canon_model(x):
    s = str(x).strip()
    low = s.lower()
    if "conditional" in low or "sc-fac" in low or "sc_fac" in low:
        return "SC-FAC"
    if "factorized" in low or "ifac" in low:
        return "IFAC"
    if "basic ppo" in low or "ja-ppo" in low or "joint" in low:
        return "JA-PPO"
    if "dqn" in low:
        return "DQN baseline"
    return s

def extract_from_path(path, key):
    text = str(path)
    if key == "seed":
        pats = [r"seed[_=\-]?(\d+)", r"[_/\-]s(\d{2,5})[_/\-]", r"[_/\-](123|323|532|777|999|2027|3407|4501|6101|8888)[_/\-]"]
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
file_index = []

for p in ROOT.rglob("*.csv"):
    if any(part in SKIP_PARTS for part in p.parts):
        continue
    try:
        df = pd.read_csv(p)
    except Exception:
        continue
    if df.empty:
        continue

    orig_cols = list(df.columns)
    df = df.rename(columns={c: norm_col(c) for c in df.columns})

    file_index.append({
        "path": str(p),
        "nrows": len(df),
        "columns": ", ".join(orig_cols),
    })

    cols = set(df.columns)

    model_col = next((c for c in ["model", "method", "policy", "agent"] if c in cols), None)
    c_col = next((c for c in ["c", "capacity"] if c in cols), None)
    k_col = "k" if "k" in cols else None
    seed_col = next((c for c in ["seed", "seeds"] if c in cols), None)
    money_col = next((c for c in ["money", "mean_money", "episode_money"] if c in cols), None)

    # common metric columns
    drops_col = next((c for c in ["drops", "mean_drops"] if c in cols), None)
    flush_col = next((c for c in ["flushes", "mean_flushes"] if c in cols), None)
    valacc_col = next((c for c in ["value_accept_ratio", "mean_valacc", "mean_value_accept_ratio"] if c in cols), None)
    regime_col = next((c for c in ["test_regime", "regime"] if c in cols), None)

    # only keep files that look relevant
    if not model_col or not money_col:
        continue

    for _, r in df.iterrows():
        model = canon_model(r.get(model_col))
        C = parse_num(r.get(c_col)) if c_col else extract_from_path(p, "C")
        k = parse_num(r.get(k_col)) if k_col else extract_from_path(p, "k")
        seed = parse_num(r.get(seed_col)) if seed_col else extract_from_path(p, "seed")
        money = parse_num(r.get(money_col))

        rows.append({
            "source_file": str(p),
            "model_raw": r.get(model_col),
            "model": model,
            "C": int(C) if pd.notna(C) else np.nan,
            "k": int(k) if pd.notna(k) else np.nan,
            "seed": int(seed) if pd.notna(seed) else np.nan,
            "test_regime": r.get(regime_col) if regime_col else np.nan,
            "money": money,
            "drops": parse_num(r.get(drops_col)) if drops_col else np.nan,
            "flushes": parse_num(r.get(flush_col)) if flush_col else np.nan,
            "value_accept_ratio": parse_num(r.get(valacc_col)) if valacc_col else np.nan,
        })

file_index_df = pd.DataFrame(file_index)
file_index_df.to_csv(OUT / "candidate_csv_file_index.csv", index=False)

all_rows = pd.DataFrame(rows)
all_rows.to_csv(OUT / "all_relevant_money_rows.csv", index=False)

print("\nSaved:")
print(OUT / "candidate_csv_file_index.csv")
print(OUT / "all_relevant_money_rows.csv")

if all_rows.empty:
    print("\nNo relevant money rows found. Search file names manually with rg/find.")
    raise SystemExit

# focus final k=24 capacities
sub = all_rows[
    all_rows["C"].isin(EXPECTED_C)
    & all_rows["k"].isin(EXPECTED_K)
    & all_rows["model"].isin(["JA-PPO", "IFAC", "SC-FAC", "DQN baseline"])
].copy()

sub.to_csv(OUT / "filtered_k24_C800_900_1000_1200.csv", index=False)

print("\n=== Filtered rows ===")
print(sub.shape)
print(sub[["model", "C", "k", "seed", "test_regime", "money", "source_file"]].head(30))

# seed coverage
coverage = (
    sub.groupby(["model", "C", "k"], dropna=False)
    .agg(
        n_rows=("money", "size"),
        n_seeds=("seed", lambda x: len(set(v for v in x.dropna().astype(int)))),
        seed_list=("seed", lambda x: ",".join(map(str, sorted(set(v for v in x.dropna().astype(int)))))),
        money_mean_raw=("money", "mean"),
        money_std_raw=("money", "std"),
    )
    .reset_index()
)

def missing_seeds(seed_list):
    got = set()
    if isinstance(seed_list, str) and seed_list:
        got = {int(x) for x in seed_list.split(",") if x.strip().isdigit()}
    miss = sorted(EXPECTED_SEEDS - got)
    return ",".join(map(str, miss))

coverage["missing_expected_seeds"] = coverage["seed_list"].apply(missing_seeds)
coverage.to_csv(OUT / "seed_coverage_k24.csv", index=False)

print("\n=== Seed coverage k=24 ===")
print(coverage.to_string(index=False))

# aggregate by seed first, then by model/C
seed_level = (
    sub.groupby(["model", "C", "k", "seed"], dropna=False)
    .agg(
        money=("money", "mean"),
        drops=("drops", "mean"),
        flushes=("flushes", "mean"),
        value_accept_ratio=("value_accept_ratio", "mean"),
        n_regime_rows=("test_regime", "count"),
    )
    .reset_index()
)

table = (
    seed_level.groupby(["C", "k", "model"], dropna=False)
    .agg(
        mean_money=("money", "mean"),
        sd_money=("money", "std"),
        n_seeds=("seed", lambda x: len(set(v for v in x.dropna().astype(int)))),
        seed_list=("seed", lambda x: ",".join(map(str, sorted(set(v for v in x.dropna().astype(int)))))),
        mean_drops=("drops", "mean"),
        mean_flushes=("flushes", "mean"),
        mean_valacc=("value_accept_ratio", "mean"),
    )
    .reset_index()
)

table["sem_money"] = table["sd_money"] / np.sqrt(table["n_seeds"])
table["ci95_halfwidth_approx"] = 1.96 * table["sem_money"]

table = table.sort_values(["C", "model"])
table.to_csv(OUT / "recomputed_table_k24_by_model_C.csv", index=False)

print("\n=== Recomputed Table k=24 by model/C ===")
print(table.to_string(index=False))

print("\nCheck these outputs:")
print(OUT / "seed_coverage_k24.csv")
print(OUT / "recomputed_table_k24_by_model_C.csv")
