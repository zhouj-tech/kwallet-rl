from pathlib import Path
import re
import pandas as pd
import numpy as np

ROOT = Path(".")
OUT = Path("paper_planning/result_audit_from_vscode_strict")
OUT.mkdir(parents=True, exist_ok=True)

EXPECTED_SEEDS = {123, 323, 532, 777, 999, 2027, 3407, 4501, 6101, 8888}
EXPECTED_C = {800, 900, 1000, 1200}
EXPECTED_K = {24}

EXCLUDE_PATH_KEYWORDS = [
    "summary",
    "总结",
    "table",
    "paper_ready",
    "submission",
    "checks",
    "report",
    "main_comparison",
    "paired_ci",
    "result_audit",
]

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
    if "basic ppo" in low or "ja-ppo" in low or "joint" in low:
        return "JA-PPO"
    if "conditional factorized" in low or "conditional_factorized" in low or "sc-fac" in low or "sc_fac" in low:
        return "SC-FAC"
    if "factorized ac" in low or "ifac" in low or "factorized_actor" in low:
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
source_diag = []

for p in ROOT.rglob("*.csv"):
    path_str = str(p)

    if any(x.lower() in path_str.lower() for x in EXCLUDE_PATH_KEYWORDS):
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

    # 严格：只接受 seed，不接受 seeds
    seed_col = "seed" if "seed" in cols else None

    money_col = next((c for c in ["money", "mean_money", "episode_money"] if c in cols), None)
    drops_col = next((c for c in ["drops", "mean_drops"] if c in cols), None)
    flush_col = next((c for c in ["flushes", "mean_flushes"] if c in cols), None)
    valacc_col = next((c for c in ["value_accept_ratio", "mean_valacc", "mean_value_accept_ratio"] if c in cols), None)
    regime_col = next((c for c in ["test_regime", "regime"] if c in cols), None)

    if not model_col or not money_col:
        continue

    file_rows = 0

    for _, r in df.iterrows():
        model = canon_model(r.get(model_col))
        C = parse_num(r.get(c_col)) if c_col else extract_from_path(p, "C")
        k = parse_num(r.get(k_col)) if k_col else extract_from_path(p, "k")

        if seed_col:
            seed = parse_num(r.get(seed_col))
        else:
            seed = extract_from_path(p, "seed")

        money = parse_num(r.get(money_col))

        if pd.isna(C) or pd.isna(k) or pd.isna(seed) or pd.isna(money):
            continue

        C = int(C)
        k = int(k)
        seed = int(seed)

        if C not in EXPECTED_C or k not in EXPECTED_K:
            continue

        # 严格：只保留 expected ten seeds
        if seed not in EXPECTED_SEEDS:
            continue

        if model not in ["JA-PPO", "IFAC", "SC-FAC", "DQN baseline"]:
            continue

        file_rows += 1

        rows.append({
            "source_file": path_str,
            "model_raw": r.get(model_col),
            "model": model,
            "C": C,
            "k": k,
            "seed": seed,
            "test_regime": r.get(regime_col) if regime_col else np.nan,
            "money": money,
            "drops": parse_num(r.get(drops_col)) if drops_col else np.nan,
            "flushes": parse_num(r.get(flush_col)) if flush_col else np.nan,
            "value_accept_ratio": parse_num(r.get(valacc_col)) if valacc_col else np.nan,
        })

    if file_rows:
        source_diag.append({
            "source_file": path_str,
            "matched_rows": file_rows,
            "columns": ", ".join(orig_cols),
        })

raw = pd.DataFrame(rows)
diag = pd.DataFrame(source_diag)

raw.to_csv(OUT / "strict_raw_rows.csv", index=False)
diag.to_csv(OUT / "strict_source_diagnostics.csv", index=False)

print("\nSaved:")
print(OUT / "strict_raw_rows.csv")
print(OUT / "strict_source_diagnostics.csv")

if raw.empty:
    print("\nNo strict rows found.")
    raise SystemExit

print("\n=== Source diagnostics ===")
print(diag.sort_values("matched_rows", ascending=False).head(80).to_string(index=False))

coverage = (
    raw.groupby(["model", "C", "k"], dropna=False)
    .agg(
        n_rows=("money", "size"),
        n_seeds=("seed", lambda x: len(set(x))),
        seed_list=("seed", lambda x: ",".join(map(str, sorted(set(x))))),
        n_source_files=("source_file", lambda x: len(set(x))),
    )
    .reset_index()
)

coverage["missing_expected_seeds"] = coverage["seed_list"].apply(
    lambda s: ",".join(map(str, sorted(EXPECTED_SEEDS - {int(x) for x in s.split(',') if x})))
)

coverage.to_csv(OUT / "strict_seed_coverage.csv", index=False)

print("\n=== Strict seed coverage ===")
print(coverage.to_string(index=False))

# seed-level aggregation: if there are 12 regime rows, average them into one seed-level number
seed_level = (
    raw.groupby(["model", "C", "k", "seed"], dropna=False)
    .agg(
        money=("money", "mean"),
        drops=("drops", "mean"),
        flushes=("flushes", "mean"),
        value_accept_ratio=("value_accept_ratio", "mean"),
        n_rows=("money", "size"),
        n_sources=("source_file", lambda x: len(set(x))),
    )
    .reset_index()
)

seed_level.to_csv(OUT / "strict_seed_level.csv", index=False)

summary = (
    seed_level.groupby(["C", "k", "model"], dropna=False)
    .agg(
        mean_money=("money", "mean"),
        sd_money=("money", "std"),
        n_seeds=("seed", lambda x: len(set(x))),
        seed_list=("seed", lambda x: ",".join(map(str, sorted(set(x))))),
        mean_drops=("drops", "mean"),
        mean_flushes=("flushes", "mean"),
        mean_valacc=("value_accept_ratio", "mean"),
    )
    .reset_index()
)

summary["sem_money"] = summary["sd_money"] / np.sqrt(summary["n_seeds"])
summary["ci95_halfwidth_approx"] = 1.96 * summary["sem_money"]

summary = summary.sort_values(["C", "model"])
summary.to_csv(OUT / "strict_recomputed_summary.csv", index=False)

print("\n=== Strict recomputed summary ===")
print(summary.to_string(index=False))

print("\nOpen these:")
print("code", OUT / "strict_seed_coverage.csv")
print("code", OUT / "strict_source_diagnostics.csv")
print("code", OUT / "strict_recomputed_summary.csv")
