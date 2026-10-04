"""Overnight screening analysis: collect panel evals, compute gates, write reports.

Usage:
  python runtime/analyze_screen.py --split SCREEN-v3 --gate screen     (default)
  python runtime/analyze_screen.py --split CHECK-v3 --gate check

Gate parameters (task spec): SCREEN delta >= max(100, .25*G), >= 3/4 seeds
positive at each capacity; CHECK delta >= max(150, .40*G), >= 4/5 seeds positive
at each capacity. No seed-level loss worse than .05*|B_C|. Mean AND median
reported; a candidate fails the "one extreme seed" clause when median <= 0.

Writes reports/overnight_screen_results.csv, overnight_paired_deltas.csv,
overnight_summary.json and the markdown report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCFAC_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = SCFAC_ROOT / "runtime" / "eval_out"
REPORTS = SCFAC_ROOT / "reports"

GATE_PRESETS = {
    "screen": {"min_delta": 100.0, "frac_gap": 0.25, "min_frac_positive": 3 / 4,
               "report": "OVERNIGHT_SCREEN_REPORT.md", "label": "SCREEN"},
    "check": {"min_delta": 150.0, "frac_gap": 0.40, "min_frac_positive": 4 / 5,
              "report": "OVERNIGHT_CHECK_REPORT.md", "label": "CHECK"},
}
MODEL_OF = {"SCM": "SC-M", "IASC": "IA-SC", "TPSC": "TP-SC"}
RISK_LOSS_FRAC = 0.05


def parse_cell_name(name: str):
    # e.g. SCM_C800_s123
    model, cap, seed = name.split("_")
    return MODEL_OF[model], float(cap[1:]), int(seed[1:])


def _unwrap(record):
    """A record file is nested as {designation: {...record...}} (eval_cell output
    schema). Return the inner per-designation record."""
    if "best_by_money" in record and isinstance(record["best_by_money"], dict) \
            and "result" in record["best_by_money"] and "cell" in record["best_by_money"]:
        return record["best_by_money"]
    return record


def load_evals(split: str):
    cells = {}
    bf = {}
    for f in sorted(EVAL_DIR.glob(f"*__{split}__best_by_money.json")):
        data = json.loads(f.read_text())
        rec = _unwrap(data)
        cells[f"{rec['cell']}"] = rec
    # SC-M cells evaluated with --designation all store their best_by_money
    # record inside the __all.json file (top-level mapping of designations)
    for f in sorted(EVAL_DIR.glob(f"*__{split}__all.json")):
        data = json.loads(f.read_text())
        money = data.get("best_by_money")
        if money is not None:
            cells[f"{money['cell']}"] = money
    for f in sorted(EVAL_DIR.glob(f"BF_C*__{split}.json")):
        data = json.loads(f.read_text())
        bf[int(data["capacity"])] = data
    return cells, bf


def cell_rows(cells):
    rows = []
    for name, d in cells.items():
        model, cap, seed = parse_cell_name(name)
        res = d["result"]
        regimes = [r for r in res if r not in ("macro12_mean_money",
                                               "macro12_value_accept_ratio")]
        money = res["macro12_mean_money"]
        settled = float(np.mean([res[r]["settled_mean"] for r in regimes]))
        flushes = float(np.mean([res[r]["flushes_mean"] for r in regimes]))
        insuff = float(np.mean([res[r]["insufficient_drops_mean"] for r in regimes]))
        var_acc = res["macro12_value_accept_ratio"]
        identity_ok = abs(money - (settled - 10.0 * flushes)) < 1e-6
        rows.append({
            "model": model, "capacity": cap, "seed": seed, "money": money,
            "settled": settled, "flushes": flushes, "insufficient": insuff,
            "value_accept_ratio": var_acc, "checkpoint_episode": d["checkpoint_episode"],
            "identity_ok": identity_ok,
        })
    rows.sort(key=lambda r: (r["model"], r["capacity"], r["seed"]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="SCREEN-v3")
    ap.add_argument("--gate", default="screen", choices=["screen", "check"])
    args = ap.parse_args()
    gate = GATE_PRESETS[args.gate]

    cells, bf = load_evals(args.split)
    rows = cell_rows(cells)

    if not rows:
        print(f"no {args.split} evals found under {EVAL_DIR}")
        return 1

    models = sorted({r["model"] for r in rows})
    capas = sorted({r["capacity"] for r in rows})

    # ---- BF per capacity ----
    bf_money = {c: bf[c]["result"]["macro12_mean_money"] for c in capas}

    # ---- SC-M per (capacity, seed), candidate deltas ----
    scm = {c: {r["seed"]: r["money"] for r in rows if r["model"] == "SC-M" and r["capacity"] == c}
           for c in capas}
    primes = {c: float(np.mean(list(scm[c].values()))) for c in capas}
    gaps = {c: max(bf_money[c] - primes[c], 0.0) for c in capas}

    per_model_cap = {}
    for m in models:
        for c in capas:
            rs = [r for r in rows if r["model"] == m and r["capacity"] == c]
            per_model_cap[(m, c)] = rs

    # ---- paired deltas / gate checks for candidates (non SC-M) ----
    deltas_rows = []
    candidates = [m for m in models if m != "SC-M"]
    gate_result = {}
    for cand in candidates:
        ok_all = True
        reasons = []
        for c in capas:
            sc = scm[c]
            cd = {}
            for r in per_model_cap[(cand, c)]:
                s = r["seed"]
                cd[s] = r["money"] - sc[s]
                deltas_rows.append({
                    "candidate": cand, "capacity": c, "seed": s,
                    "delta": cd[s], "candidate_money": r["money"],
                    "scm_money": sc[s],
                })
            ds = np.array([cd[s] for s in sorted(sc)])
            mean_d, med_d = float(ds.mean()), float(np.median(ds))
            n_pos = int((ds > 0).sum())
            worst = float(ds.min())
            threshold = max(gate["min_delta"], gate["frac_gap"] * gaps[c])
            cond_a = mean_d >= threshold
            cond_b = n_pos >= round(gate["min_frac_positive"] * len(ds))
            cond_c = worst >= -RISK_LOSS_FRAC * abs(bf_money[c])
            cond_e = med_d > 0.0
            gate_result[f"{cand}__C{int(c)}"] = {
                "mean_delta": mean_d, "median_delta": med_d, "n_positive": n_pos,
                "n_seeds": int(len(ds)), "worst_delta": worst,
                "threshold": threshold, "gap": gaps[c], "bf": bf_money[c],
                "scm_mean": primes[c],
                "gate_A": cond_a, "gate_B": cond_b, "gate_C": cond_c, "gate_E": cond_e,
            }
            if not (cond_a and cond_b and cond_c and cond_e):
                ok_all = False
                bad = []
                if not cond_a:
                    bad.append(f"A({mean_d:.1f} < threshold {threshold:.1f})")
                if not cond_b:
                    bad.append(f"B({n_pos}/{len(ds)} seeds positive)")
                if not cond_c:
                    bad.append(f"C(worst delta {worst:.1f} < {-RISK_LOSS_FRAC * abs(bf_money[c]):.1f})")
                if not cond_e:
                    bad.append("E(median <= 0)")
                reasons.append(f"C{int(c)}: " + ", ".join(bad))

        # provenance rule D: identity checks must all hold
        id_ok = all(r["identity_ok"] for r in rows)
        survivors = ok_all and id_ok
        gate_result[f"{cand}__SURVIVES_{gate['label']}"] = (survivors, reasons)

    # ---- machine-readable outputs ----
    REPORTS.mkdir(parents=True, exist_ok=True)
    csv_cells = REPORTS / "overnight_screen_results.csv" if args.split == "SCREEN-v3" \
        else REPORTS / f"overnight_{args.split.lower().replace('-', '_')}_results.csv"
    with open(csv_cells, "w") as f:
        f.write("model,capacity,seed,money,settled,flushes,insufficient_drops,"
                "value_accept_ratio,checkpoint_episode,identity_ok\n")
        for r in rows:
            f.write(f"{r['model']},{int(r['capacity'])},{r['seed']},{r['money']:.4f},"
                    f"{r['settled']:.4f},{r['flushes']:.4f},{r['insufficient']:.4f},"
                    f"{r['value_accept_ratio']:.6f},{r['checkpoint_episode']},"
                    f"{r['identity_ok']}\n")
    csv_deltas = REPORTS / "overnight_paired_deltas.csv" if args.split == "SCREEN-v3" \
        else REPORTS / f"overnight_paired_deltas_{args.split.lower().replace('-', '_')}.csv"
    with open(csv_deltas, "w") as f:
        f.write("candidate,capacity,seed,delta,candidate_money,scm_money\n")
        for d in deltas_rows:
            f.write(f"{d['candidate']},{int(d['capacity'])},{d['seed']},{d['delta']:.4f},"
                    f"{d['candidate_money']:.4f},{d['scm_money']:.4f}\n")

    summary = {
        "split": args.split, "gate": gate["label"],
        "bf_money": {int(k): v for k, v in bf_money.items()},
        "scm_mean": {int(k): v for k, v in primes.items()},
        "gap": {int(k): v for k, v in gaps.items()},
        "gates": gate_result,
        "identity_checks_ok": all(r["identity_ok"] for r in rows),
    }
    (REPORTS / "overnight_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True))

    # ---- markdown report ----
    lines = [f"# {gate['label']} REPORT — {args.split}", "",
             f"Split: {args.split}. Money identity checks (settled - 10*flushes): "
             f"{'ALL PASS' if summary['identity_checks_ok'] else 'FAIL'}.", "",
             "## BF reference (fixed BF-T0.5, once per capacity)", ""]
    for c in capas:
        lines.append(f"- C{int(c)}: macro12 Money = {bf_money[c]:.4f}")
    lines += ["", "## Primary table (best_by_money checkpoint)", "",
              "| Model | C | S123 | S323 | S777 | S999 | Mean | Median | Positive vs SC-M | Worst paired delta | BF |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in models:
        for c in capas:
            rs = per_model_cap[(m, c)]
            vals = {r["seed"]: r["money"] for r in rs}
            money_by_seed = [vals[s] for s in sorted(vals)]
            mean_m = float(np.mean(money_by_seed))
            med_m = float(np.median(money_by_seed))
            if m == "SC-M":
                pos, worst = "-", "-"
            else:
                sc = scm[c]
                deltas = [vals[s] - sc[s] for s in sorted(sc)]
                pos = f"{sum(1 for d in deltas if d > 0)}/{len(deltas)}"
                worst = f"{min(deltas):.2f}"
            s123 = f"{vals.get(123, float('nan')):.2f}"
            s323 = f"{vals.get(323, float('nan')):.2f}"
            s777 = f"{vals.get(777, float('nan')):.2f}"
            s999 = f"{vals.get(999, float('nan')):.2f}"
            lines.append(f"| {m} | {int(c)} | {s123} | {s323} | {s777} | {s999} | "
                         f"{mean_m:.2f} | {med_m:.2f} | {pos} | {worst} | {bf_money[c]:.2f} |")
    lines += ["", "## Mechanism table (mean per model/capacity)", "",
              "| Model | C | mean Money | mean settled | mean flushes | mean insufficient drops | mean value_accept_ratio | mean ckpt ep |",
              "|---|---|---|---|---|---|---|---|"]
    for m in models:
        for c in capas:
            rs = per_model_cap[(m, c)]
            lines.append(
                f"| {m} | {int(c)} | {np.mean([r['money'] for r in rs]):.2f} | "
                f"{np.mean([r['settled'] for r in rs]):.2f} | "
                f"{np.mean([r['flushes'] for r in rs]):.2f} | "
                f"{np.mean([r['insufficient'] for r in rs]):.2f} | "
                f"{np.mean([r['value_accept_ratio'] for r in rs]):.4f} | "
                f"{np.mean([r['checkpoint_episode'] for r in rs]):.1f} |")
    lines += ["", "## Gate results", ""]
    for cand in candidates:
        statuses = []
        for c in capas:
            g = gate_result[f"{cand}__C{int(c)}"]
            statuses.append(f"C{int(c)}: mean_delta={g['mean_delta']:.2f} "
                            f"(threshold {g['threshold']:.2f}), median={g['median_delta']:.2f}, "
                            f"positive={g['n_positive']}/{g['n_seeds']}, worst={g['worst_delta']:.2f}"
                            f"{', A/B/C/E OK' if all([g['gate_A'], g['gate_B'], g['gate_C'], g['gate_E']]) else ', FAIL'}")
        surv, reasons = gate_result[f"{cand}__SURVIVES_{gate['label']}"]
        lines.append(f"- **{cand}**: {'SURVIVES ' + gate['label'] if surv else 'FAILS'} — "
                     + " | ".join(statuses))
        for r in reasons:
            lines.append(f"  - {r}")
    lines += ["", "## Seed-level values", ""]
    for m in models:
        for c in capas:
            rs = per_model_cap[(m, c)]
            vals = ", ".join(f"s{r['seed']}={r['money']:.2f}" for r in rs)
            lines.append(f"- {m} C{int(c)}: {vals}")

    report_path = REPORTS / gate["report"]
    report_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {report_path}, {csv_cells.name}, {csv_deltas.name}, overnight_summary.json")
    for cand in candidates:
        surv, reasons = gate_result[f"{cand}__SURVIVES_{gate['label']}"]
        print(f"  {cand}: SURVIVES={surv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())