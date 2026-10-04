# OVERNIGHT_EXPERIMENT_PLAN.md — prioritized ladder

Time budget: ~5–7 h. Base commit 281e029. Work only under
`sandbox/aamas2027_explore/` in the `~/Desktop/kwallet-aamas-explore` worktree
(branch `aamas2027-explore-sandbox`). Do not modify frozen artifacts. Do not
retrain principal policies except in rung FOUR, and only if the ladder reaches it.

Fixed splits (infra/splits.py — never choose subsets after viewing results):
- SCREEN: episodes 0–39 per regime (480 episodes/cell)
- CHECK: episodes 40–99 per regime (720 episodes/cell, disjoint from SCREEN)
- FULL-DEV: episodes 0–199 per regime (2400 episodes/cell, finalists only)

Runtime reference: D1 full cells ran ~3 min each CPU (2400 episodes);
SCREEN ≈ 40 s/cell, CHECK ≈ 60 s/cell. Use the benchmark venv:
`/Users/zhouzhou/benchmarks/A-EVAL-JA-C800-S123/.venv/bin/python`
(py3.10.22, torch 2.5.1, numpy 2.0.2).

Runner substrate: copy `tools/aamas_stage4/d1_core.py` into
`sandbox/aamas2027_explore/code/` and add arms there. Its `d1_h1_action` already
implements H1 = BF-settle + frozen SC-flush with telemetry and parity gates;
add the guardrail as a new arm. Keep the U-parity test to prove the harness is
unchanged.

Log every attempt with infra/templates/ATTEMPT_TEMPLATE.md; append one row per
cell to a results/RESULT_LOG.csv created from
infra/templates/RESULT_LOG_TEMPLATE.csv.

---

## FIRST — E1: flush-floor guardrail on the two collapse cells (tests H-A)

Arm G0: H1, but whenever the number of usable wallets with balance ≥ current tx
is **zero** (n_feasible == 0, already computed in d1_core telemetry as
`n_feasible`), override the flush choice with the BF flush rule (lowest-balance
usable non-settle wallet; no-op if none).

Cells: C800/S777 and C1200/S323 (the two H1 collapses), plus C1200/S532 as a
success control. Split: SCREEN.

Predeclared success criterion (fix before running):
- PROMOTE if on both collapse cells G0 recovers ≥ 25% of the H1→SC-FAC Money
  deficit (S777 C800: deficit 565.8; S323 C1200: deficit 3317.2) AND the control
  cell changes Money by < ±1% of BF macro (152.99).
- Otherwise KILL H-A and jump to FOUR only if ≥ 4 h remain; else write up.

Estimated cost: ~1 h including harness copy, arm implementation, U-parity gate,
and 3 SCREEN cells.

## SECOND — E2: guardrail trigger sweep (tests H-B), only if E1 promotes

Sweep trigger ∈ {n_feasible==0, n_feasible<2, n_feasible<3} on all 10 cells
(2 capacities × 5 seeds), SCREEN. Take the best trigger by mean gap recovery vs
the SC-FAC→BF gap, then re-run best and runner-up on CHECK (disjoint episodes).
PROMOTE to FULL-DEV only if CHECK confirms ≥ 50% gap recovery at C800 with no
seed losing > 1% BF vs its own H1-U baseline.

Estimated cost: ~2 h (10 cells × 3 triggers SCREEN + 2 × 10 cells CHECK).

## THIRD — E3: C1200 flush-cost probe (tests H-C), only if E2 survives

Arm G1: E2's best guardrail + per-episode flush budget cap at BF's mean
(542 flushes; block no-op-able flushes once exceeded — implement as: when cap
reached, force flush no-op unless n_feasible==0). C1200 cells only, SCREEN.
Success: Money not worse than E2 arm by > 1% BF and flush count within 5% of BF.
This separates "flush waste" from "load-bearing flushes".

Estimated cost: ~1 h.

## FOURTH — E4 (optional, only if ≥ 2.5 h remain after E1–E3): checkpoint-selection probe

One retraining run: SC-FAC, C1200, seed 323, identical frozen config, with
(a) checkpoint saved at every validation point (every 50 eps), and (b) nothing
else changed. Then rescore all 20 checkpoints on SCREEN (Money), and compare
best-Money vs the historically selected (value_accept_ratio) checkpoint.
This is the only way to measure the checkpoint-selection effect (FACT B).
Do NOT change the reward. Do NOT use NEW12 for selection — SCREEN is for
measurement only; selection inside the run must use the historical val pool.

Success: best-Money checkpoint beats the historically selected one by > 1% BF
macro on SCREEN → checkpoint selection is a material problem; report magnitude.
Kill: difference within ±0.5% BF → selection problem small; stop.

Estimated cost: training time dominates (historical 1000-ep CPU runs took
~1–3 h each); rescoring 20 checkpoints on SCREEN ≈ 15 min.

---

## Explicitly not worth repeating tonight

- D1 same-wallet mask (M arm): gate already FAIL, effect +1.35/+13.3 Money.
- H2 (SC-settle + BF-flush): b0 pilots showed minimal/negative effect.
- Any pure-H1 rerun without a new mechanism: parity already EXACT; no new info.
- Feature-engineering/training-objective redesigns: out of scope for a 5–7 h
  evaluation-bounded sprint (see H-E demotion).

## Reporting

End of night: write `sandbox/aamas2027_explore/results/OVERNIGHT_RESULTS.md`
with per-attempt outcomes vs predeclared criteria, the final gap-recovery table,
and a stop/continue recommendation per hypothesis. Label everything post-hoc
exploratory; NEW12 was used for Stage 2B/3 selection.
