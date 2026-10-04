# HANDOFF_TO_OVERNIGHT_AGENT.md

Read this first. You have ~5–7 h. Everything you need is below.

## 1. Frozen commit and paths

- Repo worktree for all work: `/Users/zhouzhou/Desktop/kwallet-aamas-explore`
  (branch `aamas2027-explore-sandbox`, base commit
  `281e02920a32e9878177593d62c7f2d841ed6078`).
- **Do not commit. Do not push.** Work uncommitted inside
  `sandbox/aamas2027_explore/` only.
- Frozen evidence (read-only, never modify):
  `/Users/zhouzhou/Desktop/kwallet-aamas-stage4-d1/artifacts/aamas2027/`
  (stage2 raw tarball, stage3 analysis, stage3_5 h1_multiseed, stage4 d1_h1_safe).
- Extracted Stage 2B raw bundle (read-only; per-cell episodes.csv/result.json):
  `/Users/zhouzhou/Desktop/kwallet-aamas-results/artifacts/aamas2027/stage3/work/raw/KWALLET_AAMAS_STAGE2B_RAW_20261002/output/`
- Frozen checkpoints (SHA-verified, only best_model.pth per run):
  `/Users/zhouzhou/Desktop/kwallet-aamas-artifacts/checkpoints/{basic_ppo,factorized_ac,conditional_factorized_ac}/...`
  and `/Users/zhouzhou/benchmarks/A-EVAL-JA-C800-S123/data/checkpoints/`.
- Run metadata (run_info.json per historical run):
  `/Users/zhouzhou/Desktop/kwallet-aamas-artifacts/metadata/runs/...`
- NEW12-v1 pools: inside the frozen raw bundle (`new12/`) and referenced by
  manifest SHA 3a89e602...cac5. Historical train/val pools:
  `/Users/zhouzhou/Desktop/kwallet-aamas-artifacts/pools/`.
- Runtime: `/Users/zhouzhou/benchmarks/A-EVAL-JA-C800-S123/.venv/bin/python`
  (py3.10.22, torch 2.5.1, numpy 2.0.2; the approved locked runtime).

## 2. What was inspected (pre-sprint audit, ~1.5 h)

- `tools/aamas_stage2/adapter.py` (job matrix validation, checkpoint resolution,
  `validate_training_receipt`), `tools/aamas_stage4/d1_core.py` (H1 runner +
  telemetry), frozen env `src/idea3/context_attention/kwallet_ctx_attn_fair_benchmark.py`
  (reward, state, step), `run_conditional_factorized_ac_benchmark.py`
  (val_metric plumbing), all 42 Stage 2B episodes.csv, stage3/3.5/4 reports and
  CSVs, 4 B-TRAIN receipts, artifact manifest, checkpoint stores.

## 3. Important findings (details + FACT/INFERENCE labels in hypotheses/HYPOTHESIS_AUDIT.md)

1. **Checkpoint selection was flush-blind** (FACT): best model chosen by
   validation `value_accept_ratio` every 50 episodes; Money's −10·flushes term
   never entered selection.
2. **No intermediate checkpoints survive** (FACT): retrospective rescoring of
   alternative checkpoints is impossible; only a retraining probe can measure
   the selection effect.
3. **Training reward is nearly Money-aligned** (FACT): Money = 1000·R + 20·drops
   exactly; per-episode correlation 0.93–0.97. Reward redesign is low value.
4. **Gap anatomy differs by capacity** (FACT): C800 gap ≈ 763 and is ~100%
   insufficient drops (BF has zero; SC-FAC 35–43/ep). C1200 gap ≈ 750–1080 and
   is ~80% excess flush cost (SC-FAC 589–622 flushes vs BF 542).
5. **H1 collapses are flush-rate collapses** (FACT): C800/S777 H1 flushed 148
   vs SC-FAC's 224 (ins drops 36→86, Money −566); C1200/S323 H1 flushed 343 vs
   609 (ins drops 6→171, Money −3317). Success cells keep flush rate near the
   trained rate. H1 failure is NOT ordered by SC-FAC base quality.
6. **BF maintains a zero-infeasible invariant** (FACT): BF flush rule
   (lowest usable non-settle wallet below 0.5·C/k) yields 0 zero-feasible steps
   on every inspected pool; SC-flush trajectories hit ~4% zero-feasible steps.

## 4. Hypothesis ranking (full details in hypotheses/HYPOTHESIS_AUDIT.md)

1. H-A: H1 failure = flush-rate collapse under BF-settle distribution shift.
2. H-B: rule-guarded hybrid closes C800 gap without learning.
3. H-C: C1200 gap is mostly learned-flush overspend.
4. H-D: checkpoint-selection problem (real, retrospective-unfixable; one
   retraining probe can size it).
5. H-E: objective mismatch — demoted (algebra shows near-alignment).

## 5. Experiments already performed (by the pre-sprint audit — do not redo)

- Money decomposition of all frozen SC-FAC/BF cells →
  `findings/FINDING_gap_decomposition.csv`.
- H1 vs SC-FAC flush/drop matrix for all 10 cells →
  `findings/FINDING_h1_vs_scfac_flush_drops.csv`.
- Training-return vs Money correlation per cell (0.93–0.97).
- Checkpoint-store inventory + SHA verification against the frozen job summary.

No new policy evaluation was run; no rescore was possible (see finding 2).

## 6. Exact outputs/results to reuse

- Reference scores loader: `infra/reference_scores.py`
  (BF macro: C800 4661.325, C1200 15298.890; frozen CSV readers).
- Splits: `infra/splits.py` (SCREEN 0–39, CHECK 40–99, FULL 0–199).
- Metrics: `infra/metrics.py` (summarize, gap_recovery, zero_feasible_steps).
- Templates: `infra/templates/ATTEMPT_TEMPLATE.md`,
  `infra/templates/RESULT_LOG_TEMPLATE.csv`.
- H1 deficit anchors for E1 success criteria: C800/S777 deficit 565.8 Money
  (SC-FAC 4054.2 − H1 3488.4); C1200/S323 deficit 3317.2 (14381.0 − 11063.8).

## 7. Experiments NOT worth repeating

- D1 same-wallet mask (gate FAIL, +1.35/+13.3 Money). H2 hybrid (minimal/
  negative in b0 pilots). Any unchanged-H1 rerun (parity EXACT, no new info).
- Reward redesign / feature engineering (see H-E; out of scope tonight).

## 8. Recommended first overnight action

Implement and run **E1 (flush-floor guardrail)**: H1 with BF-flush override
whenever n_feasible == 0, on cells C800/S777, C1200/S323 (+ C1200/S532 control),
SCREEN split. Full ladder and predeclared criteria in
`OVERNIGHT_EXPERIMENT_PLAN.md`. Start there within your first 10 minutes.

## 9. Commands / environment

```bash
PY=/Users/zhouzhou/benchmarks/A-EVAL-JA-C800-S123/.venv/bin/python
cd /Users/zhouzhou/Desktop/kwallet-aamas-explore
cp tools/aamas_stage4/d1_core.py sandbox/aamas2027_explore/code/
# d1_core expects the frozen repo layout; run from the worktree root.
# D1's own runner scripts/configs live in artifacts/.../stage4/d1_h1_safe/code/
# (frozen, read-only — copy, don't edit in place).
```

Environment variables used by the Stage 2/4 harnesses when importing historical
modules: set writable `XDG_CACHE_HOME` and `MPLCONFIGDIR` outside the repo and
artifact roots (adapter enforces this).

## 10. Known pitfalls

- **CSV writer alias bug** (from stage 3.5): use `money` key from episode rows;
  the `eval_money` alias crash was fixed by exact-value reproduction — do not
  re-derive Money differently; identity money == settled − 10·flushes is
  asserted by the frozen validator.
- Only `best_model.pth` exists per run; any plan needing per-epoch checkpoints
  requires retraining (E4 only).
- The frozen adapter refuses to overwrite outputs (`safe_output`) — always write
  to fresh paths under `sandbox/aamas2027_explore/results/`.
- torch load: use `weights_only=True`, `map_location='cpu'`, `strict=True`
  (as the adapter does).
- Episode-index discipline: never pick subsets after seeing results; SCREEN and
  CHECK are disjoint; FULL only for finalists.
- Everything tonight is post-hoc exploratory (NEW12 already used for Stage 2B/3
  selection) — label outputs accordingly; do not call anything confirmatory.

## 11. Remaining 5–7 h strategy (summary)

Hour 0–1: E1 (harness copy, guardrail arm, U-parity, 3 SCREEN cells).
Hour 1–3: E2 trigger sweep (if E1 promotes), CHECK confirmation.
Hour 3–4: E3 flush-cap probe at C1200 (if E2 survives).
Hour 4–6.5: E4 single-cell retraining probe with checkpoint retention (only if
≥ 2.5 h remain; otherwise skip and write up).
Final 30 min: `results/OVERNIGHT_RESULTS.md` with outcomes vs predeclared
criteria, gap-recovery table, stop/continue per hypothesis.
