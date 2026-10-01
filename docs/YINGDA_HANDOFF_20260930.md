# Yingda handoff — ICASSP 2027 K-Wallet work (2026-09-30)

## Overview

This branch (`integration/yingda-icassp2027-handoff`) delivers the complete
research body produced on the `Yingda-Yu/Kwallet-Rl` fork, merged on top of
the latest `zhouj-tech/kwallet-rl:main`. It continues the original K-Wallet
project (streaming transaction collateral control) with a faithful
environment port, a clean canonical `kwallet` Python package, four learned
policy structures, strong rule baselines, a frozen multi-seed experiment
suite with paired statistics, and a compile-ready ICASSP 2027 manuscript.

No experiment was re-run during integration; all numbers below come from
committed frozen CSVs under `results/tables/` and
`paper/icassp2027/assets/paper_claims.json`. The upstream recovery baseline
(`src/ideaextra`, `src/idea5`, `research/` workflow files) is preserved
unchanged.

## Major new implementations

Canonical package `src/kwallet/` (Python >= 3.10, `pip install -e .`):

- **Faithful environment port** — `envs/kwallet.py`: `3k+2` observation,
  flush-before-settle execution, one settle + one flush per step,
  `F=3` cooldown, delayed refill-to-full, oversize/frozen/conflict/
  insufficient drop accounting, Money metric (`p x - tau flushes`).
- **Data layer** — `data/regimes.py` (faithful port of the original
  12-regime generator) and `data/pools.py` (cached, hash-checked pools;
  5000 train / 300 validation (25 per regime) / 200 test episodes per
  regime; separate seed namespaces, no train/test leakage).
- **PPO reimplementation** — `training/ppo.py`: clipped PPO with GAE,
  whole-episode rollouts, linear entropy anneal, validation checkpoint
  selection.
- **JA-PPO** — `policies/actors.py:JAPPO`: one joint categorical head over
  `(k+1)^2` (settle, flush) pairs.
- **IFAC** — `policies/actors.py:IFAC`: two independent `k+1`-categorical
  heads (`2(k+1)` logits).
- **SC-FAC** — `policies/actors.py:SCFAC`: settle-conditioned factorized
  actor-critic with the directed path
  `pi(a_s,a_f|s) = pi_s(a_s|s) pi_f(a_f|s,a_s)` and exact conditional
  entropy/log-prob; plus `no-cond` and `shuffled-conditioning` ablation
  variants.
- **Set-SC-FAC** — `policies/set_actors.py:SetSCFAC` (and `SetIFAC`):
  permutation-equivariant, `k`-independent per-wallet encoder with
  symmetric pooling; one trained model loads and runs zero-shot at unseen
  wallet counts.
- **Strong feedback baselines** — `baselines/rules.py` (FA, FWF) and
  `baselines/rules_strong.py` (BFP: best-fit settle + proactive
  most-depleted usable-wallet flush when balance `b_i < theta C/k`;
  `theta in {0.3,0.5,0.8}` selected once on validation only, `theta=0.5`
  frozen for all capacities).
- Evaluation: `evaluation/rollout.py`, `parallel.py` (sharded CPU eval),
  `stats.py` (paired differences, bootstrap CIs, seed summaries),
  `switch.py` (abrupt-regime-switch / OOD evaluation), `compute.py`
  (parameter count and per-step latency).
- CLI: `python -m kwallet.cli {doctor,gen-pools,train,evaluate,bench}`.

## New experiments

- **Multi-seed stationary evaluation** — five seeds (123, 323, 532, 777,
  999), four capacities `C in {800,900,1000,1200}`, `k=24`, `F=3`,
  deterministic argmax evaluation on fixed test pools.
- **Capacity sweep** across the four collateral levels.
- **Conditioning ablations** — no-conditioning and shuffled-conditioning
  SC-FAC vs SC-FAC at `C=1200` (n=3 seeds), matched parameters/budget.
- **Cross-`k` zero-shot transfer** — Set-SC-FAC (and flat-MLP reference)
  trained at each `k in {6,12,24}` and deployed, without retraining, at
  every other `k`; diagonal vs off-diagonal Money, retention percentages.
- **Abrupt regime switching** — six deterministic episodes spliced across
  regimes at a fixed switch point (e.g. calm <-> burst, early/late);
  post-switch Money and avoidable drops (n=3 seeds).
- **Efficiency / parameter analysis** — joint `(k+1)^2` vs factorized
  `2(k+1)` output counts, total parameters, per-step CPU latency.
- **Statistical paired tests** — seed-paired contrasts with 95% CIs and
  Holm step-down adjusted p-values within pre-specified contrast families;
  procedure documented in
  `paper/icassp2027/STATISTICAL_AUDIT.md`, raw p-values in
  `results/tables/*_paired.csv`.
- Failed seeds and null results are retained; screening and confirmatory
  results are kept separate.

## Major empirical findings

(Only claims backed by committed result files; see Tables 1-4 and
`assets/paper_claims.json` in the paper directory.)

1. **Factorization improves over the joint action head.** SC-FAC beats
   JA-PPO on Money at all four capacities (raw paired contrasts
   +677/+999/+1327/+1285; all four survive Holm correction); IFAC beats
   JA-PPO at `C in {800,900,1200}` (3/4 raw; `C=900` adjusted
   p = 0.064). Outputs drop from `(k+1)^2 = 625` to `2(k+1) = 50` logits
   at `k=24`, with lower actor parameters and millisecond-level per-step
   CPU latency.
2. **The settle -> flush conditioning path has no statistically detectable
   Money benefit.** SC-FAC vs IFAC at `C=1200` is not significant
   (n=5 paired, p = 0.357, 95% CI [-1231, 560]); no-conditioning and
   shuffled-conditioning ablations also detect no gain
   (`C=1200` p = 0.47 and 0.29; `C=800` raw p = 0.27/0.65; all Holm
   adjusted values 1.0).
3. **The permutation-equivariant set encoder enables cross-`k` deployment.**
   A single model loads and runs zero-shot at every unseen wallet count
   in `{6,12,24}` (off-diagonal cells exist; flat k-shaped MLP cannot),
   e.g. 4433 for k=6 -> 24 and 43733 for k=12 -> 6; mean zero-shot
   retention is about 65%. Transfer quality is asymmetric and weakest
   toward smaller-wallet operating points.
4. **The strong BFP rule is the strongest tested controller.** BFP0.5
   attains the highest Money at every capacity (e.g. 15327 at `C=1200`),
   zero avoidable drops in 11 of 12 method-capacity paired regimes,
   without training.
5. **No observed RL adaptation advantage over BFP under abrupt switches.**
   Learned policies transfer substantially better than naive rules
   (28-56 vs 144-148 post-switch avoidable drops), but remain worse than
   BFP on both post-switch Money and drops; the SC-FAC drops contrast is
   not significant after Holm correction (raw p = 0.035, adjusted
   p = 0.069).
6. **Failed seed retained.** The primary all-seed `k=24` Set-SC-FAC mean
   is 9190 (flat MLP 13610) because one training run (seed 532) yields
   Money 0; it is retained in the primary aggregate and reported as
   training instability. The 13785 value over the two non-failed runs is
   labeled a sensitivity calculation only.

## Repository map

- `src/kwallet/` — canonical package: `envs/`, `data/`, `policies/`,
  `training/`, `baselines/`, `evaluation/`, `cli.py`.
- `scripts/` — orchestration:
  - `run_experiments.py` — stationary matrix driver (writes manifests);
  - `run_kscale.py` — cross-`k` transfer runs;
  - `run_switching.py` — regime-switch evaluation;
  - `aggregate_results.py`, `aggregate_kscale.py`, `paired_stats.py` —
    CSV/statistics aggregation;
  - `make_paper_assets.py` — regenerates paper tables/figures from result
    CSVs (no hand-typed numbers).
- `experiments/` — frozen run manifests:
  `manifest_matrix_smoke.csv`, `manifest_matrix_main.csv`,
  `manifest_matrix_ablation.csv`, `manifest_kscale.csv`
  (method, C, k, F, seed, episodes, DONE/FAILED status, artifact path,
  wall time).
- `results/tables/` — committed compact result tables (long-format per-seed
  CSVs, main tables, regime tables, paired contrasts, post-hoc drop
  breakdowns, and `SUBMISSION_ARTIFACT_SHA256.json`). Raw run directories
  (`runs/`) are git-ignored generated artifacts.
- `paper/icassp2027/` — ICASSP 2027 manuscript: `main.tex`, compiled
  `main.pdf` (5 pages), `refs.bib`, `assets/figs/`,
  `assets/tables/` (auto-generated), `assets/paper_claims.json`
  (machine-readable claim ledger linking every headline number to its
  source CSV), plus audit documents (`FINAL_SUBMISSION_AUDIT.md`,
  `STATISTICAL_AUDIT.md`).
- `docs/` — `PAPER_CODE_MAP.md` (claim-to-code map with PORT /
  REIMPLEMENTED / NEW labels), `REPRODUCTION_REPORT.md`, `DECISIONS.md`,
  `ENVIRONMENT.md`, `AGENT_PROGRESS.md`, `BLOCKERS.md`,
  `TRAE_START_HERE.md`, `TRAE_TASK_BOARD.md`,
  `TRAE_KWALLET_ICASSP2027_EXECUTION.md`.
- `tests/` — 37 pytest tests (env semantics, data pools, policy
  factorization math vs enumerated distributions, set equivariance,
  end-to-end smoke).
- Preserved upstream material: `src/ideaextra/`, `src/idea5/` (recovered
  DQN/AC baseline), `research/` (PROJECT_STATUS, RESULTS_LEDGER,
  TASK_QUEUE), `AGENTS.md`, `TRAE_MASTER_PLAN.md`.

## Reproduction commands

Verified on Linux with Python 3.10 in an isolated user environment;
evaluation always runs on CPU.

```bash
# 1. Install (isolated env recommended)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"

# 2. Sanity
python -m kwallet.cli doctor
python -m pytest                      # 37 passed

# 3. Generate deterministic, hash-checked transaction pools
python -m kwallet.cli gen-pools      # 5000/300/200 per regime, 12 regimes

# 4. Smoke run for every learned/rule method (CPU, minutes)
python scripts/run_experiments.py --tier smoke --device cpu

# 5. Main stationary matrix: 4 capacities x methods x 5 seeds
#    (training may use --device cuda; evaluation stays on CPU workers)
python scripts/run_experiments.py --tier main --workers 1 --device cpu

# 6. Aggregate -> results/tables/
python scripts/aggregate_results.py --exp matrix_main
python scripts/paired_stats.py --exp matrix_main

# 7. Conditioning ablations (n=3)
python scripts/run_experiments.py --tier ablation
python scripts/aggregate_results.py --exp matrix_ablation
python scripts/paired_stats.py --exp matrix_ablation

# 8. Cross-k transfer (k in {6,12,24}, zero-shot matrix)
python scripts/run_kscale.py
python scripts/aggregate_kscale.py

# 9. Abrupt regime switching / OOD
python scripts/run_switching.py --seeds 123 323 532

# 10. Regenerate paper tables/figures from the CSVs, then build the PDF
python scripts/make_paper_assets.py
cd paper/icassp2027
pdflatex -interaction=nonstopmode main.tex && bibtex main && \
pdflatex -interaction=nonstopmode main.tex && \
pdflatex -interaction=nonstopmode main.tex   # 5 pages
```

Run manifests in `experiments/` record the exact executed configurations;
the committed tables correspond to the frozen runs and can be compared
against fresh reproductions (training is seeded but PPO is not
bit-reproducible across platforms).

## Known limitations

- The original manuscript is an anonymous, unpublished internal document;
  its learned-method numbers could not be regenerated from the original
  code (no PPO implementation existed in the recovered repository), so the
  three learned policies are documented reimplementations, not exact
  reproductions; missing paper hyperparameters were fixed on validation and
  labeled `chosen_for_reimplementation`.
- Ablation and switching comparisons use n=3 seeds; the stationary matrix
  uses n=5.
- One `k=24` Set-SC-FAC training run failed (seed 532, Money 0) and is
  retained in the primary aggregate; conclusions about matched-scale
  transfer superiority are not claimed.
- Scope is deliberately narrow: one flush per step, finite horizon, one
  fixed fee, no lookahead, homogeneous wallets, single-process PPO without
  vectorized rollouts; richer temporal prediction, heterogeneous wallet
  sizes/fees and multi-step lookahead are future work.
- The environment/generator are faithful ports, but the one-flush FA/FWF
  rule definitions were partially reconstructed (the native reference is
  multi-flush); see `docs/PAPER_CODE_MAP.md` and
  `docs/REPRODUCTION_REPORT.md` for evidence labels.
- The paper PDF is an internal submission candidate; author metadata,
  copyright, EDICS selection and final AI-disclosure approval remain with
  the authors (see `paper/icassp2027/FINAL_SUBMISSION_AUDIT.md`).
