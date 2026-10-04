# HYPOTHESIS_AUDIT.md — K-Wallet AAMAS 2027 exploratory sprint

Pre-sprint audit, 2026-10-03. Base: frozen commit 281e029 (branch
`aamas2027-stage4-d1`). All claims below are labeled FACT (verified in frozen
artifacts/code) or INFERENCE. Post-hoc exploratory; nothing here is confirmatory.

Supporting data files:
- `findings/FINDING_gap_decomposition.csv` — per method/C/seed Money decomposition
- `findings/FINDING_h1_vs_scfac_flush_drops.csv` — H1 vs SC-FAC flush/drop matrix

---

## Established facts used below

- FACT A (checkpoint selection): every learned-method checkpoint was selected
  by **validation `value_accept_ratio`** (settled/requested), evaluated every 50
  training episodes on the historical MIX12_EQ val pool. Pinned in
  `tools/aamas_stage2/adapter.py` `validate_training_receipt` ("Wrong checkpoint
  selection metric" unless metric == value_accept_ratio), in
  `run_conditional_factorized_ac_benchmark.py` (`val_metric` default
  `value_accept_ratio` for original reward), and in the B-TRAIN receipts
  ("checkpoint selected solely by validation value_accept_ratio").
  The metric is **flush-blind**: it ignores the −10·flushes term entirely.
- FACT B (no intermediate checkpoints): only `best_model.pth` per run survives
  (36 recovered + 4 receipt-retrained = 40 named checkpoints; verified against
  the artifact manifest and job summary SHA-256s). No per-episode or per-50-ep
  checkpoints exist anywhere inspected. **Retrospective rescoring of alternative
  checkpoints is impossible without retraining.**
- FACT C (objective algebra): training reward per step is
  `r_t = settled_tx/1000 − 0.02·dropped − 0.01·flushed` (env source
  `src/idea3/context_attention/kwallet_ctx_attn_fair_benchmark.py` lines 466-510,
  shaping disabled). With T=1000 and Money = settled − 10·flushes:
  **Money = 1000·R_train + 20·drops exactly.**
  The settle/flush price ratio is exactly Money-aligned (0.01 flush vs x/1000
  settle == 10 flush vs x settle). The only distortion is an extra −0.02/drop
  penalty (= −20 Money units per drop), i.e., training is more drop-averse than
  Money. Reconstructed per-episode training return correlates with Money at
  r = 0.93–0.97 across all frozen SC-FAC/BF cells.
- FACT D (gap anatomy): vs BF-T0.5 (NEW12 macro):
  - C800: gap ≈ +763 Money. BF has **0.0 insufficient drops/episode**; SC-FAC
    has 35–43/episode. SC-FAC actually flushes *less* than BF (217–226 vs 258).
    The C800 gap is essentially entirely missed settlements (starvation).
  - C1200: gap ≈ +750–1080 Money depending on seed. SC-FAC flushes *more* than
    BF (589–622 vs 542) and still has 2–15 insufficient drops/episode. Roughly
    80% of the C1200 gap is excess flush cost, ~20% settled shortfall.
- FACT E (H1 failure pattern): H1 (BF-settle + frozen SC-flush) outcomes are
  **not** ordered by SC-FAC base quality. Collapses (C800/S777: −566 vs its own
  SC-FAC; C1200/S323: −3317) coincide with **flush-rate collapse** of the frozen
  flush head under the BF-settle state distribution (S777: 148 vs 224 flushes;
  S323: 343 vs 609) and a resulting insufficient-drop explosion (86/ep and
  171/ep). Successes (S532 C1200: +679) keep flush rate near the trained rate.
- FACT F (state features): E0 base state = per-wallet balance, usable flag,
  freeze-remaining/F (3k) + current tx + time. No aggregate
  feasible-wallet-count or refill-pressure summary; time-to-refill exists
  per-wallet in raw form. D1: BF shows 0 zero-feasible steps on all inspected
  pools; SC-flush trajectories average ~94.6k (C800) / ~104.7k (C1200)
  zero-feasible steps per 2.4M-step cell (~4%).

---

## Ranked hypotheses

### H-A. H1 failure = flush-rate collapse under BF-settle distribution shift (starvation cascade)

- Rank: **1 (highest information gain per minute)**
- Evidence: FACT E — Δflushes and Δinsufficient-drops perfectly sign-track
  H1 success/failure across all 8 measured cells. FACT D/F — starvation is the
  C800 gap and the H1 collapse mechanism.
- Counter-evidence: D1 mask experiment showed removing same-wallet conflicts
  does NOT fix it (conflict importance LOW) — but that intervention did not
  address under-flushing, so it does not contradict H-A.
- Mechanism: frozen flush head conditions on the settle index; BF-settle visits
  balance/usable states the head never saw under SC-settle, biasing it toward
  no-op flushes; wallets drain, feasible count collapses, drops cascade.
- Cheapest discriminating experiment: **E1 — flush-floor guardrail** (see
  OVERNIGHT_EXPERIMENT_PLAN.md). H1 + override to the BF flush rule whenever
  the count of usable wallets with balance ≥ current tx is zero. Evaluation-only,
  no training. First on the two collapse cells (C800/S777, C1200/S323), SCREEN.
- Estimated runtime: ~1–2 min/cell on SCREEN (D1 full cells took ~3 min each
  on 2400 episodes; SCREEN is 480).
- Implementation complexity: low (one branch in the decision function of a
  D1-core-derived runner).
- Expected upside: if starvation is the whole story, collapse cells recover
  most of the H1 deficit and possibly beat SC-FAC; success cells should be
  nearly unchanged (guardrail rarely fires).
- Kill criterion: on SCREEN, C1200/S323 guardrail Money < SC-FAC Money − 1% BF,
  or recovery < 25% of the H1→SC-FAC deficit on both collapse cells.

### H-B. Gap closure without any learning: rule-guarded hybrid dominates H1

- Rank: 2 (condition on H-A signal)
- Evidence: FACT D — at C800 the entire SC-FAC gap is insufficient drops; any
  policy that keeps BF's zero-starvation invariant while settling greedily
  should match BF at C800. D1 masked-H1 already showed small positive deltas.
- Counter-evidence: C1200 gap is mostly flush *cost*, so a pure starvation fix
  cannot close C1200; flush-cost overspend needs a separate mechanism (H-C).
- Mechanism: BF's flush rule (flush lowest usable non-settle wallet below
  0.5·C/k) is a feasibility-maintenance rule; learned flush heads do not
  maintain it.
- Cheapest discriminating experiment: E2 — threshold sweep of the guardrail
  trigger (zero-feasible → <2 feasible → balance-thresholded) on CHECK, only
  if E1 promoted.
- Runtime: minutes per sweep point per cell.
- Complexity: low.
- Expected upside: defines the best evaluation-only policy this sprint can
  produce; bounds what learned-flush could ever add on top.
- Kill criterion: best guardrail variant recovers <50% of the SC→BF gap at
  C800 on CHECK.

### H-C. C1200 flush-cost overspend: learned flush head wastes refills

- Rank: 3
- Evidence: FACT D — SC-FAC flushes 589–622/ep vs BF 542 at C1200 (≈ +470–800
  Money of pure flush cost), and refilled-value-reused telemetry (D1 §12) shows
  SC arms refill ~5x more value than BF with higher never-reused cycles.
- Counter-evidence: some SC flushing may buy the higher settled SC achieves in
  some regimes; H1 at C1200/S999 *increased* flushes and improved Money.
- Mechanism: flush-blind checkpoint selection (FACT A) never penalized flush
  growth during model selection; the −0.01/flush reward price is exactly right
  per-step but credit assignment over the F=3 refill delay plus horizon effects
  may still under-price marginal flushes.
- Cheapest discriminating experiment: E3 — flush-budget-capped H1 (cap flushes
  per episode at BF's 542, or penalized flush logits) on SCREEN, C1200 cells
  only. Alternatively a pure "SC-settle + BF-flush" (H2-style) rescore as a
  flush-side-only control (b0 pilot showed H2 minimal/negative — weak prior).
- Runtime: minutes/cell. Complexity: low-moderate.
- Expected upside: up to ~500 Money/cell at C1200 if waste is real and
  removable without starving.
- Kill criterion: capping flushes at BF level reduces Money by >1% BF (i.e.,
  the extra flushes were load-bearing) on ≥3 of 5 seeds on SCREEN.

### H-D. Checkpoint-selection problem (flush-blind best-model choice)

- Rank: 4 (real but retrospectively unfixable; prospective only)
- Evidence: FACT A + FACT B. At C1200, cross-seed spread of the *selected*
  checkpoints is ~330 Money in flush cost alone with <2% spread in accept
  ratio — selection plausibly traded large flush cost for tiny accept-ratio
  gains. Whether any *alternative* checkpoint of the same run had better Money
  is unknowable from surviving artifacts.
- Counter-evidence: training return itself is Money-aligned (FACT C), so the
  training gradient was not misdirected; only the selection metric was.
- Mechanism: selection maximized settled/requested, ignoring −10·flushes.
- Cheapest discriminating experiment: none possible retrospectively (FACT B).
  Prospective: E4 — one retraining run with checkpoint retention every 50 eps
  + selection by eval_money, single cell (C1200/S323, the worst H1 seed), only
  if E1–E3 are done and time remains. Runtime: hours (historical 1000-ep CPU
  trainings); complexity: moderate (config change + checkpoint retention).
- Expected upside: quantifies the selection effect size; possibly a better
  SC-FAC checkpoint for free.
- Kill criterion: best-eval_money checkpoint within ±0.5% BF of the
  historically selected one on SCREEN → selection problem is small; stop.

### H-E. Objective mismatch (training reward vs Money)

- Rank: 5 (demoted by algebra)
- Evidence: FACT C — exact identity Money = 1000R + 20·drops; settle/flush
  pricing exactly aligned; observed per-episode correlation 0.93–0.97.
- Counter-evidence: the extra −0.02/drop could distort flush decisions that
  trade a future drop against a current flush; magnitude bounded by ~20 Money
  units per marginal drop.
- Conclusion: mismatch is real but small. Do not spend sprint time modifying
  the reward; only relevant if a retraining run (E4) happens anyway — then set
  `val_metric=eval_money` and keep the env reward unchanged.
- Kill criterion: n/a (already low priority).

---

## Inventory-maintenance explanation ranking (for record; feeds H-A/H-B design)

Ranked by plausibility given FACT F and D1 telemetry:

1. **No feasibility-margin trigger in the policy input/output structure** —
   the network must infer "number of wallets that can cover the current tx"
   from raw balances; BF has it as an explicit rule. Design implication:
   guardrail triggers should use feasible-count, not raw balance.
2. **Long-horizon credit assignment** — flush benefit arrives after F=3 steps
   and prevents starvation episodes later; γ=0.98 (~50-step horizon) partially
   covers this, but the reward price of a flush is immediate.
3. **Distribution shift of the frozen flush head under foreign settle** —
   proven for H1 (FACT E); the head's no-op bias under BF-settle states is the
   proximate collapse cause.
4. Missing usable-balance summary / time-to-refill aggregate — raw per-wallet
   ingredients exist in the state; likely redundant for a sufficiently trained
   network.
5. Missing action prior — true by construction of RL-from-scratch, but the
   guardrail experiments supply the prior externally and test its value.

Do NOT implement feature-engineering variants this sprint; the guardrail
experiments test the same mechanism at 1% of the cost.
