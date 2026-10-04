# STAGE 3.8 FINAL SUMMARY — SC-FAC-Centered Model Exploration

**Closed:** 2026-10-04 · **Status:** CLOSED · **Verdict:** no candidate promoted ·
Post-hoc exploratory, not confirmatory.

## A. Why Stage 3.8 existed

The campaign followed:

- **Stage 3 strong BF result** — deterministic BF-T0.5 beat every learned method
  at both capacities under NEW12-v1.
- **H1/H2 decomposition** — supplying BF settlements to frozen SC flush heads
  (H1) and excluding the settlement wallet from BF flushing (H2) did not
  recover the BF gap; H2's failure showed exclusion alone is insufficient.
- **D1 masking diagnostic** — removing executed same-wallet conflicts improved
  mean H1 Money by only +1.3505 (C800) / +13.3151 (C1200); the declared mask
  gate failed and the mismatch classification was WEAK.
- **Checkpoint-selection audit** — historical learned checkpoints had been
  selected by validation `value_accept_ratio` (flush-blind), and no intermediate
  checkpoints survived, so retrospective rescoring was impossible.

The goal of Stage 3.8 was to test whether a new SC-FAC-centered learned model
could materially reduce or exceed the BF gap **without becoming an
inference-time BF hybrid**.

## B. Historical audit findings

- Historical checkpoint selection used **value_accept_ratio**, not Money.
- Intermediate historical checkpoints **did not survive** — only
  `best_model.pth` per run.
- Training reward was **relatively Money-aligned**
  (`Money = 1000·R_train + 20·drops` exactly; settle/flush price ratio exactly
  Money-aligned; only an extra drop penalty made training more drop-averse).
- **Checkpoint selection was the stronger protocol weakness.**

## C. Candidate definitions

- **SC-M** — matched vanilla SC-FAC control with a **full checkpoint bank**
  (every 50 episodes) and **Money-based selection** (best_by_money primary).
- **IA-SC** — inventory-aware SC-FAC: 9 global inventory/refill features +
  4 selected-wallet physical features; **80,083 parameters**; no BF dependency.
- **TP-SC** — original **77,907-parameter** SC-FAC with a **training-only BF
  teacher**: λ_BF(e) = 0.01·max(1 − e/250, 0); episodes 250–999 pure PPO;
  **no BF inference component**.

## D. Protocol

- SCFAC-DESIGN-v3 (frozen split family: VAL-v3 / SCREEN-v3; CHECK-v3 and
  FULL-DEV-v3 frozen and **not consumed**).
- 1000 full training episodes; checkpoint every 50 episodes (20 per cell).
- best_by_money primary checkpoint selection; alternative selectors retained
  for the checkpoint-selection comparison.
- Matched 4-seed SCREEN (123 / 323 / 777 / 999) × 2 capacities (C800, C1200)
  × 3 models = **24 training cells, 24/24 COMPLETE**.
- Fixed deterministic BF reference (BF-T0.5, once per capacity).
- Prospectively frozen SCREEN gate.

## E. Complete per-seed SCREEN-v3 table (macro12 Money)

### C800

| Model | s123 | s323 | s777 | s999 | mean |
|---|---:|---:|---:|---:|---:|
| SC-M | 3456.1063 | 2966.3938 | 3924.3729 | 3960.0979 | **3576.7427** |
| IA-SC | 3990.0271 | 3464.1167 | 3776.6292 | 3649.4042 | **3720.0443** |
| TP-SC | 4207.3396 | 4019.1125 | 3897.3312 | 3744.2188 | **3967.0005** |
| BF | — | — | — | — | **4661.9917** |

### C1200

| Model | s123 | s323 | s777 | s999 | mean |
|---|---:|---:|---:|---:|---:|
| SC-M | 14439.0188 | 14214.5542 | 14533.6917 | 14329.8417 | **14379.2766** |
| IA-SC | 14148.6479 | 14033.3500 | 14121.0083 | 14107.0813 | **14102.5219** |
| TP-SC | 14357.6604 | 14417.1708 | 14561.1729 | 14584.5188 | **14480.1307** |
| BF | — | — | — | — | **14687.3979** |

## F. Gate interpretation (frozen SCREEN gate)

**IA-SC:**

- C800: +143.30 vs SC-M, below threshold (271.31); 2/4 positive.
- C1200: −276.75; 0/4 positive.
- **=> FAIL**

**TP-SC:**

- C800: +390.26 (above threshold), but only 2/4 positive → fails seed-consistency.
- C1200: +100.85 and 3/4 positive → passes locally.
- C1200 passes locally, C800 fails seed-consistency → **overall FAIL**.

Formal verdict:

```
IA_SC_SCREEN_PASS = NO
TP_SC_SCREEN_PASS = NO
SCREEN_SURVIVORS = NONE
RECOMMEND_HELDBACK_SEED532_CHECK = NO
```

## G. Checkpoint-selection conclusion

best_by_money improved SC-M SCREEN Money by approximately:

- **+42.4** vs the value_accept_ratio selector
- **+127.2** vs the training_return selector

Classified as **MODERATE**: a real protocol improvement, but insufficient to
close the BF gap.

## H. Research conclusions

- **IA-SC** does not support explicit inventory feature augmentation as a
  sufficient/stable solution.
- **TP-SC** provides the strongest positive signal: BF-derived finite training
  supervision can improve average learned-policy Money, but the gain is
  seed-sensitive and does not pass the frozen gate.
- **BF remains stronger at both capacities.**
- **No candidate is promoted.**
- **Do not reinterpret SCREEN after the fact.**

## I. Future idea bank — UNEXECUTED

Preserved for the record; **not results, not executed, not endorsed by data:**

- ET-SC — event/target factorization.
- AS-SC — auxiliary future-availability representation learning.
- Improved BF distillation / teacher scheduling.
- Residual learning.
- Permutation-equivariant / set-policy extensions.
- Better separation of replenishment timing and target selection.

---

## Provenance

- 24/24 training cells COMPLETE (see `protocol/OVERNIGHT_RUN_LEDGER.csv` and
  `cell_summaries/CELL_INDEX.csv`).
- All SCREEN-v3 evaluation JSONs in `screen_eval/`; compact reports in
  `results/`; executed code in `code/`.
- Full checkpoint banks preserved outside Git in
  `~/Desktop/kwallet-archives/KWallet_Stage3_8_FULL_BANK_20261004.tar.gz`
  (SHA-256 sidecar + manifest). Source banks not deleted.
- No CHECK-v3 / FULL-DEV-v3 / NEW13 / seed532 artifact was authorized or consumed.
- Scientific base commit `281e02920a32e9878177593d62c7f2d841ed6078`; frozen
  Stage 2 / 3 / 3.5 / D1 artifacts unmodified.
