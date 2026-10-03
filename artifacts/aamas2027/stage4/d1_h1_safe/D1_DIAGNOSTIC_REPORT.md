# D1-H1-SAFE-v1 Diagnostic Report (Stage 4)

**POST-HOC EXPLORATORY MECHANISM DIAGNOSTIC — NOT CONFIRMATORY.** NEW12-v1 was already used for Stage 2B/3 method selection; nothing here is a fresh test. No training occurred. The E0 environment, all SC-FAC checkpoints, the BF-T0.5 rule, and the Stage 2/3/3.5 artifacts are frozen and unmodified. The only policy change under test is a single-element flush-logit mask. Local fork numbers are one-step **local exposure only** — neither a recovery estimate nor an optimality theorem.

## 1. Purpose and hypotheses

Stage 3.5 H1 (BF settlement + frozen SC-FAC conditional flush, the flush head conditioned on the BF settle index) produced unstable seed/capacity results. D1 separates two post-hoc explanations:

- **M1 — direct same-wallet conflict.** Because E0 decodes the flush first, when the conditional flush head selects the same wallet that the BF rule selected for settlement, the flush voids a feasible current settlement (the wallet is zeroed and placed in `refresh_targets`, so the settle check fails). Masking exactly that one flush logit should remove the loss.
- **M2 — broader policy / state-distribution incompatibility.** The frozen flush head was trained against SC-FAC's own settlement distribution; under BF settlements the visited state distribution differs, and any intervention on the head can perturb useful future behaviour beyond the mechanical conflict.

The masked arm M is a surgical restriction, not a claim that the masked action is optimal: the same-wallet flush may sacrifice the current transaction deliberately for a faster future refill.

## 2. Arms and exact intervention

Per step the BF settle index `s_BF = argmin_(usable, balance≥tx) (balance, index)` (no-op `k` if none) is computed. The frozen SC model yields flush logits `L = forward_flush_given_settle(state, s_BF)`:

- **U (unmasked):** `flush = argmax L` — the archived H1 policy.
- **M (masked):** if `s_BF < k`, set **exactly** `L[0, s_BF] = -infinity` (float min), leave every other logit untouched, then `flush = argmax` with the original smallest-index tie-break. If `s_BF == k` (no feasible settle), no masking is applied.
- **BF:** frozen BF-T0.5 settle+flush under identical instrumentation (deterministic; 2 cells).

The submitted joint action `s_BF·(k+1)+flush` is executed by the **unmodified E0**. No forced no-op, no BF threshold change, no high-balance mask, no environment/ordering change, no training.

## 3. Data, frozen bindings, provenance

- Dataset: NEW12-v1, 12 regimes × 200 episodes, T=1000, k=24, F=3; manifest SHA `3a89e6023680752cd76b1eed953353a2d5163b751da1c9c4f3d407a98883cac5`.
- Frozen raw tarball SHA: `576c9baa00c4438543ab9f9c2063f65bb26b1a6f287974a9e8ab91b18189ace3`; lineage `5573ec642f0f28c218f3e6058478f62ab6db6b2b`.
- Every SC-FAC checkpoint SHA was verified against the Stage 2 job matrix AND the Stage 3 `seed_level_scores.csv` before use; each cell records the checkpoint SHA per episode row.
- The hash-pinned Stage 2 adapter was re-verified on every run; NEW12 pools and per-episode hashes were verified through the frozen adapter. See `D1_CONFIG.json` and each cell's `result.json`.

## 4. Execution matrix and runtime

- 20 H1 cells (2 arms × 2 capacities × 5 seeds) + 2 BF cells = 22 cells × 2,400 episodes = 52,800 episodes = 52.8M env steps.
- Total instrumented wall time: 4055s (67.6 min), CPU, torch 2.5.1, numpy 2.0.2.
- Full step traces are retained only for episodes 0–1 of each regime (528 traced episodes, 528k rows); all other telemetry is bounded counters/histograms.

## 5. Mandatory U-parity gate

Status: **PASS**. Each U cell was required to reproduce its archived Stage 3.5 H1 `episodes.csv` EXACTLY on 2400 episodes × 7 fields (money, settled, flushes, drops, accepted_count, insufficient_drops, oversize_drops), same regime/episode ordering.

| C | seed | result |
|---|---|---|
| 800 | 123 | EXACT |
| 800 | 323 | EXACT |
| 800 | 532 | EXACT |
| 800 | 777 | EXACT |
| 800 | 999 | EXACT |
| 1200 | 123 | EXACT |
| 1200 | 323 | EXACT |
| 1200 | 532 | EXACT |
| 1200 | 777 | EXACT |
| 1200 | 999 | EXACT |

The two BF telemetry cells additionally reproduce the frozen Stage 2 BF-T0.5 episodes EXACTLY (validated in `D1_VALIDATION.json`).

## 6. Primary outcome: paired M − U deltas (5 seeds per capacity)

| C | seed | U | M | Δ (M−U) |
|---|---|---|---|---|
| 800 | 123 | 4278.415 | 4280.231 | +1.816 |
| 800 | 323 | 4127.756 | 4128.945 | +1.189 |
| 800 | 532 | 4208.089 | 4210.479 | +2.390 |
| 800 | 777 | 3488.413 | 3488.413 | +0.000 |
| 800 | 999 | 3961.742 | 3963.099 | +1.357 |
| 1200 | 123 | 14322.539 | 14324.712 | +2.173 |
| 1200 | 323 | 11063.848 | 11067.264 | +3.416 |
| 1200 | 532 | 14900.871 | 14939.148 | +38.277 |
| 1200 | 777 | 14578.304 | 14578.879 | +0.575 |
| 1200 | 999 | 14502.778 | 14524.913 | +22.135 |

- **C=800** vs frozen BF 4661.325: mean Δ **+1.351**, SD 0.887, SE 0.397, 95% t CI [+0.249, +2.452], paired t(4) = 3.404, raw two-sided p = 0.0272, pos/neg 4/0.
- **C=1200** vs frozen BF 15298.890: mean Δ **+13.315**, SD 16.472, SE 7.367, 95% t CI [-7.138, +33.768], paired t(4) = 1.808, raw two-sided p = 0.1450, pos/neg 5/0.

These are descriptive screening statistics (n=5, df=4, raw, unadjusted) on already-seen data — not a confirmatory test.

## 7. MASK_GATE (engineering decision thresholds fixed pre-results)

PASS iff (1) mean(M−U) ≥ +1% of frozen BF macro at ≥1 capacity, (2) mean(M−U) ≥ −1% of BF at the other, and (3) no seed loses >5% of BF relative to its own U baseline.

| quantity | C=800 | C=1200 |
|---|---|---|
| frozen BF macro | 4661.325 | 15298.890 |
| +1% BF threshold | 46.613 | 152.989 |
| −1% BF threshold | -46.613 | -152.989 |
| observed mean Δ | +1.351 | +13.315 |
| per-seed 5% BF loss limit | 233.066 | 764.945 |

1. ≥1 capacity at/above +1%: **False**
2. other capacity no worse than −1%: **True**
3. no seed loses >5%: **True** (worst U→M loss: 0.000 at C800_S777)

**MASK_GATE = FAIL**

## 8. Same-wallet conflict telemetry (raw head)

Per-step events aggregated over all 2,400,000 steps per cell. `raw conflict` = unmasked flush argmax equals s_BF (<k); `pre-feasible conflict` adds that s_BF was usable and could cover the current tx; `executed` adds that E0 actually flushed the wallet; `lost settlement` adds that the feasible current tx was dropped.

| C | seed | U raw | U pre-feas | U executed | U lost | M raw | M pre-feas | M executed | M lost |
|---|---|---|---|---|---|---|---|---|---|
| 800 | 123 | 307 | 307 | 307 | 307 | 306 | 306 | 0 | 0 |
| 800 | 323 | 298 | 298 | 298 | 298 | 297 | 297 | 0 | 0 |
| 800 | 532 | 597 | 597 | 597 | 597 | 585 | 585 | 0 | 0 |
| 800 | 777 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 800 | 999 | 258 | 258 | 258 | 258 | 259 | 259 | 0 | 0 |
| 1200 | 123 | 312 | 312 | 312 | 312 | 317 | 317 | 0 | 0 |
| 1200 | 323 | 379 | 379 | 379 | 379 | 371 | 371 | 0 | 0 |
| 1200 | 532 | 6345 | 6345 | 6345 | 6345 | 6307 | 6307 | 0 | 0 |
| 1200 | 777 | 107 | 107 | 107 | 107 | 110 | 110 | 0 | 0 |
| 1200 | 999 | 3171 | 3171 | 3171 | 3171 | 3173 | 3173 | 0 | 0 |

- C=800: mean pre-feasible conflict rate 0.000122 (~0.122/episode), executed rate 0.000122, fraction of pre-feasible conflicts that lose the current settlement in U: 1.000.
- C=1200: mean pre-feasible conflict rate 0.000859 (~0.859/episode), executed rate 0.000859, fraction of pre-feasible conflicts that lose the current settlement in U: 1.000.

Rates and per-regime breakdowns: `D1_CONFLICT_TELEMETRY.csv`.

## 9. Executed conflicts and lost current settlements

Across the 5 U cells per capacity, executed same-wallet flushes number 1460 (C800) and 10314 (C1200); of these, the subset voiding a feasible current settlement totals 1460 and 10314 events. A raw conflict does not lose the current tx when the flush targets an unusable wallet (E0 ignores it), when s_BF cannot cover the tx anyway, or when the tx is oversize. In M these events cannot occur by construction (s_BF's logit is −∞), which is reflected in the M columns.

## 10. Read-only one-step local forks — LOCAL EXPOSURE ONLY

At each U/M state where the RAW head conflicts with a pre-feasible s_BF, two snapshot clones take one step: A = `(s_BF, s_BF)` and B = `(s_BF, k)` (no-op flush). Recorded: immediate settled value, flushes, and Money difference. The clones never touch the live environment (signature-checked every fork in every run and test).

| C | seed | arm | fork events | mean A−B Money | B accept rate |
|---|---|---|---|---|---|
| 800 | 123 | U | 307 | -21.088 | 1.000 |
| 800 | 123 | M | 306 | -21.088 | 1.000 |
| 800 | 323 | U | 298 | -19.661 | 1.000 |
| 800 | 323 | M | 297 | -19.663 | 1.000 |
| 800 | 532 | U | 597 | -21.858 | 1.000 |
| 800 | 532 | M | 585 | -21.836 | 1.000 |
| 800 | 777 | U | 0 | 0.000 | 0.000 |
| 800 | 777 | M | 0 | 0.000 | 0.000 |
| 800 | 999 | U | 258 | -21.318 | 1.000 |
| 800 | 999 | M | 259 | -21.363 | 1.000 |
| 1200 | 123 | U | 312 | -26.628 | 1.000 |
| 1200 | 123 | M | 317 | -26.562 | 1.000 |
| 1200 | 323 | U | 379 | -26.383 | 1.000 |
| 1200 | 323 | M | 371 | -26.348 | 1.000 |
| 1200 | 532 | U | 6345 | -25.419 | 1.000 |
| 1200 | 532 | M | 6307 | -25.436 | 1.000 |
| 1200 | 777 | U | 107 | -21.579 | 1.000 |
| 1200 | 777 | M | 110 | -21.618 | 1.000 |
| 1200 | 999 | U | 3171 | -27.564 | 1.000 |
| 1200 | 999 | M | 3173 | -27.412 | 1.000 |

Interpretation is strictly bounded: A−B measures the immediate local cost of the conflicting action at that state. It is **not** the value M would recover — M's trajectory diverges thereafter, the flushed wallet may be more valuable after refill than the forgone tx, and B changes future wallet states. These numbers only size the local mechanism that M1 posits. Full table: `D1_LOCAL_FORK_SUMMARY.csv`.

## 11. Mask-induced action changes and flush accounting

Mask validation invariant (also asserted in validation): in M the argmax changes **iff** the raw argmax was exactly s_BF. Total changed choices across the five M cells: 1447 (C800) and 10278 (C1200). At every other step U and M submit identical actions. Flush request/execution/no-op/unusable counts per cell are in `D1_CONFLICT_TELEMETRY.csv`.

| C | mean U flushes/ep | mean M flushes/ep | Δ flush cost (macro, 10/flush) |
|---|---|---|---|
| 800 | 218.075 | 218.094 | ΔMoney +1.351 = Δsettled +1.533 − 10·+0.0182 (+0.182) |
| 1200 | 524.038 | 524.112 | ΔMoney +13.315 = Δsettled +14.053 − 10·+0.0738 (+0.738) |

## 12. Refill telemetry (F = 3)

Flushing sets `freeze_until = t + F − 1`; the wallet refills at the end of the step where time first exceeds that value (measured time-to-refill = 3 decision steps; asserted in test_09). Tracked per flush cycle: refill completion, whether the refilled endowment was ever used for a later settlement, and cycles still pending at the horizon.

| arm | C | refill completions | refilled value | cycles never reused | terminal pending |
|---|---|---|---|---|---|
| U | 800 | 2611947 | 87064900.00 | 14037 | 4956 |
| U | 1200 | 6276110 | 313805500.00 | 36353 | 12350 |
| M | 800 | 2612152 | 87071733.33 | 14034 | 4970 |
| M | 1200 | 6277018 | 313850900.00 | 36436 | 12327 |
| BF | 800 | 617476 | 20582533.33 | 7649 | 1246 |
| BF | 1200 | 1297266 | 64863300.00 | 6580 | 2599 |

Histograms (ttr, starvation runs): `D1_REFILL_TELEMETRY.csv`.

## 13. Occupancy / starvation telemetry

Per-step flags: zero usable wallets (hard starvation) and zero feasible wallets for a non-oversize tx (settlement starvation), with consecutive-run histograms. Aggregate step counts (sum over the 5 seeds, then per-cell averages):

Note: the frozen BF-T0.5 policy shows zero feasible-starvation steps — its own joint flush rule replenishes low wallets often enough that a non-oversize tx is never without a covering wallet. The SC-flush arms (U/M) do reach such states. This contrast is itself part of the distribution-shift picture (section 17).

| arm | C | zero-usable steps/cell (mean) | zero-feasible steps/cell (mean) |
|---|---|---|---|
| U | 800 | 0.0 | 94589.0 |
| U | 1200 | 0.0 | 104704.0 |
| M | 800 | 0.0 | 94574.2 |
| M | 1200 | 0.0 | 104692.6 |
| BF | 800 | 0.0 | 0.0 |
| BF | 1200 | 0.0 | 0.0 |

## 14. Regime × seed patterns

Descriptive only; regimes are not independent replicates. Per-regime Δ(M−U) sign counts across the 5 seeds:

| regime | C=800 mean Δ | pos | neg | C=1200 mean Δ | pos | neg |
|---|---|---|---|---|---|---|
| US | +0.000 | 0 | 0 | +3.943 | 4 | 1 |
| TLS | +0.328 | 3 | 1 | +5.192 | 4 | 1 |
| LNS | +5.611 | 4 | 0 | +38.448 | 5 | 0 |
| TLNS | +2.517 | 4 | 0 | +33.134 | 5 | 0 |
| TPLS | +0.000 | 0 | 0 | +0.000 | 0 | 0 |
| PLS | +0.000 | 0 | 0 | +0.000 | 0 | 0 |
| UB | +0.000 | 0 | 0 | +6.918 | 5 | 0 |
| TLB | +0.575 | 4 | 0 | +4.476 | 5 | 0 |
| LNB | +5.602 | 4 | 0 | +43.033 | 5 | 0 |
| TLNB | +1.573 | 4 | 0 | +24.637 | 5 | 0 |
| TPLB | +0.000 | 0 | 0 | +0.000 | 0 | 0 |
| PLB | +0.000 | 0 | 0 | +0.000 | 0 | 0 |

Full per-regime rows: `D1_REGIME_LEVEL.csv`.

## 15. Pre-designated failure-seed inspection

Before unblinding, C800/S777, C1200/S323 and C1200/S532 were named as the archived H1 failures to inspect explicitly.

| C | seed | U | M | Δ | Δ % BF | U pre-feas conflicts | U lost |
|---|---|---|---|---|---|---|---|
| 800 | 777 | 3488.413 | 3488.413 | +0.000 | +0.00% | 0 | 0 |
| 1200 | 323 | 11063.848 | 11067.264 | +3.416 | +0.02% | 379 | 379 |
| 1200 | 532 | 14900.871 | 14939.148 | +38.277 | +0.25% | 6345 | 6345 |

## 16. Capacity contrast

Mean Δ(M−U) is +1.351 at C=800 and +13.315 at C=1200. Conflict rates, refill pressure and starvation counts by capacity are in sections 8/12/13. Any capacity dependence of the mask effect bears directly on M1 (tighter wallets make same-wallet flushes more consequential) vs M2 (distribution shift that can run in either direction).

## 17. M1 vs M2 evidence synthesis

- M1 requires that removing the same-wallet action removes a real loss and improves money coherently. Pre-feasible conflicts are common enough to be measurable (section 8), local forks show the immediate A−B exposure is negative whenever the event fires (section 10), and M eliminates the events by construction. The observed money response is summarised by the paired deltas (section 6) and the gate (section 7).
- M2 predicts weak/incoherent money response despite the surgical change, because the frozen head was trained on a different settlement-induced state distribution; the mask also withholds flushes whose future refill value the head may have learned to exploit. Evidence classification below is based on pre-registered rate/effect rules in `statistics.json`.

- **DIRECT_CONFLICT_IMPORTANCE = LOW**
- **STATE_DISTRIBUTION_MISMATCH_EVIDENCE = WEAK**
- Effects by capacity: C=800 **weak_positive**, C=1200 **weak_positive** (weak_positive = mean Δ > 0 but below the 0.5% BF moderate-gain bar).

## 18. What this experiment cannot establish

- The same-wallet flush can be a *correct* learned sacrifice: give up the current tx to bring a fresh wallet online earlier. Forbidding it is a policy restriction, not proof of suboptimality; only a retrained successor (A+) on a genuinely fresh stream release can answer the design question.
- H1/Stage 3.5 instability is not shown to be caused by M1 even if the mask helps; post-hoc data, n=5 seeds, df=4, wide CIs, unadjusted p-values.
- Fork exposure is one-step and partial-equilibrium (section 10); telemetry traces cover episodes 0–1 per regime only by design.

## 19. Decision logic trace

- Gate failed only on condition 1 (mean Δ ≥ +1% BF at some capacity); conditions 2 and 3 held (section 7).
- Direct-conflict importance classified **LOW**; mismatch evidence **WEAK** under the documented heuristic rules.
- Gate PASS → the mechanism matters enough to justify an A+ trainable successor pilot, but the masked-H1 improvement branch itself stops and goes to PI. Gate FAIL with negligible conflict effects and no coherent mismatch signal → stop the branch, no A+ pilot. Anything in between → PI review, branch retained pending review.

## 20. Required final fields

| field | value |
|---|---|
| MASK_GATE | FAIL |
| DIRECT_CONFLICT_IMPORTANCE | LOW |
| STATE_DISTRIBUTION_MISMATCH_EVIDENCE | WEAK |
| RECOMMEND_A_PLUS_PILOT | NO |
| RECOMMEND_STOP_IMPROVEMENT_BRANCH | YES |
