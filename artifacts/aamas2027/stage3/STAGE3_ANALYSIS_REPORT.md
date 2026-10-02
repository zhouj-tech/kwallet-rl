# Stage 3 — Statistics & Mechanism Analysis

This report is generated from the user-designated frozen Stage 2B raw tarball only. Historical results are not pooled with NEW12 or used to fill cells. The inferential protocol is unchanged. No new training, evaluation or stream generation occurs.

## 1. Raw-data acceptance

**Scientific completeness PASS: 42/42 jobs, 100,800 episode rows. Ledger completeness 38/42, reconciled with four approved pre-existing jobs.** See STAGE3_RAW_ACCEPTANCE.md and raw_acceptance.json. The root pilot duplicate is excluded. Raw SHA256: `576c9baa00c4438543ab9f9c2063f65bb26b1a6f287974a9e8ab91b18189ace3`. Scientific lineage: `5573ec642f0f28c218f3e6058478f62ab6db6b2b`.

All 155 internal checksum entries pass. Maximum Money identity error is zero. The embedded Mac approval and five representative episode-hash attestations support reuse of the pre-existing outputs. This does not hide provenance limitations: benchmark-only text remains in all 42 results; runtime_lock_sha256 is null; the approval header version strings differ from the consistent actual job runtime records. Pool/checkpoint/training-receipt bytes are outside this raw-results freeze, so their end-to-end regeneration is not certified by this analysis.

## 2. Frozen protocol

Primary outcome is Money = settled value − 10×executed flushes. Average 200 episodes within each regime, then equally average all twelve regimes, yielding one macro12 score per training seed. Each learned method/capacity has seeds 123,323,532,777,999; n=5. Sample SD uses ddof=1, SE=SD/√5, marginal two-sided 95% t CI uses df=4 and critical value 2.7764451051977987. Episodes and regimes are not independent learned-policy replicates.

A: six paired contrasts across both capacities (SC−JA, IF−JA, SC−IF). B: two paired full−zero contrasts. C: two one-sample t-tests of five SC seed deltas against one fixed BF score per capacity. Holm correction is applied within all six/two/two tests respectively. All ten contrasts are reported together. No additional test, bootstrap, equivalence test, interaction test, or post-hoc variance test is introduced. BF has no training-seed SD/SE/CI. Zero-variance tests are undefined, not automatically significant; their family slots are retained. All intervals below are marginal, not familywise intervals.

## 3. Descriptive results

| method | C | n_training_seeds | mean | SD | SE | ci95_low | ci95_high | df |
|---|---|---|---|---|---|---|---|---|
| JA-PPO | 800 | 5 | 3368.99 | 449.475 | 201.012 | 2810.89 | 3927.09 | 4 |
| IFAC | 800 | 5 | 3682.12 | 631.64 | 282.478 | 2897.83 | 4466.4 | 4 |
| SC-FAC | 800 | 5 | 3954.36 | 66.6665 | 29.8142 | 3871.59 | 4037.14 | 4 |
| SC-FAC-zero | 800 | 5 | 4061.45 | 299.299 | 133.851 | 3689.82 | 4433.07 | 4 |
| BF-T0.5 | 800 | 0 | 4661.32 | NA | NA | NA | NA | NA |
| JA-PPO | 1200 | 5 | 14002.2 | 324.72 | 145.219 | 13599 | 14405.3 | 4 |
| IFAC | 1200 | 5 | 14097.9 | 404.427 | 180.865 | 13595.7 | 14600 | 4 |
| SC-FAC | 1200 | 5 | 14373 | 118.143 | 52.8352 | 14226.3 | 14519.7 | 4 |
| SC-FAC-zero | 1200 | 5 | 14479.3 | 98.8024 | 44.1858 | 14356.6 | 14602 | 4 |
| BF-T0.5 | 1200 | 0 | 15298.9 | NA | NA | NA | NA | NA |

## 4. Family A — structural policy comparisons

| C | left_method | right_method | mean_difference | SD | SE | ci95_low | ci95_high | t_statistic | raw_p | holm_p | holm_reject_0_05 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 800 | SC-FAC | JA-PPO | 585.376 | 500.001 | 223.607 | -35.4574 | 1206.21 | 2.61788 | 0.0589306 | 0.353584 | False |
| 800 | IFAC | JA-PPO | 313.13 | 875.352 | 391.469 | -773.763 | 1400.02 | 0.799884 | 0.468587 | 1 | False |
| 800 | SC-FAC | IFAC | 272.246 | 632.138 | 282.701 | -512.657 | 1057.15 | 0.963018 | 0.390073 | 1 | False |
| 1200 | SC-FAC | JA-PPO | 370.828 | 353.227 | 157.968 | -67.7611 | 809.416 | 2.34749 | 0.078731 | 0.393655 | False |
| 1200 | IFAC | JA-PPO | 95.7247 | 612.287 | 273.823 | -664.53 | 855.98 | 0.349586 | 0.744279 | 1 | False |
| 1200 | SC-FAC | IFAC | 275.103 | 450.709 | 201.563 | -284.527 | 834.733 | 1.36485 | 0.244033 | 0.976131 | False |

SC-FAC − JA-PPO at C=800: Δ=+585.38, 95% CI [-35.46, +1206.21], raw p=0.0589306, Holm p=0.353584; does not reject zero under the frozen family correction.

IFAC − JA-PPO at C=800: Δ=+313.13, 95% CI [-773.76, +1400.02], raw p=0.468587, Holm p=1; does not reject zero under the frozen family correction.

SC-FAC − IFAC at C=800: Δ=+272.25, 95% CI [-512.66, +1057.15], raw p=0.390073, Holm p=1; does not reject zero under the frozen family correction.

SC-FAC − JA-PPO at C=1200: Δ=+370.83, 95% CI [-67.76, +809.42], raw p=0.078731, Holm p=0.393655; does not reject zero under the frozen family correction.

IFAC − JA-PPO at C=1200: Δ=+95.72, 95% CI [-664.53, +855.98], raw p=0.744279, Holm p=1; does not reject zero under the frozen family correction.

SC-FAC − IFAC at C=1200: Δ=+275.10, 95% CI [-284.53, +834.73], raw p=0.244033, Holm p=0.976131; does not reject zero under the frozen family correction.

All six mean differences favor the left method, but none rejects zero after Holm correction (and all six marginal intervals cross zero). Factorized policies have higher descriptive mean Money than JA-PPO at both capacities; this five-seed confirmation does not establish their inferential superiority. No claim of equivalence follows.

## 5. Family B — conditioning ablation

| C | left_method | right_method | mean_difference | SD | SE | ci95_low | ci95_high | t_statistic | raw_p | holm_p | holm_reject_0_05 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 800 | SC-FAC | SC-FAC-zero | -107.081 | 273.266 | 122.208 | -446.386 | 232.223 | -0.876222 | 0.430378 | 0.496103 | False |
| 1200 | SC-FAC | SC-FAC-zero | -106.327 | 175.981 | 78.7011 | -324.836 | 112.182 | -1.35102 | 0.248051 | 0.496103 | False |

SC-FAC − SC-FAC-zero at C=800: Δ=-107.08, 95% CI [-446.39, +232.22], raw p=0.430378, Holm p=0.496103; does not reject zero under the frozen family correction.

SC-FAC − SC-FAC-zero at C=1200: Δ=-106.33, 95% CI [-324.84, +112.18], raw p=0.248051, Holm p=0.496103; does not reject zero under the frozen family correction.

Zero conditioning has the higher mean at both capacities. There is no detected incremental benefit of explicit settlement conditioning under this protocol; neither its harm nor equivalence is established. This tests the settlement input within the SC head design, not whether factorization itself is useful. Recovered versus newly trained zero-control cohorts remain labeled and visible.

## 6. Family C — strong deterministic comparator

| C | left_method | right_method | mean_difference | SD | SE | ci95_low | ci95_high | t_statistic | raw_p | holm_p | holm_reject_0_05 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 800 | SC-FAC | BF-T0.5 | -706.96 | 66.6665 | 29.8142 | -789.738 | -624.183 | -23.7122 | 1.87555e-05 | 3.75111e-05 | True |
| 1200 | SC-FAC | BF-T0.5 | -925.906 | 118.143 | 52.8352 | -1072.6 | -779.212 | -17.5244 | 6.226e-05 | 6.226e-05 | True |

SC-FAC − BF-T0.5 at C=800: Δ=-706.96, 95% CI [-789.74, -624.18], raw p=1.87555e-05, Holm p=3.75111e-05; rejects zero under the frozen family correction.

SC-FAC − BF-T0.5 at C=1200: Δ=-925.91, 95% CI [-1072.60, -779.21], raw p=6.226e-05, Holm p=6.226e-05; rejects zero under the frozen family correction.

BF-T0.5 exceeds SC-FAC at both capacities with family-wise evidence under the declared tests. It also exceeds every learned-method mean descriptively. No extra BF-versus-JA/IF/zero significance tests were authorized or added. The rule has one fixed score per capacity; its value is reused as a constant in five SC deltas, not counted as five independently trained rules.

## 7. Regime-level analysis

Definitions are taken verbatim from frozen new12_manifest.json, not inferred from abbreviations.

| regime | dist_family | bursty | target_mean | max_tx |
|---|---|---|---|---|
| US | uniform | False | 50 | 100 |
| TLS | trunc_light | False | 50 | 120 |
| LNS | lognormal | False | 50 | 180 |
| TLNS | trunc_lognormal | False | 50 | 120 |
| TPLS | trunc_powerlaw | False | 50 | 120 |
| PLS | powerlaw | False | 50 | 250 |
| UB | uniform | True | 50 | 100 |
| TLB | trunc_light | True | 50 | 120 |
| LNB | lognormal | True | 50 | 180 |
| TLNB | trunc_lognormal | True | 50 | 120 |
| TPLB | trunc_powerlaw | True | 50 | 120 |
| PLB | powerlaw | True | 50 | 250 |

S denotes smooth and B bursty. U is uniform; TL is truncated light-tail (truncated normal), LN lognormal, TLN truncated lognormal, PL power law, TPL truncated power law. **TL does not mean time-local.** No distinct plain/time-local/patterned axis is defined in these materials. Burst/non-burst family parameters are not identical in every pair, so group differences do not isolate a pure causal burst effect.

All regimes target mean transaction value 50. They are not a predefined small-versus-large transaction experiment. regime_input_profiles.csv reports observed mean request value and mean oversized-request counts; no transaction-level size-bin effect can be recovered from episode summaries. The raw archive includes manifests rather than transaction arrays or step trajectories. Oversize rates are a descriptive feasibility proxy, not a causal size treatment.

- C=800, SC-FAC − JA-PPO: 12/12 positive descriptive regime means; range +225.62 (TLB) to +958.94 (PLS).
- C=800, IFAC − JA-PPO: 6/12 positive descriptive regime means; range -109.64 (US) to +1228.80 (PLS).
- C=800, SC-FAC − SC-FAC-zero: 2/12 positive descriptive regime means; range -303.79 (PLS) to +85.90 (TLB).
- C=800, SC-FAC − BF-T0.5: 0/12 positive descriptive regime means; range -1477.88 (LNS) to -163.25 (UB).
- C=1200, SC-FAC − JA-PPO: 12/12 positive descriptive regime means; range +268.50 (TLB) to +571.52 (TPLS).
- C=1200, IFAC − JA-PPO: 7/12 positive descriptive regime means; range -746.32 (PLS) to +666.94 (UB).
- C=1200, SC-FAC − SC-FAC-zero: 2/12 positive descriptive regime means; range -240.28 (UB) to +64.53 (TLNS).
- C=1200, SC-FAC − BF-T0.5: 0/12 positive descriptive regime means; range -1741.65 (LNB) to -239.07 (TPLB).

Smooth/bursty grouped differences (equal weighting over six regimes in each group; descriptive only):

| contrast_id | stream_group | mean_money_delta | mean_settled_delta | mean_flushes_delta |
|---|---|---|---|---|
| A_C800_SC-FAC_minus_JA-PPO | smooth | 619.572 | 996.911 | 37.734 |
| A_C800_SC-FAC_minus_JA-PPO | bursty | 551.18 | 900.655 | 34.9475 |
| A_C800_IFAC_minus_JA-PPO | smooth | 329.868 | 426.405 | 9.65367 |
| A_C800_IFAC_minus_JA-PPO | bursty | 296.392 | 385.04 | 8.86483 |
| A_C800_SC-FAC_minus_IFAC | smooth | 289.703 | 570.506 | 28.0803 |
| A_C800_SC-FAC_minus_IFAC | bursty | 254.788 | 515.615 | 26.0827 |
| A_C1200_SC-FAC_minus_JA-PPO | smooth | 372.276 | 179.289 | -19.2987 |
| A_C1200_SC-FAC_minus_JA-PPO | bursty | 369.38 | 154.26 | -21.512 |
| A_C1200_IFAC_minus_JA-PPO | smooth | -13.677 | -839.86 | -82.6183 |
| A_C1200_IFAC_minus_JA-PPO | bursty | 205.126 | -613.394 | -81.852 |
| A_C1200_SC-FAC_minus_IFAC | smooth | 385.953 | 1019.15 | 63.3197 |
| A_C1200_SC-FAC_minus_IFAC | bursty | 164.253 | 767.653 | 60.34 |
| B_C800_SC-FAC_minus_SC-FAC-zero | smooth | -120.183 | -228.509 | -10.8327 |
| B_C800_SC-FAC_minus_SC-FAC-zero | bursty | -93.9802 | -190.215 | -9.6235 |
| B_C1200_SC-FAC_minus_SC-FAC-zero | smooth | -90.2735 | 112.525 | 20.2798 |
| B_C1200_SC-FAC_minus_SC-FAC-zero | bursty | -122.38 | 111.685 | 23.4065 |
| C_C800_SC-FAC_minus_BF-T0.5 | smooth | -731.616 | -1106.87 | -37.5252 |
| C_C800_SC-FAC_minus_BF-T0.5 | bursty | -682.305 | -1024.69 | -34.2387 |
| C_C1200_SC-FAC_minus_BF-T0.5 | smooth | -919.279 | -345.187 | 57.4092 |
| C_C1200_SC-FAC_minus_BF-T0.5 | bursty | -932.533 | -255.227 | 67.7307 |

Regime plots and positive-count statements are descriptive and do not turn twelve regimes into twelve independent trained agents. Per-regime across-seed SDs and all paired differences are preserved in the tables; no regime-level significance claims are made.

## 8. Mechanism interpretation

### A. Observed accounting patterns

| contrast_id | money_delta | settled_delta | flushes_delta | flush_cost_delta | regimes_positive | regimes_negative |
|---|---|---|---|---|---|---|
| A_C800_SC-FAC_minus_JA-PPO | 585.376 | 948.783 | 36.3407 | 363.408 | 12 | 0 |
| A_C800_IFAC_minus_JA-PPO | 313.13 | 405.722 | 9.25925 | 92.5925 | 6 | 6 |
| A_C800_SC-FAC_minus_IFAC | 272.246 | 543.061 | 27.0815 | 270.815 | 8 | 4 |
| A_C1200_SC-FAC_minus_JA-PPO | 370.828 | 166.774 | -20.4053 | -204.053 | 12 | 0 |
| A_C1200_IFAC_minus_JA-PPO | 95.7247 | -726.627 | -82.2352 | -822.352 | 7 | 5 |
| A_C1200_SC-FAC_minus_IFAC | 275.103 | 893.401 | 61.8298 | 618.298 | 8 | 4 |
| B_C800_SC-FAC_minus_SC-FAC-zero | -107.081 | -209.362 | -10.2281 | -102.281 | 2 | 10 |
| B_C1200_SC-FAC_minus_SC-FAC-zero | -106.327 | 112.105 | 21.8432 | 218.432 | 2 | 10 |
| C_C800_SC-FAC_minus_BF-T0.5 | -706.96 | -1065.78 | -35.8819 | -358.819 | 0 | 12 |
| C_C1200_SC-FAC_minus_BF-T0.5 | -925.906 | -300.207 | 62.5699 | 625.699 | 0 | 12 |

Every row satisfies ΔMoney = Δsettled − 10Δflushes. This is an accounting identity, not a causal explanation.

- Conditioning, C=800: full-minus-zero differences per macro-averaged episode are -209.36 in settled value and -10.23 in flushes, yielding -107.08 Money.
- Conditioning, C=1200: full-minus-zero differences per macro-averaged episode are +112.10 in settled value and +21.84 in flushes, yielding -106.33 Money.
- BF mean insufficient-balance drops range from 0 to 0 across capacities/regimes. Thus the recorded BF trajectories accept all individually feasible requests in these test episodes. This is not a proof of optimal Money: accepting value and minimizing flush fees are distinct objectives.

### B. Plausible mechanism hypotheses

At C800, the full-versus-zero deficit is compatible with lost accepted value outweighing saved flush cost; at C1200 it is compatible with extra flush cost outweighing additional accepted value. SC-versus-JA at C800 is acceptance-led, whereas C1200 combines additional settlement with fewer flushes. These decompositions motivate hypotheses about replenishment timing and resource usage; the aggregates do not identify timing, learned representations, action conflicts or the causal reason a policy chooses a flush. The study lacks the step-level traces and intervention controls required to establish those mechanisms.

The observed pattern is consistent with compact policy structure being useful without a demonstrated incremental settlement-input benefit. Architecture width/optimization differences prevent treating SC-versus-IFAC as a pure conditioning intervention. Cohort differences in new zero trainings also remain a mechanism-study limitation.

## 9. Negative/null findings

- No Family A contrast passes Holm correction; avoid “factorization significantly improves control” for this campaign.
- Both full-minus-zero point estimates are negative, but uncertainty spans zero. There is no evidence here for an incremental conditioning benefit and no equivalence/harm conclusion.
- BF wins the prescribed comparison with SC at both capacities, and has the highest descriptive method mean. This must remain visible in the main paper.
- SC has lower descriptive seed SD than JA and IF at both capacities, but no variance test was pre-specified. It does not have the smallest learned-policy SD at C1200: zero conditioning is less variable there.
- No claim about switching, unseen regime families, zero-shot k, deployment, or faster RL computation is tested.

## 10. Implications for the AAMAS claim

The fresh evidence supports a measured structural comparison: independent and conditioned policies improve mean Money descriptively over the flat head, and SC retains broad regime-level gains versus JA, but five-seed Family A inference does not establish superiority. The conditioning-specific claim must be narrowed because zero conditioning has higher mean Money at both capacities without a resolved full-minus-zero effect. The practical controller comparison favors BF.

The question remains “How much structure does an RL policy need for coupled online decisions?” The contribution should be structured action factorization and its measured limits, not a claim that SC-FAC is universally best or that explicit conditioning is necessary. SC-FAC remains a studied proposed architecture, not a guaranteed winner. Linear output dimensionality means 2(k+1) selected-path logits versus (k+1)^2: 50 versus 625 at k=24. It does not make total RL computation linear or prove inference speedup.

## 11. Recommended paper result wording

On NEW12, both independent and settlement-conditioned policies had higher macro12 Money means than the flat joint-action policy at both capacities. None of the six paired structural contrasts rejected zero after the pre-specified Holm correction. SC-FAC-zero had higher mean Money than full SC-FAC at both capacities, and the matched tests did not establish an incremental benefit of the settlement signal. BF-T0.5 outperformed SC-FAC at both capacities under the pre-specified one-sample tests of seed-level deltas. The findings distinguish favorable descriptive performance of compact learned policies from evidence for conditioning and from practical superiority over domain-informed deterministic control.

Inferential evidence accompanying that paragraph:

SC-FAC − JA-PPO at C=800: Δ=+585.38, 95% CI [-35.46, +1206.21], raw p=0.0589306, Holm p=0.353584; does not reject zero under the frozen family correction.

IFAC − JA-PPO at C=800: Δ=+313.13, 95% CI [-773.76, +1400.02], raw p=0.468587, Holm p=1; does not reject zero under the frozen family correction.

SC-FAC − IFAC at C=800: Δ=+272.25, 95% CI [-512.66, +1057.15], raw p=0.390073, Holm p=1; does not reject zero under the frozen family correction.

SC-FAC − JA-PPO at C=1200: Δ=+370.83, 95% CI [-67.76, +809.42], raw p=0.078731, Holm p=0.393655; does not reject zero under the frozen family correction.

IFAC − JA-PPO at C=1200: Δ=+95.72, 95% CI [-664.53, +855.98], raw p=0.744279, Holm p=1; does not reject zero under the frozen family correction.

SC-FAC − IFAC at C=1200: Δ=+275.10, 95% CI [-284.53, +834.73], raw p=0.244033, Holm p=0.976131; does not reject zero under the frozen family correction.

SC-FAC − SC-FAC-zero at C=800: Δ=-107.08, 95% CI [-446.39, +232.22], raw p=0.430378, Holm p=0.496103; does not reject zero under the frozen family correction.

SC-FAC − SC-FAC-zero at C=1200: Δ=-106.33, 95% CI [-324.84, +112.18], raw p=0.248051, Holm p=0.496103; does not reject zero under the frozen family correction.

SC-FAC − BF-T0.5 at C=800: Δ=-706.96, 95% CI [-789.74, -624.18], raw p=1.87555e-05, Holm p=3.75111e-05; rejects zero under the frozen family correction.

SC-FAC − BF-T0.5 at C=1200: Δ=-925.91, 95% CI [-1072.60, -779.21], raw p=6.226e-05, Holm p=6.226e-05; rejects zero under the frozen family correction.

## 12. Recommended abstract-result sentence

“On fresh streams, factorized policies achieved higher mean net accepted value than the flat joint-action policy, although the pre-specified paired tests did not establish structural-policy superiority; explicit settlement conditioning showed no detected incremental benefit, and a domain-informed deterministic rule outperformed SC-FAC at both capacities.”

This is proposed wording only; no manuscript or abstract file was edited. It summarizes the effect estimates, marginal confidence intervals and family-adjusted p-values reported in Sections 4–6 and 11; it does not promote raw-p findings.

## 13. Limitations

Five training seeds provide limited precision. Normal-theory t inference is retained as frozen, without switching methods after outcomes. Seed matching does not imply common RNG trajectories. CIs are conditional on fixed NEW12 pools and do not quantify all generator/world uncertainty; the twelve regimes are related designed conditions, not inferential replicates. Equal episode counts make a pooled episode mean numerically identical here, but the pipeline explicitly preserves regime weighting and never uses episode-level n for learned-policy inference. Zero-controls mix three recovered and two newly trained policies per capacity; no post-hoc cohort exclusion or extra test is performed. Parent architecture comparisons also differ in head dimensions. BF is deterministic on fixed pools, but this is not a claim of zero uncertainty across hypothetical streams. Regime aggregates cannot identify action timing or transaction-level mechanism effects. Null tests do not establish equivalence. See the acceptance report for retained approval/version/benchmark-label inconsistencies and absent checkpoint/pool bytes; this analysis verifies the supplied frozen outputs, not a new end-to-end simulation replay. No PR #3 data or code enters the analysis.

## 14. Exact artifact inventory

All derived tables are listed in statistics.json under tables. artifact_inventory.csv records every deliverable's relative path, size and SHA256; SHA256SUMS.txt covers those deliverables plus the inventory. work/raw contains byte-verified extracted copies and is excluded from Git via the local .gitignore; its exact 157-file inventory is raw_file_inventory.csv. Four figures are exported as PDF/SVG/PNG; captions document uncertainty and scale. code/run_stage3_analysis.py is the deterministic entry point; code/test_stage3_analysis.py supplies independent numerical and guard tests; code/requirements.txt freezes the analysis libraries. The analysis runtime is separate from the frozen evaluator runtime.

