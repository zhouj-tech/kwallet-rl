# H1 Multi-Seed Diagnostic Report (Stage 3.5)

**POST-HOC EXPLORATORY DIAGNOSTIC — not confirmatory.** NEW12-v1 was already used for Stage 2B/Stage 3 method selection, so this is not a fresh test set. H1 is not a new method: it re-combines the frozen BF-T0.5 settlement rule with each frozen SC-FAC checkpoint's conditional flush head (conditioned on the BF settle index), executed by the unmodified E0 environment. No training, no tuning, no frozen artifact modified. p-values are raw, unadjusted, n=5 (df=4), and are reported as descriptive screening statistics only.

## 1. Provenance

- Frozen SC-FAC checkpoints for seeds 123/323/532/777/999 resolved from the frozen Stage 2 job matrix; every checkpoint SHA256 was verified against the job matrix AND the frozen Stage 3 `seed_level_scores.csv` before evaluation.
- `adapter.check_config` verified per cell: C, k=24, F=3, T=1000, training seed, cpu, original reward (money_p=1, money_tau=10), `conditional_factorized_ac`, condition_mode=full, settle_embed_dim=32, conditional_hidden_size=256.
- Before each NEW cell, the loaded checkpoint was required to reproduce its own frozen SC-FAC NEW12 episodes EXACTLY on the provenance regimes ['US', 'TPLS', 'PLB'] (200 episodes each); all passed.
- Seed 123 H1 outputs are REUSED unchanged from the validated B0 pilot (sha256-pinned in `outputs/seed123_reference_sha.json`).
- Frozen raw tarball SHA 576c9baa00c44385…, lineage 5573ec642f0f…, NEW12-v1 manifest verified per run.

## 2. Final matrix (5 seeds × 2 capacities = 10 cells)

| C | seed | H1 macro12 Money | source |
|---|---|---|---|
| 800 | 123 | 4278.4154 | B0 reuse |
| 800 | 323 | 4127.7558 | new |
| 800 | 532 | 4208.0887 | new |
| 800 | 777 | 3488.4133 | new |
| 800 | 999 | 3961.7421 | new |
| 1200 | 123 | 14322.5387 | B0 reuse |
| 1200 | 323 | 11063.8483 | new |
| 1200 | 532 | 14900.8713 | new |
| 1200 | 777 | 14578.3038 | new |
| 1200 | 999 | 14502.7779 | new |

## 3. Seed-level H1 results

| C | mean | sample SD | SE | 95% t CI | min | max | SC mean | BF (fixed) |
|---|---|---|---|---|---|---|---|---|
| 800 | 4012.88 | 316.03 | 141.33 | [3620.48, 4405.28] | 3488.41 | 4278.42 | 3954.36 | 4661.32 |
| 1200 | 13873.67 | 1584.61 | 708.66 | [11906.11, 15841.23] | 11063.85 | 14900.87 | 14372.98 | 15298.89 |

## 4. Paired H1 vs original SC-FAC (same training seed)

### C=800

| seed | H1 | SC | Δ (H1−SC) |
|---|---|---|---|
| 123 | 4278.415 | 3897.975 | +380.440 |
| 323 | 4127.756 | 3909.299 | +218.457 |
| 532 | 4208.089 | 3991.302 | +216.787 |
| 777 | 3488.413 | 4054.156 | -565.742 |
| 999 | 3961.742 | 3919.089 | +42.653 |

- positive/negative: **4/1**; mean Δ **+58.519**, SD 368.855, SE 164.957, 95% CI [-399.475, +516.513], paired t(4) = 0.355, raw two-sided p = 0.7407

### C=1200

| seed | H1 | SC | Δ (H1−SC) |
|---|---|---|---|
| 123 | 14322.539 | 14327.010 | -4.471 |
| 323 | 11063.848 | 14381.021 | -3317.173 |
| 532 | 14900.871 | 14221.868 | +679.003 |
| 777 | 14578.304 | 14547.967 | +30.337 |
| 999 | 14502.778 | 14387.056 | +115.722 |

- positive/negative: **3/2**; mean Δ **-499.316**, SD 1599.408, SE 715.277, 95% CI [-2485.244, +1486.612], paired t(4) = -0.698, raw two-sided p = 0.5236

## 5. H1 vs fixed deterministic BF-T0.5

BF-T0.5 is a deterministic rule: ONE score per capacity, no seed uncertainty is fabricated.

### C=800 (BF = 4661.325)

| seed | H1 | Δ (H1−BF) |
|---|---|---|
| 123 | 4278.415 | -382.909 |
| 323 | 4127.756 | -533.569 |
| 532 | 4208.089 | -453.236 |
| 777 | 3488.413 | -1172.911 |
| 999 | 3961.742 | -699.582 |

- mean Δ **-648.441**, SD 316.027, SE 141.332, 95% CI [-1040.841, -256.042], one-sample t(4) = -4.588, raw p = 0.0101

### C=1200 (BF = 15298.890)

| seed | H1 | Δ (H1−BF) |
|---|---|---|
| 123 | 14322.539 | -976.352 |
| 323 | 11063.848 | -4235.042 |
| 532 | 14900.871 | -398.019 |
| 777 | 14578.304 | -720.587 |
| 999 | 14502.778 | -796.113 |

- mean Δ **-1425.222**, SD 1584.614, SE 708.661, 95% CI [-3392.781, +542.336], one-sample t(4) = -2.011, raw p = 0.1146

## 6. Gap recovery  (H1−SC)/(BF−SC) per seed

Interpret cautiously: the denominator varies with the seed's own SC score; recovery > 1 can reflect a small SC baseline as much as a strong H1.

| C | seed | recovery |
|---|---|---|
| 800 | 123 | 0.498 |
| 800 | 323 | 0.290 |
| 800 | 532 | 0.324 |
| 800 | 777 | -0.932 |
| 800 | 999 | 0.057 |
| 1200 | 123 | -0.005 |
| 1200 | 323 | -3.614 |
| 1200 | 532 | 0.630 |
| 1200 | 777 | 0.040 |
| 1200 | 999 | 0.127 |

| C | mean | median | min | max | #>0 | #>30% | #>=100% |
|---|---|---|---|---|---|---|---|
| 800 | 0.048 | 0.290 | -0.932 | 0.498 | 4/5 | 2/5 | 0/5 |
| 1200 | -0.564 | 0.040 | -3.614 | 0.630 | 3/5 | 1/5 | 0/5 |

## 7. Regime-level patterns (descriptive; regimes are NOT independent replicates)

### C=800: mean H1−SC per regime (across 5 seeds)

| regime | mean Δ | SD | min | max | seeds positive | class |
|---|---|---|---|---|---|---|
| US | -408.01 | 908.42 | -1964.32 | +279.17 | 2/5 | heterogeneous |
| TLS | +55.77 | 370.46 | -562.53 | +359.95 | 3/5 | heterogeneous |
| LNS | +469.79 | 448.38 | -280.68 | +818.56 | 4/5 | heterogeneous |
| TLNS | +156.45 | 562.90 | -809.07 | +618.74 | 4/5 | heterogeneous |
| TPLS | -14.12 | 212.09 | -241.70 | +280.28 | 2/5 | heterogeneous |
| PLS | +11.75 | 298.19 | -450.91 | +318.24 | 3/5 | heterogeneous |
| UB | -338.12 | 772.52 | -1616.11 | +284.32 | 2/5 | heterogeneous |
| TLB | +27.03 | 402.29 | -669.25 | +325.25 | 4/5 | heterogeneous |
| LNB | +528.75 | 398.13 | -141.19 | +829.13 | 4/5 | heterogeneous |
| TLNB | +194.49 | 520.95 | -699.38 | +623.89 | 4/5 | heterogeneous |
| TPLB | -9.04 | 210.55 | -219.74 | +286.27 | 2/5 | heterogeneous |
| PLB | +27.50 | 281.86 | -428.62 | +325.33 | 3/5 | heterogeneous |

- consistently improved (5/5 seeds positive): none
- consistently degraded (0/5 positive): none
- high-heterogeneity (mixed signs): US, TLS, LNS, TLNS, TPLS, PLS, UB, TLB, LNB, TLNB, TPLB, PLB

### C=1200: mean H1−SC per regime (across 5 seeds)

| regime | mean Δ | SD | min | max | seeds positive | class |
|---|---|---|---|---|---|---|
| US | -67.09 | 484.56 | -776.45 | +454.59 | 2/5 | heterogeneous |
| TLS | -343.04 | 808.90 | -1717.85 | +379.93 | 1/5 | heterogeneous |
| LNS | +1135.85 | 463.09 | +646.35 | +1685.34 | 5/5 | consistently_improved |
| TLNS | +577.62 | 301.70 | +139.24 | +898.78 | 5/5 | consistently_improved |
| TPLS | -1982.87 | 3971.24 | -8963.29 | +650.52 | 2/5 | heterogeneous |
| PLS | -2419.49 | 4874.28 | -11014.50 | +610.42 | 2/5 | heterogeneous |
| UB | -58.96 | 493.01 | -696.77 | +541.32 | 2/5 | heterogeneous |
| TLB | -147.45 | 588.84 | -1014.51 | +570.83 | 2/5 | heterogeneous |
| LNB | +1126.20 | 662.27 | +461.40 | +1915.77 | 5/5 | consistently_improved |
| TLNB | +597.96 | 310.11 | +272.49 | +978.91 | 5/5 | consistently_improved |
| TPLB | -1934.71 | 3849.02 | -8743.51 | +519.64 | 1/5 | heterogeneous |
| PLB | -2475.83 | 4937.64 | -11210.58 | +499.18 | 2/5 | heterogeneous |

- consistently improved (5/5 seeds positive): LNS, TLNS, LNB, TLNB
- consistently degraded (0/5 positive): none
- high-heterogeneity (mixed signs): US, TLS, TPLS, PLS, UB, TLB, TPLB, PLB

## 8. Settlement/flush decomposition (macro12 component means)

| C | mean Δsettled vs SC | mean Δflushcost vs SC | mean Δsettled vs BF | mean Δflushcost vs BF |
|---|---|---|---|---|
| 800 | +20.08 | -38.44 | -1045.70 | -397.26 |
| 1200 | -1300.74 | -801.42 | -1600.94 | -175.72 |

Positive Δflushcost means H1 flushes MORE than the comparator (worse); negative means it saves flush cost. Per-seed values in statistics.json.

## 9. Negative / null findings

- C=800: seeds without improvement over SC: 777; H1 remains below BF at every seed: True (min Δ vs BF -1172.91, max -382.91).
- C=1200: seeds without improvement over SC: 123, 323; H1 remains below BF at every seed: True (min Δ vs BF -4235.04, max -398.02).

## 10. Did seed123 generalize? (exploratory Q1/Q2)

- C=800: seed123 Δ vs SC was +380.44. Across all 5 seeds: mean +58.52, 4/5 positive, 95% CI [-399.48, +516.51].
- C=1200: seed123 Δ vs SC was −4.47 (≈neutral). Across all 5 seeds: mean -499.32, 3/5 positive, 95% CI [-2485.24, +1486.61].

## 11. Does H1 warrant a trainable BF-settlement + learned-flush successor?

Screening opinion (not a design commitment): the multi-seed evidence above indicates whether replacing the learned settlement with the BF rule transfers a consistent, capacity-dependent gain. Any B1 model design is explicitly out of scope here and must be proposed by Codex after these results are frozen and uploaded; a future B1 must be confirmed on a fresh held-out stream release.

## 12. Caveats

- n=5 seeds, df=4: tiny sample, wide CIs; raw p-values are descriptive only and are not multiplicity-adjusted.
- NEW12-v1 is the same stream family used for Stage 2B/3 selection — results can be optimistic relative to a truly fresh stream.
- H1's flush head can flush the BF settlement wallet itself; E0 processes flush-first, voiding that settlement step (interaction kept intact by the unmodified environment).
- Provenance parity per new seed covered 3 regimes exactly; the other 9 regimes are bound by checkpoint SHA, config checks, and stream hashes, not by replay.
- Gap-recovery denominators differ across seeds (each seed has its own SC baseline); interpret the ratio with care.

## 13. Exact artifact inventory

```
artifacts/aamas2027/stage3_5/h1_multiseed/
  code/run_h1_multi.py        # verify|run|validate|analyze|all
  code/test_h1_stats.py       # numerical stats unit tests
  outputs/C{C}_S{seed}/       # 8 new cells: episodes.csv + result.json
  outputs/seed123_reference_sha.json
  outputs/H1_VALIDATION.json
  H1_SEED_LEVEL_SCORES.csv
  H1_METHOD_SUMMARY.csv
  H1_PAIRED_VS_SC.csv
  H1_VS_BF.csv
  H1_REGIME_LEVEL.csv
  statistics.json
  H1_MULTI_SEED_REPORT.md
  README.md
```

## 14. Reproduction commands

```bash
PY=/Users/zhouzhou/benchmarks/A-EVAL-JA-C800-S123/.venv/bin/python
cd /Users/zhouzhou/Desktop/kwallet-aamas-b0
M=artifacts/aamas2027/stage3_5/h1_multiseed
$PY $M/code/run_h1_multi.py verify
$PY $M/code/test_h1_stats.py
$PY $M/code/run_h1_multi.py run --all
$PY $M/code/run_h1_multi.py validate
$PY $M/code/run_h1_multi.py analyze
```

---

## Screening verdict: **MIXED SIGNAL**

_STRONG if mean paired H1-SC delta > 0 with >=4/5 seeds positive at BOTH capacities; MIXED if that holds at one capacity (>=3/5 positive counts as partial); else NO SIGNAL. Screening heuristic only._
