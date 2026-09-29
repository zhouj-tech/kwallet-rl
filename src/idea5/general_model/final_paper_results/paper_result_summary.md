# Paper Result Summary

## 1. Main Experimental Claim

Conditional action factorization provides a structured policy for settle-flush control: the model first chooses whether to settle, then conditions the flush decision on that selected settle action. In the one-pool General Collateral Model, adding threshold imitation as a warm start gives the strongest final learned policy among the compared methods, while the two-pool extension shows that the learned policy captures structure beyond a one-pool-style global threshold rule.

## 2. One-Pool Main Result

The one-pool result is the strongest evidence in this result package. For C=900 and C=1100, all compared methods use 10 seeds. For C=1000, Conditional AC + Imit100 uses 10 seeds, while the older Tuned Threshold and Stable Conditional AC baselines use 9 seeds.

| C | Variant | n_seeds | Money | Mean ValAcc | Worst ValAcc | Drops | Flushes | Source / Evidence Level |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 900 | Tuned Threshold | 10 | 42356.92 ± 0.00 | 0.9924 ± 0.0000 | 0.9776 ± 0.0000 | 3.40 ± 0.00 | 73.08 ± 0.00 | main multi-seed result |
| 900 | Stable Conditional AC | 10 | 42675.13 ± 23.79 | 0.9895 ± 0.0012 | 0.9742 ± 0.0017 | 5.14 ± 0.73 | 68.46 ± 0.58 | main multi-seed result |
| 900 | Conditional AC + Imit100 | 10 | 42714.61 ± 9.49 | 0.9885 ± 0.0013 | 0.9730 ± 0.0018 | 5.66 ± 0.75 | 67.56 ± 0.70 | main multi-seed result |
| 1000 | Tuned Threshold | 9 | 43366.67 ± 0.00 | 0.9891 ± 0.0000 | 0.9744 ± 0.0000 | 5.29 ± 0.00 | 61.38 ± 0.00 | main multi-seed result; older baseline has n=9 |
| 1000 | Stable Conditional AC | 9 | 43564.42 ± 31.86 | 0.9904 ± 0.0011 | 0.9773 ± 0.0015 | 4.74 ± 0.66 | 60.05 ± 0.59 | main multi-seed result; older baseline has n=9 |
| 1000 | Conditional AC + Imit100 | 10 | 43599.64 ± 4.57 | 0.9907 ± 0.0011 | 0.9775 ± 0.0016 | 4.53 ± 0.65 | 59.83 ± 0.58 | main multi-seed result |
| 1100 | Tuned Threshold | 10 | 44119.01 ± 0.00 | 0.9934 ± 0.0000 | 0.9811 ± 0.0000 | 2.94 ± 0.00 | 56.00 ± 0.00 | main multi-seed result |
| 1100 | Stable Conditional AC | 10 | 44244.48 ± 22.77 | 0.9916 ± 0.0016 | 0.9800 ± 0.0023 | 4.25 ± 0.91 | 53.82 ± 0.89 | main multi-seed result |
| 1100 | Conditional AC + Imit100 | 10 | 44297.44 ± 8.78 | 0.9929 ± 0.0008 | 0.9813 ± 0.0013 | 3.39 ± 0.45 | 53.94 ± 0.46 | main multi-seed result |

- Conditional AC + Imit100 achieves the highest money for C=900, C=1000, and C=1100.
- At C=900, it reaches 42714.61 ± 9.49, improving over Stable Conditional AC by +39.48 and Tuned Threshold by +357.69.
- At C=1000, it reaches 43599.64 ± 4.57, improving over Stable Conditional AC by +35.22 and Tuned Threshold by +232.98.
- At C=1100, it reaches 44297.44 ± 8.78, improving over Stable Conditional AC by +52.96 and Tuned Threshold by +178.43.
- Tuned Threshold can have higher value acceptance or fewer drops in some settings, but it flushes more; the learned policy wins on money by balancing acceptance and flush cost.

## 3. Two-Pool Extension Result

The two-pool result is an extension experiment rather than the main multi-seed result. It tests whether the learned conditional policy captures pool-specific structure when the collateral system is split into two typed pools.

| C | Global Threshold Money | Conditional AC + Imit100 Money | Pool-Specific Threshold Money | AC - Global | AC - PoolSpecific | Source / Evidence Level |
|---:|---:|---:|---:|---:|---:|---|
| 800 | 41141.79 | 41933.91 | 42023.50 | +792.12 | -89.59 | seed=123 extension result |
| 1000 | 43143.38 | 43884.77 | 43913.28 | +741.39 | -28.50 | seed=123 extension result |
| 1200 | 44452.39 | 44924.42 | 45070.21 | +472.04 | -145.78 | seed=123 extension result |

- Conditional AC + Imit100 consistently outperforms the one-pool-style Global Threshold.
- The learned policy remains below the stronger Pool-Specific Threshold, which explicitly encodes the correct pool-level pressure heuristic.
- This supports the interpretation that conditional factorization can learn useful pool-specific control structure, but does not make the learned model superior to the strongest hand-designed two-pool rule.

## 4. Ablation Summary

The ablations are best used as model-selection evidence. They explain why the final reported model uses base-state Conditional AC + Imit100 in the one-pool setting and avoids extra state features or regularization that did not improve money.

| Setting | Compared Variants | Key Numbers | Takeaway | Source / Evidence Level |
|---|---|---|---|---|
| One-pool state/imitation variant | Base Conditional AC + Imit100 vs Pressure + Imit50 | Base Imit100 money 43599.64 ± 4.57; Pressure + Imit50 money 43529.94 ± 110.29 | Pressure features plus shorter imitation were not selected for the final one-pool model; base-state Imit100 is stronger in the final main comparison. | model-selection evidence; base is 10-seed main result, pressure variant is 3-seed comparison |
| Two-pool delayed-release state features | Pressure + MaskSafe + Imit100 vs PressureRelease + MaskSafe + Imit100 | Pressure + MaskSafe + Imit100 money 43884.77; PressureRelease money 43713.06 | Adding delayed-release state features increased drops and did not help the policy use future release information effectively. | model-selection evidence; C=1000 seed=123 |
| Two-pool imitation regularization during PPO | No-reg Imit100 vs Reg050 / Reg030 / Reg010 | No-reg 43884.77; Reg050 43867.27; Reg030 43793.16; Reg010 43731.44 | Fixed imitation regularization reduced performance; the final model keeps imitation as a warm start rather than as an ongoing regularizer. | model-selection evidence; C=1000 seed=123 |
| Two-pool PPO fine-tuning schedule | Default Imit100 vs shorter/conservative PPO variants | Default Imit100 money 43884.77; shorter/conservative PPO 43457.25 | Shorter or more conservative PPO fine-tuning did not improve over the default Imit100 setting, so the final model keeps the default fine-tuning setup. | model-selection evidence; C=1000 seed=123 where available |

## 5. What Should Go Into the Paper

| Paper Section | Result/Table | Purpose | Include? |
|---|---|---|---|
| Main Results | One-pool main table | Shows Conditional AC + Imit100 is strongest in money across C=900, 1000, and 1100 | Yes |
| Extension | Two-pool global vs pool-specific table | Shows learned policy beats global pressure but not the strongest pool-specific rule | Yes |
| Ablation | pressure_release / imitation regularization / shorter PPO | Explains final model selection | Short appendix or brief paragraph |
| Appendix | Full seed-level tables | Supports reproducibility and seed-level audit | Appendix |

## 6. Final Claims to Use

1. In the one-pool General Collateral Model, threshold imitation consistently improves Conditional AC, and Conditional AC + Imit100 achieves the highest money among the compared methods across C=900, C=1000, and C=1100.
2. In the two-pool extension, Conditional AC + Imit100 outperforms the one-pool-style Global Threshold, suggesting that the learned policy captures pool-specific structure beyond aggregate pressure.
3. The strongest two-pool hand-designed rule remains Pool-Specific Threshold; this should be treated as a strong rule-based reference rather than hidden or weakened.
4. Additional state features, imitation regularization, and shorter or more conservative PPO fine-tuning did not improve the selected final model in the tested settings.
