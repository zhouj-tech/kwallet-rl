# Result Claims for Paper

## Main Claim

Conditional action factorization is a useful structure for settle-flush control. In the one-pool General Collateral Model, threshold imitation gives Conditional AC a strong prior and leads to the best money among the compared methods across C=900, C=1000, and C=1100.

## One-Pool Claim

In the one-pool setting, Conditional AC + Imit100 is the strongest method in terms of money. It improves over Stable Conditional AC and Tuned Threshold at all three tested capacities, while preserving the explicit settle-then-flush conditional policy structure.

## Two-Pool Extension Claim

In the two-pool extension, Conditional AC + Imit100 beats the one-pool-style Global Threshold at C=800, C=1000, and C=1200. This indicates that the learned policy captures pool-specific structure beyond a global committed-pressure rule.

## Limitation Claim

The learned two-pool policy does not beat Pool-Specific Threshold. The pool-specific rule remains a strong hand-designed reference because it directly encodes the correct pool-level pressure heuristic.

## Ablation Claim

Ablations support the selected final configuration. Pressure-state one-pool variants, pressure_release state features, imitation regularization, and shorter or more conservative PPO variants did not improve over the final Conditional AC + Imit100 setup in the tested settings.

## Wording Guardrails

- Do not claim the learned policy is optimal.
- Do not claim the learned policy beats the strongest two-pool threshold baseline.
- Do not hide Tuned Threshold or Pool-Specific Threshold.
- Describe the two-pool C sweep as a seed=123 extension result, not as a full multi-seed main result.
- Emphasize money as the target objective: higher value acceptance alone is not always better when it requires more flushes.
