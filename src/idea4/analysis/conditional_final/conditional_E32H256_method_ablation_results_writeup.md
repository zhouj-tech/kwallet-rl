# Conditional Factorized AC Final Write-up

## Main Model Configuration

The final Conditional Factorized Actor-Critic uses the following fixed hyperparameters:

- settle action embedding dimension: 32
- conditional flush hidden size: 256
- training reward: original environment reward
- validation metric: value acceptance ratio
- training episodes: 1000
- evaluation episodes: 200
- seeds: 123, 323, 532, 777, 999, 2027, 3407, 4501, 6101, 8888

This setting is selected based on the 5-seed top-2 parameter confirmation experiment. Although E=32,H=128 and E=32,H=256 each won two of the four tested settings by mean value acceptance ratio, E=32,H=256 achieved better global average mean value acceptance ratio, better worst-regime value acceptance ratio, higher money score, and better overall rank. Therefore, E=32,H=256 is used as the main Conditional Factorized AC configuration, while E=32,H=128 is retained as a lightweight ablation variant.

## Method: Conditional Factorized Actor-Critic

The flat Basic PPO policy directly models the joint action space over settlement and flushing decisions. For k wallets, this requires a policy output size of (k+1)^2. This becomes expensive when k grows.

The independent Factorized AC reduces the policy output size by splitting the action into two independent categorical decisions:

π(a_s, a_f | s) = π_s(a_s | s) π_f(a_f | s)

where a_s is the settlement action and a_f is the flushing action. This reduces the output size from (k+1)^2 to 2(k+1). However, the independence assumption can be too restrictive because the best flushing decision may depend on the selected settlement decision.

The proposed Conditional Factorized AC keeps the same linear output structure but restores part of the joint-action dependency:

π(a_s, a_f | s) = π_s(a_s | s) π_f(a_f | s, a_s)

The actor first predicts the settlement distribution from the state. Then the selected settlement action is embedded and concatenated with the state representation to condition the flushing distribution. In this way, the flushing head can adapt to the settlement decision while the policy output size remains 2(k+1). For k=24, this gives 50 policy outputs compared with 625 outputs for Basic PPO, corresponding to a 92% reduction.

## Ablation: Why Conditional Factorization Helps

The comparison between independent Factorized AC and Conditional Factorized AC tests whether the dependency between settlement and flushing decisions matters. Independent factorization assumes that the two decisions are conditionally independent given the state. Conditional factorization relaxes this assumption by allowing the flushing policy to observe the selected settlement action.

The 10-seed k=24 stress results show that this conditional dependency is useful. At C=1200,k=24, Conditional Factorized AC significantly improves over independent Factorized AC in mean value acceptance ratio, worst-regime value acceptance ratio, and drops. This suggests that the conditional structure recovers useful joint-action information without returning to the full quadratic action output.

## Result Interpretation

Under the harsher C=800,k=24 setting, Conditional Factorized AC significantly outperforms Basic PPO. It improves mean value acceptance ratio, worst-regime value acceptance ratio, drops, and money score, while using only 50 policy outputs instead of 625. This indicates that conditional factorization is especially effective under high-pressure large-k settings.

Under the C=1200,k=24 setting, Conditional Factorized AC is statistically comparable to Basic PPO in value acceptance and drops, while achieving a significantly higher money score. It also significantly outperforms independent Factorized AC in mean value acceptance ratio, worst-regime value acceptance ratio, and drops. This supports the main claim that conditional factorization can close the performance gap between independent factorization and flat joint-action PPO while preserving the scalability advantage.

Overall, Conditional Factorized AC provides a stronger trade-off than the previous factorized policy: it preserves the 92% output reduction at k=24 while improving the ability to model the dependency between settlement and flushing decisions.
