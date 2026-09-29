# 1. Project One-Sentence Summary / 项目一句话总结

K-Wallet RL 是一个带有 settlement 与 flushing 两部分耦合动作的结构化决策问题：直接建模 flat joint-action 虽然表达力强，但动作输出随 `A_size = (k+1)^2` 二次增长；项目最终主线是用 Conditional Factorized AC 在保持 `2(k+1)` 线性输出规模的同时建模 `flush_choice` 对 `settle_choice` 的条件依赖。

**How to use this document / 使用方式**：写 Introduction 和方法动机时优先使用 Sections 1、2、8、9；写主结果时优先使用 Section 6D 和 6E；写 ablation 或 diagnostic discussion 时使用 Section 7；需要一页式总览时使用 Section 12。

# 2. Research Problem / 研究问题

K-Wallet RL 的核心问题是在一串交易流中管理 `k` 个钱包，每一步需要决定是否把当前交易结算到某个钱包，以及是否刷新某个钱包。状态 `s` 主要包含钱包余额、钱包可用性或冻结状态、当前交易金额、时间/冻结上下文；部分 DQN 或 AC 变体还加入 recent transaction context，用来描述短期交易分布。

动作是结构化二元动作：

- `settle_choice`: `0..k-1` 表示选择一个钱包接收当前交易，`k` 表示本步不结算。
- `flush_choice`: `0..k-1` 表示选择一个钱包刷新，`k` 表示本步不刷新。

如果把二元动作展平成一个 joint action，则动作空间大小为：

`A_size = (k+1)^2`

目标是在多种交易 regime 下最大化 value acceptance ratio 或 settled value，同时降低 drops，并控制 flushes 与 money score。flushing 可以恢复或释放钱包容量，使未来交易更可能被接收，但它也会引入刷新成本、冻结期或短期容量损失，因此 flushes 应被控制在有收益的位置，而不是被简单最大化。cross-regime generalization 很关键，因为论文场景不是只拟合单一交易分布，而是希望训练于 mixed 或某一 regime 后，在 `US/TLS/LNS/...` 等静态或混合 regime 上保持稳健表现。

大 `k` 的困难来自两层：第一，flat joint-action 的输出规模随 `k` 二次增长，k=24 时是 `625` 个 policy logits；第二，settlement 与 flushing 有真实耦合，简单拆成两个独立动作虽然降低输出，却可能丢掉“先选择哪个 settle，再决定 flush 哪个钱包”的依赖关系。

# 3. Repository Scan Summary / 代码与结果扫描清单

| file path | what it contains | evidence type |
|---|---|---|
| `src/idea3/dqn/kwallet_dqn_idea3.py` | 早期 DQN baseline、KWalletEnv、`num_actions = (k+1)^2`、joint-action 解码与 cross-regime evaluation。 | exploratory evidence |
| `src/idea3/dqn/kwallet_dqn_idea3_screening_ready.py` | DQN screening 版本，用于早期 baseline 筛选。 | exploratory evidence |
| `src/idea3/context_attention/kwallet_ctx_attn_fair_benchmark.py` | Context-attention DQN fair benchmark，加入 recent transaction attention context。 | exploratory evidence |
| `src/idea3/context_attention/kwallet_attention_context12_dqn.py` and related context files | 多个 context-aware DQN 尝试版本。 | diagnostic only |
| `src/idea3/data_generation/kwallet_regime_pool_generator.py` | 生成 static、mixed、switching regime pools，是 cross-regime 实验数据基础。 | formal evidence |
| `src/ideaextra/kwallet_ideaextra_dqn_k12_formal.py`、`src/ideaextra/kwallet_ideaextra_dqn_k6_formal.py` | ideaextra DQN formal runs，用于部分 k-scaling baseline 来源。 | exploratory evidence |
| `src/ideaextra/kwallet_ideaextra_dqn_k12_mock.py` | mock run，不应作为 paper formal evidence。 | diagnostic only |
| `src/idea4/ac/code/kwallet_basic_ppo_fair_benchmark.py` | Basic PPO：flat joint-action actor head，输出 `(k+1)^2`。 | formal evidence |
| `src/idea4/ac/code/run_factorized_ac_benchmark.py` | Independent Factorized AC：settle head 与 flush head 分开建模，各输出 `k+1`。 | formal evidence |
| `src/idea4/ac/code/run_dual_branch_ac_benchmark.py` | Dual-branch AC、Gate-balanced、Gate-regularized、capacity-only、residual-risk、aux-risk 相关实现。 | formal evidence |
| `src/idea4/ac/code/run_dual_branch_ac_dual_critic.py` | Dual critic 与 aux-risk branch variants。 | diagnostic only |
| `src/idea4/ac/code/run_dual_branch_ac_dual_critic_gate_entropy.py` | Gate entropy / gate regularization follow-up。 | diagnostic only |
| `src/idea4/ac/code/kwallet_maskA_factorized_ac.py` | MaskA hard/soft feasibility mask，含 settle/flush mask 逻辑。 | diagnostic only |
| `src/idea4/ac/code/run_conditional_factorized_ac_benchmark.py` | Conditional Factorized AC：settle embedding + conditional flush head，最终主模型代码。 | formal evidence |
| `src/idea3/results/old_fair_benchmark_results/aggregates/fair_benchmark_aggregated.csv` | 早期 DQN / attention DQN 聚合结果。 | exploratory evidence |
| `src/idea4/analysis/final_tables_v1/main_k_scaling_summary.csv` | baseline 与 k-scaling 正式汇总表。 | formal evidence |
| `src/idea4/analysis/final_tables_v1/stress_complexity_summary.csv` | k=12/k=24 stress-test 正式汇总表。 | formal evidence |
| `src/idea4/analysis/final_tables_v1/ablation_summary.csv` | Dual branch、Gate-balanced、Gate-regularized、MaskA 等消融汇总。 | formal evidence |
| `src/idea4/analysis/final_tables_v1/seed_level_runs_used.csv` | final tables 使用的 seed-level runs 清单。 | formal evidence |
| `src/idea4/analysis/conditional_final/table_k24_stress_comparison_conditional_E32H256.csv` | Conditional E32,H256 k=24 10-seed final main result。 | formal evidence |
| `src/idea4/analysis/conditional_final/table_k24_paired_difference_conditional_E32H256.csv` | Conditional E32,H256 paired difference 与 95% CI。 | formal evidence |
| `src/idea4/analysis/conditional_final/conditional_E32H256_method_ablation_results_writeup.md` | Conditional final 配置、方法解释和结果解读。 | formal evidence |
| `src/idea4/analysis/conditional_summary总结表/main_result_table.csv` | Conditional summary final table 的备份来源。 | formal evidence |
| `src/idea4/analysis/experiment_audit/project_story.md` | 旧研究阶段叙事，Conditional final 之前的项目故事。 | diagnostic only |
| `src/idea4/analysis/experiment_audit/direction_summary.md` | 各实验方向的诊断总结。 | diagnostic only |
| `src/idea4/analysis/experiment_audit/gate_behavior_summary.csv` | gate saturation / gate collapse 诊断数据。 | diagnostic only |
| raw `cross_regime_results.json` files | 原始 run 证据，约 300+ 个文件；本报告只作为路径证据或 backup verification 使用。 | diagnostic only |

# 4. Research Timeline / 研究阶段时间线

## Stage 1: DQN baseline and joint-action explosion

- Motivation / 为什么做：先建立可运行的 K-Wallet RL baseline，并确认 cross-regime evaluation 能工作。
- Method / 方法是什么：DQN 直接输出 flat joint-action Q-values。
- Formula / 核心公式：`Q(s,a_s,a_f)`，输出大小 `A_size=(k+1)^2`。
- Code location / 代码位置：`src/idea3/dqn/kwallet_dqn_idea3.py`，`src/ideaextra/kwallet_ideaextra_dqn*.py`。
- Result files / 结果文件：`src/idea3/results/old_fair_benchmark_results/aggregates/fair_benchmark_aggregated.csv`；正式 k-scaling 中的 DQN 行来自 `src/idea4/analysis/final_tables_v1/main_k_scaling_summary.csv`。
- Key result / 关键结果：在 `C=1200,k=12`，DQN baseline mean value acceptance 为 `53.42 ± 10.28%`；同表中 Basic PPO 为 `85.43 ± 8.95%`，Factorized AC 为 `81.12 ± 1.51%`。
- Interpretation / 怎么理解：DQN 可作为早期 baseline，但 flat action 输出随 k 增大迅速变重，性能和稳定性也弱于后续 AC 系列。
- Decision / 最终状态：baseline。

## Stage 2: Actor-Critic / Basic PPO transition

- Motivation：检验不做 factorization 的 flat actor-critic 是否已经足够强，为后续结构化模型提供强基线。
- Method：Basic PPO 直接对 joint action 建模。
- Formula：`π(a_s,a_f | s)`，policy output 为 `(k+1)^2`。
- Code location：`src/idea4/ac/code/kwallet_basic_ppo_fair_benchmark.py`。
- Result files：`src/idea4/analysis/final_tables_v1/main_k_scaling_summary.csv`，`src/idea4/analysis/final_tables_v1/stress_complexity_summary.csv`。
- Key result：在 `C=1200,k=24` 的 2-seed stress 表中，Basic PPO mean value acceptance 为 `41.41 ± 10.10%`，Factorized AC 为 `40.66 ± 4.08%`；在 Conditional final 10-seed 表中，Basic PPO 为 `41.22 ± 0.71%`。
- Interpretation：Basic PPO 是强 baseline，尤其在部分设置下表达力强；但 k=24 需要 `625` 个 policy outputs。
- Decision：strong baseline。

## Stage 3: Independent Factorized AC

- Motivation：利用动作结构降低输出规模，缓解 large-k 下 flat joint-action 的二次扩展。
- Method：settle policy 与 flush policy 独立输出。
- Formula：`π(a_s,a_f | s)=π_s(a_s|s)π_f(a_f|s)`。
- Code location：`src/idea4/ac/code/run_factorized_ac_benchmark.py`。
- Result files：`src/idea4/analysis/final_tables_v1/main_k_scaling_summary.csv`，`src/idea4/analysis/final_tables_v1/stress_complexity_summary.csv`。
- Key result：在 `C=1200,k=12`，Factorized AC mean value acceptance 为 `81.12 ± 1.51%`，明显高于 DQN baseline 的 `53.42 ± 10.28%`；在 k=24 时 policy output 为 `50`，比 Basic PPO 的 `625` 小 92%。
- Interpretation：Independent Factorized AC 证明线性输出结构有价值，但它假设 flush 与 settle 在给定状态后条件独立。
- Decision：kept baseline。

## Stage 4: Dual-branch AC and branch-specialization attempt

- Motivation：希望把容量约束与未来风险分成两个 branch，让模型内部形成更清晰的 specialization。
- Method：capacity branch 使用 full state，risk branch 使用 engineered risk features，gate 融合两个 branch 的 settle/flush logits 和 value。
- Formula：`logits = g(s) logits_capacity + (1-g(s)) logits_risk`。
- Code location：`src/idea4/ac/code/run_dual_branch_ac_benchmark.py`。
- Result files：`src/idea4/analysis/final_tables_v1/main_k_scaling_summary.csv`，`src/idea4/analysis/final_tables_v1/ablation_summary.csv`。
- Key result：在 `C=1200,k=12`，Original Dual-branch AC mean value acceptance 为 `81.36 ± 1.03%`，Factorized AC 为 `81.12 ± 1.51%`，差距很小。
- Interpretation：dual branch 的动机合理，但没有稳定证明 branch specialization 带来主结果提升。
- Decision：diagnostic only。

## Stage 5: Gate analysis and failed gate fixes

- Motivation：观察到 dual branch 可能被单一 branch 主导，需要诊断 gate 是否真的随状态切换。
- Method：Gate-balanced 把 gate 限制到 `[0.1,0.9]`，Gate-regularized 用 target / regularization 试图控制 gate。
- Formula：`g = gate_min + (gate_max-gate_min) sigmoid(logit/temp)`，或加入 gate target regularization。
- Code location：`src/idea4/ac/code/run_dual_branch_ac_benchmark.py`；gate diagnostic output 主要使用 `src/idea4/analysis/experiment_audit/gate_behavior_summary.csv`。
- Result files：`src/idea4/analysis/final_tables_v1/ablation_summary.csv`，`src/idea4/analysis/experiment_audit/gate_behavior_summary.csv`。
- Key result：`C=1200,k=12` 下 Gate-balanced mean value acceptance 为 `82.90 ± 0.45%`，Gate-regularized 为 `81.10 ± 3.83%`；gate_behavior 显示多个 Gate-balanced runs 的 `gate_mean≈0.90`、`near_upper_rate=1.0`、`gate_std` 接近 0。
- Interpretation：Gate-balanced 可带来局部改善，但 gate 接近上边界，说明 branch 使用仍有 saturation / weak specialization 问题。
- Decision：diagnostic only。

## Stage 6: Aux-risk / risk-specialization attempts

- Motivation：让 risk branch 学到更明确的风险信号，例如未来 drops 或 dual critic 信号。
- Method：Dual critic 或 aux-risk head，对 risk branch 加辅助目标。
- Formula：主 PPO loss 加 `λ L_aux-risk` 或 dual critic 相关损失。
- Code location：`src/idea4/ac/code/run_dual_branch_ac_dual_critic.py`，`src/idea4/ac/code/run_dual_branch_ac_benchmark.py`。
- Result files：`src/idea4/analysis/experiment_audit/direction_summary.md`，`src/idea4/analysis/experiment_audit/gate_behavior_summary.csv`。
- Key result：diagnostic audit suggests Dual Critic 和 Dual Critic + Aux Risk 没有稳定压过 Gate-balanced baseline；若要报告具体百分点差异，需要回到对应 aggregate 或 raw paired comparison 重新确认，当前写作中标为 `needs verification`。
- Interpretation：辅助风险目标没有稳定转化为 actor 的策略收益，可能增加训练干扰。
- Decision：abandoned as main model; diagnostic only。

## Stage 7: MaskA or action-constraint attempts

- Motivation：把“明显不可行”的动作先验注入 policy logits，减少无效探索。
- Method：hard mask 或 soft penalty mask settle/flush logits。
- Formula：hard: `logits_invalid=-1e9`；soft: `logits_invalid -= penalty`。
- Code location：`src/idea4/ac/code/kwallet_maskA_factorized_ac.py`。
- Result files：`src/idea4/analysis/final_tables_v1/ablation_summary.csv`。
- Key result：在 `C=800,k=24` 单 seed ablation 中，MaskA hard mean value acceptance 为 `9.91%`，MaskA soft+penalty5 为 `8.22%`；同 stress 表中 2-seed Factorized AC 为 `11.17 ± 27.35%`。由于 seed 与聚合范围不同，精确胜负需 `needs verification`。
- Interpretation：硬约束可能压缩探索，且短期可行不等于长期收益。
- Decision：diagnostic only / abandoned。

## Stage 8: Conditional Factorized AC as final main model

- Motivation：Independent Factorized AC 的根本限制是 flush 不知道已选 settle；Conditional Factorized AC 直接恢复这个关键依赖，同时不回到 quadratic output。
- Method：先从 state 采样或选择 `a_s`，再将 settle action embedding 与 state representation 拼接，用 conditional flush head 输出 `π_f(a_f|s,a_s)`。
- Formula：`π(a_s,a_f | s)=π_s(a_s|s)π_f(a_f|s,a_s)`。
- Code location：`src/idea4/ac/code/run_conditional_factorized_ac_benchmark.py`。
- Result files：`src/idea4/analysis/conditional_final/table_k24_stress_comparison_conditional_E32H256.csv`，`src/idea4/analysis/conditional_final/table_k24_paired_difference_conditional_E32H256.csv`。
- Key result：k=24 时仍使用 `50` 个 policy outputs，比 Basic PPO 的 `625` 少 92%；paired CI 支持：在 `C=800,k=24` 优于 Basic PPO；在 `C=1200,k=24` 与 Basic PPO 的 value acceptance / drops 统计相当，但 money 更好，并优于 Factorized AC 的 Mean ValAcc、Worst ValAcc、Drops。
- Interpretation：Conditional Factorized AC 是当前最好的 scalability-performance trade-off 主线。
- Decision：final main model。

# 5. Model Family Summary Table / 模型家族总结表

| Model family | Formula / Policy form | Code location | Output size | Main idea | Main result | Limitation | Final status |
|---|---|---|---|---|---|---|---|
| DQN baseline | `Q(s,a_s,a_f)` | `src/idea3/dqn/kwallet_dqn_idea3.py` | `(k+1)^2` | 直接学习 joint-action Q-value | `C=1200,k=12` mean ValAcc `53.42 ± 10.28%` in `main_k_scaling_summary.csv` | large-k 输出爆炸，稳定性弱于 AC | baseline |
| Context-attention DQN | `Q(s,context,a_s,a_f)` | `src/idea3/context_attention/kwallet_ctx_attn_fair_benchmark.py` | `(k+1)^2` | 用 recent transaction attention 改善 context 表示 | formal final paper number needs verification | 仍是 flat DQN，未解决 action-size 二次增长 | exploratory baseline |
| Basic PPO | `π(a_s,a_f|s)` | `src/idea4/ac/code/kwallet_basic_ppo_fair_benchmark.py` | `(k+1)^2` | 表达完整 joint action distribution | `C=1200,k=24` 10-seed Mean ValAcc `41.22 ± 0.71%` | k=24 需要 625 outputs | strong baseline |
| Independent Factorized AC | `π_s(a_s|s)π_f(a_f|s)` | `src/idea4/ac/code/run_factorized_ac_benchmark.py` | `2(k+1)` | 线性扩展，分头输出 settle/flush | `C=1200,k=24` 10-seed Mean ValAcc `40.02 ± 0.55%` | flush 与 settle 条件独立假设过强 | kept baseline |
| Dual-branch AC | `g logits_cap + (1-g) logits_risk` | `src/idea4/ac/code/run_dual_branch_ac_benchmark.py` | `2(k+1)` | 容量 branch 与风险 branch 分工 | `C=1200,k=12` Mean ValAcc `81.36 ± 1.03%` | specialization 证据弱 | diagnostic only |
| Gate-balanced Dual AC | bounded gate fusion | `src/idea4/ac/code/run_dual_branch_ac_benchmark.py` | `2(k+1)` | 限制 gate 避免单分支垄断 | `C=1200,k=12` Mean ValAcc `82.90 ± 0.45%` | gate 常贴近上界，state-dependent 切换弱 | diagnostic only |
| Gate-regularized Dual AC | gate target regularization | `src/idea4/ac/code/run_dual_branch_ac_benchmark.py` | `2(k+1)` | 用 regularization 推动 gate 到目标范围 | `C=1200,k=12` Mean ValAcc `81.10 ± 3.83%` | 未优于 Gate-balanced，且仍有 saturation | abandoned |
| Aux-risk branch variants | PPO + `λ L_aux-risk` | `src/idea4/ac/code/run_dual_branch_ac_dual_critic.py` | `2(k+1)` | 用未来风险辅助目标推动 risk specialization | diagnostic audit suggests no stable improvement over Gate-balanced; exact percentage difference needs verification | auxiliary signal 可能干扰主目标 | diagnostic only |
| MaskA Factorized AC | factorized policy + action masks | `src/idea4/ac/code/kwallet_maskA_factorized_ac.py` | `2(k+1)` | 用 hard/soft feasibility mask 约束 logits | `C=800,k=24` hard single-seed Mean ValAcc `9.91%` | 可能限制探索；formal multi-seed support 不足 | diagnostic only |
| Conditional Factorized AC | `π_s(a_s|s)π_f(a_f|s,a_s)` | `src/idea4/ac/code/run_conditional_factorized_ac_benchmark.py` | `2(k+1)` | 用 settle embedding 条件化 flush policy | paired CI: `C=800,k=24` 优于 Basic PPO；`C=1200,k=24` 对 Basic PPO 统计相当且 money 更好 | 尚需更多 C/k/F setting paired CI | final main model |

# 6. Important Experiment Results / 重要实验结果

## A. K-scaling / main baseline comparison

Source: `src/idea4/analysis/final_tables_v1/main_k_scaling_summary.csv`

| C | k | Model | n_seeds | Mean ValAcc | Worst ValAcc | Drops |
|---:|---:|---|---:|---:|---:|---:|
| 1200 | 3 | DQN baseline | 2 | 96.41 ± 40.15% | 95.12 ± 48.63% | 33.93 ± 386.02 |
| 1200 | 3 | Basic PPO | 2 | 99.83 ± 1.70% | 99.13 ± 7.77% | 0.50 ± 5.34 |
| 1200 | 3 | Factorized AC | 2 | 99.34 ± 2.17% | 97.74 ± 4.40% | 2.30 ± 10.48 |
| 1200 | 3 | Dual-branch AC | 2 | 99.42 ± 1.58% | 97.74 ± 4.53% | 1.69 ± 5.15 |
| 1200 | 6 | DQN baseline | 2 | 94.80 ± 3.31% | 88.45 ± 1.88% | 35.34 ± 64.10 |
| 1200 | 6 | Basic PPO | 2 | 97.03 ± 3.38% | 89.80 ± 4.90% | 9.02 ± 14.94 |
| 1200 | 6 | Factorized AC | 2 | 96.70 ± 1.16% | 89.44 ± 1.07% | 10.59 ± 5.61 |
| 1200 | 12 | DQN baseline | 2 | 53.42 ± 10.28% | 49.39 ± 6.53% | 375.75 ± 124.27 |
| 1200 | 12 | Basic PPO | 2 | 85.43 ± 8.95% | 71.52 ± 6.31% | 60.67 ± 63.09 |
| 1200 | 12 | Factorized AC | 4 | 81.12 ± 1.51% | 68.40 ± 1.08% | 89.17 ± 10.80 |
| 1200 | 12 | Dual-branch AC | 4 | 81.36 ± 1.03% | 68.58 ± 0.54% | 87.31 ± 7.25 |

这组结果支持两点：第一，DQN baseline 在 k 增大后明显变弱；第二，Basic PPO、Factorized AC、Dual-branch AC 在 k=3/6/12 上形成主要可比模型族。这里不能推出 Conditional Factorized AC 的结论，因为 Conditional final 使用单独的 k=24 10-seed 表。

## B. Stress test under k=24

Source: `src/idea4/analysis/final_tables_v1/stress_complexity_summary.csv` and `src/idea4/analysis/conditional_final/table_k24_stress_comparison_conditional_E32H256.csv`

| Source | C | k | Model | n_seeds | Policy Output | Mean ValAcc | Worst ValAcc | Drops | Money |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| final_tables_v1 stress | 800 | 24 | Basic PPO | 2 | 625 | 10.51 ± 8.36% | 5.80 ± 4.19% | 798.66 ± 163.71 | needs verification |
| final_tables_v1 stress | 800 | 24 | DQN baseline | 2 | 625 | 9.26 ± 0.04% | 5.48 ± 0.92% | 821.42 ± 2.34 | needs verification |
| final_tables_v1 stress | 800 | 24 | Factorized AC | 2 | 50 | 11.17 ± 27.35% | 4.91 ± 21.25% | 787.85 ± 538.72 | needs verification |
| conditional_final | 800 | 24 | Basic PPO | 10 | 625 | 10.52 ± 1.07% | 5.77 ± 0.59% | 798.73 ± 20.46 | 3368.57 ± 320.53 |
| conditional_final | 800 | 24 | Factorized AC | 10 | 50 | 11.08 ± 1.37% | 5.35 ± 0.81% | 788.53 ± 26.28 | 3607.05 ± 404.63 |
| conditional_final | 800 | 24 | Conditional Factorized AC | 10 | 50 | 12.56 ± 0.41% | 6.60 ± 0.16% | 760.21 ± 7.85 | 3999.25 ± 123.13 |
| conditional_final | 1200 | 24 | Basic PPO | 10 | 625 | 41.22 ± 0.71% | 34.90 ± 0.58% | 402.48 ± 9.80 | 14443.01 ± 235.31 |
| conditional_final | 1200 | 24 | Factorized AC | 10 | 50 | 40.02 ± 0.55% | 32.56 ± 0.32% | 417.36 ± 7.44 | 14472.93 ± 185.72 |
| conditional_final | 1200 | 24 | Conditional Factorized AC | 10 | 50 | 41.57 ± 0.23% | 34.64 ± 0.34% | 397.12 ± 3.05 | 14687.65 ± 91.13 |

k=24 是论文最关键的 stress setting。Conditional final 10-seed 结果优先级最高，说明 Conditional Factorized AC 在只用 `50` 个 policy outputs 的情况下，保留了 92% output reduction，同时在 C=800 下明显改善性能；C=1200 下与 Basic PPO 在 value acceptance / drops 上非常接近，并在 money 上更好。

## C. Dual-branch and gate ablation results

Source: `src/idea4/analysis/final_tables_v1/ablation_summary.csv` and `src/idea4/analysis/experiment_audit/gate_behavior_summary.csv`

| C | k | Variant | n_seeds | Mean ValAcc | Worst ValAcc | Drops | Diagnostic interpretation |
|---:|---:|---|---:|---:|---:|---:|---|
| 1200 | 12 | Gate-balanced Dual AC | 4 | 82.90 ± 0.45% | 69.57 ± 0.42% | 76.75 ± 3.21 | 局部强于原始 Dual-branch，但 gate 贴近上界。 |
| 1200 | 12 | Gate-regularized Dual AC | 2 | 81.10 ± 3.83% | 68.34 ± 4.40% | 89.22 ± 24.34 | 未稳定优于 Gate-balanced。 |
| 1200 | 12 | Capacity-only Dual AC | 1 | 80.33% | 67.75% | 94.65 | 单 seed，主要是消融证据。 |
| 1200 | 12 | Residual-risk Dual AC | 1 | 80.87% | 67.95% | 90.82 | 单 seed，未形成主线证据。 |
| 800 | 24 | MaskA hard | 1 | 9.91% | 4.81% | 809.81 | action mask 没有成为正式主线。 |
| 800 | 24 | MaskA soft+penalty5 | 1 | 8.22% | 5.14% | 840.54 | 可能限制探索，需谨慎解释。 |

gate 诊断显示，多数 Gate-balanced runs 的 `gate_mean≈0.90`、`gate_near_upper_rate=1.0`、`gate_std` 接近 0。这说明 dual-branch 的风险分支虽然是有意义的设计尝试，但 branch-specialization 没有被强证据支持，因此更适合作为诊断章节或消融，而不是最终主模型。

## D. Conditional Factorized AC final 10-seed results

Source: `src/idea4/analysis/conditional_final/table_k24_stress_comparison_conditional_E32H256.csv`

| C | k | Model | Policy Output | Output Reduction | Mean ValAcc | Worst ValAcc | Drops | Flushes | Money |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 800 | 24 | Basic PPO | 625 | 0% | 10.52 ± 1.07% | 5.77 ± 0.59% | 798.73 ± 20.46 | 189.56 ± 21.88 | 3368.57 ± 320.53 |
| 800 | 24 | Factorized AC | 50 | 92% | 11.08 ± 1.37% | 5.35 ± 0.81% | 788.53 ± 26.28 | 193.24 ± 28.15 | 3607.05 ± 404.63 |
| 800 | 24 | Conditional Factorized AC | 50 | 92% | 12.56 ± 0.41% | 6.60 ± 0.16% | 760.21 ± 7.85 | 228.34 ± 8.33 | 3999.25 ± 123.13 |
| 1200 | 24 | Basic PPO | 625 | 0% | 41.22 ± 0.71% | 34.90 ± 0.58% | 402.48 ± 9.80 | 617.22 ± 36.62 | 14443.01 ± 235.31 |
| 1200 | 24 | Factorized AC | 50 | 92% | 40.02 ± 0.55% | 32.56 ± 0.32% | 417.36 ± 7.44 | 554.26 ± 9.87 | 14472.93 ± 185.72 |
| 1200 | 24 | Conditional Factorized AC | 50 | 92% | 41.57 ± 0.23% | 34.64 ± 0.34% | 397.12 ± 3.05 | 610.40 ± 9.44 | 14687.65 ± 91.13 |

这是最终论文主表的核心候选。它不支持“Conditional Factorized AC 普遍击败 Basic PPO”这种过强说法；结合 Section 6E 的 paired CI，它支持更精确的 claim：在高压力 `C=800,k=24` 下优于 Basic PPO；在 `C=1200,k=24` 下与 Basic PPO 在 value acceptance 和 drops 上统计相当，同时 money 更好。

## E. Paired difference results for Conditional E32,H256

Full source: `src/idea4/analysis/conditional_final/table_k24_paired_difference_conditional_E32H256.csv`

| C | k | Comparison | Metric | Difference ± 95% CI | Interpretation |
|---:|---:|---|---|---:|---|
| 800 | 24 | Conditional - Basic PPO | Mean ValAcc | +2.04 ± 1.22 pp | significant, Conditional better |
| 800 | 24 | Conditional - Basic PPO | Worst ValAcc | +0.83 ± 0.60 pp | significant, Conditional better |
| 800 | 24 | Conditional - Basic PPO | Drops | -38.52 ± 23.23 | significant, Conditional better |
| 800 | 24 | Conditional - Basic PPO | Money | +630.68 ± 372.07 | significant, Conditional better |
| 1200 | 24 | Conditional - Basic PPO | Mean ValAcc | +0.35 ± 0.71 pp | not significant / statistically comparable |
| 1200 | 24 | Conditional - Basic PPO | Worst ValAcc | -0.26 ± 0.58 pp | not significant / statistically comparable |
| 1200 | 24 | Conditional - Basic PPO | Drops | -5.37 ± 9.78 | not significant / statistically comparable |
| 1200 | 24 | Conditional - Basic PPO | Money | +244.64 ± 220.12 | significant, Conditional better |
| 1200 | 24 | Conditional - Factorized AC | Mean ValAcc | +1.55 ± 0.57 pp | significant, Conditional better |
| 1200 | 24 | Conditional - Factorized AC | Worst ValAcc | +2.08 ± 0.46 pp | significant, Conditional better |
| 1200 | 24 | Conditional - Factorized AC | Drops | -20.24 ± 7.68 | significant, Conditional better |

paired CI 是最终 claim 的措辞依据。C=800 对 Basic PPO 的四个指标均支持 significant improvement；C=1200 对 Basic PPO 只能说 value acceptance 和 drops 统计相当，money 显著更好；C=1200 对 independent Factorized AC 的 Mean ValAcc、Worst ValAcc、Drops 可以说显著改善。这是本文档中使用“显著”的主要依据。

# 7. Failed or Diagnostic Directions / 失败与诊断方向

## Dual-branch AC

Dual-branch AC 尝试把 full-state capacity branch 与 engineered-risk branch 分开，再用 learned gate 融合 logits。它的机制合理，但在 `C=1200,k=12` 上与 Factorized AC 的差距很小，不能证明 branch specialization 是最终解决方案。它贡献的主要 lesson 是：单纯把状态表示拆成“capacity/risk”两路，并不自动带来有效分工。

## Gate-balanced Dual AC

Gate-balanced 通过限制 gate 范围，希望避免单分支完全垄断。结果在 `C=1200,k=12` 有局部提升，但 gate diagnostics 显示 gate 经常贴近上界，`near_upper_rate` 很高，`gate_std` 接近 0。它的 lesson 是：约束 gate 的数值范围不等于获得状态依赖的 branch switching。

## Gate-regularized Dual AC

Gate-regularized 进一步使用 gate target / regularization，希望让 gate 偏离饱和区域。正式 ablation 中它没有稳定超过 Gate-balanced，且仍未解决 specialization 证据不足的问题。因此不适合作为主模型，但可以在 appendix 中作为 gate-fix negative result。

## Aux-risk / Dual critic variants

Aux-risk 与 Dual critic 试图给 risk branch 更明确的训练信号，例如预测未来 drop 风险或拆分 critic 信号。`src/idea4/analysis/experiment_audit/direction_summary.md` 的 diagnostic audit suggests 这些方向没有稳定压过 Gate-balanced baseline，但具体百分点差异不应作为 formal evidence 报告，除非进一步定位到对应 aggregate table 或重新做 matched/paired comparison。更保守的解释是：auxiliary objective 可能与主 reward 不完全一致，增加了训练干扰；这一路线的价值主要是诊断“风险分支专门化没有自然形成”。

## MaskA Factorized AC

MaskA hard/soft variants 将可行性先验注入 settle/flush logits。当前 formal ablation 只显示少量 seed，结果不支持其成为主线。原因可能是 hard mask 缩小探索空间，而短期可行性不等于长期收益最优。这个方向可以作为“action constraint prior 的负向/诊断实验”。

## Reward shaping / money reward

rewardMONEY 等实验尝试直接优化 money，但旧 audit 显示 value acceptance 明显退化。它的 lesson 是 money 作为 evaluation metric 可以保留，但训练 reward 的尺度和稀疏性需要谨慎处理；最终 Conditional E32,H256 仍使用 original environment reward，并用 money 作为结果指标之一。

# 8. Final Main Model: Conditional Factorized AC / 最终主模型

Basic PPO 直接建模完整 joint action：

`π(a_s, a_f | s)`

它的优点是表达力强，可以直接学习 settlement 与 flushing 的任意耦合关系；缺点是 policy output size 为 `(k+1)^2`，k=24 时需要 `625` 个输出。

Independent Factorized AC 将 joint action 拆成两个独立 categorical policies：

`π(a_s, a_f | s) = π_s(a_s | s)π_f(a_f | s)`

它的优点是输出规模降为 `2(k+1)`，k=24 时只需要 `50` 个输出；缺点是给定 state 后，flush policy 不知道本步已经选择了哪个 settle action。这种 independence assumption 可能过强，因为实际系统中最佳 flush decision 往往依赖 settlement decision。

Conditional Factorized AC 保留线性输出规模，同时恢复 settle-to-flush 的关键依赖：

`π(a_s, a_f | s) = π_s(a_s | s)π_f(a_f | s, a_s)`

实现上，actor 先从 shared state encoder 产生 `settle_logits`，采样或选择 `settle_action`；随后用 `settle_embedding` 将该动作嵌入为向量，与 state representation 拼接后输入 conditional flush head，得到 `flush_logits`。因此 flush head 能感知已选 settlement，同时总输出仍是 settle 的 `k+1` logits 加 flush 的 `k+1` logits，即 `2(k+1)`。

在 k=24 时，Conditional Factorized AC 使用 `50` 个 policy outputs，而 Basic PPO 使用 `625` 个，输出减少 `575/625 = 92%`。这就是最终主模型的核心 trade-off：接近 joint-action 的条件表达能力，但保持 factorized policy 的可扩展性。

# 9. Final Paper Storyline / 最终论文主线

K-Wallet RL 的关键不是普通离散动作控制，而是 settlement 与 flushing 构成的结构化耦合动作。每一步，agent 既要决定当前交易是否结算到某个钱包，又要决定是否刷新某个钱包；这两个选择在业务上互相影响，因为刷新会改变钱包可用性和未来容量，而结算会立即消耗某个钱包的余额。

最直接的方法是把 `(settle_choice, flush_choice)` 展平成一个 flat joint action。Basic PPO 就采用这种做法，因此表达力强，可以学习完整动作耦合。但它的输出规模是 `(k+1)^2`，当 k 从 3、6 增长到 12、24 时，策略头会快速变大，训练成本和大动作空间探索压力随之上升。

Independent Factorized AC 是第一步结构化改进：把 policy 拆成 `π_s(a_s|s)` 和 `π_f(a_f|s)`，输出规模从 quadratic 降为 linear。这证明了利用动作结构是可行方向，也使 k=24 的 policy outputs 从 625 降到 50。然而，这个模型把 settle 与 flush 条件独立化，无法直接表达“flush 应该根据已选 settle 动作调整”的依赖。

项目随后探索了 Dual-branch AC，希望通过 capacity branch 与 risk branch 的状态专门化来解决复杂决策问题。这个方向有诊断价值，但 gate analysis 暴露出 gate saturation 和 weak specialization：模型经常偏向单一 branch，分支结构并没有稳定转化为最终性能优势。Gate-balanced、Gate-regularized、aux-risk、dual critic 等尝试也未能形成比主 baselines 更清晰的最终 story。

最终主线应回到动作结构本身：Conditional Factorized AC 不是再拆状态 branch，而是直接修复 factorization 的关键缺陷。它仍先建模 settlement policy，但把已选 `a_s` 作为条件输入 flush policy，即 `π_f(a_f|s,a_s)`。这样，模型以线性输出规模保留了 settle-to-flush dependency。

实验上，k=24 是最能体现论文价值的 setting。根据 Section 6E 的 paired CI，Conditional E32,H256 在 `C=800,k=24` 显著优于 Basic PPO 的 Mean ValAcc、Worst ValAcc、Drops 和 Money；在 `C=1200,k=24` 与 Basic PPO 的 value acceptance 和 drops 统计相当，同时 Money 显著更好；并且在 `C=1200,k=24` 显著优于 independent Factorized AC 的 Mean ValAcc、Worst ValAcc 和 Drops。这支持最终论文 claim：Conditional Factorized AC 提供了更好的 scalability-performance trade-off，而不是简单宣称它在所有 setting 下都击败 Basic PPO。

# 10. Conference Paper Outline / 会议论文结构建议

| Section | What to include | Result table to use | Claim to make |
|---|---|---|---|
| Abstract | 一句话定义 K-Wallet RL、二元耦合动作、flat joint-action scalability 问题、Conditional Factorized AC 方法和 k=24 结果。 | Conditional final 10-seed table + paired difference key rows | Conditional Factorized AC 在 k=24 保持 92% output reduction，并在关键压力设置下改善或匹配 Basic PPO。 |
| Introduction | 介绍钱包容量管理、settlement/flushing 耦合、cross-regime generalization、large-k action explosion。 | k-scaling table | flat joint-action 在 large k 下不理想，结构化 policy 有必要。 |
| Related Work | Structured action RL、factorized policies、conditional policies、actor-critic/PPO、resource management RL。 | 不需要主实验表 | 本文贡献是对 coupled two-part action 的 conditional factorization。 |
| Problem Formulation | 定义 state、`settle_choice`、`flush_choice`、transition、reward、metrics、`A_size=(k+1)^2`。 | 无，或引用数据 pool 文件 | K-Wallet 是 structured coupled action MDP。 |
| Method | 对比 Basic PPO、Independent Factorized AC、Conditional Factorized AC；讲 settle embedding 和 conditional flush head。 | model family summary | Conditional factorization 在保持 `2(k+1)` 输出时恢复关键依赖。 |
| Experiments | 说明 regimes、C/k/F/T、seeds、metrics、baselines。 | final_tables_v1 + conditional_final | 实验覆盖 baseline、k-scaling、stress、ablation 和 final 10-seed。 |
| Ablation Study | 讨论 independent factorization、dual branch、gate、MaskA、aux-risk。 | ablation_summary + gate_behavior_summary | dual/gate/mask/risk 提供诊断，但最终不如条件动作结构直接。 |
| Results and Discussion | 聚焦 k=24 final result 与 paired CI；谨慎区分 significant、statistically comparable、not significant。 | Conditional final table + paired difference source | C=800 的 paired CI 支持优于 Basic PPO；C=1200 与 Basic PPO value/drops 相当、money 更好；对 Factorized AC 有关键改善。 |
| Limitations | seed 和 setting 覆盖、更多 paired CI、money metric 一致性、reward shaping 未完成、k=12 是否放 appendix。 | Remaining issues | 结论限于当前 formal aggregate tables，不做 universal claim。 |
| Conclusion | 回扣 structured coupled action、quadratic vs linear output、conditional factorization 的 trade-off。 | final summary table | Conditional Factorized AC 是最终推荐主模型。 |

# 11. Remaining Issues / 仍需检查的问题

- 需要确认所有 final baseline 与 Conditional 10-seed 是否严格使用相同 seed set、相同 train/eval pool、相同 evaluation episodes。
- 需要确认 mock/debug/sanity runs 没有被误纳入 final aggregate tables；当前计划只把被 final tables 纳入的结果作为 formal evidence。
- k=12 结果更适合作为 main baseline trend 还是 appendix，需要根据会议篇幅决定。
- paired CI 目前重点覆盖 Conditional E32,H256 的 k=24；若论文要声称更多 C/k/F setting，应重新计算对应 paired CI。
- `Money` 指标需要确认所有表均使用同一 `money_p` 和 `money_tau`，以及是否都来自 evaluation metric 而非 training reward。
- `final_tables_v1` 与 `conditional_final` 在 k=24 baseline 数值非常接近但 seed 数不同；正式论文应优先报告 `conditional_final` 的 10-seed baseline，并说明它覆盖 Basic PPO 和 Factorized AC。
- Context-attention DQN 的最终正式数值需要核对是否有 clean aggregate 被纳入 final paper；目前只能作为 exploratory evidence。
- MaskA 结果多为少 seed 消融，不宜写成定论；若要放 appendix，应明确 single-seed 或 low-seed。
- reward shaping / money reward 的退化来自 diagnostic audit，若要正式引用需回查对应 aggregate 或 raw result。
- 所有 reported value acceptance、drops、flushes、money 是否采用同一 `T=1000`、`F=3`、evaluation episodes=200，需要在 camera-ready 前统一核对。

# 12. Final Summary Table / 最终总结大表

| Stage | Method | Key formula | Main result file | Key result | Interpretation | Final status |
|---|---|---|---|---|---|---|
| Stage 1 | DQN baseline | `Q(s,a_s,a_f)` | `src/idea4/analysis/final_tables_v1/main_k_scaling_summary.csv` | `C=1200,k=12` Mean ValAcc `53.42 ± 10.28%` | 可运行 baseline，但 large-k 弱。 | baseline |
| Stage 2 | Basic PPO | `π(a_s,a_f|s)` | `src/idea4/analysis/final_tables_v1/stress_complexity_summary.csv` and `conditional_final/table_k24_stress_comparison_conditional_E32H256.csv` | k=24 output `625`，10-seed `C=1200` Mean ValAcc `41.22 ± 0.71%` | 强表达力、强 baseline，但输出二次增长。 | strong baseline |
| Stage 3 | Independent Factorized AC | `π_s(a_s|s)π_f(a_f|s)` | `src/idea4/analysis/final_tables_v1/main_k_scaling_summary.csv` | `C=1200,k=12` Mean ValAcc `81.12 ± 1.51%` | 线性输出有效，但独立性假设限制表达。 | kept baseline |
| Stage 4 | Dual-branch AC | `g logits_cap+(1-g)logits_risk` | `src/idea4/analysis/final_tables_v1/main_k_scaling_summary.csv` | `C=1200,k=12` Mean ValAcc `81.36 ± 1.03%` | branch specialization 未被强证据证明。 | diagnostic only |
| Stage 5 | Gate-balanced / Gate-regularized Dual AC | bounded or regularized gate | `src/idea4/analysis/final_tables_v1/ablation_summary.csv` | Gate-balanced `82.90 ± 0.45%`; Gate-regularized `81.10 ± 3.83%` | Gate-balanced 局部改善，但 gate saturation 明显。 | diagnostic only |
| Stage 6 | Aux-risk / Dual critic | PPO + auxiliary risk / critic losses | `src/idea4/analysis/experiment_audit/direction_summary.md` | diagnostic audit shows no improvement over Gate-balanced; needs verification for formal numeric claim | 辅助风险信号没有稳定转化为策略收益。 | abandoned / diagnostic only |
| Stage 7 | MaskA Factorized AC | factorized logits + hard/soft mask | `src/idea4/analysis/final_tables_v1/ablation_summary.csv` | `C=800,k=24` hard single-seed Mean ValAcc `9.91%` | action constraints 可能限制探索，证据不足。 | diagnostic only |
| Stage 8 | Conditional Factorized AC | `π_s(a_s|s)π_f(a_f|s,a_s)` | `src/idea4/analysis/conditional_final/table_k24_stress_comparison_conditional_E32H256.csv` and `table_k24_paired_difference_conditional_E32H256.csv` | k=24 output `50`; paired CI 支持 C=800 优于 Basic PPO；C=1200 对 Basic PPO value/drops 统计相当、money 更好 | 最好地平衡可扩展性与动作耦合表达。 | final main model |
