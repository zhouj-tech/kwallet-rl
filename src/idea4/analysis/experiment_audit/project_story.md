# 项目研究故事

这不是逐日流水账，而是按研究阶段记录：为什么做、试了什么、学到了什么、为什么进入下一阶段。

## 阶段 1：DQN 基线与可扩展性问题
先用旧 DQN/attention 路线建立参照。随着 k 增大，flat 动作空间会迅速变大，DQN 的可扩展性成为问题。
证据入口：`src/idea3/results/old_fair_benchmark_results/runs/baseline_trainMIX12_EQ_C1200_k3_T1000_F3_seed123/20260504_015024_611040/cross_regime_results.json`。

## 阶段 2：Factorized AC 改进
把动作拆成 settle/flush 两个头，核心动机是降低输出规模。结果显示它在若干 k 设置下能保持竞争力，因此成为新主线之一。
证据入口：`src/idea4/ac/results/factorized_ac/runs/factorized_ac_trainMIX12_EQ_C1200_k3_T1000_F3_seed123/20260505_212833_781649/cross_regime_results.json`。

## 阶段 3：Basic PPO 对比
Basic PPO 用更直接的 actor-critic 训练方式检验：是不是不需要复杂 factorization/dual branch 也能跑好。它在部分低 k 设置很强，所以必须作为强基线。
证据入口：`src/idea4/ac/results/basic_ppo/runs/basic_ppo_trainMIX12_EQ_C1200_k3_T1000_F3_seed323/20260507_213733_993592/cross_regime_results.json`。

## 阶段 4：Dual Branch 探索
Dual Branch 试图区分容量约束和未来风险。原始版本证明这个想法可以跑通，但不等于证明两个分支真的分工。
证据入口：`experiment_timeline.csv`。

## 阶段 5：Gate-balanced 突破与问题
Gate-balanced 试图约束 gate，避免单分支垄断。结果让 dual branch 进入可比较范围，但 gate 诊断也暴露了接近边界的问题。
证据入口：`src/idea4/ac/results/dual_branch_ac/runs/dual_branch_factorized_ac_gate_balanced_trainMIX12_EQ_C1200_k12_T1000_F3_seed323/20260506_211708_693134/cross_regime_results.json`。

## 阶段 6：MaskA 与硬 masking 失败
MaskA 把可行性先验硬塞进动作 logits。当前结果不理想，可能因为它减少探索，且短期可行不等于长期收益。
证据入口：`src/idea4/ac/results/factorized_ac/runs/maskA_factorized_ac_trainMIX12_EQ_C800_k24_T1000_F3_seed123_MaskA_hard/20260510_120212_307891/cross_regime_results.json`。

## 阶段 7：Dual critic 与 aux-risk 实验
Dual critic/aux-risk 想让风险分支学到更明确的信号。当前 coef=0.1 方向没有压过 gate-balanced baseline，属于负结果。
证据入口：`src/idea4/ac/results/dual_critic/runs/dual_branch_factorized_ac_auxrisk_trainMIX12_EQ_C1200_k12_T1000_F3_seed123_dualCritic_coef0p1/20260516_153742_607618/cross_regime_results.json`。

## 阶段 8：当前 gate-collapse 调查
gate_mean 接近上边界、gate_std 很小，说明 gate 很可能没有按状态动态切换。下一步应先解决 collapse，再谈分支专门化。
证据入口：`src/idea4/ac/results/dual_branch_ac/runs/dual_branch_factorized_ac_gate_balanced_trainMIX12_EQ_C1200_k12_T1000_F3_seed323/20260506_211708_693134/cross_regime_results.json`。

## 阶段 9：当前最强方向与下一步假设
当前最可信做法是保留 Basic PPO/Factorized/Gate-balanced 三条可比线，停止大规模 MaskA/money/dual-critic 主线，只做小而可诊断的变体。
证据入口：`experiment_timeline.csv`。
