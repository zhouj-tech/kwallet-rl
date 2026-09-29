# 实验方向总结

本报告只使用 Idea3/Idea4 的原始 `cross_regime_results.json` 作为主证据；聚合表只作为辅助。所有 rerun 保留，解释时同时参考去重后的 latest-per-setting。

## 1. 按时间线看实验演化

- 最早结果：2026-05-02T16:02:30.664677，来源 `src/idea3/results/old_fair_benchmark_results/main_MIX12_baseline_seed123_C1200_T1000_e300/runs/baseline_trainMIX12_EQ_C1200_k3_T1000_F3_seed123/20260502_155541_794309/cross_regime_results.json`。
- 最新结果：2026-05-17T21:18:16.086380，来源 `src/idea4/ac/results/dual_critic/runs/dual_branch_factorized_ac_gate_balanced_trainMIX12_EQ_C1200_k12_T1000_F3_seed532/20260517_203408_631821/cross_regime_results.json`。
- 大致演化：先有 DQN/attention 旧基线，然后进入 Factorized AC 与 Basic PPO 对比，再扩展到 Dual Branch、Gate-balanced、MaskA、money reward、Dual Critic 和 aux-risk。
- 详细逐 run 记录见 `experiment_timeline.csv`。

## 2. 按研究方向总结

### Basic PPO
- 假设：测试普通 PPO 是否已经足够强
- 直觉：如果 flat joint PPO 已经很强，复杂结构就必须证明自己有额外价值。
- 代码/脚本：`kwallet_basic_ppo_fair_benchmark.py`
- 设置范围：C1200.0_k12_F3_T1000_seed123; C1200.0_k12_F3_T1000_seed323; C1200.0_k12_F6_T1000_seed123; C1200.0_k24_F3_T1000_seed123; C1200.0_k24_F3_T1000_seed323; C1200.0_k3_F3_T1000_seed123; C1200.0_k3_F3_T1000_seed323; C1200.0_k6_F3_T1000_seed123; C1200.0_k6_F3_T1000_seed323; C800.0_k12_F3_T1000_seed123; C800.0_k12_F3_T1000_seed323; C800.0_k12_F6_T1000_seed123; C800.0_k24_F3_T1000_seed123; C800.0_k24_F3_T1000_seed323
- 最好结果：mean=99.96%，worst=99.75%，来源 `src/idea4/ac/results/basic_ppo/runs/basic_ppo_trainMIX12_EQ_C1200_k3_T1000_F3_seed323/20260507_213733_993592/cross_regime_results.json`
- 对比基线：`Factorized AC / DQN baseline`。基于 14 个同设置/同 seed 匹配，相对 `Factorized AC / DQN baseline` 平均高约 5.24 个百分点；若匹配数少，仍按弱证据处理。
- 证据强度：`tentative`。小差异不自动算胜利；跨 seed/setting 不一致时按弱证据处理。
- 建议：继续作为重点比较对象。
- 一句话结论：基于 14 个同设置/同 seed 匹配，相对 `Factorized AC / DQN baseline` 平均高约 5.24 个百分点；若匹配数少，仍按弱证据处理。

### Capacity-only Dual
- 假设：验证风险分支是否真的必要
- 直觉：只保留容量分支，看是否已经解释大部分收益。
- 代码/脚本：`run_dual_branch_ac_benchmark.py`
- 设置范围：C1200.0_k12_F3_T1000_seed123; C1200.0_k12_F6_T1000_seed123; C800.0_k12_F3_T1000_seed123; C800.0_k12_F6_T1000_seed123; C800.0_k24_F3_T1000_seed123
- 最好结果：mean=80.33%，worst=67.75%，来源 `src/idea4/ac/results/dual_branch_ac/runs/dual_branch_capacity_only_trainMIX12_EQ_C1200_k12_T1000_F3_seed123/20260508_191516_107694/cross_regime_results.json`
- 对比基线：`Original Dual Branch AC`。基于 5 个同设置/同 seed 匹配，相对 `Original Dual Branch AC` 的平均差异约 -0.21 个百分点，小于或接近波动，弱证据，尚不确定。
- 证据强度：`tentative`。小差异不自动算胜利；跨 seed/setting 不一致时按弱证据处理。
- 建议：作为消融保留，不建议扩大成主线。
- 一句话结论：基于 5 个同设置/同 seed 匹配，相对 `Original Dual Branch AC` 的平均差异约 -0.21 个百分点，小于或接近波动，弱证据，尚不确定。

### DQN baseline
- 假设：建立可比基线
- 直觉：先确认旧 DQN 在标准 cross-regime 上能到什么水平。
- 代码/脚本：`src/idea3/results/ 或 dqn_baseline 结果`
- 设置范围：C1200.0_k24_F3_T1000_seed123; C1200.0_k24_F3_T1000_seed323; C1200.0_k3_F3_T1000_seed123; C800.0_k12_F3_T1000_seed123; C800.0_k12_F3_T1000_seed323; C800.0_k24_F3_T1000_seed123; C800.0_k24_F3_T1000_seed323
- 最好结果：mean=93.26%，worst=91.30%，来源 `src/idea3/results/old_fair_benchmark_results/runs/baseline_trainMIX12_EQ_C1200_k3_T1000_F3_seed123/20260504_015024_611040/cross_regime_results.json`
- 对比基线：`none`。基线方向，不做胜负判断。
- 证据强度：`tentative`。小差异不自动算胜利；跨 seed/setting 不一致时按弱证据处理。
- 建议：作为参照保留。
- 一句话结论：基线方向，不做胜负判断。

### Dual Critic
- 假设：分别训练容量/风险价值头
- 直觉：希望 critic 更懂两个分支的职责。
- 代码/脚本：`run_dual_branch_ac_dual_critic.py`
- 设置范围：C1200.0_k12_F3_T1000_seed123; C1200.0_k12_F3_T1000_seed323; C1200.0_k12_F3_T1000_seed532; C1200.0_k12_F3_T1000_seed777
- 最好结果：mean=81.29%，worst=68.26%，来源 `src/idea4/ac/results/dual_critic/runs/dual_branch_factorized_ac_gate_balanced_trainMIX12_EQ_C1200_k12_T1000_F3_seed123_dualCritic_coef0p1/20260516_085346_184336/cross_regime_results.json`
- 对比基线：`Gate-balanced Dual Branch AC`。基于 4 个同设置/同 seed 匹配，相对 `Gate-balanced Dual Branch AC` 平均低约 1.60 个百分点，当前更像退化。
- 证据强度：`strong`。小差异不自动算胜利；跨 seed/setting 不一致时按弱证据处理。
- 建议：停止作为主线；只保留小规模变体复查。
- 一句话结论：基于 4 个同设置/同 seed 匹配，相对 `Gate-balanced Dual Branch AC` 平均低约 1.60 个百分点，当前更像退化。

### Dual Critic + Aux Risk
- 假设：用辅助风险任务逼出风险表征
- 直觉：让风险分支预测未来 drop 压力。
- 代码/脚本：`run_dual_branch_ac_dual_critic.py`
- 设置范围：C1200.0_k12_F3_T1000_seed123; C1200.0_k12_F3_T1000_seed323; C1200.0_k12_F3_T1000_seed532; C1200.0_k12_F3_T1000_seed777
- 最好结果：mean=82.06%，worst=69.25%，来源 `src/idea4/ac/results/dual_critic/runs/dual_branch_factorized_ac_auxrisk_trainMIX12_EQ_C1200_k12_T1000_F3_seed123_dualCritic_coef0p1/20260516_153742_607618/cross_regime_results.json`
- 对比基线：`Gate-balanced Dual Branch AC`。基于 4 个同设置/同 seed 匹配，相对 `Gate-balanced Dual Branch AC` 平均低约 1.41 个百分点，当前更像退化。
- 证据强度：`strong`。小差异不自动算胜利；跨 seed/setting 不一致时按弱证据处理。
- 建议：停止作为主线；只保留小规模变体复查。
- 一句话结论：基于 4 个同设置/同 seed 匹配，相对 `Gate-balanced Dual Branch AC` 平均低约 1.41 个百分点，当前更像退化。

### Factorized AC
- 假设：拆动作空间，缓解 k 增大后的输出爆炸
- 直觉：把 settle/flush 分头建模，降低策略输出规模。
- 代码/脚本：`run_factorized_ac_benchmark.py`
- 设置范围：C1200.0_k12_F3_T1000_seed123; C1200.0_k12_F3_T1000_seed323; C1200.0_k12_F3_T1000_seed532; C1200.0_k12_F3_T1000_seed999; C1200.0_k12_F6_T1000_seed123; C1200.0_k24_F3_T1000_seed123; C1200.0_k24_F3_T1000_seed323; C1200.0_k3_F3_T1000_seed123; C1200.0_k3_F3_T1000_seed323; C1200.0_k6_F3_T1000_seed123; C1200.0_k6_F3_T1000_seed323; C800.0_k12_F3_T1000_seed123; C800.0_k12_F3_T1000_seed323; C800.0_k12_F6_T1000_seed123; C800.0_k24_F3_T1000_seed123; C800.0_k24_F3_T1000_seed323
- 最好结果：mean=99.51%，worst=98.08%，来源 `src/idea4/ac/results/factorized_ac/runs/factorized_ac_trainMIX12_EQ_C1200_k3_T1000_F3_seed123/20260505_212833_781649/cross_regime_results.json`
- 对比基线：`DQN baseline`。基于 7 个同设置/同 seed 匹配，相对 `DQN baseline` 平均高约 14.48 个百分点；若匹配数少，仍按弱证据处理。
- 证据强度：`strong`。小差异不自动算胜利；跨 seed/setting 不一致时按弱证据处理。
- 建议：继续作为重点比较对象。
- 一句话结论：基于 7 个同设置/同 seed 匹配，相对 `DQN baseline` 平均高约 14.48 个百分点；若匹配数少，仍按弱证据处理。

### Gate-balanced Dual Branch AC
- 假设：防止 gate 偏向单分支
- 直觉：给 gate 加目标/边界，希望容量和风险分支都被使用。
- 代码/脚本：`run_dual_branch_ac_benchmark.py`
- 设置范围：C1200.0_k12_F3_T1000_seed123; C1200.0_k12_F3_T1000_seed323; C1200.0_k12_F3_T1000_seed532; C1200.0_k12_F3_T1000_seed777; C1200.0_k12_F3_T1000_seed999; C1200.0_k24_F3_T1000_seed123
- 最好结果：mean=83.13%，worst=69.84%，来源 `src/idea4/ac/results/dual_branch_ac/runs/dual_branch_factorized_ac_gate_balanced_trainMIX12_EQ_C1200_k12_T1000_F3_seed323/20260506_211708_693134/cross_regime_results.json`
- 对比基线：`Original Dual Branch AC`。基于 7 个同设置/同 seed 匹配，相对 `Original Dual Branch AC` 平均高约 1.71 个百分点；若匹配数少，仍按弱证据处理。
- 证据强度：`strong`。小差异不自动算胜利；跨 seed/setting 不一致时按弱证据处理。
- 建议：继续作为重点比较对象。
- 一句话结论：基于 7 个同设置/同 seed 匹配，相对 `Original Dual Branch AC` 平均高约 1.71 个百分点；若匹配数少，仍按弱证据处理。

### MaskA / Masked Factorized AC
- 假设：用硬/软动作 mask 注入可行性先验
- 直觉：希望减少明显不可行动作，但不改变环境语义。
- 代码/脚本：`kwallet_maskA_factorized_ac.py`
- 设置范围：C800.0_k24_F3_T1000_seed123
- 最好结果：mean=11.25%，worst=5.90%，来源 `src/idea4/ac/results/factorized_ac/runs/maskA_factorized_ac_trainMIX12_EQ_C800_k24_T1000_F3_seed123_MaskA_hard/20260510_120212_307891/cross_regime_results.json`
- 对比基线：`Factorized AC`。基于 2 个同设置/同 seed 匹配，相对 `Factorized AC` 平均低约 4.25 个百分点，当前更像退化。
- 证据强度：`tentative`。小差异不自动算胜利；跨 seed/setting 不一致时按弱证据处理。
- 建议：停止作为主线；只保留小规模变体复查。
- 一句话结论：基于 2 个同设置/同 seed 匹配，相对 `Factorized AC` 平均低约 4.25 个百分点，当前更像退化。

### Original Dual Branch AC
- 假设：把容量约束和未来风险分开建模
- 直觉：双分支可能比单一表示更能处理钱包压力。
- 代码/脚本：`run_dual_branch_ac_benchmark.py`
- 设置范围：C1200.0_k12_F3_T1000_seed123; C1200.0_k12_F3_T1000_seed323; C1200.0_k12_F3_T1000_seed532; C1200.0_k12_F3_T1000_seed999; C1200.0_k12_F6_T1000_seed123; C1200.0_k3_F3_T1000_seed123; C1200.0_k3_F3_T1000_seed323; C1200.0_k6_F3_T1000_seed123; C1200.0_k6_F3_T1000_seed323; C800.0_k12_F3_T1000_seed123; C800.0_k12_F6_T1000_seed123; C800.0_k24_F3_T1000_seed123
- 最好结果：mean=99.55%，worst=98.09%，来源 `src/idea4/ac/results/dual_branch_ac/runs/dual_branch_factorized_ac_trainMIX12_EQ_C1200_k3_T1000_F3_seed323/20260505_165828_143067/cross_regime_results.json`
- 对比基线：`Factorized AC`。基于 15 个同设置/同 seed 匹配，相对 `Factorized AC` 的平均差异约 -0.08 个百分点，小于或接近波动，弱证据，尚不确定。
- 证据强度：`strong`。小差异不自动算胜利；跨 seed/setting 不一致时按弱证据处理。
- 建议：作为参照保留。
- 一句话结论：基于 15 个同设置/同 seed 匹配，相对 `Factorized AC` 的平均差异约 -0.08 个百分点，小于或接近波动，弱证据，尚不确定。

### Residual Risk Dual
- 假设：用小残差修正容量分支
- 直觉：风险只做有限修正，避免整套双分支不稳定。
- 代码/脚本：`run_dual_branch_ac_benchmark.py`
- 设置范围：C1200.0_k12_F3_T1000_seed123; C1200.0_k12_F6_T1000_seed123; C800.0_k12_F3_T1000_seed123; C800.0_k12_F6_T1000_seed123; C800.0_k24_F3_T1000_seed123
- 最好结果：mean=80.87%，worst=67.95%，来源 `src/idea4/ac/results/dual_branch_ac/runs/dual_branch_residual_risk_trainMIX12_EQ_C1200_k12_T1000_F3_seed123/20260508_194136_365825/cross_regime_results.json`
- 对比基线：`Original Dual Branch AC`。基于 5 个同设置/同 seed 匹配，相对 `Original Dual Branch AC` 的平均差异约 0.76 个百分点，小于或接近波动，弱证据，尚不确定。
- 证据强度：`tentative`。小差异不自动算胜利；跨 seed/setting 不一致时按弱证据处理。
- 建议：作为消融保留，不建议扩大成主线。
- 一句话结论：基于 5 个同设置/同 seed 匹配，相对 `Original Dual Branch AC` 的平均差异约 0.76 个百分点，小于或接近波动，弱证据，尚不确定。

### Reward shaping / money reward
- 假设：直接优化钱的目标
- 直觉：看真实金额收益是否比原始 reward 更合适。
- 代码/脚本：`basic/factorized/dual reward_mode=money`
- 设置范围：C1200.0_k12_F3_T1000_seed123; C1200.0_k12_F3_T1000_seed323
- 最好结果：mean=0.38%，worst=0.35%，来源 `src/idea4/ac/results/dual_branch_ac/runs/dual_branch_factorized_ac_trainMIX12_EQ_C1200_k12_T1000_F3_seed323_rewardMONEY_p1_tau10/20260508_135514_790066/cross_regime_results.json`
- 对比基线：`same model under original reward`。基于 5 个同设置/同 seed 匹配，相对 `same model under original reward` 平均低约 82.59 个百分点，当前更像退化。
- 证据强度：`tentative`。小差异不自动算胜利；跨 seed/setting 不一致时按弱证据处理。
- 建议：停止作为主线；只保留小规模变体复查。
- 一句话结论：基于 5 个同设置/同 seed 匹配，相对 `same model under original reward` 平均低约 82.59 个百分点，当前更像退化。

## 3. 失败或负向方向

### MaskA / Masked Factorized AC
- 试了什么：3 个 completed run，代表来源 `src/idea4/ac/results/factorized_ac/runs/maskA_factorized_ac_trainMIX12_EQ_C800_k24_T1000_F3_seed123_MaskA_hard/20260510_120212_307891/cross_regime_results.json`。
- 观察结果：最好 mean=11.25%，平均表现见 `direction_mean_summary.csv`。基于 2 个同设置/同 seed 匹配，相对 `Factorized AC` 平均低约 4.25 个百分点，当前更像退化。
- 可能原因：硬/软 mask 可能限制了探索，且可行性先验未必等价于长期收益。
- 是否值得小改：可以小规模试一个更温和版本，但不建议作为当前主线。

### Dual Critic
- 试了什么：4 个 completed run，代表来源 `src/idea4/ac/results/dual_critic/runs/dual_branch_factorized_ac_gate_balanced_trainMIX12_EQ_C1200_k12_T1000_F3_seed123_dualCritic_coef0p1/20260516_085346_184336/cross_regime_results.json`。
- 观察结果：最好 mean=81.29%，平均表现见 `direction_mean_summary.csv`。基于 4 个同设置/同 seed 匹配，相对 `Gate-balanced Dual Branch AC` 平均低约 1.60 个百分点，当前更像退化。
- 可能原因：coef=0.1 可能把 critic 训练目标拉复杂，但没有让 actor 获得稳定收益。
- 是否值得小改：可以小规模试一个更温和版本，但不建议作为当前主线。

### Dual Critic + Aux Risk
- 试了什么：4 个 completed run，代表来源 `src/idea4/ac/results/dual_critic/runs/dual_branch_factorized_ac_auxrisk_trainMIX12_EQ_C1200_k12_T1000_F3_seed123_dualCritic_coef0p1/20260516_153742_607618/cross_regime_results.json`。
- 观察结果：最好 mean=82.06%，平均表现见 `direction_mean_summary.csv`。基于 4 个同设置/同 seed 匹配，相对 `Gate-balanced Dual Branch AC` 平均低约 1.41 个百分点，当前更像退化。
- 可能原因：aux-risk 目标可能和原 reward 的主优化目标不完全一致，增加了训练干扰。
- 是否值得小改：可以小规模试一个更温和版本，但不建议作为当前主线。

### Reward shaping / money reward
- 试了什么：5 个 completed run，代表来源 `src/idea4/ac/results/dual_branch_ac/runs/dual_branch_factorized_ac_trainMIX12_EQ_C1200_k12_T1000_F3_seed323_rewardMONEY_p1_tau10/20260508_135514_790066/cross_regime_results.json`。
- 观察结果：最好 mean=0.38%，平均表现见 `direction_mean_summary.csv`。基于 5 个同设置/同 seed 匹配，相对 `same model under original reward` 平均低约 82.59 个百分点，当前更像退化。
- 可能原因：直接 money reward 的尺度和稀疏性可能破坏了原 reward 下已学到的稳定策略。
- 是否值得小改：可以小规模试一个更温和版本，但不建议作为当前主线。

## 4. 当前最强模型

- 按 平均 value acceptance：`Basic PPO` 最好，值=0.9996，来源 `src/idea4/ac/results/basic_ppo/runs/basic_ppo_trainMIX12_EQ_C1200_k3_T1000_F3_seed323/20260507_213733_993592/cross_regime_results.json`。
- 按 最差 regime value acceptance：`Basic PPO` 最好，值=0.9975，来源 `src/idea4/ac/results/basic_ppo/runs/basic_ppo_trainMIX12_EQ_C1200_k3_T1000_F3_seed323/20260507_213733_993592/cross_regime_results.json`。
- 按 平均 eval money：`Gate-balanced Dual Branch AC` 最好，值=34375.0117，来源 `src/idea4/ac/results/dual_critic/runs/dual_branch_factorized_ac_gate_balanced_trainMIX12_EQ_C1200_k12_T1000_F3_seed323/20260516_015724_218963/cross_regime_results.json`。
- 大动作/压力设置：`Basic PPO` 当前最好，mean=64.98%，来源 `src/idea4/ac/results/stress_tests/basic_ppo/runs/basic_ppo_trainMIX12_EQ_C800_k12_T1000_F3_seed323/20260508_023233_752609/cross_regime_results.json`。
- Basic PPO vs Gate-balanced Dual：逐设置赢家见 `best_model_by_setting_seed_level.csv` 和 `best_model_by_setting_aggregate.csv`；若差异接近 seed 波动，不视为决定性胜利。

## 5. Gate 行为分析

- gate 诊断见 `gate_behavior_summary.csv`，代表来源 `src/idea4/ac/results/dual_branch_ac/runs/dual_branch_factorized_ac_trainMIX12_EQ_C1200_k3_T1000_F3_seed123/20260505_155416_193647/cross_regime_results.json`。
- 平均 gate_std=0.001142，near_upper_rate=0.878，value_disagreement=1.2520。
- 目前很多 dual run 的 gate_mean 接近上边界，state-dependent 使用不明显；这是 gate-collapse 的直接证据之一。
- value disagreement 有记录的主要在 dual critic 系列，但它没有自动转化为更好策略，因此“分支专门化有效”仍是弱证据。

## 6. 最终研究建议

- 当前主故事：Factorized/PPO 系列解决动作空间扩展问题；Dual Branch 的动机合理，但 gate collapse 让“风险分支是否真的工作”仍未完全证明。
- 最强基线：Basic PPO 和 Factorized AC 都必须保留；具体设置下以 `best_model_by_setting_aggregate.csv` 为准。
- 最强当前模型：按原始结果通常是低 k 设置下的 Basic PPO/Factorized/Dual 系列；压力设置需要单独看表，不混在一起下结论。
- 应停止：MaskA 硬 masking、money reward 主线、dual critic coef=0.1 主线。
- 应继续：gate-collapse 诊断、温和 gate variance/entropy 约束、同设置多 seed 的 Basic PPO vs Gate-balanced 对比。
- 最多 3 个下一步实验：1) gate variance/anti-collapse 小系数；2) dual critic coef 更小如 0.01；3) 同 C/k/F/T 下补齐 Basic PPO、Factorized、Gate-balanced 的 3 seed 对比。

# 当前最可信的结论

## A. 强支持结论

- `Dual Critic` 的证据量较足，但不等于方向一定有效；当前判断是：基于 4 个同设置/同 seed 匹配，相对 `Gate-balanced Dual Branch AC` 平均低约 1.60 个百分点，当前更像退化。 具体数值见 `direction_mean_summary.csv`，最好来源 `src/idea4/ac/results/dual_critic/runs/dual_branch_factorized_ac_gate_balanced_trainMIX12_EQ_C1200_k12_T1000_F3_seed123_dualCritic_coef0p1/20260516_085346_184336/cross_regime_results.json`。
- `Dual Critic + Aux Risk` 的证据量较足，但不等于方向一定有效；当前判断是：基于 4 个同设置/同 seed 匹配，相对 `Gate-balanced Dual Branch AC` 平均低约 1.41 个百分点，当前更像退化。 具体数值见 `direction_mean_summary.csv`，最好来源 `src/idea4/ac/results/dual_critic/runs/dual_branch_factorized_ac_auxrisk_trainMIX12_EQ_C1200_k12_T1000_F3_seed123_dualCritic_coef0p1/20260516_153742_607618/cross_regime_results.json`。
- `Factorized AC` 的证据量较足，但不等于方向一定有效；当前判断是：基于 7 个同设置/同 seed 匹配，相对 `DQN baseline` 平均高约 14.48 个百分点；若匹配数少，仍按弱证据处理。 具体数值见 `direction_mean_summary.csv`，最好来源 `src/idea4/ac/results/factorized_ac/runs/factorized_ac_trainMIX12_EQ_C1200_k3_T1000_F3_seed123/20260505_212833_781649/cross_regime_results.json`。
- `Gate-balanced Dual Branch AC` 的证据量较足，但不等于方向一定有效；当前判断是：基于 7 个同设置/同 seed 匹配，相对 `Original Dual Branch AC` 平均高约 1.71 个百分点；若匹配数少，仍按弱证据处理。 具体数值见 `direction_mean_summary.csv`，最好来源 `src/idea4/ac/results/dual_branch_ac/runs/dual_branch_factorized_ac_gate_balanced_trainMIX12_EQ_C1200_k12_T1000_F3_seed323/20260506_211708_693134/cross_regime_results.json`。
- `Original Dual Branch AC` 的证据量较足，但不等于方向一定有效；当前判断是：基于 15 个同设置/同 seed 匹配，相对 `Factorized AC` 的平均差异约 -0.08 个百分点，小于或接近波动，弱证据，尚不确定。 具体数值见 `direction_mean_summary.csv`，最好来源 `src/idea4/ac/results/dual_branch_ac/runs/dual_branch_factorized_ac_trainMIX12_EQ_C1200_k3_T1000_F3_seed323/20260505_165828_143067/cross_regime_results.json`。

## B. 暂定假设

- Gate-balanced 的结构动机仍值得保留，但 gate 接近边界说明风险分支未必被真正使用。
- Basic PPO 在部分设置很强，可能是当前最硬的工程基线；但跨压力设置不能直接外推。

## C. 推测性想法

- learnable masking、gate variance regularization、更小 dual critic coef 都还没有形成充分证据，只能作为下一步小实验。
