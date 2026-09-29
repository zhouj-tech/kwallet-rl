# 研究问题清单

这个文件把问题和证据分开，避免把直觉写成结论。

## Factorized AC 是否解决了动作空间扩展问题？
- 当前证据状态：`partially supported`
- 最强支持实验：Factorized AC 与 DQN/Basic PPO 的 k scaling 结果；见 `experiment_timeline.csv` 和 `direction_summary.md`。
- 未解决不确定性：压力设置和 seed 覆盖仍不均衡
- 推荐下一步实验：补齐 C/k/F/T 相同的 3 seed 对比。

## Basic PPO 是否已经是最强基线？
- 当前证据状态：`partially supported`
- 最强支持实验：Basic PPO 在若干低 k run 中表现很强；见 `experiment_timeline.csv` 和 `direction_summary.md`。
- 未解决不确定性：不同 k/C/F 下未必一致
- 推荐下一步实验：按设置输出 winner 表后补缺 seed。

## Dual Branch 的风险分支是否真的有用？
- 当前证据状态：`unclear`
- 最强支持实验：Gate-balanced run 可运行且有较好结果；见 `experiment_timeline.csv` 和 `direction_summary.md`。
- 未解决不确定性：gate 接近边界，分支专门化证据弱
- 推荐下一步实验：做 gate variance/anti-collapse 诊断实验。

## MaskA 硬 masking 是否有帮助？
- 当前证据状态：`contradicted`
- 最强支持实验：MaskA 当前最好结果弱于对应 Factorized/Basic PPO 压力设置；见 `experiment_timeline.csv` 和 `direction_summary.md`。
- 未解决不确定性：只测了少量 hard/soft 形式
- 推荐下一步实验：除非改成 learnable/soft prior，否则停止主线。

## Dual Critic coef=0.1 是否提升 Gate-balanced？
- 当前证据状态：`contradicted`
- 最强支持实验：dual critic/aux-risk run 没有压过 gate-balanced baseline；见 `experiment_timeline.csv` 和 `direction_summary.md`。
- 未解决不确定性：coef 可能过大，且只代表一种损失权重
- 推荐下一步实验：只小试 coef=0.01，不再大规模跑 0.1。

## Money reward 是否应作为主目标？
- 当前证据状态：`contradicted`
- 最强支持实验：rewardMONEY run 的 value acceptance 明显崩掉；见 `experiment_timeline.csv` 和 `direction_summary.md`。
- 未解决不确定性：money objective 的尺度和评价目标冲突仍未完全拆开
- 推荐下一步实验：若重试，只做 normalized/hybrid 小实验。
