# 未完成/暂停方向

这些方向不等于完全失败；它们只是证据不足、暂时暂停，或需要更小的诊断实验。

## gate sweep
- evidence_strength：`tentative`
- 原始动机：想找到不 collapse 的 gate 设置。
- 已部分测试：已有 gate-balanced/gate-regularized 痕迹，但系统 sweep 证据不足。
- 为什么停下：当前 gate 多接近边界，先暂停扩大。
- 当前状态：部分完成
- 是否值得以后重访：值得，用小网格和 gate variance 指标复查。
- 证据入口：`experiment_timeline.csv`、`direction_summary.md`。

## learnable masking
- evidence_strength：`speculative`
- 原始动机：希望比硬 MaskA 更温和地注入可行性先验。
- 已部分测试：当前主要是 MaskA hard/soft，learnable mask 没有充分完成。
- 为什么停下：MaskA 负结果后优先级下降。
- 当前状态：暂停
- 是否值得以后重访：可晚点重访，但必须避免硬剪探索。
- 证据入口：`experiment_timeline.csv`、`direction_summary.md`。

## smaller dual critic coef
- evidence_strength：`tentative`
- 原始动机：coef=0.1 可能太强，想测试更小辅助损失。
- 已部分测试：已有 dual critic coef=0.1。
- 为什么停下：0.1 没有证明收益，先不扩大。
- 当前状态：暂停
- 是否值得以后重访：值得小试 0.01，但只做诊断实验。
- 证据入口：`experiment_timeline.csv`、`direction_summary.md`。

## reward shaping variants
- evidence_strength：`tentative`
- 原始动机：直接优化 money 或 hybrid money。
- 已部分测试：已有 rewardMONEY/tau10 等结果。
- 为什么停下：当前 value acceptance 明显退化，money 尺度可能不稳。
- 当前状态：基本停止
- 是否值得以后重访：除非重做尺度归一化，否则不建议主线继续。
- 证据入口：`experiment_timeline.csv`、`direction_summary.md`。

## gate variance regularization
- evidence_strength：`speculative`
- 原始动机：直接惩罚 gate collapse，让 gate 随状态变化。
- 已部分测试：目前更多是 gate target/balance，不是明确 variance objective。
- 为什么停下：尚未形成完整实验。
- 当前状态：尚未开始
- 是否值得以后重访：值得作为下一步三实验之一。
- 证据入口：`experiment_timeline.csv`、`direction_summary.md`。
