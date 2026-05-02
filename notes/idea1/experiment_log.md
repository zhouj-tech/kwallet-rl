# Experiment Log

## 2026-03-24

### Run ID: setup-001
- 做了什么：
  - 归档了旧方向代码到 legacy
  - 建立了新的 src / configs / results 目录
  - 准备开始新实验框架

- 结果：
  - 旧代码已从主工作区移开
  - 新项目结构初步建立

- 问题：
  - configs 还没补齐
  - regime design 还没开始写

- 下一步：
  - 建立 smoke / quick / full 三档配置
  - 写 regime_design 第一版

###3.25
  Train on B:
- cross-regime 结果显示 A 上最好，B 次之，D 再次，C 最差
- 该结果说明 Regular 仍是最容易环境，Burst 仍是最难环境
- 当前阶段尚未观察到明显的 “train on B -> best on B” specialization
- 初步推测：环境本身难度差异目前强于训练 regime 匹配效应