# switching benchmark v1

## 1. 当前主 benchmark
- episode length = 300
- segment 1 = SL, 100 steps
- segment 2 = BH, 100 steps
- segment 3 = SH, 100 steps

## 2. 设计目的
这个 benchmark 用来测试：
- 当一个 episode 内部发生隐藏 regime 切换时，
- static policy 是否会出现适应滞后，
- context-aware policy 是否能更稳健。

## 3. 当前要求
- 切换点不告诉 agent
- 同一 seed 下可复现
- benchmark 生成逻辑固定
- Day3 只先做这一种模板，不扩展别的 switching 版本

## 4. 当前切换顺序
- 前 100 步：SL
- 中 100 步：BH
- 后 100 步：SH

## 5. 备注
后续如果要扩展：
- 可以尝试 SL -> BH -> BL
- 或者 BH -> SL -> SH
但在第一轮主实验完成前，先只保留当前这个 benchmark。

## 6. 当前实际生成结果
当前使用的 switching benchmark 为：

- sequence = SL -> BH -> SH
- weights = 20% / 50% / 30%
- total length = 300
- allocated lengths:
  - SL = 60
  - BH = 150
  - SH = 90

当前已生成文件：
- data/pools/SL_static_pool_E100_T300.npy
- data/pools/SH_static_pool_E100_T300.npy
- data/pools/BL_static_pool_E100_T300.npy
- data/pools/BH_static_pool_E100_T300.npy
- data/pools/MIXED_SLSHBLBH_pool_E400_T300.npy
- data/pools/SL_BH_SH_20_50_30_switch_pool_E100_T300.npy

