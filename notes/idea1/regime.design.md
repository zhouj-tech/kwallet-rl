# Flow Regime Design

## Regime A: Regular
- 直觉：
  平稳流，主要由小额和中额交易组成，整体波动不大，没有明显突发阶段。
- 生成思路：
  以 small + medium transaction 为主，比例比较稳定。
- 为什么重要：
  这是最基础、最正常的环境，可作为参考训练场景。

- 组成成分：以 small 和 medium 为主
- 时间结构：全程稳定，没有明显阶段变化
- 暂定思路：small 较多，medium 次之，large 很少或没有

## Regime B: Heavy-Tail
- 直觉：
  大多数时候是小额交易，但偶尔会出现较大的交易，尾部更重。
- 生成思路：
  以 small transaction 为主，加入少量 large transaction。
- 为什么重要：
  可以测试 policy 对偶发大额交易的鲁棒性。

- 组成成分：以 small 为主，加入少量 large
- 时间结构：全程都可能出现 large，但比较零散
- 暂定思路：大部分是 small，少数是 medium，更少数是 large tail

## Regime C: Burst
- 直觉：
  某一段时间里，大额交易明显变多，形成局部冲击。
- 生成思路：
  平时较平稳，但在一个时间窗口中提高 large transaction 比例。
- 为什么重要：
  可以测试 policy 对突发拥挤/冲击阶段的适应能力。

- 组成成分：平时 small/medium 为主，某一段 large 明显增多
- 时间结构：存在一个 burst window
- 暂定思路：前后较平稳，中间某段提高 large 比例

## Regime D: Switching
- 直觉：
  前后两个阶段的 flow 模式不同，例如前半段 regular，后半段 burst。
- 生成思路：
  按时间顺序拼接两种不同 regime。
- 为什么重要：
  可以直接测试 regime shift 下的性能退化和恢复能力。

- 组成成分：前后两段属于不同 regime
- 时间结构：明确分段切换
- 暂定思路：例如前半 Regular，后半 Burst

Regime A: Regular
- small: 70%
- medium: 30%
- large: 0%
- time structure: stable

Regime B: Heavy-Tail
- small: 75%
- medium: 20%
- large: 5%
- time structure: stable, large appears sparsely

Regime C: Burst
- normal phase: small 70%, medium 30%, large 0%
- burst phase: small 30%, medium 20%, large 50%
- time structure: one burst window in the middle

Regime D: Switching
- first half: same as Regular
- second half: same as Burst
- time structure: regime change at midpoint