# K-Wallet 代码设定与实验协议备忘录
3/31 

## 0. 这个文件是干什么的

这个文件用来记录当前 k-wallet 项目的核心代码设定，避免后面出现以下问题：

* 改了代码但忘了原本语义
* 结果变了却不知道是环境变了还是 agent 变了
* 写实验记录时说不清 state、action、reward 的定义
* 跑了很多次实验后混淆主协议和临时测试设定


## 1. 当前项目主线


在 regime 标签未知、存在 hidden switching 的 k-wallet 环境中，context-aware policy 是否比 static policy 更稳健、更可部署。

当前优先做的主方法是：

lightweight context-aware DQN

当前优先比较的 baseline 是：

* static generalist
* BH specialist
* oracle specialist
* classifier-router（作为 baseline，不作为主角）

当前不做的内容：

* PPO / A2C / SAC 等换 agent family
* RNN / Transformer / 大型 encoder
* 大规模扫参数
* 频繁改 reward 设计
* 同时做太多 switching 模板

---

## 2. 当前环境核心参数

环境核心参数包括：

* C：总容量
* k：wallet 数量
* wallet_size = C / k
* F：flush 后冻结时长
* T / max_transaction：用于归一化奖励和 state 的最大交易金额
* max_steps：每个 episode 的最大步数

要记住：

k 会同时影响：

* 单钱包容量
* state 维度
* action 空间大小
* 问题难度
* agent 决策自由度

F 会影响：

* flush 的代价
* 错误决策的持续影响
* hidden switching 下适应能力是否容易被看出来

---

## 3. 当前 state 定义

当前 state 由四部分组成：

第一部分：每个 wallet 的余额（归一化后）
第二部分：每个 wallet 是否可用
第三部分：每个 wallet 的剩余冻结时间（归一化后）
第四部分：当前 transaction 金额（归一化后）

如果 k=3，则 state 维度是：

3k + 1 = 10

更一般地说：

state_dim = 3k + 1

注意：
当前 state 只包含“当前时刻信息”，还不是 context-aware state。
以后如果要做 context-aware，需要在 state 中额外加入 recent-window features。

## 4. 当前动作空间定义

当前主动作空间使用：

(k + 1)^2

一个整数动作会被解码成两个部分：

* settle_choice
* flush_choice

其中：

settle_choice 的取值：

* 0 到 k-1：表示把当前交易尝试放入某个 wallet
* k：表示本步不结算

flush_choice 的取值：

* 0 到 k-1：表示刷新某个 wallet
* k：表示本步不刷新

动作解码方式是：

base = k + 1
settle_choice = action // base
flush_choice = action % base

例如当 k=3 时，base=4：

* action=3 -> (settle=0, flush=3)
* action=13 -> (settle=3, flush=1)
* action=15 -> (settle=3, flush=3)

以后如果改动作空间，这一节必须同步更新。

---

## 5. Agent 选动作规则

训练时，agent 使用 epsilon-greedy：

* 以 epsilon 概率随机选动作
* 以 1-epsilon 概率选当前 Q 值最大的动作

也就是说：

前期更偏探索，后期更偏利用。

评估时应该使用 greedy 策略，也就是：

* 不随机
* 直接选 argmax Q

注意区分两层含义：

第一层：agent 怎么选动作
这是 DQN 的事，取决于 Q 网络和 epsilon-greedy。

第二层：动作被环境怎么解释
这是 environment 的事，取决于 step() 和动作解码逻辑。

不要把这两层混在一起。

---

## 6. 当前 step 执行顺序

当前环境里，一个动作进入 step() 后，执行顺序固定为：

第一步：先 flush
第二步：再 settle
第三步：计算奖励与惩罚
第四步：时间推进
第五步：检查冻结结束的钱包是否补满
第六步：推进到下一笔 transaction

这个顺序很重要，后面如果改 step 逻辑，必须重新做 sanity check。

---

## 7. flush 规则

如果 flush_choice < k，则本步尝试刷新一个 wallet。

刷新成功的条件：

* 该 wallet 当前必须可用

刷新成功后会发生：

* 该 wallet 余额变为 0
* pending_refill 设为 True
* freeze_until 被设置
* num_flushes 增加

如果试图刷新一个冻结中的 wallet：

* 不会真的刷新成功
* 如果开启 shaping，会有额外惩罚

要注意：
flush 不是“自动发生”的，而是 agent 可以主动做的动作。

---

## 8. settle 规则

如果 settle_choice < k，则本步尝试把当前交易放入某个 wallet。

成功结算的条件必须同时满足：

1. 该 wallet 当前可用
2. 该 wallet 不是本步刚刚 flush 的那个 wallet
3. 该 wallet 余额足够覆盖当前交易金额

如果成功：

* 钱包余额减少
* total_settled 增加
* total_accepted 增加
* 获得正奖励

如果失败：

* 该交易被 drop
* 可能会记作 insufficient drop
* 获得负奖励

---

## 9. 特殊动作语义

### 9.1 不结算且不刷新

如果动作是：

settle_choice = k
flush_choice = k

则表示：

* 本步不结算
* 本步不刷新

在当前实现中，这等价于：
当前交易直接被 drop。

这不是 bug，而是当前动作设计的一部分。

以后如果觉得这个语义不合理，需要明确修改，而不是默认忽略。

---

### 9.2 同一步 flush 和 settle 同一个钱包

如果动作同时指定：

* flush_choice = i
* settle_choice = i

则当前实现中：

* flush 优先
* 当前交易不会在同一步被放入该钱包
* 这笔交易会 drop

这已经通过 sanity check 验证过。
这不是 bug，是当前动作语义的一部分。

---

### 9.3 oversize 交易

如果当前交易金额大于单个钱包容量，即：

tx > wallet_size

则这笔交易会被直接判定为 oversize drop。

当前实现中：

* oversize_dropped = True
* accepted = False
* 钱包余额不会改变
* 环境会推进到下一笔交易

注意：
当前实现里 oversize_dropped 和 dropped 是分开统计的。
以后算总 drop rate 时，要明确口径。

---

## 10. 当前 reward 设计

当前 reward 的核心部分包括：

正向部分：

* 成功结算时，奖励大致与 transaction 大小成正比

负向部分：

* 普通 drop penalty
* flush cost

可选 shaping 部分（如果 enable_shaping=True）可能还包括：

* invalid action penalty
* imbalance penalty
* wasteful refresh penalty

做环境 sanity check 时，建议先把：
enable_shaping = False

这样更方便手算和核对主逻辑。

正式实验前要确认：

* 主实验是否开启 shaping
* 一旦决定，不要随意改

---

## 11. 当前 DQN 的层次与定位

当前这份 agent 从方法层次上看，属于：

一个基于手工 state、手工 reward、自定义动作空间的标准 DQN baseline 系统

更具体地说：

* 不是 context-aware DQN（除非后面改 state）
* 不是 recurrent DQN
* 不是 meta-RL
* 不是 regime inference model
* 不是 classifier-router system

它现在的作用是：
作为 k-wallet 问题上的强 baseline 和后续改进的基础框架。

---

## 12. 当前代码里最该锁住的部分

真正和主实验协议强耦合的，不是所有代码，而是下面这些：

1. state 定义
2. action 空间定义
3. reward 设计
4. step() 的执行顺序
5. hidden switching benchmark 生成方式
6. 主 setting 的 k、F、C、T
7. 训练预算与评估方式

这些一旦进入主实验阶段，就不要随便改。

---

## 13. Day2 sanity check 已确认通过的内容

当前已经人工验证通过的内容包括：

1. reset 初始状态正确
2. 普通结算逻辑正确
3. flush 逻辑正确
4. delayed refill 逻辑正确
5. episode 结束逻辑正确
6. 同一步 flush 和 settle 同一个钱包时，flush 优先，交易会 drop
7. oversize 交易会被直接判定为 oversize drop

因此，目前环境主流程逻辑是可信的，可以进入 benchmark 固定与正式 screening 阶段。

---

## 14. 后面最容易忘、最该反复检查的点

### 14.1 改了动作空间后，必须检查

* action_size 是否同步改了
* decode_action 是否同步改了
* step() 是否还符合动作语义
* DQN 输出层维度是否同步改了

---

### 14.2 改了 k 后，必须检查

* wallet_size 是否正确变化
* state_dim 是否同步变化
* action_size 是否同步变化
* 之前保存的模型是否还能兼容（通常不能）

---

### 14.3 改了 F 后，必须检查

* freeze_until 的行为是否符合预期
* refill 时机是否仍然正确
* hidden switching 下性能变化是否只是因为 F 太大或太小

---

### 14.4 改了 state 后，必须检查

* 输入维度是否同步改了
* 归一化是否合理
* context feature 是否真的更新，而不是静态值
* 训练结果变化是否来自 state，而不是别的顺带变化

---

### 14.5 改了 reward 后，必须检查

* 正负奖励的尺度是否失衡
* flush 是否变得过于便宜或过于昂贵
* 结果变化是否只是 reward shape 变化，而不是方法更好

---

## 15. 正式实验前固定协议时，要写清楚的内容

进入主实验之前，必须写出一个 locked protocol 文件，至少包括：

* 主问题是什么
* 主 setting 是什么
* robustness setting 是什么
* 主动作空间是什么
* 主 state 版本是什么
* hidden switching benchmark 是什么
* 主 baseline 有哪些
* reward 是否开 shaping
* seed 数是多少
* eval metric 看哪些

如果这些没写清楚，不要开始大量跑实验。

---

## 16. 每次跑实验前的最小 checklist

每次正式跑实验前，先问自己这 10 个问题：

1. 我现在跑的是哪个代码版本？
2. 当前动作空间是什么？
3. 当前 state 版本是什么？
4. 当前 k、F、C、T 是多少？
5. 当前 reward 是否开 shaping？
6. 当前训练 regime 是什么？
7. 当前测试 regime / switching benchmark 是什么？
8. 当前 seed 是多少？
9. 当前结果会保存到哪个文件夹？
10. 这次实验的目的到底是 screening、主实验，还是 debug？

这 10 条不确认清楚，不要开始跑。

---

## 17. 每次跑完实验后必须记录的内容

每次实验跑完后，至少要记录：

* 日期
* 代码版本名
* 实验目的
* 参数（k、F、C、T、state 版本、动作空间）
* 训练 regime
* 测试 regime
* seed
* 关键结果
* 当前结论
* 下一步打算改什么

如果不记这些，后面结果很容易无法复现，也没法写论文。

---

## 18. 当前最现实的下一步

当前最应该做的不是再查环境 bug，而是：

1. 固定 hidden switching benchmark
2. 做小 screening 选主 setting
3. 锁主协议
4. 再把 context features 加进 state，做最小版 context-aware DQN

顺序不要乱。

---

## 19. 一句总提醒

环境定义“动作是什么意思”，DQN 学的是“什么时候该选哪个动作”。

不要把：

* 环境语义
* 动作空间设计
* state 设计
* DQN 学习效果

混成一件事。

每次结果变化时，先问清楚：
到底是 protocol 变了，还是 method 变了。

