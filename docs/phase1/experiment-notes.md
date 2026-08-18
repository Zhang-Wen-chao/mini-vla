# Phase 1 实验记录

## 评测协议

- 数据：官方 PushT image replay，206 episodes / 25,650 steps；训练/验证由固定
  `seed=0` 的 sample-level 95% / 5% split 构成（仅用来检查训练是否收敛，
  不是 rollout 泛化评测）；
- 模型：默认 2 帧 96×96 RGB（To=4 对照单独注明）、16 步二维 action chunk、条件
  flow matching、8-step midpoint Euler；图像 encoder 保留 4×4 spatial grid，并融合每帧
  agent XY proprioception（`state[:2]`）；
- 离线：masked velocity MSE 和 normalized action L1/L2；
- rollout：官方 PushT 动力学/coverage 定义的最小 headless 复现，测试 seed
  `10000..10019`，每 episode 至多 200 control steps，每次生成 16 个动作但仅执行
  前 8 个后重新观测。报告每 episode 最大 coverage 的均值，coverage > 0.95 为成功。

## Baseline 0：global-pool CNN（2026-08-18）

配置：`hidden_dim=256`、batch 128、AdamW 3e-4、10 epochs。

| 指标 | epoch 1 | epoch 10 |
| --- | ---: | ---: |
| train flow loss | 1.5059 | 0.7761 |
| val flow loss | 0.2219 | 0.2066 |
| 8-step sampled action L1 | 0.8374 | 0.8333 |
| 8-step sampled action L2 | 1.0002 | 0.9845 |

20 条固定 seed rollout：mean max coverage **0.1110**，success rate **0.0**。

结论：velocity regression 已下降，但 global average pooling 丢弃了对 absolute XY
action 必要的空间信息，不能视为控制策略训通。下一实验将池化替换为保留 4×4 grid 的
spatial encoder；其余训练和 rollout 协议不变，形成可解释的对照。

## 已验证的修复与对照

在上述两次早期对照后发现并修复：

1. 官方 `episode_ends` 为 exclusive offset，且 PushT 的监督是同时间索引
   `obs[t] -> action[t:t+Ta]`；旧实现错误地使用了 `action[t+1:]`。
2. flow sampler 从 Gaussian `x_0` 开始；旧离线采样指标错误地固定从零向量开始，
   与 rollout 不同。
3. 官方 image policy 同时使用 RGB 与 agent proprioception；当前模型已融合
   `state[:2]`，将作为下一组唯一变量的对照。

修复后、不含 proprioception 的 spatial 模型（10 epochs）最终 Gaussian-Euler
validation L1/L2 为 **0.4905 / 0.4078**，20 条固定 seed rollout mean max coverage
为 **0.2617**、success rate **0.0**。该结果说明链路有效但尚不构成任务成功。

| 10-epoch model | Gaussian-Euler val L1/L2 | mean max coverage | success rate |
| --- | --- | ---: | ---: |
| spatial, image only | 0.4905 / 0.4078 | 0.2617 | 0.0 |
| spatial + agent XY proprioception | 0.4427 / 0.3363 | 0.2241 | 0.0 |

proprioception 改善了离线 action 误差，却没有改善这 20 个固定 seed 的闭环 coverage。
这是一条刻意保留的反例：**offline regression 不能替代 rollout 验收**。下一实验固定
模型与 rollout 协议，只把训练时长从 10 增至 50 epochs，检查是否为欠训练，而不是
继续叠加架构变量。

## 环境与动作接口校准（2026-08-18）

在解释零成功 rollout 前，使用真实 PushT replay 的 1,024 个 episode 内 transition 和
headless simulator 做了独立校准（`python -m mini_vla.calibration`）：

| 检查 | 结果 |
| --- | ---: |
| replay action 范围 | [18, 509]，在 [0, 512] workspace 内 |
| action 到 agent 的平均距离 | 23.139 → 13.849（一步后） |
| 距离下降的 replay transition 比例 | 99.61% |
| simulator 对绝对 XY target 的响应 | 206.155 → 145.201 |
| 将 T 块设为精确 goal pose 的 coverage | 1.0 |

因此 `data/action` 的语义已确认是绝对 workspace target `(x, y)`，并且当前
simulator 的坐标和 coverage 几何能通过这些不变量；此前 rollout 的零成功不能归咎于
把 action 当位移、坐标反向或 coverage 公式显然失效。原始结果保存在 L20 的
`outputs/phase1/calibration.json`，并以可追踪副本归档为
[`results/calibration-20260818.json`](results/calibration-20260818.json)。

进一步从真实 replay 随机抽取 128 条**非 episode 边界** transition，以 `state[t]`
重置 headless simulator、执行 `action[t]`，并与 `state[t+1]` 对齐。一步平均误差为
agent **2.45 px**、T-block position **0.86 px**、T-block angle **0.0101 rad**
（最坏 agent 9.45 px、block 6.44 px、angle 0.0617 rad）。这证明环境不仅坐标语义
正确，动力学/接触参数也与 replay 足够一致；当前低 rollout 不能合理归因于 simulator
未复现数据。完整产物：
[`results/calibration-dynamics-128-20260818.json`](results/calibration-dynamics-128-20260818.json)。

补充辨析：官方 image-PushT runner 的 benchmark 配置使用 `legacy_test=true`，因此
本项目**正式 policy rollout**显式记录并默认 `legacy=true`。但对 zarr 内的已记录
`state` 重新渲染时，`legacy=false` 与存储 RGB 更接近；这是 dataset state 的历史
赋值语义，与随机 seed 上的官方 benchmark 语义不同。两者不可混用。

因此示教 action 的整段 simulator replay 只保留为诊断，不能被设为
"demonstration 必须成功"的 evaluator gate：在 20 条固定示教上，legacy true/false 的
mean max coverage 分别为 **0.2109 / 0.3648**，均为 0 success；而记录 state 自身在
本 coverage 定义下也几乎不越过 0.95（legacy true 仅 1/206，non-legacy 0/206）。
这说明该 replay 流不是逐动作可精确复演的 success oracle，不能据此否定或证明 policy。
保留的有效 evaluator 校准是 action 坐标、single-step transition 以及 exact-goal coverage。

### 时长对照：50 epochs（完成）

`phase1-proprio-50e-20260818` 使用与 proprioception 10-epoch 对照完全相同的
模型、数据、seed、优化器和 rollout 协议，只将训练轮数改为 50。最终 epoch 50 的
Gaussian-Euler validation L1/L2 为 **0.2117 / 0.0797**，远优于 10-epoch
proprioception 对照的 0.4427 / 0.3363。

该 checkpoint 曾在 `legacy=false` 的环境语义下得到 20-seed mean max coverage
**0.2279**、success rate **0.0**。此数值和以下控制频率 sweep 都已归档，但在发现
官方 image-PushT runner 明确配置 `legacy_test=true` 后，必须标注为**非 canonical 环境的
诊断结果**，不能再用它判断"欠训练"或模型失败。该 checkpoint 将按 canonical
`legacy=true` 协议重新评测，再与 To=4 比较。

### 闭环重规划频率：1 / 4 / 8（完成）

对相同 50-epoch checkpoint、相同 20 个环境 seed 与 flow-noise seed，只修改每次
重新观测前执行的动作数，曾在**旧 `legacy=false` 诊断环境**得到：

| execute steps | mean max coverage | success rate |
| ---: | ---: | ---: |
| 1 | 0.1113 | 0.0 |
| 4 | 0.1833 | 0.0 |
| 8 | **0.2279** | 0.0 |

该诊断中 `execute_steps=8` 数值最高，但该结论也须在 canonical `legacy=true` 下复核；
在复核完成前，后续训练暂按官方 runner 的 `n_action_steps=8` 保持 8。完整可追踪产物：
[`proprio-50e-metrics-20260818.jsonl`](results/proprio-50e-metrics-20260818.jsonl)、
[`proprio-50e-rollout-20260818.json`](results/proprio-50e-rollout-20260818.json)、
[`proprio-50e-control-frequency-sweep-20260818.json`](results/proprio-50e-control-frequency-sweep-20260818.json)。

### 观测历史：To=2 → 4（完成）

固定官方 runner 的 `execute_steps=8`、其余模型和优化配置，唯一将 image 与 agent XY
condition 的观测历史从 2 帧增加为 4 帧。目的是给动作头加入更明确的短时运动趋势，
而不同时混入 action horizon 或网络宽度变化。训练完成后先按相同 20-seed rollout
协议评测，再决定是否改 action horizon。

训练使用 `checkpoint_every=1`，所以 `last.pt` 是**当前 epoch**而非最终 epoch。最初的
外部 watcher 因等待 `last.pt` 而在 epoch 1 过早跑了一次 rollout；该结果为 mean max
coverage **0.1128**、success rate **0.0**，只作为早期快照归档为
[`proprio-to4-early-epoch0001-rollout-20260818.json`](results/proprio-to4-early-epoch0001-rollout-20260818.json)，
不能和最终 50-epoch 结果比较。修正后的 watcher 只等待 `epoch_0050.pt`，再以该文件
生成标准 `rollout.json`。

最终 To=4 checkpoint 的 Gaussian-Euler validation L1/L2 为 **0.1651 / 0.0468**，
优于 To=2 50-epoch checkpoint 的 **0.2117 / 0.0797**。但按官方兼容的
`legacy=true` 协议、完全相同的 20 个 seed 重新评测后，闭环结果反而更差：

| 50-epoch model | observation horizon | offline sampled L1/L2 | canonical mean max coverage | success rate |
| --- | ---: | ---: | ---: | ---: |
| proprioception baseline | 2 | 0.2117 / 0.0797 | **0.2674** | 0.0 |
| To=4 对照 | 4 | **0.1651 / 0.0468** | 0.1587 | 0.0 |

所以 To=4 虽将离线 L1 降低约 22%、L2 降低约 41%，却使 canonical coverage 降低约
0.1088（约 41%）。在这个固定训练预算和随机种子集合中，**增加观测历史不能改善闭环**；
这再次说明不能以 sampled action error 替代 rollout 验收。可追踪原始产物：
[`proprio-to4-50e-metrics-20260818.jsonl`](results/proprio-to4-50e-metrics-20260818.jsonl)、
[`proprio-to4-50e-rollout-canonical-legacy-20260818.json`](results/proprio-to4-50e-rollout-canonical-legacy-20260818.json)、
[`proprio-to2-50e-rollout-canonical-legacy-20260818.json`](results/proprio-to2-50e-rollout-canonical-legacy-20260818.json)。

两组均为零成功，Phase 1 的“训通并有 rollout 成功”验收尚未达成。

### action horizon：Ta=16 → 8（完成）

To=4 的最终 canonical rollout 和 To=2 的 canonical 复评均无成功轨迹，下一轮保持
To=4、`execute_steps=8` 及其余训练配置不变，只把预测 action chunk 从 16 步减至 8 步。
这样一次生成的动作数量恰好等于执行数量，可检验长 horizon joint prediction 是否造成动作块
后段质量或条件容量问题。训练输出为
`outputs/phase1-proprio-to4-ta8-50e-20260818`：50 epochs、To=4、batch 128、hidden
256、8 Euler steps 与每次执行 8 个 action 均保持不变；仅 action horizon 改为 8。独立
watcher 在最终 `epoch_0050.pt` 后以 `legacy=true` 跑固定 20-seed rollout。

最终离线 sampled L1/L2 为 **0.1313 / 0.0304**，是目前这组 To=4 对照中最小的离线误差。
canonical mean max coverage 为 **0.2185**、success 为 **0.0**。因此，缩短 chunk 相较于
To=4/Ta=16（coverage 0.1587）有 +0.0598 的改善，但仍低于 To=2/Ta=16 基线（0.2674），
且没有产生成功轨迹。结论是：Ta=8 缓解了 To=4/Ta=16 的闭环退化，但单独不足以“训通”
PushT。原始产物：[`proprio-to4-ta8-50e-metrics-20260818.jsonl`](results/proprio-to4-ta8-50e-metrics-20260818.jsonl)、
[`proprio-to4-ta8-50e-rollout-canonical-legacy-20260818.json`](results/proprio-to4-ta8-50e-rollout-canonical-legacy-20260818.json)。

## Ta=8 结束后的诊断优先级（准备中）

当前 Ta=8 是必要但有意保持很小的单变量对照：官方 image PushT 的常见协议同样是
action horizon 16、每次执行 8 步（并非因为执行 8 步就必须预测 8 步）。因此，若 Ta=8
仍无闭环成功，不应继续随意扫描 horizon；下一优先级应转向训练协议/表征能力差异。

已核对的参考配置与本项目的主要差别包括：参考 image policy 使用 ResNet18、GroupNorm、
RGB crop/augmentation、EMA、cosine learning-rate schedule 及远长于 50 epoch 的训练；本项目
刻意使用小型空间 CNN、直接条件 flow MLP、固定 AdamW 学习率和 50 epochs，以保持教学实现
可读。它们会共同影响闭环质量，不能同时改动。Ta=8 出结果后，先以 canonical rollout 判定，
再从其中选择**一项**高信息量变量（优先是 EMA 或学习率 schedule 的训练稳定化；编码器升级
应另作单独架构对照），并先增加对应测试和文档说明。

### 并行的单变量矩阵（2026-08-18，50-epoch 对照已完成）

L20 的四张 L20 均为空闲后，GPU 0--3 分别启动了互不共享输出目录的受控训练；每一项都
有独立 watcher，只会等各自的最终 checkpoint 出现后，按 `legacy=true`、固定 20 个
seed、`execute_steps=8` 跑 rollout。它们的共同基线均为空间 CNN + agent XY、hidden 256、
batch 128、AdamW 3e-4、8 Euler steps、To=4、Ta=16；例外在表中明确列出。

| GPU | 输出目录 | 唯一变量 | 训练预算 | 目标 |
| ---: | --- | --- | ---: | --- |
| 0 | `phase1-proprio-to4-ta8-50e-20260818` | Ta=16 → 8（完成） | 50 epochs | coverage 0.2185 / success 0；相较 To=4/Ta=16 改善但未超过 To=2 基线 |
| 1 | `phase1-proprio-to4-ta16-ema-50e-20260818` | checkpoint/评测采用 EMA（decay 0.999） | 50 epochs | 完成：coverage 0.1207 / success 0，未改善 |
| 2 | `phase1-proprio-to4-ta16-cosine-50e-20260818` | cosine LR（3e-4 → 0） | 50 epochs | 完成：coverage 0.2207 / success 0，部分回升但不超过 To=2 基线 |
| 3 | `phase1-proprio-to4-ta16-200e-20260818` | epochs=50 → 200 | 200 epochs | 排除训练步数仍不足 |

GPU 0 在 To=4/Ta=8 完成并释放后，补充启动了缺失的 To=2/Ta=8 组合：
`phase1-proprio-to2-ta8-50e-20260818`。它固定 50 epochs、batch 128、hidden 256、
AdamW 3e-4、8 Euler steps、execute_steps=8 与 canonical legacy=true rollout；唯一把
observation horizon 设为 2。这样将得到完整的 To × Ta 2×2 对照：

| observation horizon | action horizon 16 | action horizon 8 |
| ---: | --- | --- |
| 2 | 已完成：coverage 0.2674，success 0 | 已完成：coverage 0.2098，success 0 |
| 4 | 已完成：coverage 0.1587，success 0 | 已完成：coverage 0.2185，success 0 |

To=2/Ta=8 的最终 offline L1/L2 是 **0.1671 / 0.0515**，canonical coverage 为
**0.2098**、success **0.0**。至此 2×2 矩阵完成：Ta 从 16 缩至 8 时，在 To=2 下
coverage 降低 0.0577（0.2674 → 0.2098），在 To=4 下则提高 0.0598（0.1587 → 0.2185）。
反之，To 从 2 增至 4 时，在 Ta=16 下 coverage 降低 0.1088，在 Ta=8 下只提高 0.0087。
因此 Ta 与 To 存在明显交互，不能宣布“短 action chunk 普遍更好”或“长 observation history
普遍更好”；四组均无 success，下一步仍应等待 200-epoch 时长对照，避免将训练预算不足错归因
为结构选择。原始产物：[`proprio-to2-ta8-50e-metrics-20260818.jsonl`](results/proprio-to2-ta8-50e-metrics-20260818.jsonl)、
[`proprio-to2-ta8-50e-rollout-canonical-legacy-20260818.json`](results/proprio-to2-ta8-50e-rollout-canonical-legacy-20260818.json)。

EMA 与 cosine 均为新加入、默认关闭的训练开关；EMA 权重仅用于 validation、checkpoint 与
rollout，原始模型仍被正常优化。首次 L20 回归验证为 **28 passed, 1 pygame warning**。
EMA 的早期 validation 会因平均模型由随机初始权重起步而暂时偏差，必须以完成 50 epoch 后
的 canonical rollout 判断，不能与其他训练的前几轮 metrics 横比。所有四项完成前，不再启动
新的训练变量或将中间 action L1/L2 视为策略成功。

EMA 与 cosine 的最终 canonical 结果现已完成。EMA 的最终 offline L1/L2 为
**0.1673 / 0.0493**，但 coverage 仅 **0.1207**、success **0.0**；相较 To=4/Ta=16
固定 LR 基线的 0.1587，它是负向结果。Cosine LR 的 offline L1/L2 较差，为
**0.2313 / 0.0947**，但 coverage 为 **0.2207**、success **0.0**，高于 To=4/Ta=16
及 EMA，也略高于 To=4/Ta=8 的 0.2185，却仍低于 To=2/Ta=16 的 0.2674。这个结果再次
说明离线 action error 与闭环 coverage 的排序不一致，不能根据 L1/L2 筛选策略。原始可追踪
产物：[`proprio-to4-ta16-ema-50e-metrics-20260818.jsonl`](results/proprio-to4-ta16-ema-50e-metrics-20260818.jsonl)、
[`proprio-to4-ta16-ema-50e-rollout-canonical-legacy-20260818.json`](results/proprio-to4-ta16-ema-50e-rollout-canonical-legacy-20260818.json)、
[`proprio-to4-ta16-cosine-50e-metrics-20260818.jsonl`](results/proprio-to4-ta16-cosine-50e-metrics-20260818.jsonl)、
[`proprio-to4-ta16-cosine-50e-rollout-canonical-legacy-20260818.json`](results/proprio-to4-ta16-cosine-50e-rollout-canonical-legacy-20260818.json)。

### 200-epoch 时长对照：运行快照（2026-08-18 18:47 CST）

GPU 3 上的 `phase1-proprio-to4-ta16-200e-20260818` 仍在运行，训练进程和其 DataLoader
worker 均存活。当前已写入 **epoch 90 / 200**；该 epoch 的 train flow loss 为 **0.1312**，
Gaussian-Euler validation sampled L1/L2 为 **0.1135 / 0.0215**。输出目录此时只有会随 epoch
覆盖的 `last.pt` 与 `metrics.jsonl`，尚无 `epoch_0200.pt` 或 `rollout.json`。因此这只是用于
恢复和监控的中间快照，不能和完成的 50-epoch rollout 作策略优劣比较，也不能据此宣布
“训练时长解决了零 success”。训练结束后必须固定沿用 `legacy=true`、20 个 seed、200 control
steps、`execute_steps=8` 的 canonical rollout，并将最终 metrics/JSON 归档后才更新结论。
