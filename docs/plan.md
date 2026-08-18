# mini-vla 计划（Phase 0-4）

> 决策记录（2026-08-18）：动作头走 π0 式 flow matching（复用 mini-diffusion 目标函数与少步采样结论）；
> 数据用 PushT（diffusion policy 标准 benchmark，真实示教轨迹，可测成功率）；分 Phase 渐进。
> 面试叙事见 resume 仓库 `interview/15_多模态岗位-学习准备.md` 7.1 节。

## Phase 0：数据与评测基线（本仓库当前状态）

- PushT zarr 加载：`data/img` (N,96,96,3)、`data/action` (N,2)、`data/state` (N,5)、`meta/episode_ends`；
- 序列采样：观测窗口 To、动作分块 Ta，episode 边界 padding（头尾 mask）；
- 归一化：按通道/维度统计 mean/std（LinearNormalizer 的简化版）；
- 指标定义：动作 L1 / L2（归一化空间）、分块内一致性；rollout 覆盖率留给 Phase 1。

验收：合成 zarr 单测全绿 + 真实数据 inspect 输出合理。

## Phase 1：单卡训练闭环（约 1 周）

- 模型：小 CNN 观测编码器（96x96 → 特征）+ 条件 flow matching 动作头；
- 目标：`||v_θ(x_t, t, obs) - (a1 - a0)||²`，`x_t = (1-t)a0 + t·a1`（照搬 mini-diffusion objectives）；
- 采样：Euler 8/16 步出动作块；
- 验证：训练 loss 下降、动作误差收敛、PushT rollout 能推动 T 块（覆盖率指标）。

## Phase 2：分布式训练（约 4 天）

- 复用 mini-megatron TP 原语切动作头；torch-distributed-internals 的 FSDP/HSDP；可选 mini-deepspeed ZeRO；
- 等价性验证套路：TP=2 前向逐元素一致、100 步 loss 差 ~1e-4。

## Phase 3：后训练（约 5 天）

- SFT：文本指令条件（"把 T 推到目标点"）微调；
- 蒸馏：teacher 多步 flow → student 1~2 步（复用 DMD 回归 + 流形项）；
- 量化：INT8 / 权重 INT4（复用 AWQ 流程），记录量化前后动作误差退化曲线。

## Phase 4：实时推理（约 4 天）

- 用 mini-vllm 思路写 VLA 推理循环（KV cache 换成观测缓存 + 动作块）；
- 指标：TTFT、单步推理耗时、可达控制频率（机器人刚需 >10 Hz）。

## 复用清单

| 来源 | 复用什么 |
| --- | --- |
| mini-diffusion | flow 目标函数、Euler 采样器、DMD 蒸馏、DiT 块 |
| mini-megatron | TP 原语、等价性验证、1F1B（如需 PP） |
| torch-distributed-internals | FSDP / HSDP 手写实现 |
| mini-deepspeed | ZeRO 分片 |
| mini-vllm | 推理循环、CUDA Graph 思路 |
| 得物经验 | AWQ / PTQ 量化流程 |
