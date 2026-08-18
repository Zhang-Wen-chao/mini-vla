# Phase 0 数据笔记：PushT

## 数据集来源

- 官方下载：`https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip`（解压后 `pusht_cchi_v7_replay.zarr`，约 1.5 GB）
- 出处：Diffusion Policy（Chi et al., 2023）真实机器人示教数据，206 条 episode、共 25650 步

## zarr 布局（diffusion-policy 官方 README 记录）

```
pusht_cchi_v7_replay.zarr
├── data
│   ├── action    (25650, 2)   float32   # 末端执行器位移 (dx, dy)
│   ├── img       (25650, 96, 96, 3) float32  # 俯视 RGB 观测
│   ├── keypoint  (25650, 9, 2) float32  # T 块 9 个关键点
│   ├── n_contacts (25650, 1)  float32
│   └── state     (25650, 5)   float32   # agent 位置 + T 块位姿/角度
└── meta
    └── episode_ends (206,) int64        # 每条 episode 的结束下标（闭区间）
```

- 所有字段按时间轴拼接，`meta/episode_ends` 是 episode 边界；
- PushT 任务：用圆头推杆把 T 形块推到目标点（成功率 = 覆盖率，diffusion policy CNN 基线 ~0.9+）。

## 采样约定（对齐 diffusion-policy 接口）

- 观测窗口 `To`（默认 2 帧图像，也可以只取最后一帧——本项目 Phase 1 先用 To=2 或 1，实验对比）；
- 动作分块 `Ta`（默认 16 步）；
- episode 起始处观测用"重复首帧"填充；结尾处动作用 0 填充 + validity mask；
- 归一化：img 按 [-1,1] 或 [0,1]（决定后再定）；action 用训练集 mean/std。

## 与 diffusion-policy 的差异（如实记录）

- 不用 hydra / wandb / robomimic 那套工程，纯 numpy + torch 最小实现；
- 本仓库目标是"教学闭环 + 可复用 mini 生态"，不追求与官方 CNN 基线同分，但 Phase 1 会跑 rollout 覆盖率做相对比较。

## 复现命令

```bash
bash scripts/download_pusht.sh
python -m mini_vla.data --data-dir data/pusht/pusht_cchi_v7_replay.zarr
pytest -q
```
