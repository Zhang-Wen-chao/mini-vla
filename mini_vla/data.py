"""PushT zarr 加载、序列采样与归一化（Phase 0）。

对齐 diffusion-policy 的接口约定：观测窗口 To 帧、动作分块 Ta 步；
episode 起始处观测用"重复首帧"填充，结尾处动作用 0 填充并带 validity mask。
"""

from __future__ import annotations

import argparse
import bisect

import numpy as np
import torch
import zarr

from .config import PushTConfig


class LinearNormalizer:
    """按维度统计 mean/std 的线性归一化（LinearNormalizer 的教学简化版）。"""

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, arr: np.ndarray) -> "LinearNormalizer":
        self.mean = arr.mean(axis=0)
        self.std = arr.std(axis=0)
        self.std = np.where(self.std < 1e-6, 1.0, self.std)
        return self

    def normalize(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self.mean) / self.std

    def denormalize(self, arr: np.ndarray) -> np.ndarray:
        return arr * self.std + self.mean


class PushTDataset(torch.utils.data.Dataset):
    """从 PushT zarr 采样 (obs, action, action_mask) 三元组。

    采样规则：样本起点 s 保证观测窗口 [s-To+1, s] 完整落在 episode 内；
    动作块从 s+1 开始取 Ta 步，超出 episode 结尾的部分补 0、mask 置 0。
    """

    def __init__(self, cfg: PushTConfig) -> None:
        self.cfg = cfg
        self.root = zarr.open(cfg.zarr_path, mode="r")

        ends = np.asarray(self.root["meta"]["episode_ends"], dtype=np.int64)
        if cfg.max_episodes > 0:
            ends = ends[: cfg.max_episodes]
        self.episode_ends = ends
        self.episode_starts = np.concatenate([[0], ends[:-1] + 1])

        self.img = self.root["data"]["img"]
        self.action = self.root["data"]["action"]
        if self.img.shape[1] != cfg.img_size:
            raise ValueError(
                f"img_size={cfg.img_size} 与数据 {self.img.shape[1]} 不符"
            )

        self.starts = self._build_starts()
        self.action_normalizer = LinearNormalizer().fit(
            np.asarray(self.action)
        )

    def _build_starts(self) -> np.ndarray:
        starts: list[int] = []
        for st, en in zip(self.episode_starts, self.episode_ends):
            length = en - st + 1
            if length < self.cfg.obs_horizon + 1:
                continue
            last_start = en - self.cfg.obs_horizon
            starts.extend(range(int(st), int(last_start) + 1))
        return np.array(starts, dtype=np.int64)

    def _episode_range_of(self, s: int) -> tuple[int, int]:
        idx = bisect.bisect_right(self.episode_ends, s)
        return int(self.episode_starts[idx]), int(self.episode_ends[idx])

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        s = int(self.starts[index])
        st, _ = self._episode_range_of(s)
        to, ta = self.cfg.obs_horizon, self.cfg.action_horizon

        obs_start = max(s - to + 1, st)
        obs = np.asarray(self.img[obs_start : s + 1])
        if obs.shape[0] < to:
            obs = np.concatenate(
                [np.repeat(obs[:1], to - obs.shape[0], axis=0), obs], axis=0
            )

        action = np.asarray(self.action[s + 1 : s + 1 + ta])
        mask = np.zeros(ta, dtype=np.float32)
        mask[: action.shape[0]] = 1.0
        if action.shape[0] < ta:
            action = np.concatenate(
                [action, np.zeros((ta - action.shape[0], action.shape[1]),
                                  dtype=np.float32)], axis=0
            )

        lo, hi = self.cfg.img_range
        obs = obs.astype(np.float32) / 255.0
        obs = obs * (hi - lo) + lo
        obs = obs.transpose(0, 3, 1, 2)  # (To, C, H, W)

        return {
            "obs": torch.from_numpy(obs),
            "action": torch.from_numpy(action),
            "action_mask": torch.from_numpy(mask),
        }


def inspect(data_dir: str) -> None:
    """打印数据集概况：形状、episode 数、归一化统计。"""
    root = zarr.open(data_dir, mode="r")
    ends = np.asarray(root["meta"]["episode_ends"])
    img = np.asarray(root["data"]["img"])
    action = np.asarray(root["data"]["action"])
    state = np.asarray(root["data"]["state"])
    norm = LinearNormalizer().fit(action)
    print(f"episodes        : {len(ends)}")
    print(f"total steps     : {img.shape[0]}")
    print(f"img shape       : {img.shape}")
    print(f"action shape    : {action.shape}, range {action.min(0)} ~ {action.max(0)}")
    print(f"state shape     : {state.shape}")
    print(f"action mean/std : {norm.mean} / {norm.std}")


def main() -> None:
    parser = argparse.ArgumentParser(description="inspect PushT zarr")
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()
    inspect(args.data_dir)


if __name__ == "__main__":
    main()
