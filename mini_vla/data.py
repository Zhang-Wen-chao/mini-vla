"""PushT zarr 加载、序列采样与归一化（Phase 0）。

对齐 diffusion-policy 的接口约定：观测窗口 To 帧、动作分块 Ta 步；
episode 起始处观测用"重复首帧"填充，结尾处动作用 0 填充并带 validity mask。
``meta/episode_ends`` 遵循 diffusion-policy 约定，保存的是 exclusive end
offset，因此最后一个值等于数据总步数。
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

    采样规则：样本起点 s 的观测窗口 [s-To+1, s] 在左侧按首帧 padding；
    动作块与最近 observation 同时间对齐，从 s 开始取 Ta 步；这与
    diffusion-policy 的 PushT ``SequenceSampler`` 对齐约定一致。超出
    episode 结尾的部分补 0、mask 置 0。
    ``episode_ends`` 是 exclusive。
    """

    def __init__(self, cfg: PushTConfig) -> None:
        self.cfg = cfg
        self.root = zarr.open(cfg.zarr_path, mode="r")

        ends = np.asarray(self.root["meta"]["episode_ends"], dtype=np.int64)
        if cfg.max_episodes > 0:
            ends = ends[: cfg.max_episodes]
        self.img = self.root["data"]["img"]
        self.state = self.root["data"]["state"]
        self.action = self.root["data"]["action"]
        if len(ends) == 0 or ends[-1] > self.action.shape[0]:
            raise ValueError("invalid exclusive episode_ends in zarr metadata")
        self.episode_ends = ends
        self.episode_starts = np.concatenate([[0], ends[:-1]])
        if self.img.shape[1] != cfg.img_size:
            raise ValueError(
                f"img_size={cfg.img_size} 与数据 {self.img.shape[1]} 不符"
            )

        self.starts = self._build_starts()
        self.action_normalizer = LinearNormalizer().fit(np.asarray(self.action[: ends[-1]]))

    def _build_starts(self) -> np.ndarray:
        starts: list[int] = []
        for st, en in zip(self.episode_starts, self.episode_ends):
            length = en - st
            if length < 1:
                continue
            # Observation history is left-padded at an episode start; include
            # every control step, including the final one, and mask the action
            # chunk tail as needed.
            last_start = en - 1
            starts.extend(range(int(st), int(last_start) + 1))
        return np.array(starts, dtype=np.int64)

    def _episode_range_of(self, s: int) -> tuple[int, int]:
        idx = bisect.bisect_right(self.episode_ends, s)
        return int(self.episode_starts[idx]), int(self.episode_ends[idx])

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        s = int(self.starts[index])
        st, en = self._episode_range_of(s)
        to, ta = self.cfg.obs_horizon, self.cfg.action_horizon

        obs_start = max(s - to + 1, st)
        obs = np.asarray(self.img[obs_start : s + 1])
        agent_pos = np.asarray(self.state[obs_start : s + 1, :2], dtype=np.float32)
        if obs.shape[0] < to:
            obs = np.concatenate(
                [np.repeat(obs[:1], to - obs.shape[0], axis=0), obs], axis=0
            )
            agent_pos = np.concatenate(
                [np.repeat(agent_pos[:1], to - agent_pos.shape[0], axis=0), agent_pos], axis=0
            )

        # NumPy/zarr slicing has no notion of episode boundaries.  Clamp the
        # end explicitly: otherwise the tail of one demonstration would use
        # actions from the next one, while its mask still claimed they were
        # valid.
        action_end = min(s + ta, en)
        action = np.asarray(self.action[s:action_end])
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
            # PushT's native proprioception: absolute agent position in the
            # same 512x512 workspace as the action.  Map to [-1, 1] so its
            # scale matches the normalized image conditioning.
            "agent_pos": torch.from_numpy(agent_pos / 256.0 - 1.0),
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
