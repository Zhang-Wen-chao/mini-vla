"""合成 zarr 上的数据层与指标单测（CPU，不需要下载 PushT）。"""

import numpy as np
import pytest
import torch
import zarr

from mini_vla.config import PushTConfig
from mini_vla.data import LinearNormalizer, PushTDataset
from mini_vla.metrics import action_l1, action_l2, chunk_consistency


@pytest.fixture
def synthetic_zarr(tmp_path):
    """两条 episode 的合成数据：episode 长度 6 与 4。"""
    path = tmp_path / "pusht_test.zarr"
    root = zarr.open_group(str(path), mode="w")
    n = 10
    img = np.arange(n * 96 * 96 * 3, dtype=np.float32).reshape(n, 96, 96, 3) % 255
    action = np.stack([np.arange(n, dtype=np.float32),
                       np.arange(n, dtype=np.float32) * 0.5], axis=1)
    data = root.require_group("data")
    data.create_array("img", data=img)
    data.create_array("action", data=action)
    data.create_array("state", data=np.zeros((n, 5), dtype=np.float32))
    meta = root.require_group("meta")
    # Diffusion Policy stores exclusive ends: [0, 6) and [6, 10).
    meta.create_array("episode_ends", data=np.array([6, 10], dtype=np.int64))
    return str(path)


def make_cfg(path, obs_horizon=2, action_horizon=16):
    return PushTConfig(zarr_path=path, obs_horizon=obs_horizon,
                       action_horizon=action_horizon)


def test_shapes_and_ranges(synthetic_zarr):
    ds = PushTDataset(make_cfg(synthetic_zarr))
    assert len(ds) > 0
    sample = ds[0]
    assert sample["obs"].shape == (2, 3, 96, 96)
    assert sample["agent_pos"].shape == (2, 2)
    assert sample["action"].shape == (16, 2)
    assert sample["action_mask"].shape == (16,)
    lo, hi = ds.cfg.img_range
    assert sample["obs"].min() >= lo and sample["obs"].max() <= hi + 1e-6


def test_episode_boundary_no_leak(synthetic_zarr):
    ds = PushTDataset(make_cfg(synthetic_zarr, obs_horizon=2, action_horizon=16))
    for i in range(len(ds)):
        mask = ds[i]["action_mask"]
        if mask.sum() < len(mask):
            assert mask.sum() > 0 and (mask[mask.sum().int():] == 0).all()


def test_obs_padding_repeats_first_frame(synthetic_zarr):
    ds = PushTDataset(make_cfg(synthetic_zarr, obs_horizon=2, action_horizon=16))
    first = ds[0]["obs"]
    assert torch.equal(first[0], first[1])


def test_action_padding_zeros(synthetic_zarr):
    ds = PushTDataset(make_cfg(synthetic_zarr, obs_horizon=1, action_horizon=16))
    for i in range(len(ds)):
        sample = ds[i]
        n_valid = int(sample["action_mask"].sum())
        assert (sample["action"][n_valid:] == 0).all()


def test_action_tail_does_not_leak_into_next_episode(synthetic_zarr):
    ds = PushTDataset(make_cfg(synthetic_zarr, obs_horizon=1, action_horizon=16))
    last_start_of_first_episode = int(np.where(ds.starts == 5)[0][0])
    sample = ds[last_start_of_first_episode]
    assert sample["action_mask"].sum() == 1
    assert torch.equal(sample["action"][0], torch.tensor([5.0, 2.5]))
    assert (sample["action"][1:] == 0).all()


def test_action_is_aligned_with_current_observation(synthetic_zarr):
    ds = PushTDataset(make_cfg(synthetic_zarr, obs_horizon=1, action_horizon=2))
    sample = ds[int(np.where(ds.starts == 2)[0][0])]
    assert torch.equal(sample["action"], torch.tensor([[2.0, 1.0], [3.0, 1.5]]))


def test_normalizer_roundtrip(synthetic_zarr):
    ds = PushTDataset(make_cfg(synthetic_zarr))
    norm = ds.action_normalizer
    arr = np.array([[0.5, -1.0], [2.0, 3.0]], dtype=np.float32)
    out = norm.denormalize(norm.normalize(arr))
    assert np.allclose(out, arr, atol=1e-5)


def test_metrics_known_values():
    pred = torch.zeros(4, 2)
    target = torch.ones(4, 2)
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
    assert action_l1(pred, target, mask) == pytest.approx(1.0)
    assert action_l2(pred, target, mask) == pytest.approx(1.0)
    assert chunk_consistency(torch.zeros(2, 3, 2)) == 0.0


def test_metrics_accept_batched_action_mask():
    pred = torch.zeros(2, 3, 2)
    target = torch.ones(2, 3, 2)
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    assert action_l1(pred, target, mask) == pytest.approx(1.0)
    assert action_l2(pred, target, mask) == pytest.approx(1.0)
