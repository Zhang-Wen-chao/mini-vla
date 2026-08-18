"""Focused CPU checks for Phase 1 training artifacts."""

import numpy as np
import torch

from mini_vla.config import PushTConfig, TrainConfig
from mini_vla.train import _ModelEMA, _checkpoint
from mini_vla.model import FlowActionHead


def test_checkpoint_contains_rollout_metadata():
    cfg = TrainConfig()
    model = FlowActionHead(hidden_dim=32)
    checkpoint = _checkpoint(
        model, cfg, np.array([1.0, 2.0]), np.array([3.0, 4.0]), epoch=7
    )
    assert checkpoint["epoch"] == 7
    assert checkpoint["model_config"]["proprio_dim"] == 2
    assert torch.equal(checkpoint["model"]["velocity.0.weight"], model.state_dict()["velocity.0.weight"])


def test_checkpoint_preserves_nondefault_observation_horizon():
    cfg = TrainConfig(data=PushTConfig(obs_horizon=4))
    checkpoint = _checkpoint(
        FlowActionHead(obs_horizon=4, hidden_dim=32), cfg,
        np.array([1.0, 2.0]), np.array([3.0, 4.0]), epoch=1,
    )
    assert checkpoint["model_config"]["obs_horizon"] == 4


def test_checkpoint_preserves_nondefault_action_horizon():
    cfg = TrainConfig(data=PushTConfig(action_horizon=8))
    checkpoint = _checkpoint(
        FlowActionHead(action_horizon=8, hidden_dim=32), cfg,
        np.array([1.0, 2.0]), np.array([3.0, 4.0]), epoch=1,
    )
    assert checkpoint["model_config"]["action_horizon"] == 8


def test_ema_updates_toward_latest_model_weights():
    model = FlowActionHead(hidden_dim=32)
    ema = _ModelEMA(model, decay=0.5)
    parameter = next(model.parameters())
    average = next(ema.model.parameters())
    before = average.detach().clone()
    with torch.no_grad():
        parameter.add_(2.0)
    ema.update(model)
    assert torch.allclose(average, before + 1.0)
