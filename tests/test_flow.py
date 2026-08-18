"""CPU tests for Phase 1 flow-matching math and action-head interfaces."""

import pytest
import torch

from mini_vla.flow import flow_matching_loss, sample_euler
from mini_vla.model import FlowActionHead


class ZeroVelocity(torch.nn.Module):
    def forward(self, x, t, obs, agent_pos):
        return torch.zeros_like(x)


class UnitVelocity(torch.nn.Module):
    def forward(self, x, t, obs, agent_pos):
        return torch.ones_like(x)


def test_action_head_output_shape():
    model = FlowActionHead(obs_horizon=2, action_horizon=4, hidden_dim=32)
    output = model(
        torch.randn(3, 4, 2), torch.rand(3), torch.randn(3, 2, 3, 96, 96),
        torch.randn(3, 2, 2),
    )
    assert output.shape == (3, 4, 2)
    assert model.encoder.backbone[-2].in_features == 128 * 4 * 4


def test_flow_loss_uses_only_valid_actions():
    action = torch.tensor([[[1.0, 2.0], [100.0, 100.0]]])
    obs = torch.zeros(1, 2, 3, 96, 96)
    mask = torch.tensor([[1.0, 0.0]])
    loss = flow_matching_loss(
        ZeroVelocity(), action, obs, torch.zeros(1, 2, 2), mask,
        t=torch.tensor([0.5]), noise=torch.zeros_like(action)
    )
    assert loss.item() == pytest.approx((1.0 + 4.0) / 2)


def test_euler_integrates_velocity():
    obs = torch.zeros(2, 2, 3, 96, 96)
    start = torch.full((2, 3, 2), -2.0)
    result = sample_euler(
        UnitVelocity(), obs, torch.zeros(2, 2, 2), action_horizon=3, action_dim=2,
        steps=8, noise=start,
    )
    assert torch.allclose(result, torch.full_like(start, -1.0))
