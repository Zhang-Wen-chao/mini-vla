"""Conditioned flow-matching action head for PushT (Phase 1).

The model deliberately keeps the VLA action head small and explicit: an image
history is encoded once, then an MLP predicts the velocity of a full action
chunk conditioned on that history and on continuous flow time.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    """Map continuous flow time in ``[0, 1]`` to Fourier features."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim < 2:
            raise ValueError("time embedding dimension must be at least 2")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        frequencies = torch.exp(
            -math.log(10_000)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / max(half - 1, 1)
        )
        angles = t[:, None] * frequencies[None]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if self.dim % 2:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return embedding


class ObservationEncoder(nn.Module):
    """Small CNN that jointly encodes an RGB observation history.

    PushT actions are absolute workspace coordinates, so the encoder must retain
    where the agent, block, and goal appear in the image.  The 4x4 spatial grid
    is deliberately flattened rather than globally averaged.
    """

    def __init__(self, obs_horizon: int, feature_dim: int, proprio_dim: int) -> None:
        super().__init__()
        channels = 3 * obs_horizon
        self.backbone = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, feature_dim),
            nn.SiLU(),
        )
        self.proprio = nn.Sequential(
            nn.Linear(obs_horizon * proprio_dim, feature_dim),
            nn.SiLU(),
        )

    def forward(self, obs: torch.Tensor, agent_pos: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 5:
            raise ValueError("obs must have shape (batch, horizon, 3, height, width)")
        batch, horizon, channels, height, width = obs.shape
        if channels != 3:
            raise ValueError(f"expected RGB observations, got {channels} channels")
        if agent_pos.shape != (batch, horizon, 2):
            raise ValueError("agent_pos must have shape (batch, horizon, 2)")
        return self.backbone(obs.reshape(batch, horizon * channels, height, width)) + self.proprio(
            agent_pos.flatten(1)
        )


class FlowActionHead(nn.Module):
    """Predict ``v_theta(x_t, t, obs)`` for a normalized action chunk.

    Args:
        obs_horizon: Number of RGB frames in the conditioning history.
        action_horizon: Number of future actions generated at once.
        action_dim: Dimension of each action (two for PushT pixel positions).
        hidden_dim: Width of the conditional MLP.
    """

    def __init__(
        self,
        obs_horizon: int = 2,
        action_horizon: int = 16,
        action_dim: int = 2,
        hidden_dim: int = 256,
        proprio_dim: int = 2,
    ) -> None:
        super().__init__()
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.proprio_dim = proprio_dim
        self.encoder = ObservationEncoder(obs_horizon, hidden_dim, proprio_dim)
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        action_features = action_horizon * action_dim
        self.velocity = nn.Sequential(
            nn.Linear(action_features + 2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_features),
        )

    def forward(
        self, noisy_action: torch.Tensor, t: torch.Tensor, obs: torch.Tensor,
        agent_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Return velocity with the same shape as ``noisy_action``."""
        expected = (self.action_horizon, self.action_dim)
        if tuple(noisy_action.shape[1:]) != expected:
            raise ValueError(
                f"noisy_action must have shape (batch, {expected[0]}, {expected[1]})"
            )
        if t.ndim != 1 or t.shape[0] != noisy_action.shape[0]:
            raise ValueError("t must have one time value per batch item")
        condition = torch.cat((self.encoder(obs, agent_pos), self.time_embedding(t)), dim=-1)
        velocity_input = torch.cat((noisy_action.flatten(1), condition), dim=-1)
        return self.velocity(velocity_input).view_as(noisy_action)
