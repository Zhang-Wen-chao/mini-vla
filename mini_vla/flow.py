"""Flow-matching objective and Euler sampler for action chunks.

This is the action-shaped counterpart of ``mini-diffusion/flow_matching``:
``x_t = (1 - t) a0 + t a1`` and the regression target is ``a1 - a0``.
"""

from __future__ import annotations

import torch


def flow_matching_loss(
    model: torch.nn.Module,
    action: torch.Tensor,
    obs: torch.Tensor,
    agent_pos: torch.Tensor,
    action_mask: torch.Tensor | None = None,
    *,
    t: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return masked velocity MSE for a batch of normalized actions.

    ``action_mask`` has shape ``(batch, action_horizon)`` and excludes padded
    demonstration tails from both numerator and denominator.
    """
    if action.ndim != 3:
        raise ValueError("action must have shape (batch, horizon, action_dim)")
    batch = action.shape[0]
    if t is None:
        t = torch.rand(batch, device=action.device, dtype=action.dtype)
    if noise is None:
        noise = torch.randn_like(action)
    t_view = t[:, None, None]
    noisy_action = (1 - t_view) * noise + t_view * action
    target_velocity = action - noise
    error = (model(noisy_action, t, obs, agent_pos) - target_velocity).square()
    if action_mask is None:
        return error.mean()
    mask = action_mask.to(device=action.device, dtype=action.dtype)[..., None]
    denominator = (mask.sum() * action.shape[-1]).clamp_min(1)
    return (error * mask).sum() / denominator


@torch.no_grad()
def sample_euler(
    model: torch.nn.Module,
    obs: torch.Tensor,
    agent_pos: torch.Tensor,
    *,
    action_horizon: int,
    action_dim: int,
    steps: int = 8,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generate normalized action chunks by integrating ``dx/dt=v_theta``."""
    if steps < 1:
        raise ValueError("steps must be positive")
    shape = (obs.shape[0], action_horizon, action_dim)
    x = torch.randn(shape, device=obs.device, dtype=obs.dtype) if noise is None else noise
    if tuple(x.shape) != shape:
        raise ValueError(f"noise must have shape {shape}")
    dt = 1.0 / steps
    for index in range(steps):
        t = torch.full(
            (obs.shape[0],), (index + 0.5) * dt, device=obs.device, dtype=obs.dtype
        )
        x = x + dt * model(x, t, obs, agent_pos)
    return x
