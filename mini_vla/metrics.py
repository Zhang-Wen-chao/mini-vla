"""动作预测指标（Phase 0）。

Phase 1 起与 rollout 覆盖率一起构成完整评测；
这里只定义张量级指标，供单测与后续训练循环复用。
"""

from __future__ import annotations

import torch


def _apply_mask(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None):
    if mask is not None:
        mask = mask.to(pred.dtype).reshape(-1, 1).to(pred.device)
        pred, target = pred * mask, target * mask
    return pred, target, mask


def action_l1(pred: torch.Tensor, target: torch.Tensor,
              mask: torch.Tensor | None = None) -> float:
    """归一化空间下的动作 L1 误差（被 mask 的 padding 不计入）。"""
    pred, target, mask = _apply_mask(pred, target, mask)
    if mask is not None:
        return float((pred - target).abs().sum() / mask.sum())
    return float((pred - target).abs().mean())


def action_l2(pred: torch.Tensor, target: torch.Tensor,
              mask: torch.Tensor | None = None) -> float:
    """归一化空间下的动作 L2 误差（被 mask 的 padding 不计入）。"""
    pred, target, mask = _apply_mask(pred, target, mask)
    if mask is not None:
        return float(((pred - target) ** 2).sum() / mask.sum())
    return float(((pred - target) ** 2).mean())


def chunk_consistency(action_chunk: torch.Tensor) -> float:
    """分块内平滑度：相邻步差的 L2 均值，越小越连贯。"""
    diff = action_chunk[:, 1:] - action_chunk[:, :-1]
    return float((diff ** 2).mean())
