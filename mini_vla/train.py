"""Train and evaluate the Phase 1 flow-matching PushT action head."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .config import PushTConfig, TrainConfig
from .data import PushTDataset
from .flow import flow_matching_loss, sample_euler
from .metrics import action_l1, action_l2
from .model import FlowActionHead


def _normalize_action(action: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (action - mean) / std


class _ModelEMA:
    """Maintain an exponential moving average used only for evaluation/checkpoints."""

    def __init__(self, model: FlowActionHead, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("ema_decay must be between 0 and 1")
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: FlowActionHead) -> None:
        for averaged, current in zip(self.model.parameters(), model.parameters(), strict=True):
            averaged.lerp_(current, 1.0 - self.decay)


def _checkpoint(
    model: FlowActionHead, cfg: TrainConfig, mean: np.ndarray, std: np.ndarray, *, epoch: int
) -> dict[str, object]:
    """Build a self-contained checkpoint for rollout or later resume support."""
    return {
        "epoch": epoch,
        "model": model.state_dict(),
        "model_config": {
            "obs_horizon": cfg.data.obs_horizon,
            "action_horizon": cfg.data.action_horizon,
            "action_dim": 2,
            "hidden_dim": cfg.hidden_dim,
            "proprio_dim": 2,
        },
        "action_mean": mean,
        "action_std": std,
        "train_config": asdict(cfg),
    }


def _split_dataset(dataset: PushTDataset, fraction: float, seed: int) -> tuple[Subset, Subset]:
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(dataset) < 2:
        raise ValueError("dataset needs at least two samples for a validation split")
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed)).tolist()
    validation_size = min(max(1, round(len(dataset) * fraction)), len(dataset) - 1)
    return Subset(dataset, indices[validation_size:]), Subset(dataset, indices[:validation_size])


def evaluate(
    model: FlowActionHead, loader: DataLoader, mean: torch.Tensor, std: torch.Tensor, *,
    sample_steps: int, max_batches: int, device: torch.device, seed: int = 0,
) -> dict[str, float]:
    """Report flow loss and Euler-sampled action errors with fixed Gaussian noise."""
    model.eval()
    totals = {"flow_loss": 0.0, "action_l1": 0.0, "action_l2": 0.0}
    batches = 0
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        for batch in loader:
            if batches == max_batches:
                break
            obs = batch["obs"].to(device, non_blocking=True)
            agent_pos = batch["agent_pos"].to(device, non_blocking=True)
            action = _normalize_action(batch["action"].to(device), mean, std)
            mask = batch["action_mask"].to(device)
            noise = torch.randn(action.shape, device=device, dtype=action.dtype, generator=generator)
            loss = flow_matching_loss(
                model, action, obs, agent_pos, mask, noise=noise,
                t=torch.full((len(obs),), 0.5, device=device),
            )
            prediction = sample_euler(
                model, obs, agent_pos, action_horizon=action.shape[1],
                action_dim=action.shape[2], steps=sample_steps, noise=noise,
            )
            totals["flow_loss"] += float(loss)
            totals["action_l1"] += action_l1(prediction, action, mask)
            totals["action_l2"] += action_l2(prediction, action, mask)
            batches += 1
    if batches == 0:
        raise ValueError("evaluation loader was empty")
    return {key: value / batches for key, value in totals.items()}


def train(cfg: TrainConfig) -> list[dict[str, float]]:
    """Run single-GPU Phase 1 optimization and return per-epoch metrics."""
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; pass --device cpu for a smoke test")
    random.seed(cfg.data.seed)
    np.random.seed(cfg.data.seed)
    torch.manual_seed(cfg.data.seed)
    dataset = PushTDataset(cfg.data)
    train_set, validation_set = _split_dataset(dataset, cfg.validation_fraction, cfg.data.seed)
    loader_args = {
        "batch_size": cfg.batch_size, "num_workers": cfg.num_workers,
        "pin_memory": device.type == "cuda", "persistent_workers": cfg.num_workers > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    validation_loader = DataLoader(validation_set, shuffle=False, **loader_args)
    model = FlowActionHead(
        obs_horizon=cfg.data.obs_horizon, action_horizon=cfg.data.action_horizon,
        hidden_dim=cfg.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    mean = torch.tensor(dataset.action_normalizer.mean, device=device, dtype=torch.float32)
    std = torch.tensor(dataset.action_normalizer.std, device=device, dtype=torch.float32)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ema = _ModelEMA(model, cfg.ema_decay) if cfg.use_ema else None
    scheduler = None
    if cfg.cosine_lr:
        total_steps = cfg.epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: 0.5 * (1.0 + math.cos(math.pi * step / total_steps)),
        )
    history: list[dict[str, float]] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            obs = batch["obs"].to(device, non_blocking=True)
            agent_pos = batch["agent_pos"].to(device, non_blocking=True)
            action = _normalize_action(batch["action"].to(device), mean, std)
            mask = batch["action_mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = flow_matching_loss(model, action, obs, agent_pos, mask)
            loss.backward()
            optimizer.step()
            if ema is not None:
                ema.update(model)
            if scheduler is not None:
                scheduler.step()
            total_loss += float(loss.detach())
        metrics = evaluate(
            ema.model if ema is not None else model,
            validation_loader, mean, std, sample_steps=cfg.sample_steps,
            max_batches=cfg.eval_batches, device=device, seed=cfg.data.seed,
        )
        metrics = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_flow_loss": total_loss / len(train_loader),
            **metrics,
        }
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        with (output_dir / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(metrics) + "\n")
        if epoch % cfg.checkpoint_every == 0 or epoch == cfg.epochs:
            checkpoint = _checkpoint(
                ema.model if ema is not None else model,
                cfg, dataset.action_normalizer.mean, dataset.action_normalizer.std, epoch=epoch
            )
            torch.save(checkpoint, output_dir / "last.pt")
            torch.save(checkpoint, output_dir / f"epoch_{epoch:04d}.pt")
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/phase1")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument("--obs-horizon", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--cosine-lr", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=-1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    train(TrainConfig(
        data=PushTConfig(
            zarr_path=args.data_dir, max_episodes=args.max_episodes,
            obs_horizon=args.obs_horizon, action_horizon=args.action_horizon,
        ),
        batch_size=args.batch_size, epochs=args.epochs, learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim, num_workers=args.num_workers, sample_steps=args.sample_steps,
        output_dir=args.output_dir, use_ema=args.use_ema, ema_decay=args.ema_decay,
        cosine_lr=args.cosine_lr, device=args.device,
    ))


if __name__ == "__main__":
    main()
