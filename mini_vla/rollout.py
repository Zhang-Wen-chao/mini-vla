"""Evaluate a trained flow action head in the headless PushT simulator.

Requires ``pip install -e '.[rollout]'``. The reported ``max_coverage`` is the
maximum T/goal area overlap over an episode; success means coverage > 0.95.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .flow import sample_euler
from .model import FlowActionHead
from .pusht_env import PushTImageEnv


def _image_tensor(history: deque[np.ndarray], device: torch.device) -> torch.Tensor:
    image = np.stack(history).astype(np.float32) / 255.0
    image = image * 2.0 - 1.0
    return torch.from_numpy(image.transpose(0, 3, 1, 2)).unsqueeze(0).to(device)


def _agent_pos_tensor(history: deque[np.ndarray], device: torch.device) -> torch.Tensor:
    positions = np.stack(history).astype(np.float32) / 256.0 - 1.0
    return torch.from_numpy(positions).unsqueeze(0).to(device)


def load_model(checkpoint_path: str, device: torch.device) -> tuple[FlowActionHead, torch.Tensor, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = FlowActionHead(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    mean = torch.as_tensor(checkpoint["action_mean"], dtype=torch.float32, device=device)
    std = torch.as_tensor(checkpoint["action_std"], dtype=torch.float32, device=device)
    return model, mean, std


@torch.no_grad()
def rollout_episode(
    model: FlowActionHead, mean: torch.Tensor, std: torch.Tensor, *, seed: int,
    max_steps: int = 200, execute_steps: int = 8, sample_steps: int = 8, device: torch.device,
    legacy: bool = True,
) -> dict[str, float | int | bool]:
    """Run one deterministic-seed episode, executing the first actions of each chunk.

    ``legacy=True`` is the official image-PushT evaluation setting for the
    ``pusht_cchi_v7_replay`` dataset.  It preserves legacy assignment order
    when loading a block pose with a non-zero center of gravity.
    """
    if not 1 <= execute_steps <= model.action_horizon:
        raise ValueError(
            "execute_steps must be between 1 and the model action horizon "
            f"({model.action_horizon}), got {execute_steps}"
        )
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    env = PushTImageEnv(render_size=96, legacy=legacy)
    first_image = env.reset(seed=seed)
    history: deque[np.ndarray] = deque(
        [first_image] * model.obs_horizon, maxlen=model.obs_horizon,
    )
    position_history: deque[np.ndarray] = deque(
        [env.agent_position()] * model.obs_horizon, maxlen=model.obs_horizon,
    )
    max_coverage, steps, success = env.coverage(), 0, False
    while steps < max_steps and not success:
        action = sample_euler(
            model, _image_tensor(history, device), _agent_pos_tensor(position_history, device),
            action_horizon=model.action_horizon,
            action_dim=model.action_dim, steps=sample_steps,
        )[0]
        action = (action * std + mean).cpu().numpy().clip(0, 512)
        for target in action[:execute_steps]:
            image, _, success, info = env.step(target)
            history.append(image)
            position_history.append(env.agent_position())
            steps += 1
            max_coverage = max(max_coverage, float(info["coverage"]))
            if success or steps == max_steps:
                break
    env.close()
    return {
        "seed": seed, "steps": steps, "max_coverage": max_coverage,
        "success": success,
    }


def evaluate(
    checkpoint_path: str, *, episodes: int, seed: int, max_steps: int, execute_steps: int,
    sample_steps: int, device: torch.device, legacy: bool = True,
) -> dict[str, object]:
    # Euler sampling starts from Gaussian noise, so seed it independently of
    # environment seeds to make a reported checkpoint evaluation reproducible.
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model, mean, std = load_model(checkpoint_path, device)
    results = [rollout_episode(
        model, mean, std, seed=seed + index, max_steps=max_steps, execute_steps=execute_steps,
        sample_steps=sample_steps, device=device, legacy=legacy,
    ) for index in range(episodes)]
    return {
        "checkpoint": str(Path(checkpoint_path)), "episodes": episodes, "seed_start": seed,
        "max_steps": max_steps, "execute_steps": execute_steps, "euler_steps": sample_steps,
        "legacy": legacy,
        "mean_max_coverage": float(np.mean([item["max_coverage"] for item in results])),
        "success_rate": float(np.mean([item["success"] for item in results])),
        "episodes_detail": results,
    }


def evaluate_control_frequency_sweep(
    checkpoint_path: str, *, episodes: int, seed: int, max_steps: int,
    execute_steps_list: list[int], sample_steps: int, device: torch.device, legacy: bool = True,
) -> dict[str, object]:
    """Evaluate one checkpoint under several closed-loop replanning rates.

    Every setting gets the identical environment seed range and a reset flow-noise
    RNG.  Its trajectory may still diverge after the first different replan, which
    is exactly the closed-loop effect this controlled comparison measures.
    """
    if not execute_steps_list:
        raise ValueError("execute_steps_list must not be empty")
    if len(set(execute_steps_list)) != len(execute_steps_list):
        raise ValueError("execute_steps_list must not contain duplicates")
    results: dict[str, object] = {}
    for execute_steps in execute_steps_list:
        results[str(execute_steps)] = evaluate(
            checkpoint_path, episodes=episodes, seed=seed, max_steps=max_steps,
            execute_steps=execute_steps, sample_steps=sample_steps, device=device, legacy=legacy,
        )
    return {
        "checkpoint": str(Path(checkpoint_path)),
        "comparison": "control_replanning_frequency",
        "execute_steps_list": execute_steps_list,
        "episodes": episodes,
        "seed_start": seed,
        "max_steps": max_steps,
        "euler_steps": sample_steps,
        "legacy": legacy,
        "by_execute_steps": results,
    }


def _parse_execute_steps_list(value: str) -> list[int]:
    """Parse the compact CLI representation used by the sweep command."""
    try:
        values = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "execute steps must be a comma-separated list of integers, e.g. 1,4,8"
        ) from error
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("execute steps must all be positive")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("execute steps must not contain duplicates")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--max-steps", type=int, default=200)
    execution_group = parser.add_mutually_exclusive_group()
    execution_group.add_argument("--execute-steps", type=int)
    execution_group.add_argument(
        "--execute-steps-list", type=_parse_execute_steps_list,
        help="compare several chunk execution lengths, for example 1,4,8",
    )
    parser.add_argument("--sample-steps", type=int, default=8)
    parser.add_argument(
        "--legacy", action=argparse.BooleanOptionalAction, default=True,
        help="use the official PushT image-evaluation legacy state semantics (default: true)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output")
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.execute_steps_list is not None:
        result = evaluate_control_frequency_sweep(
            args.checkpoint, episodes=args.episodes, seed=args.seed, max_steps=args.max_steps,
            execute_steps_list=args.execute_steps_list, sample_steps=args.sample_steps, device=device,
            legacy=args.legacy,
        )
    else:
        result = evaluate(
            args.checkpoint, episodes=args.episodes, seed=args.seed, max_steps=args.max_steps,
            execute_steps=8 if args.execute_steps is None else args.execute_steps,
            sample_steps=args.sample_steps, device=device, legacy=args.legacy,
        )
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n")


if __name__ == "__main__":
    main()
