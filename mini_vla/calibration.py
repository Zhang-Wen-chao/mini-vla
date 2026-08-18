"""Calibrate PushT replay action semantics against the rollout simulator.

This diagnostic exists to distinguish a learned-policy failure from an evaluator
or action-interface mismatch.  It establishes three invariants before a rollout
score is interpreted: replay actions are absolute workspace targets, a target
reduces the agent's distance under the simulator PD controller, and exact goal
geometry has unit coverage.  It requires the optional ``mini-vla[rollout]``
dependencies because it instantiates the headless simulator.

The replay action stream is not treated as an executable simulator oracle: it is
recorded alongside observations, and full replay can accumulate timing/contact
differences even when single-step state transitions align.  Demonstration replay
statistics are reported as diagnostics, not a pass/fail criterion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr

from .pusht_env import PushTImageEnv


def valid_transition_indices(episode_ends: np.ndarray, total_steps: int) -> np.ndarray:
    """Return replay indices with a next state in the same episode."""
    ends = np.asarray(episode_ends, dtype=np.int64)
    if ends.ndim != 1 or len(ends) == 0 or ends[-1] > total_steps:
        raise ValueError("invalid exclusive episode_ends")
    valid = np.ones(total_steps - 1, dtype=bool)
    valid[ends[:-1] - 1] = False
    return np.flatnonzero(valid)


def episode_ranges(episode_ends: np.ndarray, total_steps: int) -> list[tuple[int, int]]:
    """Convert exclusive replay offsets to non-empty half-open episode ranges."""
    ends = np.asarray(episode_ends, dtype=np.int64)
    if ends.ndim != 1 or len(ends) == 0 or ends[-1] > total_steps:
        raise ValueError("invalid exclusive episode_ends")
    starts = np.concatenate(([0], ends[:-1]))
    ranges = [(int(start), int(end)) for start, end in zip(starts, ends)]
    if any(end <= start for start, end in ranges):
        raise ValueError("episodes must be non-empty")
    return ranges


def replay_target_metrics(
    action: np.ndarray, state: np.ndarray, episode_ends: np.ndarray, *, max_samples: int, seed: int,
) -> dict[str, float | int]:
    """Measure whether each recorded XY action attracts the next agent state."""
    indices = valid_transition_indices(episode_ends, len(action))
    if max_samples > 0 and len(indices) > max_samples:
        indices = np.random.default_rng(seed).choice(indices, max_samples, replace=False)
    target = np.asarray(action[indices], dtype=np.float64)
    current = np.asarray(state[indices, :2], dtype=np.float64)
    following = np.asarray(state[indices + 1, :2], dtype=np.float64)
    before = np.linalg.norm(target - current, axis=1)
    after = np.linalg.norm(target - following, axis=1)
    return {
        "transition_samples": int(len(indices)),
        "action_min": float(target.min()),
        "action_max": float(target.max()),
        "mean_target_distance_before": float(before.mean()),
        "mean_target_distance_after": float(after.mean()),
        "fraction_target_distance_decreases": float(np.mean(after < before)),
    }


def replay_rollout_metrics(
    action: np.ndarray, state: np.ndarray, episode_ends: np.ndarray, *, max_episodes: int, seed: int,
    legacy: bool = False,
) -> dict[str, float | int]:
    """Replay full demonstrations through the simulator from their recorded initial states.

    The score is intentionally separate from policy rollout: it checks whether
    long-horizon simulator drift can explain a learned policy's low coverage.
    """
    ranges = episode_ranges(episode_ends, len(action))
    indices = np.arange(len(ranges))
    if max_episodes > 0 and len(indices) > max_episodes:
        indices = np.sort(np.random.default_rng(seed).choice(indices, max_episodes, replace=False))
    coverages: list[float] = []
    successes: list[bool] = []
    final_state_errors: list[float] = []
    env = PushTImageEnv(render_size=96, legacy=legacy)
    try:
        for episode_index in indices:
            start, end = ranges[int(episode_index)]
            env.reset(state=np.asarray(state[start], dtype=np.float64))
            max_coverage = env.coverage()
            # ``action[t]`` leads from recorded ``state[t]`` to ``state[t+1]``.
            # The terminal action has no following state in the replay, so omit
            # it from this fidelity measurement.
            for target in action[start : end - 1]:
                _, _, _, info = env.step(target)
                max_coverage = max(max_coverage, float(info["coverage"]))
            expected_final = np.asarray(state[end - 1, 2:4], dtype=np.float64)
            observed_final = np.asarray(env.block.position, dtype=np.float64)
            final_state_errors.append(float(np.linalg.norm(observed_final - expected_final)))
            coverages.append(max_coverage)
            successes.append(max_coverage > env.success_threshold)
    finally:
        env.close()
    return {
        "episodes": int(len(indices)),
        "mean_max_coverage": float(np.mean(coverages)),
        "success_rate": float(np.mean(successes)),
        "mean_final_block_position_error": float(np.mean(final_state_errors)),
        "max_final_block_position_error": float(np.max(final_state_errors)),
    }


def _angle_error(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Return signed shortest angular error in radians."""
    return (actual - expected + np.pi) % (2 * np.pi) - np.pi


def replay_dynamics_metrics(
    action: np.ndarray, state: np.ndarray, episode_ends: np.ndarray, *, max_samples: int, seed: int,
) -> dict[str, float | int]:
    """Compare one official replay step to a simulator step from the same state.

    This is a stricter evaluator check than merely testing target-coordinate
    semantics: it detects incorrect controller or contact parameters before an
    imitation policy's rollout score is attributed to its model.
    """
    indices = valid_transition_indices(episode_ends, len(action))
    if max_samples > 0 and len(indices) > max_samples:
        indices = np.random.default_rng(seed).choice(indices, max_samples, replace=False)
    agent_error: list[float] = []
    block_position_error: list[float] = []
    block_angle_error: list[float] = []
    env = PushTImageEnv(render_size=96)
    try:
        for index in indices:
            env.reset(state=np.asarray(state[index], dtype=np.float64))
            _, _, _, info = env.step(action[index])
            expected = np.asarray(state[index + 1], dtype=np.float64)
            agent_error.append(float(np.linalg.norm(np.asarray(info["pos_agent"]) - expected[:2])))
            observed_block = np.asarray(info["block_pose"], dtype=np.float64)
            block_position_error.append(float(np.linalg.norm(observed_block[:2] - expected[2:4])))
            block_angle_error.append(float(abs(_angle_error(observed_block[2], expected[4]))))
    finally:
        env.close()
    return {
        "transition_samples": int(len(indices)),
        "mean_agent_position_error": float(np.mean(agent_error)),
        "max_agent_position_error": float(np.max(agent_error)),
        "mean_block_position_error": float(np.mean(block_position_error)),
        "max_block_position_error": float(np.max(block_position_error)),
        "mean_block_angle_error_rad": float(np.mean(block_angle_error)),
        "max_block_angle_error_rad": float(np.max(block_angle_error)),
    }


def simulator_invariants() -> dict[str, float | bool]:
    """Check absolute-target motion and the coverage geometry in isolation."""
    env = PushTImageEnv(render_size=96)
    try:
        # Keep the block far away so this is a pure controller-coordinate test.
        initial_state = np.array([100.0, 100.0, 400.0, 400.0, 0.0])
        target = np.array([300.0, 150.0])
        env.reset(state=initial_state)
        before = float(np.linalg.norm(target - env.agent_position()))
        _, _, _, info = env.step(target)
        after = float(np.linalg.norm(target - info["pos_agent"]))

        exact_goal = np.array([100.0, 100.0, *env.goal_pose])
        env.reset(state=exact_goal)
        goal_coverage = env.coverage()
    finally:
        env.close()
    return {
        "simulator_target_distance_before": before,
        "simulator_target_distance_after": after,
        "simulator_target_moves_agent_closer": after < before,
        "exact_goal_coverage": goal_coverage,
    }


def calibrate(
    data_dir: str, *, max_samples: int = 1_024, replay_episodes: int = 20, seed: int = 0,
    legacy: bool = False,
) -> dict[str, object]:
    """Run replay and simulator calibration checks and return JSON-safe results."""
    root = zarr.open(data_dir, mode="r")
    replay = replay_target_metrics(
        np.asarray(root["data"]["action"]), np.asarray(root["data"]["state"]),
        np.asarray(root["meta"]["episode_ends"]), max_samples=max_samples, seed=seed,
    )
    simulator = simulator_invariants()
    dynamics = replay_dynamics_metrics(
        np.asarray(root["data"]["action"]), np.asarray(root["data"]["state"]),
        np.asarray(root["meta"]["episode_ends"]), max_samples=max_samples, seed=seed,
    )
    demonstration_rollout = replay_rollout_metrics(
        np.asarray(root["data"]["action"]), np.asarray(root["data"]["state"]),
        np.asarray(root["meta"]["episode_ends"]), max_episodes=replay_episodes, seed=seed,
        legacy=legacy,
    )
    checks = {
        "replay_actions_within_workspace": 0.0 <= replay["action_min"]
        and replay["action_max"] <= PushTImageEnv.window_size,
        "replay_actions_move_agent_toward_target": replay["fraction_target_distance_decreases"] > 0.5,
        "simulator_target_coordinate_semantics": simulator["simulator_target_moves_agent_closer"],
        "simulator_exact_goal_coverage": bool(np.isclose(simulator["exact_goal_coverage"], 1.0)),
    }
    return {
        "data_dir": str(Path(data_dir)), "max_samples": max_samples,
        "replay_episodes": replay_episodes, "seed": seed, "legacy": legacy,
        "replay": replay,
        "simulator": simulator, "dynamics": dynamics,
        "demonstration_rollout_diagnostic": demonstration_rollout, "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=1_024)
    parser.add_argument("--replay-episodes", type=int, default=20)
    parser.add_argument(
        "--legacy", action=argparse.BooleanOptionalAction, default=False,
        help="use legacy state assignment for optional replay diagnostics (default: false)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = calibrate(
        args.data_dir, max_samples=args.max_samples, replay_episodes=args.replay_episodes, seed=args.seed,
        legacy=args.legacy,
    )
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
