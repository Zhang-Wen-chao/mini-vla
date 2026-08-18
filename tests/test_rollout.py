"""Optional simulator checks; skipped unless the rollout extra is installed."""

import importlib.util
import json

import pytest
import torch

from mini_vla import rollout
from mini_vla.calibration import (
    _angle_error, episode_ranges, replay_rollout_metrics, replay_target_metrics, valid_transition_indices,
)


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pymunk") is None, reason="requires mini-vla[rollout]"
)


def test_pusht_reset_step_and_coverage_range():
    from mini_vla.pusht_env import PushTImageEnv

    env = PushTImageEnv(render_size=96)
    image = env.reset(seed=10_000)
    next_image, reward, done, info = env.step([256, 256])
    assert image.shape == next_image.shape == (96, 96, 3)
    assert image.dtype.name == "uint8"
    assert 0.0 <= reward <= 1.0
    assert isinstance(done, bool)
    assert 0.0 <= info["coverage"] <= 1.0
    env.close()


def test_policy_rollout_defaults_to_official_legacy_environment(monkeypatch):
    created_legacy = []

    class FakeEnv:
        success_threshold = 0.95

        def __init__(self, **kwargs):
            created_legacy.append(kwargs["legacy"])

        def reset(self, *, seed):
            return torch.zeros(96, 96, 3, dtype=torch.uint8).numpy()

        def agent_position(self):
            return torch.zeros(2).numpy()

        def coverage(self):
            return 1.0

        def step(self, action):
            return self.reset(seed=0), 1.0, True, {"coverage": 1.0}

        def close(self):
            return None

    monkeypatch.setattr(rollout, "PushTImageEnv", FakeEnv)
    monkeypatch.setattr(
        rollout, "sample_euler", lambda *args, **kwargs: torch.zeros(1, 1, 2)
    )
    model = type("Model", (), {"action_horizon": 1, "action_dim": 2, "obs_horizon": 1})()
    rollout.rollout_episode(
        model, torch.zeros(2), torch.ones(2), seed=0, max_steps=1, execute_steps=1,
        sample_steps=1, device=torch.device("cpu"),
    )
    assert created_legacy == [True]


def test_calibration_excludes_episode_boundary_transitions():
    ends = torch.tensor([3, 6]).numpy()
    assert valid_transition_indices(ends, total_steps=6).tolist() == [0, 1, 3, 4]
    assert episode_ranges(ends, total_steps=6) == [(0, 3), (3, 6)]


def test_replay_target_metrics_uses_only_within_episode_steps():
    action = torch.tensor([[2.0, 0.0]] * 6).numpy()
    state = torch.tensor([
        [0.0, 0.0, 0, 0, 0], [1.0, 0.0, 0, 0, 0], [2.0, 0.0, 0, 0, 0],
        [0.0, 0.0, 0, 0, 0], [1.0, 0.0, 0, 0, 0], [2.0, 0.0, 0, 0, 0],
    ]).numpy()
    metrics = replay_target_metrics(action, state, torch.tensor([3, 6]).numpy(), max_samples=-1, seed=0)
    assert metrics["transition_samples"] == 4
    assert metrics["fraction_target_distance_decreases"] == pytest.approx(1.0)


def test_replay_rollout_metrics_passes_legacy_to_environment(monkeypatch):
    created_legacy = []

    class FakeEnv:
        success_threshold = 0.95

        def __init__(self, **kwargs):
            created_legacy.append(kwargs["legacy"])
            self.block = type("Block", (), {"position": [0.0, 0.0]})()

        def reset(self, *, state):
            return None

        def coverage(self):
            return 0.0

        def step(self, action):
            return None, 0.0, False, {"coverage": 0.0}

        def close(self):
            return None

    from mini_vla import calibration

    monkeypatch.setattr(calibration, "PushTImageEnv", FakeEnv)
    replay_rollout_metrics(
        torch.zeros(3, 2).numpy(), torch.zeros(3, 5).numpy(), torch.tensor([3]).numpy(),
        max_episodes=-1, seed=0, legacy=False,
    )
    assert created_legacy == [False]


def test_angle_error_wraps_across_pi_boundary():
    assert _angle_error(torch.tensor(-3.13).numpy(), torch.tensor(3.13).numpy()) == pytest.approx(
        0.0231853, abs=1e-6
    )


def test_calibration_cli_creates_output_parent(monkeypatch, tmp_path):
    from mini_vla import calibration

    monkeypatch.setattr(calibration, "calibrate", lambda *args, **kwargs: {"passed": True})
    output = tmp_path / "nested" / "calibration.json"
    monkeypatch.setattr(
        "sys.argv", ["calibration", "--data-dir", "ignored", "--output", str(output)]
    )
    calibration.main()
    assert json.loads(output.read_text()) == {"passed": True}


def test_control_frequency_sweep_reuses_the_same_protocol(monkeypatch):
    calls = []

    def fake_evaluate(checkpoint_path, **kwargs):
        calls.append((checkpoint_path, kwargs))
        return {"execute_steps": kwargs["execute_steps"]}

    monkeypatch.setattr(rollout, "evaluate", fake_evaluate)
    result = rollout.evaluate_control_frequency_sweep(
        "checkpoint.pt", episodes=20, seed=10_000, max_steps=200,
        execute_steps_list=[1, 4, 8], sample_steps=8, device=torch.device("cpu"),
    )

    assert result["comparison"] == "control_replanning_frequency"
    assert result["by_execute_steps"] == {
        "1": {"execute_steps": 1},
        "4": {"execute_steps": 4},
        "8": {"execute_steps": 8},
    }
    assert [kwargs for _, kwargs in calls] == [
        {
            "episodes": 20, "seed": 10_000, "max_steps": 200,
            "execute_steps": execute_steps, "sample_steps": 8,
            "device": torch.device("cpu"), "legacy": True,
        }
        for execute_steps in [1, 4, 8]
    ]


@pytest.mark.parametrize("value", ["", "0", "1,1", "1,wrong"])
def test_execute_steps_list_rejects_invalid_values(value):
    with pytest.raises(Exception):
        rollout._parse_execute_steps_list(value)
