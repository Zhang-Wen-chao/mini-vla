from dataclasses import dataclass, field


@dataclass
class PushTConfig:
    """PushT 数据与采样配置。"""

    zarr_path: str = "data/pusht_cchi_v7_replay.zarr"
    obs_horizon: int = 2
    action_horizon: int = 16
    img_size: int = 96
    seed: int = 0
    max_episodes: int = -1
    img_range: tuple = field(default_factory=lambda: (-1.0, 1.0))


@dataclass
class TrainConfig:
    """Single-GPU Phase 1 training settings."""

    data: PushTConfig = field(default_factory=PushTConfig)
    batch_size: int = 128
    epochs: int = 50
    learning_rate: float = 3e-4
    hidden_dim: int = 256
    num_workers: int = 4
    validation_fraction: float = 0.05
    eval_batches: int = 20
    sample_steps: int = 8
    checkpoint_every: int = 1
    use_ema: bool = False
    ema_decay: float = 0.999
    cosine_lr: bool = False
    device: str = "cuda"
    output_dir: str = "outputs/phase1"
