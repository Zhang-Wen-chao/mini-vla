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
