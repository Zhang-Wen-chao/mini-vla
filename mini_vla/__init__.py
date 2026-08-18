"""mini-vla：从 PushT 数据到实时推理的 VLA 全流程教学实现。

Phase 0 提供数据层（PushT zarr 加载 + 序列采样 + 归一化）与指标定义；
Phase 1 起接入 flow-matching 动作头（目标函数复用 mini-diffusion）。
"""

__version__ = "0.1.0"

from .config import PushTConfig
from .data import PushTDataset, LinearNormalizer

__all__ = ["PushTConfig", "PushTDataset", "LinearNormalizer", "__version__"]
