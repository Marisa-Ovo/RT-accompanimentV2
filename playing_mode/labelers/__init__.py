"""
Labelers — 训练时 playing 标签生成策略

已实现:
  (均为 TODO，Phase 1 起逐步填入)

未来导出示例:
    from playing_mode.labelers import NoiseLabeler, NLLLabeler
"""

from playing_mode.labelers.base import BaseLabeler, LabelContext  # noqa: F401

__all__ = [
    "BaseLabeler",
    "LabelContext",
    # Phase 1+:
    # "MusicTheoryLabeler",
    # "WeakModelLabeler",
    # "NoiseLabeler",
    # "NLLLabeler",
]
