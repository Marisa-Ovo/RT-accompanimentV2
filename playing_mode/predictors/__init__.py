"""
Predictors — 推理时 playing token 预测架构

已实现:
  (均为 TODO，Phase 2 起逐步填入)

未来导出示例:
    from playing_mode.predictors import AuxHeadPredictor
"""

from playing_mode.predictors.base import BasePredictor  # noqa: F401

__all__ = [
    "BasePredictor",
    # Phase 2+:
    # "SeparateModelPredictor",
    # "AuxHeadPredictor",
]
