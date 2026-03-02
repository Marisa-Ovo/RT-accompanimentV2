"""
Playing Mode 模块
=================
为每个生成的 Part1 beat 预测 / 标注一个 playing token（PLAY / SKIP）。

子模块:
  labelers   - 训练时自动生成 playing 标签的各类策略
  predictors - 推理时预测 playing token 的各类架构

快速使用示例（待 Phase 1 实现后填入）:
    from playing_mode.labelers import NoiseLabeler
    from playing_mode.predictors import AuxHeadPredictor
"""

from playing_mode.labelers.base import BaseLabeler, LabelContext  # noqa: F401
from playing_mode.predictors.base import BasePredictor  # noqa: F401

# playing token 的语义常量（实际 token id 应在 config.py 中分配后引入）
PLAY = 1
SKIP = 0
