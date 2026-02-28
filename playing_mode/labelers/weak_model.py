"""
WeakModelLabeler — 基于弱模型对比的 playing 标签生成策略

思路（待设计）:
  1. 用低质量（如更少训练步、更小模型）的 PianoLLaMA 为同一 Part0 生成 Part1
  2. 对比弱模型生成结果与 GT：
     - 差异大（弱模型"猜错了"）的 beat → 标为 SKIP（这些 beat 模型不擅长，应跳过）
     - 差异小的 beat → 标为 PLAY
  3. 差异度量可选：token 完全匹配 / edit distance / piano-roll IoU

参数设计思路:
  - weak_model_path: 弱模型 checkpoint 路径
  - diff_threshold: 差异超过此值标为 SKIP
  - diff_metric: "exact" | "edit_distance" | "iou"

TODO (Phase 1/2):
  [ ] 确定差异度量方式
  [ ] 实现弱模型推理（复用 inference.py 的 generate_accompaniment）
  [ ] 实现 beat 级差异计算
  [ ] 确定 diff_threshold 超参的选取方式（可由验证集 F1 调优）
"""

from __future__ import annotations

from playing_mode.labelers.base import BaseLabeler, LabelContext


class WeakModelLabeler(BaseLabeler):
    """
    基于弱模型生成结果对比的 playing 标签生成策略。

    Parameters
    ----------
    weak_model_path : str
        弱模型 checkpoint 路径。
    diff_threshold : float
        beat 级差异超过此值时标为 SKIP（具体量纲由 diff_metric 决定）。
    diff_metric : str
        差异度量方式："exact" | "edit_distance" | "iou"
    """

    def __init__(
        self,
        weak_model_path: str,
        diff_threshold: float = 0.5,
        diff_metric: str = "exact",
    ) -> None:
        self.weak_model_path = weak_model_path
        self.diff_threshold = diff_threshold
        self.diff_metric = diff_metric
        # TODO: 加载弱模型

    def label_beat(
        self,
        beat_tokens: list[int],
        context: LabelContext,
    ) -> int:
        # TODO (Phase 2): 实现弱模型推理 + beat 差异计算
        raise NotImplementedError("WeakModelLabeler 待 Phase 2 实现。")
