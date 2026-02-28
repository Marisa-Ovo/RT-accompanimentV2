"""
NLLLabeler — 基于 NLL 值划分的 playing 标签生成策略

思路（待设计）:
  1. 用已训练好的 PianoLLaMA 计算每个 Part1 beat 的 NLL（负对数似然）
  2. NLL 高 → 模型认为此 beat 难以预测 → 标为 SKIP
  3. NLL 低 → 模型对此 beat 有把握 → 标为 PLAY
  4. 阈值可固定或按全曲 NLL 分布自适应（如取中位数）

此策略的数据流与其他 labeler 不同：
  - 需要在 label_sequence() 中先计算全曲所有 beat 的 NLL（需要模型推理）
  - 再根据分布阈值批量划分 PLAY/SKIP
  - 因此覆写了 label_sequence() 而非逐 beat 调用 label_beat()

TODO (Phase 2/3):
  [ ] 实现 NLL 计算（复用 model.py 的 forward pass，teacher forcing 模式）
  [ ] 确定阈值策略：fixed / percentile / adaptive
  [ ] 考虑：NLL 是否按 token 平均还是取 max？
  [ ] 考虑：是否需要对 BPM/拍号归一化 NLL？
"""

from __future__ import annotations

from playing_mode.labelers.base import BaseLabeler, LabelContext


class NLLLabeler(BaseLabeler):
    """
    基于模型 NLL 值的 playing 标签生成策略。

    Parameters
    ----------
    model_path : str
        用于计算 NLL 的已训练 PianoLLaMA checkpoint 路径。
    threshold_strategy : str
        阈值策略："fixed" | "percentile" | "median"
    threshold : float
        仅 threshold_strategy="fixed" 时使用，NLL 超过此值标为 SKIP。
    percentile : float
        仅 threshold_strategy="percentile" 时使用，取全曲 NLL 分布的第 N 百分位作阈值。
    device : str
        推理设备，如 "cuda" 或 "cpu"。
    """

    def __init__(
        self,
        model_path: str,
        threshold_strategy: str = "median",
        threshold: float = 2.0,
        percentile: float = 50.0,
        device: str = "cuda",
    ) -> None:
        assert threshold_strategy in ("fixed", "percentile", "median"), f"未知 threshold_strategy: {threshold_strategy}"
        self.model_path = model_path
        self.threshold_strategy = threshold_strategy
        self.threshold = threshold
        self.percentile = percentile
        self.device = device
        # TODO: 加载模型

    def label_beat(
        self,
        beat_tokens: list[int],
        context: LabelContext,
    ) -> int:
        """
        单 beat 模式：要求 context.nll_score 已被预填入（由 label_sequence 统一计算）。
        """
        if context.nll_score is None:
            raise ValueError(
                "NLLLabeler.label_beat() 要求 context.nll_score 已预填入。请使用 label_sequence() 而非逐 beat 调用。"
            )
        return 0 if context.nll_score > self.threshold else 1

    def label_sequence(
        self,
        all_beat_tokens: list[list[int]],
        all_contexts: list[LabelContext],
    ) -> list[int]:
        """
        覆写：先批量计算全曲 NLL，再根据分布阈值划分。
        """
        # TODO (Phase 2): 实现 NLL 批量计算
        # 1. 构造 teacher-forcing 序列
        # 2. 前向传播计算每 beat 的平均 NLL
        # 3. 根据 threshold_strategy 确定阈值
        # 4. 填入 context.nll_score 并调用 label_beat
        raise NotImplementedError("NLLLabeler 待 Phase 2/3 实现。")
