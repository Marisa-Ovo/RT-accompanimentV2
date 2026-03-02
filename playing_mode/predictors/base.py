"""
BasePredictor — 推理时 playing token 预测的抽象基类

接口约定:
  输入: token_sequence（当前自回归生成上下文）+ 当前 beat 的 tokens
  输出: float — 当前 beat 为 PLAY 的概率 [0, 1]

推理流程示意（由 inference.py 调用，Phase 2 实现）:
    predictor = SomePredictor(...)
    prob = predictor.predict(context_tokens, beat_tokens)
    playing_token = PLAY if prob > threshold else SKIP
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BasePredictor(ABC):
    """
    推理时 playing token 预测架构基类。

    使用方式（由 model.generate_accompaniment() 调用）:
        predictor = SomePredictor(...)
        play_prob = predictor.predict(context_tokens, current_beat_tokens)
        # play_prob > threshold → 插入 PLAY token，否则插入 SKIP token
    """

    @abstractmethod
    def predict(
        self,
        context_tokens: list[int],
        current_beat_tokens: list[int],
    ) -> float:
        """
        预测当前 beat 的 playing 概率。

        Parameters
        ----------
        context_tokens : list[int]
            当前生成上下文（BOS 到当前位置的所有 token）。
        current_beat_tokens : list[int]
            刚生成的 Part1 beat token 序列（不含 playing token）。

        Returns
        -------
        float
            PLAY 的概率，范围 [0, 1]。
        """
        ...

    def decide(
        self,
        context_tokens: list[int],
        current_beat_tokens: list[int],
        threshold: float = 0.5,
    ) -> int:
        """
        便捷方法：在 predict() 基础上直接返回 PLAY=1 / SKIP=0。
        """
        prob = self.predict(context_tokens, current_beat_tokens)
        return 1 if prob >= threshold else 0
