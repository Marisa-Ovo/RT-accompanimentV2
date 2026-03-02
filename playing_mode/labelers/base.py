"""
BaseLabeler — 标签生成策略的抽象基类

所有 labeler 实现必须继承此类并实现 `label_beat()`。

接口约定:
  输入: beat_tokens (单个 beat 的 token id 列表), context (辅助上下文)
  输出: int — PLAY=1 / SKIP=0
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class LabelContext:
    """
    labeler 可用的辅助上下文信息。
    各 labeler 按需使用，不需要的字段可忽略。
    """

    beat_index: int = 0  # 当前 beat 在整首曲子中的索引
    part0_beat_tokens: list[int] = field(default_factory=list)  # 同位置的 Part0 melody beat
    prev_part1_tokens: list[int] = field(default_factory=list)  # 前若干 beat 的 Part1 历史
    gt_part1_beat_tokens: list[int] = field(default_factory=list)  # GT Part1（用于对比）
    nll_score: float | None = None  # 该 beat 的 NLL（由 NLLLabeler 预计算填入）
    extra: dict = field(default_factory=dict)  # 扩展字段，各 labeler 自定义


class BaseLabeler(ABC):
    """
    playing 标签生成策略基类。
    子类实现 `label_beat()` 即可插入 PianoDataset 的数据构造流程。

    使用方式（由 PianoDataset 调用，Phase 1 实现）:
        labeler = SomeLabeler(...)
        label = labeler.label_beat(beat_tokens, context)
        # label: 1 = PLAY, 0 = SKIP
    """

    @abstractmethod
    def label_beat(
        self,
        beat_tokens: list[int],
        context: LabelContext,
    ) -> int:
        """
        为单个 Part1 beat 生成 playing 标签。

        Parameters
        ----------
        beat_tokens : list[int]
            该 beat 经 PianoRollTokenizer 编码后的 token id 序列。
        context : LabelContext
            辅助上下文，包含 beat 位置、melody、历史等信息。

        Returns
        -------
        int
            1 = PLAY（保留此 beat），0 = SKIP（跳过/静音）
        """
        ...

    def label_sequence(
        self,
        all_beat_tokens: list[list[int]],
        all_contexts: list[LabelContext],
    ) -> list[int]:
        """
        批量对整首曲子的所有 beat 生成标签（默认逐 beat 调用 label_beat）。
        子类可覆写此方法以实现需要全局信息的策略（如 NLL 阈值划分）。
        """
        assert len(all_beat_tokens) == len(all_contexts)
        return [self.label_beat(tokens, ctx) for tokens, ctx in zip(all_beat_tokens, all_contexts)]
