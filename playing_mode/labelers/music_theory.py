"""
MusicTheoryLabeler — 基于乐理的 playing 标签生成策略

思路（待设计）:
  - 将 Part1 beat 对应到 lead sheet 的和弦骨架，保留和声功能性强的 beat（PLAY）
  - 弱拍、装饰音密集、与和弦不协和的 beat 标为 SKIP
  - 具体规则参考 reduce lead sheet 方法论

参考方向:
  - 节拍权重：强拍（1、3拍）倾向 PLAY，弱拍倾向 SKIP
  - 音符密度：空拍（empty beat）标为 SKIP
  - 和声匹配度：与 Part0 melody 的协和度评分

TODO (Phase 1):
  [ ] 确定输入格式：是否需要原始音高信息（需要在 context 中携带）
  [ ] 实现节拍权重规则
  [ ] 实现音符密度过滤（空 beat → SKIP）
  [ ] 实现和声协和度评分（可选）
"""

from __future__ import annotations

from playing_mode.labelers.base import BaseLabeler, LabelContext


class MusicTheoryLabeler(BaseLabeler):
    """
    基于乐理规则的 playing 标签生成策略。

    Parameters
    ----------
    empty_token_id : int
        空 beat 对应的 token id（config 中的 empty_marker，默认 169）。
    skip_empty : bool
        是否将空 beat 直接标为 SKIP（默认 True）。
    """

    def __init__(
        self,
        empty_token_id: int = 169,
        skip_empty: bool = True,
    ) -> None:
        self.empty_token_id = empty_token_id
        self.skip_empty = skip_empty

    def label_beat(
        self,
        beat_tokens: list[int],
        context: LabelContext,
    ) -> int:
        # TODO (Phase 1): 实现完整乐理规则

        # 最基础规则：空 beat 直接 SKIP
        if self.skip_empty and beat_tokens == [self.empty_token_id]:
            return 0  # SKIP

        raise NotImplementedError("MusicTheoryLabeler 完整实现待 Phase 1 完成，目前仅支持空 beat 检测。")
