"""
NoiseLabeler — 手动加噪的 playing 标签生成策略

思路（最简单，推荐 Phase 1 优先实现验证 token 格式）:
  直接对 GT Part1 按一定策略随机或规律地将部分 beat 标为 SKIP，
  并在训练序列中对应位置插入空 beat。

  这相当于将"不稳定/不确定的 beat"人为制造为训练数据，
  让模型学会在 SKIP 位置不输出音符。

噪声策略:
  - random:    以固定概率 p 随机将 beat 标为 SKIP
  - periodic:  每隔 N 个 beat 标为 SKIP（模拟规律简化）
  - density:   音符数少于阈值的 beat 标为 SKIP（稀疏 beat 跳过）

TODO (Phase 1):
  [ ] 实现 random 策略
  [ ] 实现 density 策略（需要从 beat_tokens 解析音符数，或在 context 中携带）
  [ ] 实现 periodic 策略
  [ ] 确定：SKIP 时训练序列中对应 Part1 beat 是替换为全空 beat，还是保留原始 GT？
"""

from __future__ import annotations

import random

from playing_mode.labelers.base import BaseLabeler, LabelContext


class NoiseLabeler(BaseLabeler):
    """
    手动加噪策略：以指定方式随机/规律标注 SKIP。

    Parameters
    ----------
    strategy : str
        "random" | "periodic" | "density"
    skip_prob : float
        仅 strategy="random" 时使用，每个 beat 被标为 SKIP 的概率。
    period : int
        仅 strategy="periodic" 时使用，每隔 period 个 beat 标一个 SKIP。
    density_threshold : int
        仅 strategy="density" 时使用，beat 中音符数低于此值时标为 SKIP。
    empty_token_id : int
        空 beat token id（context 中无音符时的标志）。
    seed : int | None
        随机种子，None 表示不固定。
    """

    def __init__(
        self,
        strategy: str = "random",
        skip_prob: float = 0.3,
        period: int = 4,
        density_threshold: int = 1,
        empty_token_id: int = 169,
        seed: int | None = None,
    ) -> None:
        assert strategy in ("random", "periodic", "density"), f"未知 strategy: {strategy}"
        self.strategy = strategy
        self.skip_prob = skip_prob
        self.period = period
        self.density_threshold = density_threshold
        self.empty_token_id = empty_token_id
        self._rng = random.Random(seed)

    def label_beat(
        self,
        beat_tokens: list[int],
        context: LabelContext,
    ) -> int:
        # TODO (Phase 1): 实现完整逻辑，目前仅为占位

        if self.strategy == "random":
            return 0 if self._rng.random() < self.skip_prob else 1

        elif self.strategy == "periodic":
            return 0 if (context.beat_index % self.period == 0) else 1

        elif self.strategy == "density":
            # 空 beat → SKIP
            if beat_tokens == [self.empty_token_id]:
                return 0
            # TODO: 从 beat_tokens 解析实际音符数后与 density_threshold 比较
            raise NotImplementedError("density 策略中的音符数解析待 Phase 1 实现。")

        return 1  # fallback: PLAY
