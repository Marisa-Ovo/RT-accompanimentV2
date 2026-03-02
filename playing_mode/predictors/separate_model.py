"""
SeparateModelPredictor — 独立轻量分类模型

思路（待设计）:
  训练一个独立的轻量分类器（如小型 Transformer 或 MLP），
  专门根据上下文序列预测当前 beat 的 PLAY/SKIP 概率。

  优点:
    - 主模型（PianoLLaMA）无需修改
    - 可独立迭代、替换
  缺点:
    - 需要额外的训练流程
    - 无法利用主模型的表示（除非用其 hidden states 作为输入特征）

  候选架构:
    - 轻量 Transformer encoder（4层）
    - 主模型最后一层 hidden state → MLP binary head（需要改造主模型推理接口）

TODO (Phase 2):
  [ ] 确定输入特征：原始 token 序列 / 主模型 hidden states / 二者混合
  [ ] 设计模型架构
  [ ] 实现训练脚本（可以复用 trainer.py 的框架）
  [ ] 实现 predict()
"""

from __future__ import annotations

from playing_mode.predictors.base import BasePredictor


class SeparateModelPredictor(BasePredictor):
    """
    独立分类模型预测器。

    Parameters
    ----------
    model_path : str
        独立分类模型 checkpoint 路径。
    device : str
        推理设备。
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
    ) -> None:
        self.model_path = model_path
        self.device = device
        # TODO (Phase 2): 加载分类模型

    def predict(
        self,
        context_tokens: list[int],
        current_beat_tokens: list[int],
    ) -> float:
        # TODO (Phase 2): 实现独立模型推理
        raise NotImplementedError("SeparateModelPredictor 待 Phase 2 实现。")
