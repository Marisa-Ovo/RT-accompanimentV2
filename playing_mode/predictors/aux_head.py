"""
AuxHeadPredictor — PianoLLaMA 辅助二分类预测头

思路（推荐 Phase 2 优先实现）:
  在现有 PianoLLaMA 主干上添加一个轻量 binary classification head，
  与主语言模型头共享所有 Transformer 层的参数（特征提取）。

  训练时:
    total_loss = lm_loss + α * aux_bce_loss
    - lm_loss:      Part1 next-token prediction（现有）
    - aux_bce_loss: playing token 的二进制交叉熵（新增）

  推理时:
    - 主模型正常自回归生成 Part1 beat tokens
    - 每个 beat 结束时，aux_head 读取该 beat 最后一个位置的 hidden state
    - 输出 PLAY/SKIP 概率，决定是否静音该 beat

  模型结构变化（model.py 中 PianoLLaMA 需添加）:
    self.aux_head = nn.Linear(hidden_size, 1)  # binary logit

  结构示意:
    token_embeds → LLaMA layers → hidden_states
                                       ↓              ↓
                                   lm_head        aux_head
                                  (vocab_size)      (1)

TODO (Phase 2):
  [ ] 在 model.py 的 PianoLLaMA 中添加可选 aux_head（use_aux_head=False 时不影响现有功能）
  [ ] 确定 aux_head 读取哪个位置的 hidden state（beat 的 end_marker 位置）
  [ ] 修改 trainer.py：添加 aux_bce_loss 计算 & alpha 超参
  [ ] 修改 inference.py：推理时调用 aux_head 决定 PLAY/SKIP
  [ ] 实现此类的 predict()，封装对 PianoLLaMA.aux_head 的调用
"""

from __future__ import annotations

from playing_mode.predictors.base import BasePredictor


class AuxHeadPredictor(BasePredictor):
    """
    PianoLLaMA 辅助预测头，共享主模型参数。

    Parameters
    ----------
    model : object
        已加载的 PianoLLaMA 实例（需已添加 aux_head 属性）。
    aux_loss_alpha : float
        训练时 aux_bce_loss 的权重系数（仅训练阶段使用，推理无关）。
    """

    def __init__(
        self,
        model,  # PianoLLaMA，Phase 2 实现后添加类型注解
        aux_loss_alpha: float = 0.1,
    ) -> None:
        self.model = model
        self.aux_loss_alpha = aux_loss_alpha
        # TODO (Phase 2): 检查 model 是否有 aux_head 属性

    def predict(
        self,
        context_tokens: list[int],
        current_beat_tokens: list[int],
    ) -> float:
        """
        调用主模型的 aux_head，返回 PLAY 概率。
        要求主模型已在最近一次 forward 中缓存了对应位置的 hidden state。
        """
        # TODO (Phase 2): 实现 aux_head 推理
        # 大致逻辑:
        #   hidden = self.model.last_hidden_states[:, -1, :]  # beat end marker 位置
        #   logit = self.model.aux_head(hidden)
        #   prob = torch.sigmoid(logit).item()
        #   return prob
        raise NotImplementedError("AuxHeadPredictor 待 Phase 2 实现。")
