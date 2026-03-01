import os
import torch
import torch.nn as nn
from transformers import (
    LlamaForCausalLM,
    PreTrainedModel
)
from typing import Optional

from my_tokenizer import PianoMusicTokenizer
import numpy as np

class PianoLLaMA(PreTrainedModel):
    """基于LLaMA架构的Piano生成模型"""

    def __init__(self,config):
        super().__init__(config)

        # 配置LLaMA
        self.config = config

        # 初始化LLaMA模型
        self.model = LlamaForCausalLM(self.config)

        # 特殊token
        self.pad_token_id = config.pad_token_id
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id

        # 初始化权重（使用更好的初始化）
        self.model.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        """前向传播"""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False if labels is not None else True,
        )

    # ==================== 采样工具 ====================

    def _sample_token(self, logits, generated, temperature, top_k, top_p, repetition_penalty):
        """对 logits 应用温度、重复惩罚、top-k/p 并采样。"""
        logits = logits / temperature

        if repetition_penalty != 1.0:
            for token_id in set(generated[0].tolist()):
                logits[0, token_id] /= repetition_penalty

        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float('Inf')

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[0, indices_to_remove] = -float('Inf')

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def _generate_one_beat(self, generated, vocab, temperature, top_k, top_p, repetition_penalty):
        """
        自回归生成一个 acc beat，直到遇到 track_marker_acc。

        Returns:
            (beat_tokens, generated):
                beat_tokens: 生成的 token ID 列表
                generated: 更新后的 (1, seq_len) tensor
        """
        # track_marker_acc 为正常终止符，bar/beat_marker 为安全终止符
        end_markers = {vocab.track_marker_acc, vocab.bar_token_id, vocab.beat_marker}
        beat_tokens = []
        past_kv = None

        for _ in range(200):  # per-beat safety limit
            outputs = self.model(
                input_ids=generated[:, -1:] if past_kv else generated,
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv = outputs.past_key_values

            next_token = self._sample_token(
                outputs.logits[:, -1, :], generated,
                temperature, top_k, top_p, repetition_penalty)

            generated = torch.cat([generated, next_token], dim=1)
            token_id = next_token.item()
            beat_tokens.append(token_id)

            if token_id in end_markers:
                break

        return beat_tokens, generated

    # ==================== 主生成方法 ====================

    @torch.no_grad()
    def generate_accompaniment(
        self,
        dataset,
        condition_idx: int,
        gt_prefix_beats: int = 12,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        device: str = 'cuda',
        verbose: bool = True,
    ):
        """
        基于条件（mel）生成伴奏（acc）。
        使用 tokenizer 提供的 schedule 驱动生成，调度与执行分离。

        Schedule 结构:
          inject[bar] → inject[beat] → generate acc → inject mel → ...
        """
        self.eval()

        # 1. 加载数据
        tokenizer = dataset.tokenizer
        file_path = os.path.join(dataset.root_dir, dataset.data_files[condition_idx])
        save_dict = np.load(file_path, allow_pickle=True)
        metadata = save_dict['metadata'].item()
        measures = [save_dict[f'measure_{i}'] for i in range(metadata['num_measures'])]

        # 2. 构建生成计划（调度逻辑全在 tokenizer）
        gen_data = tokenizer.build_generation_schedule(
            measures=measures,
            metadata=metadata,
            gt_prefix_beats=gt_prefix_beats,
        )

        schedule = gen_data['schedule']

        if verbose:
            n_gen = sum(1 for s in schedule if s.action == "generate")
            n_gt = sum(1 for s in schedule if s.action == "inject_gt")
            print(f"生成计划: {len(schedule)} 步, {n_gen} 步自回归生成, {n_gt} 步GT前缀")

        # 3. 顺序执行生成计划
        generated = gen_data['initial_tokens'].unsqueeze(0).to(device)
        acc_beats = []

        for step in schedule:
            if step.action in ("inject", "inject_gt"):
                generated = torch.cat(
                    [generated, step.data.unsqueeze(0).to(device)], dim=1)
                if step.action == "inject_gt":
                    acc_beats.append(step.data.cpu().tolist())

            elif step.action == "generate":
                beat_tokens, generated = self._generate_one_beat(
                    generated, tokenizer.vocab,
                    temperature, top_k, top_p, repetition_penalty)
                acc_beats.append(beat_tokens)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 4. 汇总结果
        idx_value = metadata['time_signature_idx']
        if idx_value == 9:
            idx_value = 4

        return {
            'generated_sequence': generated.cpu(),
            'mel_beats': gen_data['mel_beats'],
            'acc_beats': acc_beats,
            'GT_path': file_path,
            'metadata': {
                'time_signature_idx': idx_value,
                'bpm': metadata['bpm'],
                'num_measures': metadata['num_measures'],
                'num_mel_beats': len(gen_data['mel_beats']),
                'num_acc_beats': len(acc_beats),
                'num_acc_gt_beats': sum(1 for s in schedule if s.action == "inject_gt"),
                'num_acc_generated_beats': sum(1 for s in schedule if s.action == "generate"),
            }
        }
