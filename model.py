import os
import torch
import torch.nn as nn
from transformers import (
    LlamaForCausalLM,
    PreTrainedModel
)
from typing import Optional

from PianoDataset import encode_bpm, process_measure_with_beat_interleaving
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
    
    @torch.no_grad()
    def generate_accompaniment(
        self,
        dataset,
        condition_idx: int,
        delay_beats: int = -1,
        gt_prefix_beats: int = 12,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        device: str = 'cuda',
        verbose: bool = True,
    ):
        """
        基于条件（part0）生成伴奏（part1）

        Args:
            dataset: 数据集对象
            condition_idx: 用作条件的数据索引
            delay_beats: part1相对part0的延迟拍数
                        正数表示part1延迟（part0先出现）
                        负数表示part1提前（part1先出现）
                        例如：-1表示part1提前1拍
            gt_prefix_beats: 使用ground truth的前缀拍数，默认为12
                            前gt_prefix_beats拍的part1将使用GT，之后才开始生成
            temperature: 温度参数
            top_k: Top-k采样
            top_p: Top-p采样
            repetition_penalty: 重复惩罚
            device: 设备

        Returns:
            dict: {
                'generated_sequence': 完整生成序列,
                'part0_beats': 条件part0的拍列表,
                'part1_beats': 生成的part1拍列表,
                'metadata': 元数据信息
            }
        """
        self.eval()
        
        # ========== 1. 从数据集提取条件 ==========
        file_path = os.path.join(dataset.root_dir, dataset.data_files[condition_idx])
        
        save_dict = np.load(file_path, allow_pickle=True)
        metadata = save_dict['metadata'].item()
        
        idx_value = metadata['time_signature_idx']
        if idx_value == 9:
            idx_value = 4
        bpm_value = metadata['bpm']
        num_measures = metadata['num_measures']
        
        # 提取所有part0的beat tokens（条件）和part1的beat tokens（GT）
        part0_beats_list = []
        part1_beats_gt_list = []  # 存储part1的GT beats
        for i in range(num_measures):
            measure = save_dict[f'measure_{i}']
            part0_beats, part1_beats = process_measure_with_beat_interleaving(
                measure,
                tokenizer= dataset.tokenizer,
                timesteps_per_beat=4
            )
            # 添加bar token
            part0_beats_list.append(torch.tensor([dataset.bar_token], dtype=torch.long))
            part0_beats_list.extend(part0_beats)

            # 同样处理part1 GT
            part1_beats_gt_list.append(torch.tensor([dataset.bar_token], dtype=torch.long))
            part1_beats_gt_list.extend(part1_beats)

        pad_marker = 173

        # 根据delay_beats决定padding位置
        if delay_beats >= 0:
            # part1延迟：part0后面添加padding
            part0_beats_list.append(torch.tensor([pad_marker] * delay_beats, dtype=torch.long))
            # part1_beats_gt_list前面添加padding
            part1_beats_gt_list.insert(0, torch.tensor([pad_marker] * delay_beats, dtype=torch.long))
        else:
            # part1提前：part0前面添加padding
            advance_beats = -delay_beats
            part0_beats_list.insert(0, torch.tensor([pad_marker] * advance_beats, dtype=torch.long))
            # part1_beats_gt_list后面添加padding
            part1_beats_gt_list.append(torch.tensor([pad_marker] * advance_beats, dtype=torch.long))


        print(f"提取条件: {len(part0_beats_list)} 个part0 beats, {len(part1_beats_gt_list)} 个part1 GT beats")
        
        # ========== 2. 构造初始prompt ==========
        bpm_token = encode_bpm(bpm_value) + dataset.bpm_offset_id
        
        initial_tokens = [
            torch.tensor([dataset.bos_token], dtype=torch.long),
            torch.tensor([idx_value + dataset.time_sig_offset_id], dtype=torch.long),
            torch.tensor([bpm_token], dtype=torch.long),
        ]
        
        generated = torch.cat(initial_tokens, dim=0).unsqueeze(0).to(device)
        
        # ========== 3. 交错生成part0和part1 ==========
        part0_idx = 0  # 当前注入到第几拍part0
        part1_idx = 0  # 当前处理到第几拍part1（用于GT注入）
        part1_beats_generated = []  # 收集生成的part1 beats
        current_part1_beat = []  # 当前正在生成的part1 beat
        
        
        part1_end_marker = 171
        part1_empty_marker = 169
        bar_token_id = 255

        position = 0  # 0=part0位置, 1=part1位置（交错）
        past_key_values = None
        l =  len(part0_beats_list)
        max_iterations = 7000   # 防止无限循环

        if verbose:
            print(f"开始生成 (总拍数: {l}, 序列将包含~{l*2}拍)")

        # 定期清理显存的计数器
        cache_reset_count = 0

        with torch.no_grad():
            for iteration in range(max_iterations):
                
                # ===== 决定当前位置应该做什么 =====
                if position == 0:  # part0位置
                    # 如果delay_beats < 0，前abs(delay_beats)拍的part0位置需要填充pad
                    if delay_beats < 0 and part0_idx <= -delay_beats:
                        pad_token = torch.tensor([[pad_marker]], device=device)
                        generated = torch.cat([generated, pad_token], dim=1)
                        part0_idx += 1
                        position = 1  # 切换到part1位置
                        past_key_values = None
                        cache_reset_count += 1
                        continue

                    if part0_idx < l:
                        # 注入下一拍part0
                        next_part0_beat = part0_beats_list[part0_idx].to(device)
                        generated = torch.cat([generated, next_part0_beat.unsqueeze(0)], dim=1)
                        part0_idx += 1
                        position = 1  # 切换到part1位置
                        past_key_values = None  # 重置cache（因为我们手动添加了tokens）
                        cache_reset_count += 1
                        continue

                    else:
                        # part0全部注入完毕
                        break

                else:  # part1位置（需要生成或注入GT）
                    # 如果delay_beats >= 0，前delay_beats拍的part1位置填充pad_marker
                    if delay_beats >= 0 and part0_idx <= delay_beats:
                        # 直接注入pad marker
                        pad_token = torch.tensor([[pad_marker]], device=device)
                        generated = torch.cat([generated, pad_token], dim=1)
                        part1_idx += 1  # 更新part1计数
                        position = 0  # 切换回part0位置
                        past_key_values = None
                        cache_reset_count += 1
                        continue

                    # 检查是否应该使用GT（前gt_prefix_beats拍）
                    if part1_idx < gt_prefix_beats and part1_idx < len(part1_beats_gt_list):
                        # 直接注入GT beat
                        gt_beat = part1_beats_gt_list[part1_idx].to(device)
                        generated = torch.cat([generated, gt_beat.unsqueeze(0)], dim=1)
                        # 将GT beat记录到生成结果中
                        part1_beats_generated.append(gt_beat.cpu().tolist())
                        part1_idx += 1
                        position = 0  # 切换回part0位置
                        past_key_values = None
                        cache_reset_count += 1
                        continue

                    # 开始真正生成part1（过了GT前缀后）
                    # 生成一个token
                    outputs = self.model(
                        input_ids=generated[:, -1:] if past_key_values else generated,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    
                    logits = outputs.logits[:, -1, :]
                    past_key_values = outputs.past_key_values
                    
                    # 温度缩放
                    logits = logits / temperature
                    
                    # 重复惩罚
                    if repetition_penalty != 1.0:
                        for token_id in set(generated[0].tolist()):
                            logits[0, token_id] /= repetition_penalty
                    
                    # Top-k采样
                    if top_k > 0:
                        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                        logits[indices_to_remove] = -float('Inf')
                    
                    # Top-p采样
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        logits[0, indices_to_remove] = -float('Inf')
                    
                    # 采样
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # 添加到生成序列
                    generated = torch.cat([generated, next_token], dim=1)
                    current_part1_beat.append(next_token.item())
                    
                    # 检查是否完成一拍part1
                    if next_token.item() in [part1_end_marker, part1_empty_marker,bar_token_id]:
                        # 一拍part1生成完毕
                        part1_beats_generated.append(current_part1_beat.copy())
                        current_part1_beat = []
                        part1_idx += 1  # 更新part1计数
                        position = 0  # 切换回part0位置，注入下一拍part0        
        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        num_gt_beats = min(gt_prefix_beats, len(part1_beats_generated))
        num_generated_beats = len(part1_beats_generated) - num_gt_beats

        return {
            'generated_sequence': generated.cpu(),
            'part0_beats': part0_beats_list,
            'part1_beats': part1_beats_generated,
            'GT_path': file_path,
            'metadata': {
                'time_signature_idx': idx_value,
                'bpm': bpm_value,
                'num_measures': num_measures,
                'num_part0_beats': part0_idx,
                'num_part1_beats': len(part1_beats_generated),
                'num_part1_gt_beats': num_gt_beats,
                'num_part1_generated_beats': num_generated_beats
            }
        }