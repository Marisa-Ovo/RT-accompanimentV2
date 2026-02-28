# dataset.py - 修改后的完整代码

import os
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
import pickle
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict
from dataclasses import dataclass
from my_tokenizer import PianoRollTokenizer

def process_measure_with_beat_interleaving(
    measure,
    tokenizer,
    timesteps_per_beat=4
):
    """
    在每一拍级别交错处理高声部和低声部

    Args:
        measure: (4, 88, t) - 前2通道part0(高声部)，后2通道part1(低声部)
        tokenizer: PianoRollTokenizer实例
        timesteps_per_beat: 每拍的时间步数

    Returns:
        part0_beat_tokens: part0的beat级别token列表
        part1_beat_tokens: part1的beat级别token列表
    """
    part0_beat_tokens = []
    part1_beat_tokens = []

    t = measure.shape[2]
    beat_length = timesteps_per_beat

    # 计算实际有多少拍
    num_beats = (t + beat_length - 1) // beat_length  # 向上取整

    for beat_idx in range(num_beats):
        # 计算当前拍的时间范围
        start_t = beat_idx * beat_length
        end_t = min(start_t + beat_length, t)

        # 如果最后一拍不够长，需要padding
        beat_measure = measure[:, :, start_t:end_t]
        current_length = end_t - start_t

        if current_length < beat_length:
            # Padding到完整的拍长度
            pad_width = ((0, 0), (0, 0), (0, beat_length - current_length))
            beat_measure = np.pad(beat_measure, pad_width, mode='constant', constant_values=0)

        # === 处理part0 (高声部) ===
        part0_beat = beat_measure[:2]  # (2, 88, beat_length)
        tokens_0 = tokenizer.image_to_patch_tokens(part0_beat, strict_mode=True)
        compressed_tokens_0 = tokenizer.compress_tokens(tokens_0, end_marker=170)
        part0_beat_tokens.append(torch.tensor(compressed_tokens_0, dtype=torch.long))

        # === 处理part1 (低声部) ===
        part1_beat = beat_measure[2:]  # (2, 88, beat_length)
        tokens_1 = tokenizer.image_to_patch_tokens(part1_beat, strict_mode=True)
        compressed_tokens_1 = tokenizer.compress_tokens(tokens_1, end_marker=171)
        part1_beat_tokens.append(torch.tensor(compressed_tokens_1, dtype=torch.long))

    return part0_beat_tokens, part1_beat_tokens


def encode_bpm(bpm):
        if bpm is None:
            return 3  # UNK token
        bpm = int(bpm)
        if bpm < 90:
            return 0  # 慢速
        elif bpm <= 200:
            return 1  # 中速
        else:
            return 2  # 快速
        


class PianoDataset(Dataset):
    """支持长度感知的数据集"""

    def __init__(self, data_dir, config, cache_lengths=True, mode='train',
                 test_split_ratio=0.05, random_seed=42):
        """
        Args:
            data_dir: 数据目录
            config: 模型配置
            cache_lengths: 是否使用长度缓存
            mode: 'train' 或 'test'，决定使用训练集还是测试集
            test_split_ratio: 测试集划分比例（0-1之间）
            random_seed: 随机种子，用于可重复的数据集划分
        """
        self.root_dir = data_dir
        self.patch_h = config.patch_h
        self.patch_w = config.patch_w
        self.max_seq_len = config.train_cutoff_len
        self.pad_token = config.pad_token_id
        self.bos_token = config.bos_token_id
        self.eos_token = config.eos_token_id
        self.bar_token = config.bar_token_id
        self.time_sig_offset_id = config.time_sig_offset_id
        self.bpm_offset_id = config.bpm_offset_id
        self.mode = mode
        self.test_split_ratio = test_split_ratio
        self.random_seed = random_seed

        # 创建tokenizer实例
        self.tokenizer = PianoRollTokenizer(
            patch_h=self.patch_h,
            patch_w=self.patch_w,
            marker_offset=81,
            measures_length=88,
            end_marker_part0=170,
            end_marker_part1=171,
            empty_marker=169,
            img_h=88
        )

        self.data_files = [f for f in os.listdir(self.root_dir) if f.endswith('.npz')]
        print(f"找到 {len(self.data_files)} 个有效的npz文件")

        # 预计算长度信息
        cache_file = os.path.join(data_dir, '.lengths_cache.pkl')

        if cache_lengths and os.path.exists(cache_file):
            print("加载长度缓存...")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                
            # 验证patch参数是否匹配
            if (cache_data['patch_h'] != self.patch_h or 
                cache_data['patch_w'] != self.patch_w):
                raise ValueError(
                    f"缓存的patch参数({cache_data['patch_h']}x{cache_data['patch_w']}) "
                    f"与配置({self.patch_h}x{self.patch_w})不匹配，请重新运行precompute_lengths.py"
                )
            
            self.data_files = cache_data['data_files']
            self.file_lengths = cache_data['lengths']
            self.sorted_indices = cache_data['sorted_indices']
            
            print(f"加载 {len(self.data_files)} 个文件的长度信息")
            
        elif cache_lengths:
            raise FileNotFoundError(
                f"长度缓存不存在: {cache_file}\n"
                f"请先运行: python precompute_lengths.py"
            )
        else:
            # 不使用缓存，传统方式
            self.file_lengths = None
            self.sorted_indices = None
            print(f"找到 {len(self.data_files)} 个文件（未使用长度缓存）")

        # 划分训练集和测试集
        self._split_train_test()

    def _split_train_test(self):
        """根据mode参数划分训练集和测试集"""
        total_files = len(self.data_files)

        # 设置随机种子以确保可重复性
        np.random.seed(self.random_seed)

        # 创建索引数组并打乱
        indices = np.arange(total_files)
        np.random.shuffle(indices)

        # 计算测试集大小
        test_size = int(total_files * self.test_split_ratio)
        train_size = total_files - test_size

        if self.mode == 'train':
            # 使用前train_size个样本作为训练集
            selected_indices = indices[:train_size]
            print(f"使用训练集: {len(selected_indices)} 个文件 ({train_size}/{total_files})")
        elif self.mode == 'test':
            # 使用后test_size个样本作为测试集
            selected_indices = indices[train_size:]
            print(f"使用测试集: {len(selected_indices)} 个文件 ({test_size}/{total_files})")
        else:
            raise ValueError(f"mode必须是'train'或'test'，当前为: {self.mode}")

        # 更新data_files和相关索引
        self.data_files = [self.data_files[i] for i in selected_indices]

        # 如果使用了长度缓存，也需要更新相关信息
        if self.file_lengths is not None:
            self.file_lengths = [self.file_lengths[i] for i in selected_indices]

            # 重新创建sorted_indices（在新的子集中的排序）
            self.sorted_indices = sorted(
                range(len(self.file_lengths)),
                key=lambda i: self.file_lengths[i]
            )

    def __len__(self):
        return len(self.data_files)
    
    def _find_next_bar_token(self, sequence, start_pos, target_len):
        """
        从指定位置开始查找下一个bar_token，并返回从该位置开始的target_len长度序列
        
        Args:
            sequence: 原始token序列
            start_pos: 开始搜索的位置
            target_len: 目标序列长度
        
        Returns:
            从找到的bar_token开始的序列切片
        """
        search_end = min(len(sequence), start_pos + target_len)
        
        for i in range(start_pos, search_end):
            if sequence[i] == self.bar_token:
                return sequence[i:i + target_len]
        
        # 找不到bar_token时，从原始位置截取（降级策略）
        return sequence[start_pos:start_pos + target_len]

    def _truncate_sequence(self, sequence, target_len):
        """
        智能截断序列，确保截取点从完整小节开始
        
        截断策略：
        - 短序列(≤2*max_len): 50%从头截取，50%从尾部找bar_token截取
        - 长序列: 30%从头截取，70%随机位置找bar_token截取，剩余从尾部找bar_token截取
        """
        seq_len = len(sequence)
        
        # ========== 策略1: 处理较短序列 (≤2倍max_seq_len) ==========
        if seq_len <= self.max_seq_len * 2:
            if np.random.random() < 0.5:
                # 从开头截取 - 无需对齐（序列开头已是结构化的）
                return sequence[:target_len]
            else:
                # 从末尾截取 - 需要对齐到bar_token
                start_pos = max(0, seq_len - target_len)
                return self._find_next_bar_token(sequence, start_pos, target_len)
        
        # ========== 策略2: 处理较长序列 ==========
        
        # 30%概率：从开头截取（无需对齐）
        if np.random.random() < 0.3:
            return sequence[:target_len]
        
        # 70%概率：随机位置截取（需要对齐到bar_token）
        if np.random.random() < 0.7:        
            # 随机选择起始位置（避免边界）
            start_idx = np.random.randint(2, seq_len - target_len)
            return self._find_next_bar_token(sequence, start_idx, target_len)
        
        # 剩余概率：从末尾截取（需要对齐到bar_token）
        start_pos = max(0, seq_len - target_len)
        return self._find_next_bar_token(sequence, start_pos, target_len)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.data_files[idx])
        file_name = self.data_files[idx]

        save_dict = np.load(file_path, allow_pickle=True)
        metadata = save_dict['metadata'].item()
        idx_value = metadata['time_signature_idx']

        add_bos = True
        if '_' in file_name:
            # 提取"_"后面的部分
            suffix = file_name.split('_')[-1].replace('.npz', '')
            if suffix != '1' and suffix.isdigit():  # 是数字但不是1
                add_bos = False

        if idx_value == 9:
            idx_value = 4  # 2/2拍

        is_continuation = metadata.get('is_continuation', False)
        
        bpm_value = metadata['bpm']
        num_measures = metadata['num_measures']
        shift = 0
        delay_beats = -1  # 负数表示part1提前(-1表示提前1拍)
        pad_marker = 173
        
        # 收集所有part0和part1的beat tokens
        all_part0_beats = []
        all_part1_beats = []

        if np.random.random() < 0.7:
            shift = np.random.randint(-5, 6)

        for i in range(num_measures):
            measure = save_dict[f'measure_{i}']

            if shift != 0:
                measure = np.roll(measure, shift, axis=1)
                if shift > 0:
                    measure[:, :shift, :] = 0
                else:
                    measure[:, shift:, :] = 0

            part0_beats, part1_beats = process_measure_with_beat_interleaving(
                measure,
                tokenizer=self.tokenizer,
                timesteps_per_beat=4
            )
            all_part0_beats.append(torch.tensor([self.bar_token], dtype=torch.long))
            all_part0_beats.extend(part0_beats)
            all_part1_beats.append(torch.tensor([self.bar_token], dtype=torch.long))
            all_part1_beats.extend(part1_beats)
        
        # 根据delay_beats的正负决定谁先出现
        pad_token = torch.tensor([pad_marker], dtype=torch.long)

        if delay_beats >= 0:
            # delay_beats为正数：part1延迟（part0先出现）
            # part1前面加N个pad，part0后面加N个pad
            part1_padded = [pad_token] * delay_beats + all_part1_beats
            part0_padded = all_part0_beats + [pad_token] * delay_beats
        else:
            # delay_beats为负数：part1提前（part1先出现）
            # part0前面加N个pad，part1后面加N个pad
            advance_beats = -delay_beats  # 转换为正数
            part0_padded = [pad_token] * advance_beats + all_part0_beats
            part1_padded = all_part1_beats + [pad_token] * advance_beats
        
        # 交错添加并分别记录input和label
        all_input_tokens = []
        all_label_tokens = []

        for p0, p1 in zip(part0_padded, part1_padded):
            # input包含part0和part1
            all_input_tokens.append(p0)
            all_input_tokens.append(p1)
            
            # label中part0部分用pad_token替换（忽略损失），part1保持原样
            all_label_tokens.append(torch.full_like(p0, self.pad_token))
            all_label_tokens.append(p1)
        
        all_measure_tokens = torch.cat(all_input_tokens, dim=0)
        all_label_tokens_cat = torch.cat(all_label_tokens, dim=0)

        bpm_token = encode_bpm(bpm_value) + self.bpm_offset_id

        input_token_list = []
        label_token_list = []
        
        # BOS token (根据文件名决定)
        if add_bos:
            bos_tensor = torch.tensor([self.bos_token], dtype=torch.long)
            input_token_list.append(bos_tensor)
            label_token_list.append(torch.full_like(bos_tensor, self.pad_token))  # BOS不参与loss
        
        # 拍号和BPM总是添加
        time_sig_tensor = torch.tensor([idx_value + self.time_sig_offset_id], dtype=torch.long)
        bpm_tensor = torch.tensor([bpm_token], dtype=torch.long)
        
        input_token_list.extend([time_sig_tensor, bpm_tensor])
        label_token_list.extend([
            torch.full_like(time_sig_tensor, self.pad_token),  # 拍号不参与loss
            torch.full_like(bpm_tensor, self.pad_token)  # BPM不参与loss
        ])
        
        # 小节内容
        input_token_list.append(all_measure_tokens)
        label_token_list.append(all_label_tokens_cat)
        
        # EOS token (根据continuation标志决定)
        if not is_continuation:
            eos_tensor = torch.tensor([self.eos_token], dtype=torch.long)
            input_token_list.append(eos_tensor)
            label_token_list.append(eos_tensor)  # EOS参与loss
        
        compressed_input = torch.cat(input_token_list, dim=0)
        compressed_labels = torch.cat(label_token_list, dim=0)

        # 使用改进的截断策略
        seq_len = len(compressed_input)

        if seq_len > self.max_seq_len:
            compressed_input = compressed_input[:self.max_seq_len]
            compressed_labels = compressed_labels[:self.max_seq_len]  # labels与input保持同样长度
    
        return {
            'input_ids': compressed_input,
            'labels': compressed_labels,
        }


class BucketBatchSampler(Sampler):
    """长度感知的批采样器"""
    
    def __init__(self, dataset, batch_size=16, bucket_size=100, shuffle=True):
        """
        Args:
            dataset: PianoDataset实例
            batch_size: 实际训练的batch大小
            bucket_size: 每个长度bucket的大小
            shuffle: 是否随机化
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.shuffle = shuffle
        
        if dataset.sorted_indices is None:
            raise ValueError("Dataset需要启用cache_lengths=True")
        
        self._create_buckets()
    
    def _create_buckets(self):
        """将相近长度的样本分组到buckets中"""
        self.buckets = []
        sorted_indices = self.dataset.sorted_indices
        
        # 将排序后的索引分割成buckets
        for i in range(0, len(sorted_indices), self.bucket_size):
            bucket = sorted_indices[i:i + self.bucket_size]
            self.buckets.append(bucket)
        
        print(f"创建了 {len(self.buckets)} 个长度buckets")
    
    def __iter__(self):
        """生成batch索引"""
        # 随机打乱buckets的顺序
        if self.shuffle:
            bucket_order = np.random.permutation(len(self.buckets))
        else:
            bucket_order = range(len(self.buckets))
        
        for bucket_idx in bucket_order:
            bucket = self.buckets[bucket_idx].copy()
            
            # 在bucket内部随机打乱
            if self.shuffle:
                np.random.shuffle(bucket)
            
            # 从bucket中生成batches
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i + self.batch_size]
                if len(batch) > 0:  # 确保不是空batch
                    yield batch
    
    def __len__(self):
        return sum(len(bucket) for bucket in self.buckets) // self.batch_size


@dataclass
class DataCollatorForVariableLengthLM:
    """数据整理器，支持动态padding"""
    
    def __init__(self, config):
        self.pad_token_id = config.pad_token_id
        self.max_length = config.train_cutoff_len
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # 获取批次中的最大长度
        max_len = min(
            max(len(feature["input_ids"]) for feature in features),
            self.max_length
        )
        
        batch = {
            "input_ids": [],
            "labels": [],
            "attention_mask": []
        }
        
        for feature in features:
            input_ids = feature["input_ids"]
            labels = feature["labels"]
            seq_len = len(input_ids)
            
            if seq_len < max_len:
                padding_len = max_len - seq_len
                
                input_ids = torch.cat([
                    input_ids,
                    torch.full((padding_len,), self.pad_token_id, dtype=torch.long)
                ])
                
                labels = torch.cat([
                    labels,
                    torch.full((padding_len,), -100, dtype=torch.long)
                ])
                
                attention_mask = torch.cat([
                    torch.ones(seq_len, dtype=torch.long),
                    torch.zeros(padding_len, dtype=torch.long)
                ])
            else:
                attention_mask = torch.ones(seq_len, dtype=torch.long)
            
            batch["input_ids"].append(input_ids)
            batch["labels"].append(labels)
            batch["attention_mask"].append(attention_mask)
        
        batch = {k: torch.stack(v) for k, v in batch.items()}
        
        return batch