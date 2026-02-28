"""
Piano Roll Tokenizer - 钢琴卷帘编解码器

统一管理piano roll与token之间的编解码逻辑，包括：
- Patch-based tokenization (三进制编码)
- 相对位置压缩编码
- 双通道piano roll重建

Author: Optimized from original implementation
"""

import numpy as np
import torch
from typing import Optional, Union, Tuple


class PianoRollTokenizer:
    """
    钢琴卷帘(Piano Roll)的Token编解码器

    功能：
    1. 将双通道piano roll (sustain + onset) 转换为patch tokens
    2. 使用相对位置编码压缩token序列
    3. 支持完整的编码-解码循环

    参数：
        patch_h: patch高度（默认2，对应2个音高）
        patch_w: patch宽度（默认4，对应4个时间步）
        marker_offset: 相对位置标记的偏移量（默认81）
        measures_length: 每个measure的pitch数量（默认88键）
        end_marker_part0: part0的结束标记ID（默认170）
        end_marker_part1: part1的结束标记ID（默认171）
        empty_marker: 空measure的标记ID（默认169）
        img_h: 图像高度，即钢琴键数（默认88）

    示例：
        >>> tokenizer = PianoRollTokenizer(patch_h=2, patch_w=4)
        >>> # 编码
        >>> image = np.random.randint(0, 2, (2, 88, 16))
        >>> compressed = tokenizer.encode(image)
        >>> # 解码
        >>> reconstructed = tokenizer.decode(compressed, num_measures=4)
    """

    def __init__(
        self,
        patch_h: int = 1,
        patch_w: int = 4,
        marker_offset: int = 81,
        measures_length: int = 88,
        end_marker_part0: int = 170,
        end_marker_part1: int = 171,
        empty_marker: int = 169,
        img_h: int = 88
    ):
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.marker_offset = marker_offset
        self.measures_length = measures_length
        self.end_marker_part0 = end_marker_part0
        self.end_marker_part1 = end_marker_part1
        self.empty_marker = empty_marker
        self.img_h = img_h

        # 预计算patch相关参数
        self.patch_size = patch_h * patch_w
        self.powers_3 = 3 ** np.arange(self.patch_size - 1, -1, -1)

        # 特殊token替换规则（用于strict模式）
        self.special_token_ids = [
            13, 12, 59, 31, 64, 11, 55, 73, 37, 30,
            28, 5, 15, 46, 16, 17, 10, 14, 32, 19,
            3, 9, 1, 57, 4
        ]
        self.replacement_ids = [0, 67, 7, 40, 63]

    def encode(
        self,
        image: Union[np.ndarray, torch.Tensor],
        use_strict_mode: bool = True
    ) -> np.ndarray:
        """
        完整编码流程：piano roll → compressed tokens

        Args:
            image: shape (2, 88, t) 的双通道piano roll
                   ch0: sustain, ch1: onset
            use_strict_mode: 是否使用严格模式（替换特殊token）

        Returns:
            compressed_sequence: 一维压缩token序列
        """
        tokens = self.image_to_patch_tokens(image, strict_mode=use_strict_mode)
        compressed = self.compress_tokens(tokens)
        return compressed

    def decode(
        self,
        compressed_sequence: Union[np.ndarray, list],
        end_marker_id: Optional[int] = None
    ) -> np.ndarray:
        """
        完整解码流程：compressed tokens → piano roll

        Args:
            compressed_sequence: 压缩的token序列
            end_marker_id: 结束标记ID（如果为None，支持part0和part1两种）

        Returns:
            image: shape (2, 88, t) 的双通道piano roll
        """
        tokens = self.decompress_tokens(compressed_sequence, end_marker_id)
        image = self.patch_tokens_to_image(tokens)
        return image

    # ==================== 编码相关方法 ====================

    def image_to_patch_tokens(
        self,
        image: Union[np.ndarray, torch.Tensor],
        strict_mode: bool = True
    ) -> np.ndarray:
        """
        将双通道piano roll转换为patch tokens（三进制编码）

        Args:
            image: shape (2, 88, t) 的双通道piano roll
                   ch0: sustain, ch1: onset
            strict_mode: 是否替换特殊token

        Returns:
            tokens: shape (num_time_patches, num_pitch_patches) 的token矩阵

        编码规则：
            0: 无音符 (sustain=0, onset=0)
            1: 只有sustain (sustain=1, onset=0)
            2: onset和sustain都有 (sustain=1, onset=1)
        """
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        # 确保输入是双通道
        assert image.shape[0] == 2, f"Expected 2 channels, got {image.shape[0]}"

        sustain_channel = image[0].copy()  # shape: (88, t)
        onset_channel = image[1].copy()    # shape: (88, t)

        # onset只能出现在sustain为1的地方
        onset_channel[sustain_channel == 0] = 0

        img_h, img_w = sustain_channel.shape

        # 处理宽度padding（确保可以整除patch_w）
        padding_w = (self.patch_w - img_w % self.patch_w) % self.patch_w
        if padding_w > 0:
            sustain_channel = np.pad(
                sustain_channel,
                ((0, 0), (0, padding_w)),
                mode='constant',
                constant_values=0
            )
            onset_channel = np.pad(
                onset_channel,
                ((0, 0), (0, padding_w)),
                mode='constant',
                constant_values=0
            )
            img_w = sustain_channel.shape[1]

        num_patch_rows = img_h // self.patch_h
        num_patch_cols = img_w // self.patch_w

        # 重塑为patches
        sustain_patches = self._reshape_to_patches(sustain_channel, num_patch_rows, num_patch_cols)
        onset_patches = self._reshape_to_patches(onset_channel, num_patch_rows, num_patch_cols)

        # 组合成三进制编码
        combined_patches = sustain_patches.astype(np.int64) + onset_patches.astype(np.int64)

        # 使用三进制计算token值
        tokens = np.dot(combined_patches, self.powers_3)

        # 处理特殊token（strict模式）
        if strict_mode:
            tokens = self._replace_special_tokens(tokens)

        return tokens

    def _reshape_to_patches(
        self,
        channel: np.ndarray,
        num_patch_rows: int,
        num_patch_cols: int
    ) -> np.ndarray:
        """将通道重塑为patches"""
        patches = channel.reshape(num_patch_rows, self.patch_h, num_patch_cols, self.patch_w)
        patches = patches.transpose(2, 0, 1, 3)  # (cols, rows, h, w)
        patches = patches.reshape(num_patch_cols, num_patch_rows, self.patch_size)
        return patches

    def _replace_special_tokens(self, tokens: np.ndarray) -> np.ndarray:
        """随机替换特殊token"""
        mask = np.isin(tokens, self.special_token_ids)
        if np.any(mask):
            num_replacements = np.sum(mask)
            random_replacements = np.random.choice(self.replacement_ids, size=num_replacements)
            tokens = tokens.copy()
            tokens[mask] = random_replacements
        return tokens

    def compress_tokens(
        self,
        token_indices_flat: np.ndarray,
        end_marker: Optional[int] = None
    ) -> np.ndarray:
        """
        使用相对位置编码压缩token序列

        Args:
            token_indices_flat: shape (num_measures, measures_length) 的token矩阵
            end_marker: 结束标记（如果为None，使用end_marker_part0）

        Returns:
            compressed_sequence: 压缩后的一维序列

        编码格式：
            非空: [相对pos0] [token0] [相对pos1] [token1] ... [end_marker]
                  - 第一个音符：相对于0的位置（即绝对位置）
                  - 后续音符：相对于上一个音符的距离
            空:   [empty_marker]
        """
        if end_marker is None:
            end_marker = self.end_marker_part0

        compressed_sequences = []

        for measure_tokens in token_indices_flat:
            # 找到所有非零token的位置
            non_zero_indices = np.where(measure_tokens != 0)[0]

            if len(non_zero_indices) == 0:
                # 空measure
                compressed = [self.empty_marker]
            else:
                # 非空measure
                compressed = []
                prev_idx = 0

                # 添加 [相对位置, token值] 对
                for idx in non_zero_indices:
                    relative_position = idx - prev_idx
                    position_marker = self.marker_offset + relative_position
                    token_value = measure_tokens[idx]

                    compressed.extend([position_marker, token_value])
                    prev_idx = idx

                # 添加结束标记
                compressed.append(end_marker)

            compressed_sequences.append(np.array(compressed, dtype=np.int64))

        # 连接所有measures
        flattened_sequence = np.concatenate(compressed_sequences)
        return flattened_sequence

    # ==================== 解码相关方法 ====================

    def decompress_tokens(
        self,
        compressed_sequence: Union[np.ndarray, list],
        end_marker_id: Optional[int] = None
    ) -> np.ndarray:
        """
        解压缩token序列（相对位置编码 → token矩阵）

        Args:
            compressed_sequence: 压缩的token序列
            end_marker_id: 结束标记（如果为None，支持part0和part1两种）

        Returns:
            decompressed_measures: shape (num_measures, measures_length) 的token矩阵
        """
        if isinstance(compressed_sequence, list):
            compressed_sequence = np.array(compressed_sequence, dtype=np.int64)

        decompressed_measures = []
        i = 0

        while i < len(compressed_sequence):
            current_token = compressed_sequence[i]

            if current_token == self.empty_marker:
                # 空measure
                measure = np.zeros(self.measures_length, dtype=np.int64)
                decompressed_measures.append(measure)
                i += 1
            else:
                # 非空measure - 当前token是第一个相对位置标记
                measure = np.zeros(self.measures_length, dtype=np.int64)
                current_abs_pos = 0

                # 读取位置-token对，直到遇到end_marker
                while i < len(compressed_sequence):
                    position_marker = compressed_sequence[i]
                    i += 1

                    # 检查是否是结束标记
                    if self._is_end_marker(position_marker, end_marker_id):
                        break

                    if i >= len(compressed_sequence):
                        break

                    # 读取token值
                    token_value = compressed_sequence[i]
                    i += 1

                    # 计算绝对位置（累加相对位置）
                    relative_pos = position_marker - self.marker_offset
                    current_abs_pos += relative_pos

                    # 填充token
                    if 0 <= current_abs_pos < self.measures_length:
                        measure[current_abs_pos] = token_value

                decompressed_measures.append(measure)

        return np.stack(decompressed_measures, axis=0)

    def _is_end_marker(self, token: int, end_marker_id: Optional[int]) -> bool:
        """检查是否是结束标记"""
        if end_marker_id is not None:
            return token == end_marker_id
        # 支持part0和part1的结束标记
        return token in [self.end_marker_part0, self.end_marker_part1]

    def patch_tokens_to_image(
        self,
        tokens: np.ndarray
    ) -> np.ndarray:
        """
        从tokens重建双通道piano roll

        Args:
            tokens: shape (num_time_patches, num_pitch_patches) 的token矩阵

        Returns:
            image: shape (2, 88, t) 的双通道piano roll
                   ch0: sustain, ch1: onset
        """
        num_patch_cols, num_patch_rows = tokens.shape

        # 解码tokens为三进制表示
        combined_patches = np.zeros(
            (num_patch_cols, num_patch_rows, self.patch_size),
            dtype=np.int64
        )
        temp_tokens = tokens.copy()

        for i in range(self.patch_size):
            combined_patches[:, :, i] = temp_tokens // self.powers_3[i]
            temp_tokens = temp_tokens % self.powers_3[i]

        # 从三进制值恢复双通道
        # 0 -> sustain=0, onset=0
        # 1 -> sustain=1, onset=0
        # 2 -> sustain=1, onset=1
        sustain_patches = (combined_patches >= 1).astype(np.float32)
        onset_patches = (combined_patches == 2).astype(np.float32)

        # 重建图像
        sustain_channel = self._patches_to_channel(
            sustain_patches,
            num_patch_cols,
            num_patch_rows
        )
        onset_channel = self._patches_to_channel(
            onset_patches,
            num_patch_cols,
            num_patch_rows
        )

        # 组合成双通道
        image = np.stack([sustain_channel, onset_channel], axis=0)
        return image

    def _patches_to_channel(
        self,
        patches: np.ndarray,
        num_patch_cols: int,
        num_patch_rows: int
    ) -> np.ndarray:
        """将patches重建为通道"""
        patches = patches.reshape(num_patch_cols, num_patch_rows, self.patch_h, self.patch_w)
        patches = patches.transpose(1, 2, 0, 3)  # (rows, h, cols, w)
        channel = patches.reshape(self.img_h, num_patch_cols * self.patch_w)
        return channel

    # ==================== 工具方法 ====================

    def get_config(self) -> dict:
        """返回配置字典"""
        return {
            'patch_h': self.patch_h,
            'patch_w': self.patch_w,
            'marker_offset': self.marker_offset,
            'measures_length': self.measures_length,
            'end_marker_part0': self.end_marker_part0,
            'end_marker_part1': self.end_marker_part1,
            'empty_marker': self.empty_marker,
            'img_h': self.img_h
        }

    def __repr__(self) -> str:
        return (
            f"PianoRollTokenizer("
            f"patch_size={self.patch_h}x{self.patch_w}, "
            f"img_h={self.img_h}, "
            f"marker_offset={self.marker_offset})"
        )


# ==================== 便捷函数（保持向后兼容） ====================

# 创建全局默认tokenizer实例
_default_tokenizer = PianoRollTokenizer()


def image_to_patch_tokens_vectorized_strict(image, H=2, W=4):
    """向后兼容的函数包装"""
    tokenizer = PianoRollTokenizer(patch_h=H, patch_w=W)
    return tokenizer.image_to_patch_tokens(image, strict_mode=True)


def patch_tokens_to_image_vectorized(tokens, H=2, W=4, img_h=88):
    """向后兼容的函数包装"""
    tokenizer = PianoRollTokenizer(patch_h=H, patch_w=W, img_h=img_h)
    return tokenizer.patch_tokens_to_image(tokens)


def compress_tokens_v2(token_indices_flat, marker_offset=81, measures_length=88,
                       end_marker=None, empty_marker=169):
    """向后兼容的函数包装"""
    if end_marker is None:
        end_marker = 170
    tokenizer = PianoRollTokenizer(
        marker_offset=marker_offset,
        measures_length=measures_length,
        end_marker_part0=end_marker,
        empty_marker=empty_marker
    )
    return tokenizer.compress_tokens(token_indices_flat, end_marker=end_marker)


def decompress_tokens_v3(compressed_sequence, marker_offset=81, measures_length=88,
                         end_marker_id_1=None, end_marker_id_2=None, empty_marker=169):
    """向后兼容的函数包装"""
    tokenizer = PianoRollTokenizer(
        marker_offset=marker_offset,
        measures_length=measures_length,
        end_marker_part0=end_marker_id_1 or 170,
        end_marker_part1=end_marker_id_2 or 171,
        empty_marker=empty_marker
    )
    return tokenizer.decompress_tokens(compressed_sequence)
