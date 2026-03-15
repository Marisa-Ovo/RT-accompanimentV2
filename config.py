from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass
class TrainingConfig:
    """训练配置类"""

    # 基础配置
    time = datetime.now().strftime("%m%d_%H%M")
    output_dir: str = "./checkpoints-shift-3"
    num_epochs: int = 4
    save_model_epochs: int = 1
    train_batch_size = 16
    gradient_accumulation_steps: int = 32  # effective batch = 16×32=512
    lr_warmup_steps: int = 200
    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"  # H200 原生支持 bf16，更稳定

    # 学习率配置
    learning_rate: float = 5e-5

    # 日志配置
    log: bool = True
    log_every_n_steps: int = 20
    tensorboard_log_dir: str = "./logs"
    tensorboard_log_name: str = f"music_transformer_shift3_{time}"
    # standard (192K files): allxml_npz_dual_track_optimized_no_underscore
    # augmented (342K files): allxml_npz_dual_track_optimized
    data_dir = "/data/home/yuanxin/data/allxml_npz_dual_track_optimized_no_underscore"
    save_steps = -1

    # 测试集配置
    use_test_set: bool = True  # 是否使用测试集
    test_split_ratio: float = 0.10  # 测试集划分比例（5%作为测试集）
    test_frequency: float = 0.10  # 测试频率（每0.10个epoch测试一次）
    test_batch_size: int = 4  # 测试时的batch size
    test_save_results: bool = True  # 是否保存测试结果到tensorboard

    # Drop Accompaniment 训练增强
    # 训练时以此概率将某 beat 的 acc 替换为空，强迫模型减少对 acc history 的依赖
    # 0.0 = 关闭（默认），推荐值 0.1~0.2
    acc_drop_prob: float = 0

    # Position Shift 训练增强
    # 每首曲子随机采样 Δ ∈ {0,...,pos_shift_max}，整体偏移 beat 内的位置标记
    # 使 BEAT token 不再是严格的"位置0"锚点，增强泛化
    # 0 = 关闭（默认），推荐值 3
    pos_shift_max: int = 3
    random_seed: int = 42  # 数据集划分的随机种子


@dataclass
class ModelConfig:
    """模型架构配置"""

    # Token-level GPT配置
    vocab_size: int = 268
    hidden_size: int = 768  #
    num_hidden_layers: int = 18
    num_attention_heads: int = 6
    intermediate_size: int = 3072
    max_position_embeddings: int = 3500
    markoffset = 81  # 位置标记起始ID
    measures_length = 88  # 每个小节的最大长度（单位：位置标记数）
    patch_h = 1  # 音高方向的patch大小
    patch_w = 4  # 时间方向的patch大小

    # Token IDs
    track_marker_acc_id: int = 170  # acc 轨结束标记
    track_marker_mel_id: int = 171  # mel 轨结束标记
    beat_marker_id: int = 172  # 拍分隔符
    bar_token_id: int = 255
    eos_token_id: int = 256
    bos_token_id: int = 257
    pad_token_id: int = 258
    time_sig_offset_id: int = 259
    bpm_offset_id: int = 264
    train_cutoff_len = 2048  # 训练时的截断长度
    rope_theta: float = 10000.0  # RoPE base
    dropout = 0.1

    # 拍号 191: 3/4 192: 4/4
    # 和弦作为小节bar  190
