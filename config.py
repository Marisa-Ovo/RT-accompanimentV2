from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass
class TrainingConfig:
    """训练配置类"""

    # 基础配置
    time = datetime.now().strftime("%m%d_%H%M")
    output_dir: str = "./checkpoints"
    num_epochs: int = 8
    save_model_epochs: int = 1
    train_batch_size = 2
    gradient_accumulation_steps: int = 127
    lr_warmup_steps: int = 50
    mixed_precision: Literal["no", "fp16", "bf16"] = "fp16"

    # 学习率配置
    learning_rate: float = 5e-5

    # 日志配置
    log: bool = False
    log_every_n_steps: int = 20
    tensorboard_log_dir: str = "./logs"
    tensorboard_log_name: str = f"music_transformer_{time}"
    # data_dir = "/home/lab-wei.zhenao/boyu/Dataset/DATA6_6T/cby/musicxml/chord_retrieve_two_chords_NAfliter3"
    data_dir = "/DATA2_4T/cby/home/lab-wei.zhenao/boyu/Dataset/allxml_npz_dual_track_optimized"
    # data_dir = "/home/lab-wei.zhenao/boyu/Dataset/allxml_npz_optimized"
    save_steps = -1

    # 测试集配置
    use_test_set: bool = True  # 是否使用测试集
    test_split_ratio: float = 0.05  # 测试集划分比例（5%作为测试集）
    test_frequency: float = 0.25  # 测试频率（每0.25个epoch测试一次）
    test_batch_size: int = 4  # 测试时的batch size
    test_save_results: bool = True  # 是否保存测试结果到tensorboard
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
    measures_length = 88  # 每个小节的最大长度（单位：位置标记数） 257
    offset = 257
    patch_h = 1  # 音高方向的patch大小
    patch_w = 4  # 时间方向的patch大小

    pad_token_id: int = 257 + 1
    bos_token_id: int = 257
    eos_token_id: int = 257 - 1
    bar_token_id: int = 257 - 2
    time_sig_offset_id: int = 259
    bpm_offset_id: int = 259 + 5
    train_cutoff_len = 2048  # 训练时的截断长度
    rope_theta: float = 10000.0  # RoPE base
    dropout = 0.1

    # ---------------------------------------------------------------------------
    # Playing Mode tokens（Phase 1）
    # 启用时需同步将 vocab_size 改为 270，并重新训练（会破坏旧 checkpoint）
    # ---------------------------------------------------------------------------
    use_playing_token: bool = False  # 是否在序列中插入 playing token
    play_token_id: int = 268  # PLAY：保留此 beat
    skip_token_id: int = 269  # SKIP：跳过/静音此 beat
    # vocab_size 扩充提示：启用 playing token 时请将上方 vocab_size 改为 270

    # 拍号 191: 3/4 192: 4/4
    # 和弦作为小节bar  190
