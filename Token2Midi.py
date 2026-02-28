import numpy as np
import torch
import pretty_midi
from my_tokenizer import PianoRollTokenizer

def map_piano_to_midi(piano_roll):
    """
    将88键钢琴pianoroll映射到完整的128维MIDI空间
    
    Args:
        piano_roll: (88, time_steps) 的88键钢琴pianoroll
    
    Returns:
        full_midi_roll: (128, time_steps) 的完整MIDI pianoroll
    """
    if isinstance(piano_roll, torch.Tensor):
        piano_roll = piano_roll.cpu().numpy()
    
    # 创建128维的MIDI pianoroll (MIDI 0-127)
    full_midi_roll = np.zeros((128, piano_roll.shape[1]), dtype=piano_roll.dtype)
    
    # 标准88键钢琴从A0(MIDI 21)到C8(MIDI 108)
    # 映射88键到MIDI 21-108
    midi_start = 21  # A0
    
    # 将88键数据映射到MIDI 21-108范围
    full_midi_roll[midi_start:midi_start+88, :] = piano_roll
    
    return torch.tensor(full_midi_roll, dtype=torch.float32)


def process_part_beats_to_pianoroll(beats_list, tokenizer=None,
                                     split_marker_id_1=170, split_marker_id_2=171):
    """
    将单个part的beats列表处理成pianoroll

    Args:
        beats_list: beat tokens列表（每个元素是一个tensor或列表）
        tokenizer: PianoRollTokenizer实例（如果为None，创建默认实例）
        split_marker_id_1, split_marker_id_2: 分隔标记

    Returns:
        pianoroll: (2, 88, t) 的pianoroll数组
    """
    # 如果没有提供tokenizer，创建默认实例
    if tokenizer is None:
        tokenizer = PianoRollTokenizer(
            patch_h=1,
            patch_w=4,
            marker_offset=81,
            measures_length=88,
            end_marker_part0=split_marker_id_1,
            end_marker_part1=split_marker_id_2,
            empty_marker=169,
            img_h=88
        )

    # 1. 展平并过滤tokens（去除>=173的特殊标记，保留255 bar token和音乐tokens）
    all_tokens = []
    for beat in beats_list:
        if isinstance(beat, torch.Tensor):
            all_tokens.extend(beat.cpu().tolist())
        elif isinstance(beat, (list, np.ndarray)):
            all_tokens.extend(beat if isinstance(beat, list) else beat.tolist())
        else:
            # 单个数值
            all_tokens.append(beat)

    all_tokens = np.array(all_tokens, dtype=np.int64)
    filtered_tokens = np.array([t for t in all_tokens if t < 173])

    # 2. 解压缩tokens
    decompressed_matrix = tokenizer.decompress_tokens(
        filtered_tokens,
        end_marker_id=None  # 支持两种结束标记
    )
    print(f"解压后beat数量: {decompressed_matrix.shape[0]}")

    # 3. 转换为pianoroll
    pianoroll = tokenizer.patch_tokens_to_image(decompressed_matrix)
    print(f"Part pianoroll形状: {pianoroll.shape}")
    return pianoroll


def pianoroll_to_midi_notes(pianoroll, tempo, velocity, track_name="Piano"):
    """
    将pianoroll转换为MIDI notes列表

    Args:
        pianoroll: (2, 88, t) 的pianoroll数组 [sustain, onset]
        tempo: BPM速度
        velocity: MIDI音符力度
        track_name: track名称

    Returns:
        instrument: pretty_midi.Instrument对象
    """
    sustain_roll = pianoroll[0]  # shape: (88, t)
    onset_roll = pianoroll[1]    # shape: (88, t)

    # 创建乐器track
    instrument = pretty_midi.Instrument(program=0, name=track_name)  # 0 = Acoustic Grand Piano

    # 计算每个时间单位的秒数（1/16音符）
    seconds_per_16th = 60.0 / tempo / 4

    # 遍历每个音高
    note_count = 0
    for pitch_idx in range(88):
        pitch = pitch_idx + 21  # MIDI音高 (A0=21开始)

        # 找到所有onset位置
        onset_positions = np.where(onset_roll[pitch_idx] > 0)[0]

        for onset_pos in onset_positions:
            # 找到音符结束位置（sustain结束）
            end_pos = onset_pos + 1

            # 继续向后查找直到sustain结束
            while end_pos < sustain_roll.shape[1] and sustain_roll[pitch_idx, end_pos] > 0:
                end_pos += 1

            # 转换为时间（秒）
            start_time = onset_pos * seconds_per_16th
            end_time = end_pos * seconds_per_16th

            # 创建MIDI音符
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=start_time,
                end=end_time
            )
            instrument.notes.append(note)
            note_count += 1

    print(f"  {track_name}: {note_count} 个音符")
    return instrument


def tokens_to_midi(result_dict, save_path="temp.mid", velocity=64, tempo=120):
    """
    将生成结果转换为MIDI文件（part0和part1分别保存为两个track）

    Args:
        result_dict: generate_accompaniment返回的字典，包含:
            - 'part0_beats': 条件part0的beats列表
            - 'part1_beats': 生成的part1 beats列表
            - 'metadata': 元数据（包含bpm等信息）
        save_path: MIDI文件保存路径
        velocity: MIDI音符力度 (1-127)
        tempo: 节拍速度 (BPM)，如果为None则从metadata读取
    """
    print("="*50)
    print("开始处理tokens到MIDI（双track模式）...")

    # 1. 提取数据
    part0_beats = result_dict['part0_beats']
    part1_beats = result_dict['part1_beats']
    metadata = result_dict.get('metadata', {})

    # 从metadata获取BPM
    if tempo is None or tempo == 120:
        tempo = metadata.get('bpm', 120)
    if tempo is None:
        tempo = 120  # 默认BPM
    print(f"Part0 beats数量: {len(part0_beats)}")
    print(f"Part1 beats数量: {len(part1_beats)}")
    print(f"BPM: {tempo}")

    # 2. 分别处理part0和part1
    part0_pianoroll = process_part_beats_to_pianoroll(
        part0_beats,
        tokenizer=None,  # 使用默认tokenizer
        split_marker_id_1=170,  # part0的end marker
        split_marker_id_2=171   # part1的end marker (虽然part0不应该有171)
    )

    part1_pianoroll = process_part_beats_to_pianoroll(
        part1_beats,
        tokenizer=None,  # 使用默认tokenizer
        split_marker_id_1=170,
        split_marker_id_2=171   # part1的end marker
    )

    # 3. 对齐长度（以较长者为准）
    target_length = max(part0_pianoroll.shape[2], part1_pianoroll.shape[2])
    part0_length = part0_pianoroll.shape[2]
    part1_length = part1_pianoroll.shape[2]

    print(f"Part0 长度: {part0_length}, Part1 长度: {part1_length}, 目标长度: {target_length}")

    if part0_length < target_length:
        pad_length = target_length - part0_length
        print(f"Part0较短，填充 {pad_length} 个时间步")
        part0_pianoroll = np.pad(
            part0_pianoroll,
            ((0, 0), (0, 0), (0, pad_length)),
            mode='constant',
            constant_values=0
        )

    if part1_length < target_length:
        pad_length = target_length - part1_length
        print(f"Part1较短，填充 {pad_length} 个时间步")
        part1_pianoroll = np.pad(
            part1_pianoroll,
            ((0, 0), (0, 0), (0, pad_length)),
            mode='constant',
            constant_values=0
        )

    # 4. 创建MIDI对象
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)

    # 5. 为part0创建track（高声部）
    print("创建Track:")
    part0_instrument = pianoroll_to_midi_notes(
        part0_pianoroll,
        tempo=tempo,
        velocity=velocity,
        track_name="Part0 (Melody)"
    )
    midi.instruments.append(part0_instrument)

    # 6. 为part1创建track（低声部/伴奏）
    part1_instrument = pianoroll_to_midi_notes(
        part1_pianoroll,
        tempo=tempo,
        velocity=velocity,
        track_name="Part1 (Accompaniment)"
    )
    midi.instruments.append(part1_instrument)

    # 7. 保存MIDI文件
    midi.write(save_path)

    total_notes = len(part0_instrument.notes) + len(part1_instrument.notes)
    print("="*50)
    print(f"✓ MIDI文件已保存到: {save_path}")
    print(f"✓ 总时长: {midi.get_end_time():.2f} 秒")
    print(f"✓ 总音符数量: {total_notes}")
    print(f"✓ Track数量: 2 (Part0高声部 + Part1低声部)")
    print(f"✓ BPM: {tempo}")
    print("="*50)
    
