import numpy as np
import torch
import pretty_midi
from my_tokenizer import PianoMusicTokenizer


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
    sustain_roll = pianoroll[0]
    onset_roll = pianoroll[1]

    instrument = pretty_midi.Instrument(program=0, name=track_name)
    seconds_per_16th = 60.0 / tempo / 4

    note_count = 0
    for pitch_idx in range(88):
        pitch = pitch_idx + 21

        onset_positions = np.where(onset_roll[pitch_idx] > 0)[0]

        for onset_pos in onset_positions:
            end_pos = onset_pos + 1
            while end_pos < sustain_roll.shape[1] and sustain_roll[pitch_idx, end_pos] > 0:
                end_pos += 1

            start_time = onset_pos * seconds_per_16th
            end_time = end_pos * seconds_per_16th

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
    将生成结果转换为MIDI文件（mel 和 acc 分别保存为两个 track）

    Args:
        result_dict: generate_accompaniment返回的字典
        save_path: MIDI文件保存路径
        velocity: MIDI音符力度
        tempo: BPM速度
    """
    print("="*50)
    print("开始处理tokens到MIDI（双track模式）...")

    mel_beats = result_dict['mel_beats']
    acc_beats = result_dict['acc_beats']
    metadata = result_dict.get('metadata', {})

    if tempo is None or tempo == 120:
        tempo = metadata.get('bpm', 120)
    if tempo is None:
        tempo = 120
    print(f"Melody beats数量: {len(mel_beats)}")
    print(f"Accompaniment beats数量: {len(acc_beats)}")
    print(f"BPM: {tempo}")

    # 使用默认 tokenizer 解码
    tokenizer = PianoMusicTokenizer()
    v = tokenizer.vocab

    mel_pianoroll = tokenizer.decode_beats_to_pianoroll(mel_beats, track_marker_id=v.track_marker_mel)
    acc_pianoroll = tokenizer.decode_beats_to_pianoroll(acc_beats, track_marker_id=v.track_marker_acc)

    # 对齐长度
    mel_len = mel_pianoroll.shape[2]
    acc_len = acc_pianoroll.shape[2]
    target_length = max(mel_len, acc_len)

    print(f"Melody 长度: {mel_len}, Accompaniment 长度: {acc_len}, 目标长度: {target_length}")

    if mel_len < target_length:
        pad_len = target_length - mel_len
        print(f"Melody较短，填充 {pad_len} 个时间步")
        mel_pianoroll = np.pad(mel_pianoroll,
                                 ((0, 0), (0, 0), (0, pad_len)), constant_values=0)

    if acc_len < target_length:
        pad_len = target_length - acc_len
        print(f"Accompaniment较短，填充 {pad_len} 个时间步")
        acc_pianoroll = np.pad(acc_pianoroll,
                                 ((0, 0), (0, 0), (0, pad_len)), constant_values=0)

    # 创建MIDI
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)

    print("创建Track:")
    mel_instrument = pianoroll_to_midi_notes(
        mel_pianoroll, tempo=tempo, velocity=velocity, track_name="Melody")
    midi.instruments.append(mel_instrument)

    acc_instrument = pianoroll_to_midi_notes(
        acc_pianoroll, tempo=tempo, velocity=velocity, track_name="Accompaniment")
    midi.instruments.append(acc_instrument)

    midi.write(save_path)

    total_notes = len(mel_instrument.notes) + len(acc_instrument.notes)
    print("="*50)
    print(f"✓ MIDI文件已保存到: {save_path}")
    print(f"✓ 总时长: {midi.get_end_time():.2f} 秒")
    print(f"✓ 总音符数量: {total_notes}")
    print(f"✓ Track数量: 2 (Melody + Accompaniment)")
    print(f"✓ BPM: {tempo}")
    print("="*50)
