"""
MidiConverter — Token/Pianoroll → MIDI 转换器

职责单一：解码 + 写 MIDI，不涉及模型或数据集。
"""

import os
import numpy as np
import pretty_midi
from my_tokenizer import PianoMusicTokenizer


class MidiConverter:
    """Token beats / pianoroll → MIDI 文件。构造时注入 tokenizer。"""

    def __init__(self, tokenizer: PianoMusicTokenizer):
        self.tokenizer = tokenizer
        self.vocab = tokenizer.vocab

    def beats_to_midi(self, mel_beats, acc_beats, tempo=120,
                      save_path="temp.mid", velocity=64):
        """
        将 mel/acc beat tokens 解码并保存为双轨 MIDI。

        Args:
            mel_beats: List[List[int]] — melody beat tokens
            acc_beats: List[List[int]] — accompaniment beat tokens
            tempo: BPM
            save_path: 输出路径
            velocity: MIDI velocity
        """
        mel_pr = self.tokenizer.decode_beats_to_pianoroll(
            mel_beats, track_marker_id=self.vocab.track_marker_mel)
        acc_pr = self.tokenizer.decode_beats_to_pianoroll(
            acc_beats, track_marker_id=self.vocab.track_marker_acc)

        # 对齐长度
        target_len = max(mel_pr.shape[2], acc_pr.shape[2])
        mel_pr = self._pad_time(mel_pr, target_len)
        acc_pr = self._pad_time(acc_pr, target_len)

        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        midi.instruments.append(
            self._pianoroll_to_instrument(mel_pr, tempo, velocity, "Melody"))
        midi.instruments.append(
            self._pianoroll_to_instrument(acc_pr, tempo, velocity, "Accompaniment"))

        self._ensure_dir(save_path)
        midi.write(save_path)

        total = sum(len(inst.notes) for inst in midi.instruments)
        print(f"MIDI saved: {save_path} | {tempo} BPM | {total} notes | {midi.get_end_time():.1f}s")

    def gt_to_midi(self, npz_path, save_path, velocity=80):
        """从 GT npz 文件直接生成双轨 MIDI。"""
        data = np.load(npz_path, allow_pickle=True)
        meta = data['metadata'].item()
        tempo = meta.get('bpm', 120) or 120

        measures = [data[f'measure_{i}'] for i in range(meta['num_measures'])]
        full_pr = np.concatenate(measures, axis=2)  # (4, 88, t)

        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        midi.instruments.append(
            self._pianoroll_to_instrument(full_pr[:2], tempo, velocity, "Melody"))
        midi.instruments.append(
            self._pianoroll_to_instrument(full_pr[2:], tempo, velocity, "Accompaniment"))

        self._ensure_dir(save_path)
        midi.write(save_path)
        print(f"GT MIDI saved: {save_path} | {tempo} BPM | {meta['num_measures']} measures")

    # ---- internal ----

    @staticmethod
    def _pad_time(arr, target_len):
        if arr.shape[2] < target_len:
            return np.pad(arr, ((0, 0), (0, 0), (0, target_len - arr.shape[2])))
        return arr

    @staticmethod
    def _pianoroll_to_instrument(pianoroll, tempo, velocity, name="Piano"):
        """(2, 88, t) pianoroll → pretty_midi.Instrument"""
        sustain, onset = pianoroll[0], pianoroll[1]
        inst = pretty_midi.Instrument(program=0, name=name)
        sec_per_16th = 60.0 / tempo / 4

        for pitch_idx in range(88):
            for onset_pos in np.where(onset[pitch_idx] > 0)[0]:
                end_pos = onset_pos + 1
                while end_pos < sustain.shape[1] and sustain[pitch_idx, end_pos] > 0:
                    end_pos += 1
                inst.notes.append(pretty_midi.Note(
                    velocity=velocity,
                    pitch=pitch_idx + 21,
                    start=onset_pos * sec_per_16th,
                    end=end_pos * sec_per_16th,
                ))
        return inst

    @staticmethod
    def _ensure_dir(path):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
