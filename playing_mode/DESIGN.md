# Playing Mode 设计文档（Beat-Marker V2 版）

## 1. 动机与概念

### 1.1 问题：自回归实时伴奏的 Exposure Bias

实时伴奏的自回归生成存在训练-推理不一致：

- **训练时**：模型上下文全部来自 GT → $P(y_t \mid y_{1:t-1}^{\text{GT}},\; x_{1:t-1})$
- **推理时**：模型上下文来自自己的生成 → $P(y_t \mid \hat{y}_{1:t-1},\; x_{1:t-1})$

一旦某拍的 acc 生成偏离合理范围，后续所有拍的 context 都被污染。模型从未在训练中见过这种"带误差的 context"，无法修正，导致 **误差持续积累、越写越差**。

这是自回归生成的经典 exposure bias 问题，但在实时伴奏场景中尤为严重——因为生成是 beat-by-beat 持续进行的，没有机会回头修改。

### 1.2 解决思路：Playing Token 作为 acc-mel 匹配性判断

在每拍的 mel 注入之后，插入一个 **playing token（PLAY / SKIP）**。此时模型同时拥有自己生成的 acc 和实际到来的 mel，可以判断"我的伴奏和旋律匹配吗？"。

- **PLAY**：acc 与 mel 匹配，保留此 beat 的 acc
- **SKIP**：acc 与 mel 不匹配，此 beat 输出静音，同时告知后续 beat "这拍的 acc 不可靠"

**训练时**：labeler 标注哪些 beat 的 acc "不够好"（加噪/NLL/弱模型），SKIP beat 的 acc 用降质版本，与 GT mel 天然不匹配。模型学会：(1) 识别 acc-mel 不匹配 → 预测 SKIP；(2) 在含 SKIP 的 context 下继续合理生成。

**推理时**：模型生成 acc → mel 注入 → 模型判断 acc-mel 是否匹配 → PLAY/SKIP。

核心思想：**让模型认识到先前生成的伴奏并不完美，学会在不完美 context 下继续合理生成。**

---

## 2. 在 Beat-Marker 格式中的位置

### 2.1 当前序列格式（无 playing mode）

```
[BOS][TS][BPM] [BAR][BEAT][acc₀...TRK_ACC][mel₀...TRK_MEL] [BEAT][acc₁...TRK_ACC][mel₁...TRK_MEL] ... [EOS]
```

每拍内部：`[BEAT] [acc compressed tokens] [TRK_ACC(170)] [mel compressed tokens] [TRK_MEL(171)]`

### 2.2 Playing token 插入位置分析

有四个候选位置：

| 位置 | 格式 | 决策依据 | 问题 |
|------|------|---------|------|
| A: TRK_ACC 之后 | `[BEAT][acc...TRK_ACC][PLAY/SKIP][mel...TRK_MEL]` | 只有 acc（自评估） | 认知闭环：让出错的系统评估自己；且未见 mel 无法判断匹配性 |
| B: BEAT 之后 | `[BEAT][PLAY/SKIP][acc...TRK_ACC][mel...TRK_MEL]` | 只有历史 context | 信息最少，无法评估当前 beat |
| C: Aux head | `[BEAT][acc...TRK_ACC][mel...TRK_MEL]`（不变） | 取决于读取位置 | 需修改模型结构 |
| **D: TRK_MEL 之后** | **`[BEAT][acc...TRK_ACC][mel...TRK_MEL][PLAY/SKIP]`** | **acc + mel** | **唯一能做 acc-mel 匹配性判断的位置** |

**选定：方案 D（Post-mel）**

关键洞察：在实时伴奏中，模型先生成 acc（看不到当前拍 mel），然后当前拍 mel 到来。此时模型同时拥有 **自己生成的 acc** 和 **实际到来的 mel**，可以判断"我的伴奏和旋律匹配吗？"。

这不是纯粹的"自评"——它有**明确的外部参照**（mel），是一个 well-defined 的 acc-mel compatibility 判断，完全可学。

类比现实：伴奏师先弹（acc），听到独奏者实际演奏（mel），然后判断"我弹对了没有"。

### 2.3 序列格式

每拍结构：`[BEAT][acc...TRK_ACC][mel...TRK_MEL][PLAY/SKIP]`

```
训练序列:
[BOS][TS][BPM] [BAR][BEAT][acc...TRK_ACC][mel...TRK_MEL][PLAY] [BEAT][acc...TRK_ACC][mel...TRK_MEL][SKIP] ... [EOS]

Labels（selective loss）:
[PAD][PAD][PAD] [PAD][PAD] [acc...TRK_ACC][PAD  PAD     ][PLAY] [PAD] [acc...TRK_ACC][PAD  PAD     ][SKIP] ... [EOS]
```

**关键设计**：
- playing token **参与 loss**——它是自回归序列的一部分，模型在 TRK_MEL 位置预测下一个 token 为 PLAY 或 SKIP
- mel 部分不参与 loss（条件输入），但 playing token 参与——语义上它属于"acc-mel 匹配性判断"
- 每个 beat 是一个完整的自包含单元：`[BEAT][acc][mel][judgment]`

### 2.4 Token ID 分配

| Token | ID | 说明 |
|-------|:---:|------|
| PLAY | 268 | 保留此 beat 的 acc |
| SKIP | 269 | 静音此 beat 的 acc |

`vocab_size`: 268 → **270**

---

## 3. 整体架构（两个正交维度）

```
playing_mode/
├── labelers/      # 训练时：如何自动标注 PLAY/SKIP
│   ├── base.py
│   ├── noise.py           # 随机加噪
│   ├── nll_based.py       # 基于模型 NLL
│   ├── weak_model.py      # 弱模型对比
│   └── music_theory.py    # 乐理规则
└── predictors/    # 推理时：如何预测 PLAY/SKIP
    ├── base.py
    ├── aux_head.py         # PianoLLaMA 辅助分类头
    └── separate_model.py   # 独立轻量分类器
```

两个维度**完全解耦**——任意 labeler 可配合任意 predictor。

---

## 4. Labelers（标签生成策略）

### 4.1 接口（更新为 acc/mel 语义）

```python
@dataclass
class LabelContext:
    beat_index: int = 0
    mel_beat_tokens: list[int] = field(default_factory=list)    # 同位置的 mel（条件）
    acc_beat_tokens: list[int] = field(default_factory=list)    # GT acc（用于对比）
    prev_acc_tokens: list[int] = field(default_factory=list)    # 前若干 beat 的 acc 历史
    nll_score: float | None = None                              # NLL（仅 nll_based 使用）
    extra: dict = field(default_factory=dict)

class BaseLabeler(ABC):
    @abstractmethod
    def label_beat(self, acc_tokens: list[int], context: LabelContext) -> int:
        """返回 PLAY=1 / SKIP=0"""
```

### 4.2 四种策略的适配说明

| 策略 | 核心逻辑 | Beat-Marker 适配要点 |
|------|---------|---------------------|
| **noise** | 以概率 p 将 GT acc 替换为 `[EMPTY(169) TRK_ACC(170)]`，替换的标 SKIP | 直接操作压缩后 token；density 策略需解析 `marker_offset` 计数音符数 |
| **nll_based** | 用已训练模型 teacher-forcing 计算每 beat acc 的平均 NLL | 需定位 acc 段：`BEAT` 到 `TRK_ACC` 之间的 token；NLL 只算 acc 位置 |
| **weak_model** | 用弱模型（少 epoch / 小模型）生成 acc，与 GT 对比 | 对比粒度：beat 级 patch token 完全匹配 or 解码后 piano roll 距离 |
| **music_theory** | 按乐理规则：弱拍/填充型伴奏标 SKIP | 需从 mel tokens 提取和声信息（patch 解码 → 识别音高集合） |

### 4.3 Noise Labeler 详细设计（Phase 1 优先）

```python
class NoiseLabeler(BaseLabeler):
    """三种模式：
    - 'random':     每 beat 独立以概率 p 标为 SKIP
    - 'density':    空拍（只含 EMPTY + TRK_ACC）标为 SKIP
    - 'contiguous': 连续 N 拍标为 SKIP（模拟段落缺失）
    """
```

空拍判定（Beat-Marker 编码）：
```python
def is_empty_beat(acc_tokens: list[int], vocab) -> bool:
    # 空拍压缩为 [EMPTY(169), TRK_ACC(170)]
    return acc_tokens == [vocab.empty_marker, vocab.track_marker_acc]
```

---

## 5. Predictors（推理时预测）

### 5.1 Inline Token（Post-mel，主方案）

**无需独立 predictor 模块**——playing token 就是正常的 next-token prediction，与 acc tokens 共用同一个 lm_head。

推理流程（在 `model.py::generate_accompaniment` 中实现）：

```
每拍:
  1. inject [BEAT]
  2. 自回归 generate acc → 直到生成 TRK_ACC(170)
  3. inject mel → 注入当前拍的 GT 旋律 [mel...TRK_MEL(171)]
  4. generate 1 token → 模型综合 acc + mel 信息，预测 PLAY(268) 或 SKIP(269)
  5. if PLAY: 保留 acc 输出到 MIDI
     if SKIP: 输出静音，SKIP token 告诉后续 beats "这拍的 acc 不可靠"
```

**Step 4 的关键**：此时模型的 attention 同时覆盖了 acc tokens 和 mel tokens，判断的是"我生成的伴奏和实际旋律是否匹配"——有明确外部参照的 compatibility 判断。

### 5.2 Aux Head（对照实验）

在 `model.py::PianoLLaMA` 添加：

```python
self.aux_head = nn.Linear(hidden_size, 1)  # sigmoid → P(PLAY)
```

读取 **TRK_MEL 位置**的 hidden state（此时已融合 acc + mel 信息），BCE loss 独立反向传播：

```python
# trainer.py
total_loss = lm_loss + alpha * F.binary_cross_entropy_with_logits(
    aux_logit,                    # aux_head(hidden_at_trk_mel)
    playing_label.float()         # 1.0=PLAY, 0.0=SKIP
)
```

推理时：注入 mel 后，读 aux_head 输出 → 概率 > threshold 则 PLAY。

---

## 6. 具体实现流程

### Phase 1：数据管线（先跑通格式，不训练）

**Step 1.1** — config.py
```python
# 新增
play_token_id: int = 268
skip_token_id: int = 269
vocab_size: int = 270        # 268 → 270
use_playing_mode: bool = False  # 开关
```

**Step 1.2** — labelers/base.py
- `LabelContext` 字段重命名：`part0_beat_tokens` → `mel_beat_tokens`，`gt_part1_beat_tokens` → `acc_beat_tokens`
- `label_beat()` 参数：`beat_tokens` → `acc_tokens`（语义更清晰）

**Step 1.3** — labelers/noise.py（最简实现）
- `random` 模式：`return 0 if random.random() < self.skip_prob else 1`
- `density` 模式：空拍返回 SKIP，有音符返回 PLAY

**Step 1.4** — my_tokenizer.py::build_training_sequence
```python
# playing token 在 mel 之后（Post-mel 位置）
for acc, mel in beats:
    # [beat_marker]
    inp_parts.append(bm);  lbl_parts.append(PAD)

    if self.use_playing_mode:
        label = labeler.label_beat(acc.tolist(), context)

    # [acc tokens]（SKIP 时替换为降质版本）
    if self.use_playing_mode and label == 0:
        degraded_acc = torch.tensor([vocab.empty_marker, vocab.track_marker_acc], dtype=torch.long)
        inp_parts.append(degraded_acc);  lbl_parts.append(torch.full_like(degraded_acc, PAD))
    else:
        inp_parts.append(acc);  lbl_parts.append(acc)

    # [mel tokens]（条件输入，不参与 loss）
    inp_parts.append(mel);  lbl_parts.append(torch.full_like(mel, PAD))

    # [playing token]（在 mel 之后，模型能综合 acc + mel 判断）
    if self.use_playing_mode:
        playing_tok = play_token_id if label == 1 else skip_token_id
        pt = torch.tensor([playing_tok], dtype=torch.long)
        inp_parts.append(pt);  lbl_parts.append(pt)  # 参与 loss
```

**Step 1.5** — 验证
- 单元测试：构造样本 → 打印序列 → 人工检查格式
- 长度估算：每 beat 多 1 token，典型序列增长 ~3-5%

### Phase 2：模型训练

**Step 2.1** — 无需改 model.py（方案 A）
- PLAY/SKIP 只是词表中的两个新 token，lm_head 自动覆盖
- Embedding 层自动扩展（vocab_size=270）

**Step 2.2** — 训练策略
- 先用 noise labeler（skip_prob=0.3）生成训练数据
- 正常训练 PianoLLaMA，playing token 的 loss 自然包含在 acc loss 中
- **不需要额外 loss 权重**：playing token 只有 1 个/beat，自动被 acc tokens 平衡

**Step 2.3** — PianoDataset.py 集成
```python
class PianoDataset:
    def __init__(self, ..., labeler: BaseLabeler = None):
        self.labeler = labeler

    def __getitem__(self, idx):
        # ... 加载 measures ...
        input_ids, labels = self.tokenizer.build_training_sequence(
            measures, metadata, labeler=self.labeler)
        return {'input_ids': input_ids, 'labels': labels}
```

### Phase 3：推理集成

**Step 3.1** — model.py::generate_accompaniment
```python
# 每拍的生成流程
acc_tokens = self._generate_one_beat(...)   # Step 1-2: 自回归直到 TRK_ACC
self._inject_tokens(mel_tokens)             # Step 3: 注入当前拍 mel
playing_token = self._sample_token(...)     # Step 4: 模型综合 acc+mel 预测 PLAY/SKIP

is_play = (playing_token.item() == vocab.play_token_id)
if not is_play:
    # MIDI 输出：静音。KV cache 中 acc tokens 保留 + SKIP 信号告知后续 beats
    acc_tokens_for_midi = [vocab.empty_marker, vocab.track_marker_acc]
```

**Step 3.2** — build_generation_schedule 更新
- schedule 中每拍的步骤顺序：`inject[BEAT]` → `generate acc` → `inject mel` → `generate playing`
- 新增 GenerationStep action: `"generate_playing"` — 采样一个 token（仅限 PLAY/SKIP）

**Step 3.3** — Token2Midi.py
- beats_to_midi 中自动忽略 PLAY/SKIP token（ID >= 268 直接跳过）

### Phase 4：NLL Labeler（需要已训练模型）

**Step 4.1** — 用 Phase 2 训练好的模型（不含 playing mode）计算 NLL
```python
# teacher forcing：输入完整 GT 序列，计算每个 acc token 的 NLL
# 按 beat 聚合：mean NLL per beat
nll_per_beat = []
for beat_start, beat_end in beat_boundaries:
    beat_nll = -log_probs[beat_start:beat_end].mean()
    nll_per_beat.append(beat_nll)
```

**Step 4.2** — 阈值策略
- `median`：NLL > 全曲中位数 → SKIP（自适应，约 50% SKIP）
- `percentile(75)`：仅标记最差 25% 为 SKIP
- `fixed(2.0)`：绝对阈值

**Step 4.3** — 用 NLL 标签重训含 playing mode 的模型

### Phase 5：消融实验

| 实验 | 变量 | 评估 |
|------|------|------|
| Labeler 对比 | noise(0.3) vs nll(median) vs nll(p75) | beat_exact_match, FMD |
| SKIP 比例影响 | skip_prob = 0.1/0.2/0.3/0.5 | acc 质量 vs 静音比 |
| Inline vs Aux | 方案 A vs 方案 C | 精度、延迟、模型大小 |
| SKIP 行为 | 静音 vs sustain hold | 听感主观评价 |

---

## 7. 关键设计决策（已确定）

### Q1: PLAY/SKIP token ID 如何分配？
**A**: `PLAY=268, SKIP=269, vocab_size=270`。扩充词表 2 个 token，对模型影响可忽略（embedding 增加 2×768=1536 参数）。

### Q2: Playing token 是否参与 loss？
**A**: **是，参与 next-token loss**。它位于 TRK_MEL 之后，是序列中的一个普通 token，模型通过标准自回归学习预测它。无需 aux loss。

Aux Head 对照：单独 BCE loss，读 TRK_MEL 位置 hidden state，权重 α=0.1。

### Q3: SKIP 时推理行为？
**A**: **默认：静音（空拍）**。即 `[EMPTY(169), TRK_ACC(170)]` 替换生成的 acc tokens。
后续可实验 sustain hold（保持上一拍延音），但初版用最简方案。

### Q4: 需要 soft label 吗？
**A**: **初版用 hard label（PLAY/SKIP 二分类）**。NLL labeler 天然产生连续分值，但模型侧只需预测两个 token ID 之一。未来可尝试 label smoothing 或 temperature scaling。

### Q5: SKIP 的 beat 在训练时，acc 用什么？

这是最关键的设计决策，有两种方案，各有道理：

**方案 α：SKIP 位置用 degraded acc（推荐）**

```
PLAY: [BEAT][acc_gt......TRK_ACC][mel...TRK_MEL][PLAY]  ← GT acc 与 GT mel 匹配
SKIP: [BEAT][acc_degraded.TRK_ACC][mel...TRK_MEL][SKIP]  ← 降质 acc 与 GT mel 不匹配
```

模型在训练中就接触到"不匹配的 acc-mel + SKIP"，推理时遇到自身错误不会崩溃。

**方案 β：SKIP 位置仍用 GT acc**

```
SKIP: [BEAT][acc_gt...TRK_ACC][mel...TRK_MEL][SKIP]  ← acc 和 mel 都是 GT（其实匹配），但标签说 SKIP
```

劣势：训练时 SKIP 前面的 acc-mel 是匹配的，推理时 SKIP 前面是真的不匹配 → 信号与实际脱节。

**推荐方案 α**：noise labeler 直接将选中 beat 的 acc 替换为 `[EMPTY TRK_ACC]`（最极端的降质）。更精细的降质可在 Phase 4（weak_model / self-play）实现。

### Q6: 防止模式坍塌（总预测 PLAY）？
**A**: 两道保险：
1. **训练数据平衡**：noise labeler 控制 skip_prob，nll labeler 用 percentile 控制比例
2. **评估监控**：跟踪 PLAY/SKIP 预测比例，若 PLAY > 95% 则 labeler 参数需调整

---

## 8. 待探索问题

- [ ] **最优 SKIP 比例**：skip_prob 过高→模型什么都不弹，过低→门控无效。需实验确定 sweet spot（猜测 0.2-0.3）
- [ ] **NLL 与生成质量的相关性**：高 NLL 的 beat 是否真的生成质量差？需人工验证
- [ ] **Playing token 对后续 beat 生成的影响**：SKIP 信号是否会让模型在后续 beat 更保守？
- [ ] **多级门控**：是否需要 PLAY / SOFT_PLAY / SKIP 三级（控制 velocity）？

---

## 9. 实施路线图汇总

| Phase | 内容 | 改动文件 | 估时 |
|:-----:|------|---------|:----:|
| 1 | 数据管线 + noise labeler | config.py, base.py, noise.py, my_tokenizer.py | 1 天 |
| 2 | 模型训练（noise 标签） | PianoDataset.py, (model.py: vocab_size only) | 0.5 天 + 训练时间 |
| 3 | 推理集成 + Token2Midi | model.py, inference.py, Token2Midi.py | 0.5 天 |
| 4 | NLL labeler + 重训 | nll_based.py, 重跑 Phase 2-3 | 1 天 + 训练时间 |
| 5 | 消融实验 | eval scripts | 1-2 天 |
