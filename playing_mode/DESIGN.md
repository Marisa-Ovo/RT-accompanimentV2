# Playing Mode 模块设计文档

## 1. 概念

在当前的 beat-level 交错生成序列中，每个 Part1 beat 之后插入一个 **playing token**，
用于标记该 beat 的生成结果应被「**保留（play）**」还是「**跳过/静音（skip）**」。

```
原序列:    [P0_beat_i] [P1_beat_i] [P0_beat_i+1] [P1_beat_i+1] ...
新序列:    [P0_beat_i] [P1_beat_i] [playing_i] [P0_beat_i+1] [P1_beat_i+1] [playing_i+1] ...
```

**playing token 定义（待定，在 config.py 中统一分配 token id）**

| Token | 含义 |
|-------|------|
| `PLAY`  | 保留此 beat，输出到伴奏 |
| `SKIP`  | 跳过此 beat（静音/让 player 自由发挥）|

---

## 2. 整体架构（两个正交维度）

```
playing_mode/
├── labelers/      # 训练时：如何自动生成 playing 标签
└── predictors/    # 推理时：如何预测 playing token
```

两个维度**完全解耦**——任意 labeler 生成的数据可以配合任意 predictor 训练。

---

## 3. Labelers（标签生成策略）

### 接口（见 `labelers/base.py`）

```python
class BaseLabeler:
    def label_beat(self, beat_tokens: list[int], context: LabelContext) -> int:
        """返回 PLAY=1 / SKIP=0"""
```

### 各策略进度

| 策略 | 文件 | 状态 | 说明 |
|------|------|------|------|
| 乐理 reduce lead sheet | `music_theory.py` | ⬜ TODO | 根据和声功能、节拍重要性等乐理规则判断 |
| 弱模型替换 | `weak_model.py` | ⬜ TODO | 用低质量模型生成 Part1，对比 GT，差异大的 beat 标为 SKIP |
| 手动加噪 | `noise.py` | ⬜ TODO | 随机或按规律将 GT beat 置为空白，对应标 SKIP |
| 基于 NLL 划分 | `nll_based.py` | ⬜ TODO | 用已训练模型计算每 beat 的 NLL，高于阈值标为 SKIP |

---

## 4. Predictors（推理时预测架构）

### 接口（见 `predictors/base.py`）

```python
class BasePredictor:
    def predict(self, token_sequence: list[int]) -> float:
        """返回当前 beat 为 PLAY 的概率 [0, 1]"""
```

### 各架构进度

| 架构 | 文件 | 状态 | 说明 |
|------|------|------|------|
| 独立模型 | `separate_model.py` | ⬜ TODO | 单独训练一个轻量分类器 |
| PianoLLaMA 辅助预测头 | `aux_head.py` | ⬜ TODO | 在现有模型上添加 binary classification head，共享主干 |

---

## 5. 实施路线图

### Phase 0 — 框架搭建（✅ 已完成）
- [x] 目录结构 & 抽象接口定义
- [x] 本设计文档

### Phase 1 — 标签生成（部分完成）
推荐先做 **noise.py**（最简单，无需额外模型），快速验证 token 格式是否合理

- [x] 在 `config.py` 中分配 PLAY/SKIP token id（play=268, skip=269, `use_playing_token=False`）
- [x] 修改 `PianoDataset.py`：支持接入 labeler，可选地在序列中插入 playing token
- [ ] 在 `my_tokenizer.py` 中添加 playing token 的编解码支持（Token2Midi 跳过 playing token）
- [ ] 完善 `noise.py` labeler 的 density 策略（需从 beat_tokens 解析音符数）
- [ ] 编写 playing mode 的单元测试（验证序列格式 & labeler 输出）

### Phase 2 — 预测头集成
推荐先做 **aux_head.py**（利用现有主干，无需额外训练流程）

- [ ] 在 `model.py` 的 PianoLLaMA 中添加可选的 aux_head
- [ ] 修改 `trainer.py`：支持 aux loss（playing token 的 BCE loss）
- [ ] 修改 `inference.py`：推理时读取 playing 预测值，决定是否静音该 beat

### Phase 3 — 验证 & 消融
- [ ] 对比 4 种 labeler 生成数据的训练效果
- [ ] 对比 2 种 predictor 架构的推理精度 & 延迟

---

## 6. 关键设计决策（待讨论）

- [ ] PLAY/SKIP 的 token id 分配（不破坏现有 vocab_size=268 或扩充？）
- [ ] playing token 是否参与 Part1 loss 计算？（建议单独 aux loss）
- [ ] SKIP 时推理行为：全静音 / 保持上一 beat sustain / 让用户演奏？
- [ ] 是否需要 SOFT label（概率值）而非 HARD label（0/1）？
