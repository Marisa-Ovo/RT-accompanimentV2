# Drop Accompaniment 训练增强

## 动机

自回归伴奏生成存在 **exposure bias**：训练时模型始终能看到完整的历史 acc context，
推理时一旦某拍生成偏差，后续所有拍都依赖这个有误差的 context，误差持续积累。

老师提出的 Drop Accompaniment 方案：**训练时随机将某些 beat 的 acc 替换为空**，
强迫模型在 acc history 不完整的情况下依然能生成合理的伴奏，从而减少对 acc 历史的过度依赖，
更多地依靠 mel 信息。

---

## 与 Playing Mode 的区别

| | Drop Accompaniment（本文档） | Playing Mode SKIP |
|---|---|---|
| 阶段 | **训练增强**（数据构造） | 推理时自我纠错 |
| 目的 | 减少对 acc history 的依赖 | 识别坏 acc 并静音 |
| 新增 token | 无 | PLAY(268) / SKIP(269) |
| vocab_size | 不变 | 268 → 270 |
| 推理行为 | 无影响，正常生成 | 接受/拒绝每拍 acc |

两者**可以叠加**，不互斥。

---

## 实现方案

### 序列对比

```
正常 beat t:
  input:  [BEAT][acc_gt ... TRK_ACC][mel ... TRK_MEL]
  label:  [PAD] [acc_gt ... TRK_ACC][PAD  PAD       ]

Drop beat t（以概率 p 触发）:
  input:  [BEAT][EMPTY(169), TRK_ACC(170)][mel ... TRK_MEL]  ← input 替换为空
  label:  [PAD] [PAD                     ][PAD  PAD       ]  ← 不参与 loss
```

- Drop beat 的 acc **不参与 loss**（label 填 PAD），模型不学习"如何生成空 acc"
- 下一拍的 label 照常——模型必须在看不到这拍 acc 的情况下继续生成正确的下一拍
- 推理时无任何影响，模型正常自回归生成

### 代码位置

| 文件 | 改动 |
|------|------|
| `config.py` | `TrainingConfig.acc_drop_prob: float = 0.0`（开关） |
| `my_tokenizer.py` | `build_training_sequence(acc_drop_prob=0.0)` 实现替换逻辑 |
| `PianoDataset.py` | `__init__` 收 `acc_drop_prob`，`__getitem__` 转发给 tokenizer |
| `train.py` | 训练集传 `train_config.acc_drop_prob`，**测试集不传**（默认 0.0） |

---

## 使用方式

### 开启 Drop（修改 `config.py` 一行即可）

```python
# config.py — TrainingConfig
acc_drop_prob: float = 0.15   # 每 beat 有 15% 概率被 drop
```

### 关闭（默认）

```python
acc_drop_prob: float = 0.0
```

### 推荐值

| 值 | 说明 |
|---|---|
| `0.0` | 关闭，基线训练 |
| `0.1` | 轻度增强，建议先用这个 |
| `0.15` | 适中，推荐实验值 |
| `0.3+` | 过强，可能导致 acc 生成质量下降 |

---

## 测试集行为

测试集 **始终不使用 drop**（`acc_drop_prob=0.0`），原因：
- 评估需要在标准条件下进行，保证 loss/perplexity 可比较
- drop 只是训练时的数据增强，不是模型结构变化

---

## 两阶段实验建议

1. **Stage 1**：`acc_drop_prob=0.0`，训练 baseline，保存 checkpoint
2. **Stage 2**：`acc_drop_prob=0.15`，从 Stage 1 checkpoint 继续训练（fine-tune）
   - 在 `train.py` 里设置 `checkpoint_path` 加载 Stage 1 权重
   - TensorBoard 曲线会从 Stage 2 起点续上

对比两阶段在测试集上的 loss/perplexity，以及主观听感，评估 drop 的效果。
