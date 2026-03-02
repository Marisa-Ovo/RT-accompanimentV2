# Playing Mode 扩展方向

核心问题不变：训练用 GT context，推理用自回归 context → 误差累积。
PLAY/SKIP token 是最简的一步。以下是沿同一方向的更深层设计。

---

## 方向 1：Post-mel Refinement（看到旋律后修正伴奏）

### 思路

PLAY/SKIP 只做判断，**不修正**。但既然模型在 mel 注入后已经知道 acc-mel 不匹配，为什么不让它**重新生成一版 acc**？

```
当前（判断）:
  [BEAT][acc_draft...TRK_ACC][mel...TRK_MEL][PLAY/SKIP]

扩展（修正）:
  [BEAT][acc_draft...TRK_ACC][mel...TRK_MEL][REFINE][acc_refined...TRK_ACC2]
  或
  [BEAT][acc_draft...TRK_ACC][mel...TRK_MEL][KEEP]
```

- `KEEP`：acc_draft 足够好，直接用（等价于 PLAY）
- `REFINE`：acc_draft 不行，模型看着 mel 重新生成一段 acc

### 为什么有意义

实时伴奏中有一个**天然的信息不对称**：
- 生成 acc 时看不到当前 mel → 盲猜
- mel 到来后信息完整 → 有能力修正

PLAY/SKIP 浪费了这个信息——发现不匹配后选择放弃（静音），而不是利用 mel 来修正。Refinement 把这个信息用起来了。

### 训练构造

```
KEEP:   [BEAT][acc_gt...TRK_ACC][mel...TRK_MEL][KEEP]                              ← acc loss 正常
REFINE: [BEAT][acc_degraded...TRK_ACC][mel...TRK_MEL][REFINE][acc_gt...TRK_ACC2]   ← draft 不参与 loss，refined 参与
```

模型学三件事：
1. 盲生成 acc（和现在一样）
2. 判断 acc-mel 是否匹配（KEEP vs REFINE）
3. **在已知 mel 的条件下重新生成 acc**（这是一个 encoder-decoder 式的 seq2seq 任务，但嵌入在自回归框架中）

### 代价

- 序列变长（REFINE 拍长度翻倍）
- 推理延迟增加（REFINE 需要额外生成步骤）
- 但在实时场景中，mel 到来意味着这拍已经过去了——refinement 的结果用于**修正 KV cache context**而非实时输出

### 关键洞察

Refinement 不是为了改善**当前拍**的输出（来不及，已经弹过了），而是为了**修正后续拍能看到的 context**。即使当前拍的 MIDI 输出已经是 draft 版本，后续拍的 attention 看到的是 refined 版本 → 切断误差传播链。

这就像一个人弹错了一个音，虽然已经弹出去了收不回来，但他在脑子里把正确的版本"覆盖"上去，这样后续的演奏不会被之前的错误带偏。

---

## 方向 2：Beat-level Scheduled Sampling

### 思路

经典 scheduled sampling 在 token 级别混入模型自身输出。但在 beat-marker 格式中，有一个更自然的粒度——**beat 级别**。

训练时，以概率 $p$（随 epoch 递增）将某拍的 GT acc 替换为**模型自身的 inference 输出**：

```
epoch 1: p=0.0 → 全 GT（标准 teacher forcing）
epoch 3: p=0.2 → 20% 的 beat 用模型自己的生成
epoch 5: p=0.5 → 50/50 混合
```

### 与 playing token 的结合

被替换的 beat 自然成为 SKIP 候选——模型自己的输出与 GT 的差异直接决定 PLAY/SKIP 标签：

```python
# 训练时
if random.random() < scheduled_p:
    draft_acc = model.inference_one_beat(context)  # 模型自身生成
    similarity = compare(draft_acc, gt_acc)        # 与 GT 对比
    label = PLAY if similarity > threshold else SKIP
    acc_for_training = draft_acc                   # 用模型自身的输出
else:
    label = PLAY
    acc_for_training = gt_acc
```

### 优势

- 直接弥合 train-test gap：模型在训练中就见过自身的错误
- Playing token 的标签不再是随机的（noise）或预计算的（NLL），而是**在线生成的、与实际推理误差直接对应的**
- 自然的课程学习：p 从 0 递增到 0.5，逐步增加难度

### 代价

- 训练变慢（需要 inference 步骤）
- 实现复杂（需要在 training loop 中嵌入 inference）
- 可用 gradient detach 保证 scheduled sampling 部分不反传

---

## 方向 3：Confidence-weighted Context（连续信号替代离散 token）

### 思路

PLAY/SKIP 是二值的，但 acc 质量是连续的。如果模型能输出一个**连续的 confidence score**，后续 beat 的 attention 可以直接利用这个分数来加权对前面 acc 的关注程度。

### 实现

在 TRK_MEL 之后，不是预测一个离散的 PLAY/SKIP token，而是通过 aux head 输出一个 confidence scalar $c \in [0, 1]$：

```python
# model.py
hidden_at_trk_mel = transformer_output[:, trk_mel_pos, :]  # (B, H)
confidence = torch.sigmoid(self.confidence_head(hidden_at_trk_mel))  # (B, 1)
```

然后将 $c$ 编码为一个**可学习的 embedding**，注入到序列中：

```python
# confidence 转换为 embedding
conf_embed = self.confidence_embedding(confidence)  # 连续 → 向量
# 追加到序列中，替代离散的 PLAY/SKIP token
```

### 高级用法：Attention Modulation

更激进的方案：用 confidence score 直接调制后续 beat 对前面 acc 的 attention weight：

```python
# 对 beat i 的 acc tokens，如果 confidence_i = 0.3
# 后续所有 beat 对这些位置的 attention logits 乘以 0.3
attention_weights[:, :, future_positions, beat_i_acc_positions] *= confidence_i
```

这是 PLAY/SKIP 的"软版本"——不是 0/1 二值切断，而是连续调制。

### 与 PLAY/SKIP 的关系

可以共存：
- 离散 PLAY/SKIP token：粗粒度决策（输出还是静音）
- 连续 confidence：细粒度调制（后续 attention 权重）

但初版可能不值得做这么复杂。留作消融实验。

---

## 方向 4：Self-Play 迭代训练

### 思路

方向 2 的离线版本，但更彻底：

```
Round 0: 训练 baseline 模型（无 playing mode）
Round 1: 用 baseline 对全训练集 inference → 每首曲子得到 generated acc
         对比 generated acc 与 GT → 标注 PLAY/SKIP
         generated acc 作为 SKIP beat 的降质版本
         训练 model_v1（含 playing mode）
Round 2: 用 model_v1 inference → 新的 generated acc
         重新标注 → 训练 model_v2
...
```

### 为什么比单次训练好

- Round 0 的模型犯的错 → 训练出的 model_v1 学会避免这些错误
- 但 model_v1 会犯新的、不同的错误 → Round 2 用这些新错误继续训练
- 每轮模型都在上一轮的基础上修正，类似 GAN 的对抗训练但更稳定

### 实际考虑

- 每轮需要对全训练集 inference → 计算成本高
- 2-3 轮通常足够（收益递减）
- 可以只对一个子集做 inference（比如 10%），减少成本

---

## 方向对比

| 方向 | 创新度 | 实现复杂度 | 与 playing token 的关系 |
|------|:------:|:---------:|----------------------|
| PLAY/SKIP token（当前） | 低 | 低 | baseline |
| Post-mel Refinement | **高** | 中 | 替代 SKIP → 改为 REFINE |
| Beat-level Scheduled Sampling | 中 | 中 | 提供更真实的 SKIP 标签 |
| Confidence-weighted Context | 中 | 高 | PLAY/SKIP 的连续化 |
| Self-Play 迭代 | 中 | 中（代码简单，算力重） | 提供更真实的训练数据 |

## 推荐组合

**最小可行方案**：PLAY/SKIP token + noise labeler（Phase 1-3，验证格式可行）

**完整方案**（论文级）：
1. PLAY/SKIP token 作为基础机制
2. **Post-mel Refinement** 作为核心创新——"看到旋律后修正伴奏"是一个非常直觉、非常音乐化的 idea
3. **Self-play 训练** 提供真实降质数据
4. 消融：SKIP-only vs REFINE, noise vs self-play, 有无 refinement 对后续 beat 质量的影响

Post-mel Refinement 是最有论文价值的方向——它不只是一个 trick，而是提出了"实时生成中利用延迟到达的信息修正 context"这个 general 的框架。
