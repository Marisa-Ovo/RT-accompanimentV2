# Playing Mode 分析

## 1. 推理流程

以 3 拍为例，beat 2 的 acc 与实际 mel 不匹配：

```
Beat 1:
  inject [BEAT]
  generate acc → [tok_a, tok_b, TRK_ACC]
  inject mel   → [mel_1...TRK_MEL]          ← 真实旋律到来
  generate 1 token → PLAY(268)               ← 模型看到 acc+mel，判断：匹配 ✓
  → MIDI 输出 beat 1 的 acc

Beat 2:
  inject [BEAT]
  generate acc → [tok_x, tok_y, TRK_ACC]    ← 生成了与旋律不搭的 acc
  inject mel   → [mel_2...TRK_MEL]          ← 真实旋律到来
  generate 1 token → SKIP(269)               ← 模型看到 acc+mel，判断：不匹配 ✗
  → MIDI 输出静音

Beat 3:
  inject [BEAT]
  模型 attention 看到:
    beat 1: [acc_ok][mel_1][PLAY]   → 可靠 context
    beat 2: [acc_bad][mel_2][SKIP]  → SKIP 信号：这拍的 acc 不可靠
  generate acc → 模型倾向于更多依赖 beat 1 和 mel 信息
  inject mel → generate → PLAY/SKIP
```

**核心**：PLAY/SKIP 的判断发生在 acc 和 mel 都可见的时刻——有外部参照（mel）的 compatibility 判断，不是自评。

---

## 2. 位置选择

```
每拍信息流：
  [BEAT] → [acc...TRK_ACC] → [mel...TRK_MEL] → [PLAY/SKIP]
  结构标记    盲生成（无当前mel）   条件注入           判断（acc+mel）
```

Post-mel 是**唯一能做 acc-mel 匹配性判断的位置**：
- Pre-acc：既没 acc 也没 mel → 信息最少
- Post-acc：只见 acc → 纯自评，认知闭环（如果能准确自评，也应该能避免出错）
- Post-mel：acc + mel 都有 → well-defined 的 compatibility 判断

---

## 3. 训练构造

```
PLAY: [BEAT][acc_gt......TRK_ACC][mel_gt...TRK_MEL][PLAY]  ← GT acc 与 GT mel 匹配
SKIP: [BEAT][acc_degraded.TRK_ACC][mel_gt...TRK_MEL][SKIP]  ← 降质 acc 与 GT mel 不匹配
```

模型学两件事：
1. **生成能力**：PLAY beat 正常学 acc generation
2. **判断能力**：看完 acc+mel 后识别不匹配 → 预测 SKIP

---

## 4. 关键问题

### 4.1 KV cache 中坏 acc 残留

SKIP 只决定 MIDI 输出静音，生成的坏 acc tokens 仍在 KV cache 中，后续 beat 的 attention 能看到。

SKIP token 是**语义标记**而非物理切断。模型需通过训练学会：SKIP → 降低对前面 acc 的依赖。初版先这样做，效果不足再考虑 attention mask。

### 4.2 Noise labeler 的捷径风险

Noise labeler 将 SKIP beat 的 acc 替换为 EMPTY。问题：
- SKIP 总是 EMPTY acc → 模型可能学到"非 EMPTY 就是 PLAY"的表面规则
- 推理时坏 acc 不是 EMPTY，而是非空但错误的 tokens

解决：两阶段训练（self-play）——用 baseline 模型 inference 得到真实的推理误差，用 generated acc 替代 EMPTY 作为降质版本。

### 4.3 两阶段训练

```
Stage 1: 正常训练 baseline（不含 playing mode）
Stage 2: 用 baseline 对训练集 inference → 得到真实的推理误差
Stage 3: 对比 generated acc 与 GT → 差异大的标 SKIP，acc 用 generated 版本
Stage 4: 带 playing mode 重训
```

产生最真实的"非空但不匹配"训练样本，直接消除捷径问题。

### 4.4 连续 SKIP 恢复

连续多个 SKIP 后 acc context 信息越来越少。可设 `max_consecutive_skips` 强制 PLAY，避免死循环。

### 4.5 SKIP beat 的 acc loss

SKIP beat 的 acc 是降质版本，不参与 loss（labels 填 PAD）。模型只在 PLAY beat 上学习 acc generation，在 SKIP beat 上只学习判断能力（预测 SKIP token）。
