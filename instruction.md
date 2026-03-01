# 实时钢琴伴奏生成系统：方法论

## 一、任务定义

给定实时旋律 $X$（mel），生成伴奏 $Y$（acc），满足低延迟约束。

核心建模：
$$P(Y|X) = \prod_{t=1}^{T} P(y_t | x_{1:t-1}, y_{1:t-1})$$

即生成 beat $t$ 的伴奏时，模型已见到 beat $0$ 到 $t{-}1$ 的旋律和伴奏，但尚未见当前 beat 的旋律——天然因果，无需额外延迟机制。

---

## 二、序列格式（Beat-Marker）

### 2.1 格式定义

```
[BOS][TS][BPM] [BAR][BEAT][acc₀...TRK_ACC][mel₀...TRK_MEL] [BEAT][acc₁...TRK_ACC][mel₁...TRK_MEL] ... [BAR]... [EOS]
```

每一拍的结构：`[BEAT] [acc compressed tokens] [track_marker_acc] [mel compressed tokens] [track_marker_mel]`

### 2.2 设计要点

| 设计 | 说明 |
|------|------|
| **acc 先于 mel** | 模型生成 acc 时尚未见当前拍 mel → 天然实时 |
| **beat_marker 分隔** | 显式拍边界，模型"知道"每拍起点 |
| **track_marker 终止** | acc 以 `track_marker_acc`(170) 结尾，mel 以 `track_marker_mel`(171) 结尾，声部边界清晰 |
| **bar 分隔** | 每小节开头一个 `bar_token`(255) |
| **无需 padding** | 旧方案用 delay_beats + interleave_pad 实现时间偏移，新格式通过 acc-before-mel 顺序天然解决 |

### 2.3 Token 词表

| ID 范围 | 含义 |
|---------|------|
| 0-80 | patch 三进制编码（$3^4=81$ 种） |
| 81-168 | 相对位置标记（marker_offset=81，88 个位置） |
| 169 | empty_marker（空拍） |
| 170 | track_marker_acc |
| 171 | track_marker_mel |
| 172 | beat_marker |
| 255 | bar_token |
| 256/257/258 | EOS/BOS/PAD |
| 259-263 | 拍号（5 种） |
| 264-267 | BPM 分桶（4 种） |

vocab_size = 268

### 2.4 选择性 Loss

```
Input:  [BOS][TS][BPM] [BAR][BEAT][acc₀ TRK_ACC][mel₀ TRK_MEL] ...
Labels: [PAD][PAD][PAD] [PAD][PAD] [acc₀ TRK_ACC][PAD  PAD    ] ...
```

- **acc 部分**：参与 loss（预测目标）
- **mel / bar / beat_marker / 元数据**：masked（条件输入）

$$\mathcal{L} = -\frac{1}{|\mathcal{A}|} \sum_{t \in \mathcal{A}} \log P_\theta(x_t | x_{<t})$$

其中 $\mathcal{A}$ 为 acc token 位置集合。

---

## 三、编码：三进制 Patch + 相对位置压缩

### 3.1 Patch 编码

将 (sustain, onset) 双通道 piano roll 的 $1 \times 4$ 时间窗口编码为单个 token：

$$\text{token} = \sum_{i=0}^{3} v_i \cdot 3^{3-i}, \quad v_i \in \{0, 1, 2\}$$

- 0 = 静音，1 = 延音（sustain only），2 = 起音（sustain + onset）

### 3.2 相对位置压缩

只编码非零位置，用相对距离替代绝对位置：

```
原始 (88维): [0, 0, 50, 0, 0, 60, 0, 40, 0, ...]
压缩:        [POS+2, 50, POS+3, 60, POS+2, 40, TRACK_MARKER]
```

空拍压缩为 `[EMPTY, TRACK_MARKER]`。

压缩比 $\approx 2(1{-}s)$，稀疏度 $s \approx 95\%$ 时压缩到 ~10%。

---

## 四、训练与推理

### 4.1 模型架构

LLaMA 因果语言模型（768d, 18层, 6头, SwiGLU, RoPE）。因果注意力保证训练推理一致。

### 4.2 训练

- 数据增强：70% 概率音高平移 $\Delta \in [-5, 5]$
- Bucket batch sampler：按序列长度分桶减少 padding 开销
- 有效 batch = 2 × 127 = 254，AdamW, lr=5e-5, fp16

### 4.3 推理（Schedule-Driven Generation）

Tokenizer 预计算生成计划（schedule），Model 只执行：

```
初始化: inject [BOS, TS, BPM]

每小节:
  inject [BAR]
  每拍:
    inject [BEAT]
    generate acc  → 自回归直到生成 track_marker_acc
    inject mel    → 注入 GT 旋律
```

前 $N$ 拍可注入 GT acc（warm-up），之后全部自回归生成。

采样：temperature + top-k + top-p + repetition penalty。

---

## 五、为什么这样设计

### 5.1 为什么 acc 先于 mel

- 推理时：模型先生成 acc，再"看到"当前拍 mel → 纯因果，无信息泄露
- 训练时：因果 attention mask 自动保证 acc 位置只能看到之前的 mel
- 不需要 delay_beats padding，格式即机制

### 5.2 为什么选拍级粒度

| 粒度 | 优劣 |
|------|------|
| 音符级 | 序列过长 |
| 拍级 | 符合音乐节奏结构，长度适中 |
| 小节级 | 粒度过粗 |

### 5.3 为什么必须因果 LM

| 架构 | 能否实时生成 | 原因 |
|------|:---:|------|
| Encoder-Decoder | ✗ | 需要完整输入 |
| 双向 Transformer | ✗ | 训练时有未来信息泄露 |
| Diffusion | ✗ | 全局去噪 |
| **因果 LM** | **✓** | 因果 attention + 自回归 + KV cache |

### 5.4 为什么其他编码不行

| 编码 | 声部边界 | 能否交错 |
|------|:---:|:---:|
| MIDI 事件序列 | 混合 | ✗ |
| REMI | 混合 | ✗ |
| **Beat-Marker** | **位置决定** | **✓** |

关键：声部归属由**序列结构**决定，不是模型预测。

---

## 六、局限性

1. 仅支持钢琴双轨
2. 超长序列（>3500 tokens）需截断
3. 缺乏标准化实时伴奏评估基准

---

*文档版本：3.0 — Beat-Marker 格式*
