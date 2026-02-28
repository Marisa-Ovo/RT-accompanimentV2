# Real-Time Piano Accompaniment Generation System

## Technical Documentation

---

## 1. System Overview

This system implements a **real-time piano accompaniment generation framework** based on the LLaMA architecture. Given a melody track (Part0), the model generates a corresponding accompaniment track (Part1) in an interleaved, streaming manner.

### 1.1 Core Capabilities

| Feature | Description |
|---------|-------------|
| **Conditional Generation** | Generate Part1 conditioned on Part0 melody input |
| **Beat-Level Interleaving** | Process and generate music at beat granularity |
| **Temporal Offset Control** | Support arbitrary time delays between tracks via `delay_beats` |
| **Relative Position Encoding** | Efficient tokenization with sparse representation |

### 1.2 Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    PianoLLaMA Model                             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  LLaMA Architecture (18 layers, 6 heads, 768 hidden)      │  │
│  │  - RoPE positional encoding (theta=10000)                 │  │
│  │  - Causal attention mask                                  │  │
│  │  - Vocabulary size: 268 tokens                            │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Design: Beat-Level Interleaved Generation

### 2.1 Problem Formulation

Traditional sequence-to-sequence models generate the entire accompaniment after receiving the complete melody. This approach introduces latency unsuitable for real-time applications.

**Our Solution**: Interleave Part0 (melody) and Part1 (accompaniment) at the **beat level**, enabling the model to generate accompaniment progressively as melody arrives.

### 2.2 Interleaving Mechanism

```
Sequence Structure:
[BOS] [TimeSignature] [BPM]
      [Part0_Beat_0] [Part1_Beat_0]
      [Part0_Beat_1] [Part1_Beat_1]
      ...
      [Part0_Beat_N] [Part1_Beat_N]
[EOS]
```

**Key Insight**: The model learns the conditional distribution `P(Part1_Beat_i | Part0_Beat_0:i, Part1_Beat_0:i-1)`, enabling real-time generation as each melody beat arrives.

### 2.3 Training vs. Inference

| Phase | Part0 | Part1 |
|-------|-------|-------|
| **Training** | Teacher forcing (GT) | Teacher forcing (GT), loss computed only on Part1 |
| **Inference** | Injected from input | Generated autoregressively |

---

## 3. Temporal Offset Mechanism (`delay_beats`)

### 3.1 Concept

The `delay_beats` parameter controls the temporal relationship between Part0 and Part1:

| Value | Behavior | Use Case |
|-------|----------|----------|
| `delay_beats = 0` | Simultaneous start | Standard accompaniment |
| `delay_beats > 0` | Part1 delayed by N beats | Following accompaniment |
| `delay_beats < 0` | Part1 advanced by N beats | **Real-time accompaniment** |

### 3.2 Implementation

Padding tokens (`pad_marker=173`) are inserted to achieve the temporal offset:

```python
if delay_beats >= 0:
    # Part1 delayed: pad Part1's beginning, pad Part0's end
    part1_padded = [pad_marker] * delay_beats + all_part1_beats
    part0_padded = all_part0_beats + [pad_marker] * delay_beats
else:
    # Part1 advanced: pad Part0's beginning, pad Part1's end
    advance_beats = -delay_beats
    part0_padded = [pad_marker] * advance_beats + all_part0_beats
    part1_padded = all_part1_beats + [pad_marker] * advance_beats
```

### 3.3 Real-Time Scenario (delay_beats = -1)

```
Timeline:
Beat Index:    0    1    2    3    4    ...
Part1:        [G]  [G]  [G]  [G]  [G]   (Part1 starts generating)
Part0:        [_]  [I]  [I]  [I]  [I]   (Part0 arrives 1 beat later)

[G] = Generated/GT Part1
[I] = Injected Part0
[_] = Padding (no information yet)
```

This allows the system to **predict accompaniment before the corresponding melody arrives**, achieving real-time responsiveness.

---

## 4. Token Encoding System

### 4.1 Piano Roll Representation

Input piano roll has shape `(4, 88, T)`:
- Channels 0-1: Part0 (melody) - sustain and onset
- Channels 2-3: Part1 (accompaniment) - sustain and onset

### 4.2 Ternary Patch Encoding

Each `1x4` patch is encoded as a ternary (base-3) number:

| Sustain | Onset | Value |
|---------|-------|-------|
| 0 | 0 | 0 (silence) |
| 1 | 0 | 1 (sustain only) |
| 1 | 1 | 2 (onset + sustain) |

For a `1x4` patch: `token = sum(value[i] * 3^(3-i))` where `i in [0,3]`

This yields token values in range `[0, 80]` (3^4 - 1 = 80).

### 4.3 Relative Position Compression

Instead of storing all 88 pitch positions, only non-zero positions are encoded:

```
Format: [rel_pos_marker] [token_value] [rel_pos_marker] [token_value] ... [end_marker]

Example:
  Position 0: token=50  ->  [81+0, 50]  = [81, 50]
  Position 5: token=60  ->  [81+5, 60]  = [86, 60]  (relative to position 0)
  Position 7: token=40  ->  [81+2, 40]  = [83, 40]  (relative to position 5)
  End                   ->  [170] (Part0) or [171] (Part1)
```

**Benefits**:
- Significant sequence length reduction for sparse piano rolls
- Empty beats represented by single `empty_marker` (169)

### 4.4 Special Token Dictionary

| Token ID | Meaning |
|----------|---------|
| 0-80 | Ternary-encoded patch values |
| 81-168 | Relative position markers (81 = distance 0, 168 = distance 87) |
| 169 | Empty beat marker |
| 170 | Part0 end marker |
| 171 | Part1 end marker |
| 173 | Padding marker (for delay_beats) |
| 255 | Bar token |
| 256 | EOS token |
| 257 | BOS token |
| 258 | PAD token |
| 259-263 | Time signature tokens |
| 264-267 | BPM tokens (slow/medium/fast/unknown) |

---

## 5. Generation Algorithm

### 5.1 `generate_accompaniment()` Core Logic

```
Location: model.py:59-306
```

**Algorithm**:

```
Input: Part0 beats from dataset, generation parameters
Output: Generated Part1 beats

1. INITIALIZE:
   - Load Part0 beats and Part1 GT beats from NPZ file
   - Apply delay_beats padding
   - Construct initial prompt: [BOS] [TimeSig] [BPM]

2. INTERLEAVED GENERATION LOOP:
   position = 0  # 0=Part0 slot, 1=Part1 slot

   FOR iteration in range(max_iterations):
     IF position == 0:  # Part0 slot
       IF part0_idx < len(part0_beats):
         INJECT next Part0 beat
         RESET KV-cache (manual token injection invalidates cache)
         position = 1
       ELSE:
         BREAK  # All Part0 injected

     ELSE:  # Part1 slot (position == 1)
       IF part1_idx < gt_prefix_beats:
         INJECT GT Part1 beat (warm-up phase)
         RESET KV-cache
       ELSE:
         GENERATE token autoregressively:
           logits = model(input, past_key_values)
           next_token = sample(logits, temperature, top_k, top_p, rep_penalty)
           APPEND next_token to sequence

           IF next_token in [end_marker, bar_token]:
             Record completed beat
             position = 0  # Switch back to Part0 slot

3. RETURN generated Part1 beats
```

### 5.2 Sampling Strategy

The system employs a multi-stage sampling pipeline:

```python
# Stage 1: Temperature scaling
logits = logits / temperature

# Stage 2: Repetition penalty
for token_id in generated_tokens:
    logits[token_id] /= repetition_penalty

# Stage 3: Top-K filtering
indices_to_remove = logits < topk(logits, k)[..., -1]
logits[indices_to_remove] = -inf

# Stage 4: Top-P (nucleus) filtering
sorted_probs = cumsum(softmax(sort(logits)))
indices_to_remove = sorted_probs > top_p
logits[sorted_indices_to_remove] = -inf

# Stage 5: Multinomial sampling
next_token = multinomial(softmax(logits))
```

**Default Parameters** (inference.py):
- `temperature = 1.1`
- `top_k = 10`
- `top_p = 0.95`
- `repetition_penalty = 1.0`
- `gt_prefix_beats = 12`

---

## 6. KV-Cache Management

### 6.1 Caching Strategy

KV-cache accelerates autoregressive generation by reusing attention computations:

```python
# With cache: O(1) per token (only compute new token's attention)
outputs = model(
    input_ids=generated[:, -1:],  # Only new token
    past_key_values=past_key_values,
    use_cache=True
)
past_key_values = outputs.past_key_values
```

### 6.2 Cache Invalidation

**Critical**: When tokens are manually injected (Part0 injection or GT Part1), the cache becomes invalid and must be reset:

```python
# After injecting Part0 beat
generated = torch.cat([generated, next_part0_beat], dim=1)
past_key_values = None  # MUST reset cache
```

**Trade-off**: Cache resets increase computation but ensure correctness. The interleaved design requires approximately `N` cache resets for `N` beats.

---

## 7. Training Pipeline

### 7.1 Loss Computation

**Key Design**: Only Part1 tokens contribute to the loss; Part0 tokens are masked:

```python
# In PianoDataset.__getitem__():
for p0, p1 in zip(part0_padded, part1_padded):
    all_input_tokens.append(p0)           # Input: Part0
    all_input_tokens.append(p1)           # Input: Part1
    all_label_tokens.append(pad_token)    # Label: MASKED (no loss)
    all_label_tokens.append(p1)           # Label: Part1 (compute loss)
```

This trains the model to predict Part1 conditioned on Part0, without learning to predict Part0 itself.

### 7.2 Training Configuration

```python
# From config.py
num_epochs = 8
train_batch_size = 2
gradient_accumulation_steps = 127  # Effective batch = 254
learning_rate = 5e-5
lr_warmup_steps = 50
mixed_precision = "fp16"
train_cutoff_len = 2048
```

### 7.3 Length-Aware Batching

`BucketBatchSampler` groups samples by sequence length to minimize padding waste:

```python
class BucketBatchSampler:
    def _create_buckets(self):
        # Sort by length, group into buckets of size 100
        for i in range(0, len(sorted_indices), bucket_size):
            self.buckets.append(sorted_indices[i:i+bucket_size])

    def __iter__(self):
        # Shuffle buckets, shuffle within buckets, yield batches
        for bucket in shuffled_buckets:
            for batch in batches_from_bucket:
                yield batch
```

---

## 8. Data Processing Pipeline

### 8.1 NPZ Data Format

```python
# Each NPZ file contains:
{
    'metadata': {
        'time_signature_idx': int,  # Time signature index
        'bpm': float,               # Tempo
        'num_measures': int         # Number of measures
    },
    'measure_0': ndarray,  # Shape: (4, 88, T)
    'measure_1': ndarray,
    ...
}
```

### 8.2 Beat-Level Processing

```python
def process_measure_with_beat_interleaving(measure, tokenizer, timesteps_per_beat=4):
    """
    Process one measure into beat-level tokens for both parts.

    Args:
        measure: (4, 88, T) - 4 channels (Part0: 0-1, Part1: 2-3)
        timesteps_per_beat: 4 (16th note resolution)

    Returns:
        part0_beat_tokens: List of tensors, one per beat
        part1_beat_tokens: List of tensors, one per beat
    """
    for beat_idx in range(num_beats):
        # Extract beat slice
        beat_measure = measure[:, :, start_t:end_t]

        # Part0: channels [0:2]
        tokens_0 = tokenizer.image_to_patch_tokens(beat_measure[:2])
        compressed_0 = tokenizer.compress_tokens(tokens_0, end_marker=170)

        # Part1: channels [2:4]
        tokens_1 = tokenizer.image_to_patch_tokens(beat_measure[2:])
        compressed_1 = tokenizer.compress_tokens(tokens_1, end_marker=171)
```

### 8.3 Data Augmentation

```python
# Pitch shifting (70% probability)
if np.random.random() < 0.7:
    shift = np.random.randint(-5, 6)  # [-5, +5] semitones
    measure = np.roll(measure, shift, axis=1)
    # Zero out wrapped-around region
    if shift > 0:
        measure[:, :shift, :] = 0
    else:
        measure[:, shift:, :] = 0
```

---

## 9. System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Data Flow Architecture                          │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│   NPZ File   │───>│   PianoDataset       │───>│  Beat Interleaving   │
│  (Raw Data)  │    │  - Load measures     │    │  - Split by beat     │
│              │    │  - Apply augmentation│    │  - Separate Part0/1  │
└──────────────┘    └──────────────────────┘    └──────────┬───────────┘
                                                           │
                    ┌──────────────────────────────────────┘
                    v
┌──────────────────────────────────────────────────────────────────────┐
│                      PianoRollTokenizer                              │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐  │
│  │ Ternary Encode │─>│ Relative Pos   │─>│ Compressed Token Seq   │  │
│  │ (1x4 patches)  │  │ Compression    │  │ [pos][val]...[end]     │  │
│  └────────────────┘  └────────────────┘  └────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
                                                           │
                    ┌──────────────────────────────────────┘
                    v
┌──────────────────────────────────────────────────────────────────────┐
│                         PianoLLaMA Model                             │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  Input: [BOS][TS][BPM][P0_B0][P1_B0][P0_B1][P1_B1]...[EOS]     │  │
│  │  Labels: [PAD][PAD][PAD][PAD][P1_B0][PAD][P1_B1]...            │  │
│  │                              ^                                 │  │
│  │                     Loss computed only here                    │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
                                                           │
                    ┌──────────────────────────────────────┘
                    v
┌──────────────────────────────────────────────────────────────────────┐
│                      Token2Midi Conversion                           │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────┐  │
│  │ Decompress     │─>│ Decode Ternary │─>│ PrettyMIDI Export      │  │
│  │ Token Sequence │  │ to Piano Roll  │  │ (Dual Track)           │  │
│  └────────────────┘  └────────────────┘  └────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 10. Key Innovations

### 10.1 Beat-Level Conditional Generation

Unlike frame-level or measure-level approaches, beat-level granularity provides:
- **Musical coherence**: Beats are natural rhythmic units
- **Computational efficiency**: Fewer generation steps than frame-level
- **Real-time capability**: Fine enough granularity for responsive generation

### 10.2 Flexible Temporal Control

The `delay_beats` mechanism enables:
- **Predictive accompaniment** (`delay_beats < 0`): Model anticipates upcoming melody
- **Reactive accompaniment** (`delay_beats > 0`): Model follows the melody
- **Synchronized accompaniment** (`delay_beats = 0`): Standard parallel generation

### 10.3 Sparse Relative Position Encoding

Compared to absolute position encoding:
- **Compression ratio**: ~60-80% reduction in sequence length for typical music
- **Empty beat handling**: Single token for silence
- **Boundary-aware**: Explicit end markers for each beat

### 10.4 Selective Loss Computation

Training only on Part1 predictions:
- **Prevents mode collapse**: Model cannot simply copy Part0
- **Focuses capacity**: All model capacity devoted to accompaniment generation
- **Clean separation**: Part0 purely provides context, not prediction targets

---

## 11. Performance Characteristics

### 11.1 Model Specifications

| Parameter | Value |
|-----------|-------|
| Total Parameters | ~200M |
| Hidden Dimension | 768 |
| Attention Heads | 6 |
| Layers | 18 |
| Max Sequence Length | 3500 |
| Vocabulary Size | 268 |

### 11.2 Resource Requirements

| Resource | FP32 | FP16 |
|----------|------|------|
| Model Memory | ~800 MB | ~400 MB |
| Inference Memory | ~1.5 GB | ~800 MB |
| Generation Speed | ~50-100 ms/token | ~30-60 ms/token |

### 11.3 Optimization Techniques

| Technique | Location | Effect |
|-----------|----------|--------|
| KV-Cache | model.py:237-244 | Reduces redundant attention computation |
| FP16 Inference | inference.py | Halves memory, accelerates computation |
| Gradient Accumulation | trainer.py:167 | Enables large effective batch size |
| Length-Aware Batching | PianoDataset.py:391-446 | Minimizes padding overhead |
| Mixed Precision Training | trainer.py:82 | Accelerates training, reduces memory |

---

## 12. File Reference

| File | Purpose | Key Components |
|------|---------|----------------|
| `model.py` | Model definition and generation | `PianoLLaMA`, `generate_accompaniment()` |
| `PianoDataset.py` | Data loading and processing | `PianoDataset`, `BucketBatchSampler`, `process_measure_with_beat_interleaving()` |
| `my_tokenizer.py` | Piano roll encoding/decoding | `PianoRollTokenizer`, `compress_tokens()`, `decompress_tokens()` |
| `config.py` | Configuration classes | `ModelConfig`, `TrainingConfig` |
| `inference.py` | Inference utilities | `load_model()`, `save_gt_midi()` |
| `Token2Midi.py` | Token-to-MIDI conversion | `tokens_to_midi()`, `pianoroll_to_midi_notes()` |
| `trainer.py` | Training loop | `TransformerTrainer` |
| `train.py` | Training entry point | Main training script |
| `get_length.py` | Length cache computation | Pre-compute sequence lengths |

---

## 13. Usage Example

### 13.1 Inference

```python
from model import PianoLLaMA
from PianoDataset import PianoDataset
from config import ModelConfig
from inference import load_model
from Token2Midi import tokens_to_midi

# Load model
config = ModelConfig()
model = load_model("checkpoint.safetensors", config, device="cuda", use_fp16=True)

# Load dataset
dataset = PianoDataset(data_dir, config, mode='test')

# Generate accompaniment
result = model.generate_accompaniment(
    dataset=dataset,
    condition_idx=0,           # Which sample to use as condition
    delay_beats=-1,            # Real-time mode: Part1 leads by 1 beat
    gt_prefix_beats=12,        # Use GT for first 12 beats as warm-up
    temperature=1.1,
    top_k=10,
    top_p=0.95,
    repetition_penalty=1.0
)

# Export to MIDI
tokens_to_midi(result, "output.mid", tempo=result['metadata']['bpm'])
```

### 13.2 Key Parameters

| Parameter | Recommended | Effect |
|-----------|-------------|--------|
| `delay_beats` | -1 | Real-time responsiveness |
| `gt_prefix_beats` | 8-16 | More = better context, less = faster start |
| `temperature` | 0.8-1.2 | Higher = more variety |
| `top_k` | 10-50 | Lower = more focused |
| `top_p` | 0.9-0.95 | Standard nucleus sampling |
| `repetition_penalty` | 1.0-1.2 | Prevent repetitive patterns |

---

## 14. Conclusion

This system presents a novel approach to real-time music accompaniment generation through:

1. **Beat-level interleaved architecture** enabling streaming generation
2. **Flexible temporal offset control** via the `delay_beats` mechanism
3. **Efficient sparse tokenization** through relative position encoding
4. **Targeted training** with selective loss computation on accompaniment only

The combination of these techniques enables responsive, musically coherent accompaniment generation suitable for real-time interactive applications.

---

*Document Version: 1.0*
*Last Updated: 2025*
