# CNN Architecture — Sound Type Classifier

How a convolutional neural network learns to distinguish hums, whistles, and claps
from their Mel-spectrogram shapes.

---

## The core idea

A Mel-spectrogram is an image. A hum looks like a horizontal band in the low-frequency
region. A whistle is a thin band up high. A clap is a vertical burst across all frequencies.

A CNN scans for those shapes the same way an image classifier scans for edges, textures,
and objects — starting with simple local patterns (edges, blobs) in early layers and
combining them into more abstract features in later layers.

No manual feature engineering needed. The model learns which shapes matter from the labels.

---

## Building blocks

### Conv2d — the scanning layer

A convolutional layer slides a small **kernel** (e.g. 3×3) across the input and computes
a dot product at every position. Each kernel learns to detect one pattern — a horizontal
edge, a diagonal frequency smear, a broadband burst.

```
Input:  (batch, channels_in, height, width)
Output: (batch, channels_out, height', width')
```

Key parameters:
- **`kernel_size`** — how large the pattern detector is. `3×3` is standard; catches
  local structure without seeing too much context at once.
- **`padding=1`** — adds a border of zeros around the input so the output stays the
  same spatial size as the input. Without it, each layer shrinks the feature map.
- **`channels_out`** — how many distinct patterns to learn. 16 in the first layer,
  32 in the second, 64 in the third. More channels = more patterns, more parameters.

> **Coming from C:** A Conv2d is a nested loop: for each output channel, for each position
> in the output, compute a dot product between the kernel and the input patch at that
> position. The kernel weights are learned, not hand-coded. The loop is fused into
> matrix multiplication and runs on the GPU.

> **Coming from JS/TS:** Like `Array.map` applied to a sliding window over a 2D grid.
> Each kernel is a transform function; `channels_out` is how many transforms run in
> parallel. The transforms are learned, not written by hand.

### BatchNorm2d — stabilising activations between layers

After each conv layer, the distribution of activations can shift dramatically —
large weights produce huge values, small weights produce near-zeros. This makes
training slow and unstable.

BatchNorm normalises each channel's activations to mean≈0, std≈1 across the
batch, then applies a learned scale and shift. The model gets to decide how spread
out the activations should be, but training starts from a stable baseline.

```python
# Applied after Conv2d, before ReLU
nn.BatchNorm2d(num_channels)
```

> **In practice:** BatchNorm is one of the reasons deep networks became trainable.
> Before it (2015), you had to be very careful with weight initialisation and
> learning rates. With it, training is more forgiving and converges faster.
> The same principle (normalise inputs between stages) applies in data pipelines:
> z-score normalisation in our feature extraction step does the same job for
> the raw spectrogram.

### ReLU — the non-linearity

Without non-linearity, stacking linear layers is mathematically equivalent to
one big linear layer. ReLU (`max(0, x)`) introduces the non-linearity that lets
the network learn complex decision boundaries.

```python
nn.ReLU()   # replaces all negatives with 0
```

Simple but effective. The network uses combinations of ReLUs to approximate
any function.

### MaxPool2d — downsampling

After each conv+BN+ReLU block, MaxPool halves the spatial dimensions by keeping
only the maximum activation in each 2×2 window.

```python
nn.MaxPool2d(kernel_size=2, stride=2)
# (B, C, H, W) → (B, C, H//2, W//2)
```

Why downsample?
- **Reduces computation** — smaller feature maps in later layers
- **Builds spatial invariance** — a pattern detected at position (10, 20) and
  position (11, 20) both survive as "that pattern exists in this region"
- **Increases receptive field** — later layers effectively see a larger area
  of the original input

### Global Average Pooling — replacing Flatten

After the three conv blocks, we have a feature map `(B, 64, H', W')`. To feed
it into a Linear layer, we need a flat vector.

**Option A — Flatten:** `(B, 64, H', W')` → `(B, 64 × H' × W')`. The size depends
on input dimensions. Large, fixed, overfits easily.

**Option B — Global Average Pooling (GAP):** Average across all spatial positions
per channel. `(B, 64, H', W')` → `(B, 64)`. Fixed size regardless of input. Far
fewer parameters into the linear head.

```python
nn.AdaptiveAvgPool2d(1)   # output spatial size = (1, 1)
# then: x = x.squeeze(-1).squeeze(-1)  → (B, 64)
```

For our task, the spectrogram's spatial layout matters less than *which frequency
patterns are present*. GAP answers "how much of this pattern exists overall" —
exactly what we want for sound type classification.

> **Coming from C/JS/TS:** GAP is a reduce operation — collapse `H×W` values per
> channel to a single average. Like `array.reduce((sum, x) => sum + x, 0) / len`
> applied independently to each channel's 2D grid.

### Dropout — regularisation for small datasets

Dropout randomly zeros a fraction of neuron activations during training.
With `p=0.3`, each activation has a 30% chance of being set to zero on each
forward pass.

```python
nn.Dropout(p=0.3)   # applied between the two linear layers
```

Why? With only ~150 training samples, the model can memorise them. Dropout
forces it to learn redundant representations — if neuron A is randomly off,
neuron B should also carry that information. This generalises better.

Dropout is **only active during training** (`model.train()`). During evaluation
(`model.eval()`), all neurons are active and outputs are scaled to compensate.

> **In practice:** Dropout is the simplest form of ensemble learning. Each
> forward pass uses a different random subset of the network — you're effectively
> training many overlapping sub-networks and averaging their predictions at
> inference time.

---

## Our architecture

```
Input: (B, 1, 128, 65)
         ↓
Conv2d(1→16, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)
         ↓  (B, 16, 64, 32)
Conv2d(16→32, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)
         ↓  (B, 32, 32, 16)
Conv2d(32→64, 3×3, pad=1) → BN → ReLU → MaxPool(2×2)
         ↓  (B, 64, 16, 8)
AdaptiveAvgPool2d(1)  →  squeeze
         ↓  (B, 64)
Linear(64→32) → ReLU → Dropout(0.3)
         ↓  (B, 32)
Linear(32→3)
         ↓
Output: (B, 3) logits
```

**Parameter count:** ~25,700 — well under the 500K target. Appropriate for a
3-class problem with ~150 training samples.

| Layer | Parameters |
|-------|-----------|
| Conv1 (1→16) | 160 |
| BN1 | 32 |
| Conv2 (16→32) | 4,640 |
| BN2 | 64 |
| Conv3 (32→64) | 18,496 |
| BN3 | 128 |
| Linear (64→32) | 2,080 |
| Linear (32→3) | 99 |
| **Total** | **~25,700** |

**Why so few?** 3-class classification from spectrograms is not a hard problem —
hums, whistles, and claps look very different. A small model forces the network
to learn *meaningful* features rather than memorising training examples.
If we used a large model (millions of params), it would overfit immediately on
150 samples.

> **In practice:** Model size should match dataset size. A rule of thumb: you
> want at least 10× more training samples than parameters. We have ~150 samples
> and ~25K params — that's 6×, on the tight side, which is why we use dropout
> and early stopping in training (M9).

---

## Why not just a linear classifier?

A linear classifier on the raw spectrogram `(1×128×65 = 8320 features)` would
need to learn a separate weight for every pixel-position per class. It can't
share knowledge: "a horizontal band at frequency 440 Hz" and "a horizontal band
at frequency 500 Hz" would be completely unrelated to it.

Conv layers solve this with **weight sharing** — the same kernel scans every
position in the spectrogram. If the pattern detector for "horizontal band" fires
at position (20, 10), it also fires at (20, 30). One kernel, many positions.
This is what makes CNNs parameter-efficient and translation-invariant.
