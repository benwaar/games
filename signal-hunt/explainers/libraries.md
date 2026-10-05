# Key Libraries

What each dependency does and why we use it. Written for people coming from general Python — you might know NumPy but not the ML/audio stack.

---

## NumPy

Multi-dimensional array library. The foundation everything else builds on.

A NumPy array is a grid of numbers — all the same type, stored contiguously in memory. That's what makes it fast: operations run in compiled C, not Python loops. When we load audio, it arrives as a 1D NumPy array of float32 values (one number per sample). When we compute a spectrogram, it becomes a 2D array (frequency bins × time frames).

**Key concept — shape:** `array.shape` tells you the dimensions. A 1.5s audio clip at 22050 Hz has shape `(33075,)` — one dimension, 33075 samples. A spectrogram might be `(128, 65)` — 128 Mel bins × 65 time frames.

---

## Tensors (the concept)

A **tensor** is just a multi-dimensional array — conceptually the same as a NumPy array. The word comes from maths, but in ML it means "the thing we feed into a model."

| Dimensions | Name | Example |
|-----------|------|---------|
| 0 | Scalar | a single loss value: `3.72` |
| 1 | Vector | an audio waveform: `(33075,)` |
| 2 | Matrix | a spectrogram: `(128, 65)` |
| 3 | 3D tensor | a batch of spectrograms: `(32, 128, 65)` |
| 4 | 4D tensor | a batch with channel dim: `(32, 1, 128, 65)` |

**Why not just use NumPy arrays?** PyTorch tensors add two things NumPy doesn't have:
1. **Automatic differentiation** — tracks every operation so it can compute gradients for training (backpropagation)
2. **GPU acceleration** — `.to("cuda")` moves computation to the graphics card

In Phase 1, we convert NumPy arrays to PyTorch tensors at the end of the pipeline. In Phase 2, everything stays as tensors because the model needs gradients.

---

## librosa

Audio analysis toolkit built on NumPy. The heavy lifter for this project.

**What it gives us:**
- `librosa.load()` — read any audio format, resample in one call
- `librosa.effects.trim()` — silence trimming based on decibel threshold
- `librosa.effects.pitch_shift()` / `time_stretch()` — augmentation primitives
- `librosa.stft()` — Short-Time Fourier Transform, the first step in feature extraction. Breaks audio into overlapping windows and FFTs each one.
- `librosa.feature.melspectrogram()` — Mel-scaled spectrogram in one call. Applies the STFT, then a Mel filterbank to compress frequency bins into perceptual bands.
- `librosa.power_to_db()` — converts power spectrogram to decibels. Compresses dynamic range so quiet sounds are visible alongside loud ones.
- `librosa.display` — spectrogram and waveform plotting helpers

**Mental model:** librosa is to audio what pandas is to tabular data — a high-level API over lower-level ops. Under the hood it uses `soundfile` for I/O and NumPy for computation.

**Docs:** https://librosa.org/doc/latest/

---

## soundfile

Reads and writes audio files (WAV, FLAC, OGG). librosa uses it internally for file I/O. We also use it directly in tests to create fixture files.

**Why not just librosa?** `sf.write()` is simpler for writing — librosa is read-focused.

---

## PyTorch (`torch`)

The deep learning framework. In Phase 1 we only use it for the final tensor conversion (`torch.Tensor`). Phase 2 is where it takes over — model architecture, training loops, GPU acceleration.

**Phase 1 usage:**
- `torch.from_numpy()` — convert a NumPy spectrogram to a PyTorch tensor
- `tensor.unsqueeze(0)` — add the channel dimension: `(128, 65)` → `(1, 128, 65)`
- `torch.save()` / `torch.load()` — persist and reload tensors as `.pt` files. Binary format, fast, preserves dtype and shape exactly.
- `TensorDataset` + `DataLoader` — wrap tensors into a dataset and iterate in shuffled batches. The batch pipeline produces tensors that slot directly into this.

**Why PyTorch over TensorFlow?** Both do the same job — tensor math, automatic differentiation, GPU acceleration. The difference is how you write code:

- **TensorFlow** (historically) builds a computation graph first, then executes it. Like writing a full recipe before cooking. You can't easily inspect intermediate values or step through with a debugger. TF2 added eager mode, but the ecosystem still leans on graph-based patterns.
- **PyTorch** runs each operation immediately as you write it — standard Python. You can `print()` a tensor mid-computation, set breakpoints, use normal `if/else` and `for` loops. No special "session" or "graph compilation" step.

For learning, this matters a lot. When something goes wrong (and it will), you want to inspect the actual numbers at each step — not wrestle with framework abstractions. PyTorch also dominates in research papers, so most tutorials and examples you'll find use it.

---

## matplotlib

Plotting. We use it for visual sanity checks — waveform plots, spectrogram heatmaps, loss curves later. Nothing fancy, just `plt.savefig()` to confirm the pipeline is producing sensible output.

---

## pytest

Test runner. Each pipeline module gets a matching test file. Run with `python -m pytest tests/ -v`.

---

## torch.utils.data — Dataset and DataLoader

The PyTorch data pipeline. Two classes do the heavy lifting:

**`Dataset`** — abstract base class. Subclass it and implement two methods:
- `__len__()` → total number of items
- `__getitem__(idx)` → load and return item at index `idx`

PyTorch calls those two methods; everything else is your logic. Our `SignalDataset` reads from the manifest and returns `(tensor, integer_label)` pairs.

**`DataLoader`** — wraps a `Dataset` and handles:
- **Batching** — collects `batch_size` items into a single tensor `(B, 1, 128, 65)`
- **Shuffling** — randomises order each epoch (train only — val/test use `shuffle=False`)
- **Parallel loading** — `num_workers=N` pre-fetches batches in background subprocesses so the GPU (or CPU) is never waiting on disk I/O

```python
from torch.utils.data import Dataset, DataLoader

loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
for tensors, labels in loader:
    # tensors: (32, 1, 128, 65), labels: (32,)
    ...
```

**Docs:** https://pytorch.org/docs/stable/data.html

---

## scikit-learn (sklearn) — train/test split

We use one function: `sklearn.model_selection.train_test_split`.

```python
from sklearn.model_selection import train_test_split

train, rest = train_test_split(records, test_size=0.30, random_state=42, stratify=labels)
val, test   = train_test_split(rest,    test_size=0.50, random_state=42, stratify=[r["label"] for r in rest])
```

The `stratify` argument is the key one — it ensures each split contains the same proportion of each class. Without it, a random split might under-represent a class in the test set, making metrics misleading.

`random_state=42` makes the split reproducible — same split every run, on every machine.

**Docs:** https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

---

## torch.nn — building neural networks

`torch.nn` is PyTorch's layer library. Every layer is an `nn.Module` — a callable that
holds learnable parameters and can be stacked into a model.

**Layers used in `model/cnn.py`:**

| Layer | What it does |
|-------|-------------|
| `nn.Conv2d(in, out, kernel_size, padding)` | Learnable kernel that scans the input for local patterns |
| `nn.BatchNorm2d(num_features)` | Normalises activations per channel across the batch |
| `nn.ReLU()` | Non-linearity: `max(0, x)`. Makes stacking layers non-trivial |
| `nn.MaxPool2d(kernel_size, stride)` | Downsamples by keeping the max in each window |
| `nn.AdaptiveAvgPool2d(output_size)` | Global Average Pooling — averages each channel to a target spatial size |
| `nn.Linear(in, out)` | Fully-connected layer: `y = xW + b` |
| `nn.Dropout(p)` | Randomly zeros activations during training (inactive at eval time) |
| `nn.Sequential(*layers)` | Chains layers so `forward` calls them in order |

**Model definition pattern:**

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(torch.relu(self.conv(x)))
```

`super().__init__()` must be called — it registers the layer as an `nn.Module` so
PyTorch can find its parameters for optimisation.

**Docs:** https://pytorch.org/docs/stable/nn.html
