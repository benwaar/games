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
- `librosa.feature.melspectrogram()` — the core feature extraction (M4)
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

**Why PyTorch over TensorFlow?** More Pythonic, easier to debug (eager execution by default), dominant in research. For a learning project, the imperative style makes it clearer what's happening.

---

## matplotlib

Plotting. We use it for visual sanity checks — waveform plots, spectrogram heatmaps, loss curves later. Nothing fancy, just `plt.savefig()` to confirm the pipeline is producing sensible output.

---

## pytest

Test runner. Each pipeline module gets a matching test file. Run with `python -m pytest tests/ -v`.
