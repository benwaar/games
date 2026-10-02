# Key Libraries

What each dependency does and why we use it. If you know NumPy, the others slot in around it.

---

## NumPy

Array math. Everything flows through NumPy arrays before it becomes a tensor. You know this one.

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
