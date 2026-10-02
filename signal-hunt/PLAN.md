# Signal Hunt — Plan

## Learning Objectives

- **Data Pipeline Engineering for Production:** Ingest, clean, and transform unstructured real-world data streams into structured numerical features at scale.
- **Predictive Modeling & Sequence Learning:** Build custom neural networks in PyTorch to solve multi-label classification and temporal pattern recognition.

---

## Phase 1: Data Pipelines & Feature Extraction

Turn raw audio (hums, whistles, claps) into clean, normalised Mel-spectrogram tensors.

**Business goal:** Learn unstructured data ingestion and preprocessing — required for any ML feature pipeline.
**Business value:** Teaches you how to turn raw, messy customer or sensor inputs into clean machine-readable data while preventing model failure due to real-world edge cases.

### The problem

Real-world audio is messy. Background noise, mic quality, room acoustics — all vary wildly. A model trained on clean studio recordings fails the moment it meets a laptop mic in a kitchen. Phase 1 builds the pipeline that handles this before a model ever sees the data.

### What to build

1. **Ingestion wrapper** — load or record 1–2 second audio clips via `librosa` / `soundfile`
2. **Augmentation engine** — inject white noise, ambient room sound, pitch shift, time stretch
3. **Feature transform** — STFT → Mel-spectrogram (log-dB) → normalise (mean=0, std=1) → PyTorch tensor

### Milestones

#### M1: Environment & hello-audio (~1 hr) ✅
Load a sample audio file, print its shape and sample rate. Plot the raw waveform. Confirm librosa, soundfile, torch all import cleanly in the venv.

**Gate:** `python hello_audio.py` runs, prints shape, saves a waveform plot. **PASSED**

#### M2: Ingestion wrapper (~2 hrs) ✅
`pipeline/ingest.py` — functions to load audio from file or record from mic. Resample to a standard rate (22050 Hz). Trim silence. Output a consistent numpy array.

**Gate:** Unit tests pass for shape, dtype, sample rate. Handles both mono and stereo input. **PASSED — 9/9 tests**

#### M3: Augmentation engine (~2 hrs) ✅
`pipeline/augment.py` — composable augmentation functions:
- `add_noise(signal, snr_db)` — white noise at specified SNR
- `add_ambient(signal, ambient_path, snr_db)` — mix with ambient recording
- `pitch_shift(signal, sr, n_steps)` — shift pitch up/down
- `time_stretch(signal, rate)` — speed up/slow down without pitch change

**Gate:** Unit tests verify output shape matches input. Augmented audio sounds different but recognisable (manual listen check). **PASSED — 12/12 tests**

#### M4: Feature extraction (~2 hrs)
`pipeline/features.py` — STFT, Mel-spectrogram, log-dB conversion, normalisation. Output: PyTorch tensor ready for a CNN.

**Gate:** Unit tests verify tensor shape `(1, n_mels, time_frames)`, mean ≈ 0, std ≈ 1. Spectrogram plot saved for visual sanity check.

#### M5: Batch processing & dataset (~2 hrs)
`pipeline/batch.py` — process a folder of audio files through the full pipeline (ingest → augment → features → save). Output `.pt` tensor files. Generate augmented variants per source file.

`data/raw/` — a handful of baseline audio clips (hums, whistles, claps).
`data/processed/` — tensor output from batch processing.

**Gate:** `python -m pipeline.batch data/raw data/processed` produces tensors. A `DataLoader` can iterate them.

#### M6: Integration & documentation (~1 hr)
End-to-end: record or load → augment → extract → tensor. README updated with usage. All tests green.

**Gate:** `pytest` passes. `python -m pipeline.demo` runs the full pipeline on a sample file and prints summary stats.

### What's tricky

1. **Mic access.** `sounddevice` needs PortAudio. May need `brew install portaudio`. Don't block on this — file-based ingestion is the core path.
2. **Spectrogram dimensions.** Different audio lengths produce different time frames. Need to decide: pad/truncate to fixed length, or handle variable?
3. **Augmentation realism.** Too much noise = garbage. Too little = no robustness. SNR range matters.

### When to stop

| After | Stop if | Meaning |
|---|---|---|
| M1 | Can't get libs installed | Environment issue, not a learning blocker — fix and retry |
| M2 | Ingestion works for files | Mic recording is nice-to-have, file loading is the core |
| M4 | Features look wrong in plots | Spectrogram not showing expected patterns — debug before proceeding |
| M5 | Batch pipeline works | Dataset is ready for Phase 2 |

### Effort

~10 hours across 2–3 sessions.

---

## Phase 2: Core Model Architecture & Training

Take the Mel-spectrogram tensors from Phase 1 and train a hybrid CNN-RNN to classify acoustic events and sequences.

**Business goal:** Master core predictive modeling, training loops, loss functions, and multi-label classification using industry-standard PyTorch.
**Business value:** Build custom, proprietary AI models tailored to specific business logic rather than relying purely on generic, expensive third-party APIs.

### The problem

Spectrograms have two dimensions of information: spatial frequency shapes (what notes/sounds are present) and temporal transitions (how they change over time). A pure CNN sees shapes but not sequences. A pure RNN sees sequences but not shapes. The hybrid architecture combines both — CNN extracts frequency features, RNN learns how those features evolve.

### What to build

1. **Hybrid CNN-RNN model** — 2D CNN blocks → bidirectional GRU/LSTM → classification head
2. **Training loop** — loss, backprop, validation splits, metric logging
3. **Evaluation** — accuracy curves, confusion matrices, overfitting detection

### Milestones

#### M7: Dataset & DataLoader (~2 hrs)
`model/dataset.py` — a PyTorch `Dataset` class that loads the `.pt` tensors from Phase 1, maps them to labels, and splits into train/val/test sets. Labels come from folder structure or a manifest file.

**Gate:** `DataLoader` iterates batches of `(tensor, label)` pairs. Shapes are consistent. Train/val/test split is reproducible.

#### M8: CNN feature extractor (~2 hrs)
`model/architecture.py` — the convolutional front-end:
- 2–3 Conv2d blocks with BatchNorm, ReLU, MaxPool
- Input: `(batch, 1, n_mels, time_frames)`
- Output: `(batch, features, compressed_time)` — spatial features per time step

**Gate:** Forward pass on a dummy batch produces expected output shape. No NaNs or exploding values.

#### M9: RNN temporal layer + classification head (~2 hrs)
Extend `model/architecture.py`:
- Bidirectional GRU or LSTM on the CNN output sequence
- Linear classification head with Dropout
- Output: multi-class probabilities

**Gate:** Full forward pass (CNN → RNN → head) produces `(batch, num_classes)` logits. Softmax sums to 1.

#### M10: Training loop (~3 hrs)
`model/train.py` — complete training script:
- `CrossEntropyLoss` + `Adam` optimizer
- Train/validation split per epoch
- Metric logging (loss curves, accuracy per epoch)
- Early stopping on validation loss plateau
- Model checkpoint saving (best val loss)

**Gate:** Model trains for N epochs without crashing. Loss decreases. Validation accuracy above random baseline.

#### M11: Evaluation & analysis (~2 hrs)
`model/evaluate.py` — load best checkpoint, run on test set:
- Accuracy, precision, recall, F1 per class
- Confusion matrix plot
- Overfitting analysis (train vs val loss curves)
- Sample predictions with spectrograms for visual inspection

**Gate:** Evaluation report generated. Model performs meaningfully above random on held-out test data.

#### M12: Dry-run & documentation (~1 hr)
End-to-end: raw audio → pipeline → tensor → model → prediction. A single script or notebook that demonstrates the full path. README updated.

**Gate:** `python -m model.predict data/raw/sample.wav` outputs a classification with confidence. All tests green.

### What's tricky

1. **Label scheme.** What are we classifying? Notes, sound types (clap/whistle/hum), chord progressions? Define this before M7 — it shapes the whole model.
2. **Dataset size.** Deep models need data. Augmentation from Phase 1 helps, but may need to source or generate more samples.
3. **CNN→RNN reshape.** The transition from 2D conv feature maps to a sequence for the RNN is the fiddly part. Need to collapse the frequency axis while preserving the time axis.
4. **Overfitting.** Small dataset + expressive model = memorisation risk. Dropout, augmentation, and early stopping are the defences.

### When to stop

| After | Stop if | Meaning |
|---|---|---|
| M7 | Dataset too small for meaningful training | Need more data — go back to augmentation or source more clips |
| M8 | CNN outputs look wrong (all zeros, NaNs) | Architecture bug — debug before adding RNN |
| M10 | Loss doesn't decrease after 10+ epochs | Architecture or hyperparameter issue — simplify before adding complexity |
| M11 | Model no better than random | Fundamental problem — revisit features, labels, or architecture |

### Effort

~12 hours across 2–3 sessions.
