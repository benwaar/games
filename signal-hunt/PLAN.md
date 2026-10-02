# Signal Hunt — Plan

## Learning Objectives

- **Data Pipeline Engineering for Production:** Ingest, clean, and transform unstructured real-world data streams into structured numerical features at scale.
- **Predictive Modeling & Sequence Learning:** Build custom neural networks in PyTorch to solve multi-label classification and temporal pattern recognition.

---

## Phase 1: Data Pipelines & Feature Extraction ✅

Turn raw audio (hums, whistles, claps) into clean, normalised Mel-spectrogram tensors.

**Business goal:** Learn unstructured data ingestion and preprocessing — required for any ML feature pipeline.
**Business value:** Teaches you how to turn raw, messy customer or sensor inputs into clean machine-readable data while preventing model failure due to real-world edge cases.

### What was built

A four-module pipeline that takes a folder of `.wav` files and produces augmented, normalised `(1, 128, 65)` PyTorch tensors ready for a CNN:

| Module | What it does |
|--------|-------------|
| `pipeline/ingest.py` | Load any audio format, resample to 22050 Hz mono, trim silence, pad/truncate to 1.5s |
| `pipeline/augment.py` | Four composable transforms: white noise, ambient mixing, pitch shift, time stretch |
| `pipeline/features.py` | STFT → 128-band Mel-spectrogram → log-dB → z-score normalisation → PyTorch tensor |
| `pipeline/batch.py` | Walk a folder, generate 7 augmented variants per file, save `.pt` tensors + `manifest.json` |

**End-to-end:** `python -m pipeline.batch data/raw data/processed` (3 source files → 21 tensors). `python -m pipeline.demo` shows the full pipeline on a single file with summary stats.

**Tests:** 57 passing across 4 test files. All gates met.

**Decisions made along the way:**
- Fixed-length tensors (pad/truncate) over variable-length — simplifies batching
- Mel scale over linear STFT — perceptual weighting matches human hearing and our label scheme
- Per-spectrogram normalisation — removes mic/volume bias between recordings
- Deterministic augmentation (seeded RNG) — reproducible datasets across machines
- Manifest pattern — JSON sidecar tracks data provenance

**Docs:** [explainers/](explainers/README.md) covers each step with C/JS/TS callouts and business parallels.

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
