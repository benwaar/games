# Explainers

Brief notes on each step of the pipeline — what it does and why. Detailed explainers linked where the concept needs more depth.

→ [Key Libraries](libraries.md) — what each dependency does and when you'd reach for it
→ [Python Concepts](python-concepts.md) — tuples, type hints, fixtures, and other patterns used in this codebase
→ [Data Collection](data-collection.md) — why real recordings, why these sounds, how to record and add more
→ [Train / Val / Test Split](train-val-test-split.md) — why three sets, why random isn't enough, the no-peeking rule
→ [Evaluation](evaluation.md) — precision, recall, F1, confusion matrix, loss curves, and what our run proved

---

## 1. Audio Loading & Resampling

The entry point of the pipeline — takes audio in any format, sample rate, or channel layout and produces a consistent fixed-length mono numpy array at 22050 Hz. Everything downstream depends on this uniformity.

Raw audio comes in many formats, sample rates, and channel layouts. The ingestion step normalises all of that:

- **Resample to 22050 Hz** — a standard rate that captures frequencies up to ~11 kHz (Nyquist). Human speech and most musical content sit well below this. Higher rates waste compute; lower rates lose detail.
- **Mono conversion** — stereo channels carry spatial info we don't need. Averaging to mono halves the data and keeps the model focused on *what* sounds are present, not *where*.
- **Silence trimming** — leading/trailing silence varies by recording. Trimming it means the model sees signal, not dead air.
- **Fixed-length output** — pad short clips with zeros, truncate long ones. Consistent tensor shapes simplify batching and model architecture.

> **In practice:** This is the same problem as normalising data from different sources. Financial tick data arrives at different rates per exchange; IoT sensors report at different intervals; customer event logs have different schemas. The ingestion step — resample to a common rate, trim noise, enforce a fixed shape — is the universal first step in any ML pipeline.

See: [pipeline/ingest.py](../pipeline/ingest.py)

---

## 1a. Reading Waveforms & Spectrograms

The two core visualisations for understanding audio. A waveform shows amplitude over time (when things happen); a spectrogram shows frequency over time with brightness for intensity (what frequencies are present). If you can read these, you can debug every step of the pipeline.

→ [How to read waveforms and spectrograms](reading-plots.md) — with annotated examples from our pipeline output

---

## 2. Augmentation

Four composable transforms that inject realistic variation into clean recordings — noise, ambient sound, pitch shift, and time stretch. Applied before feature extraction so the model trains on diverse conditions instead of memorising one mic in one room.

A model trained on clean recordings fails on real-world audio. Augmentation injects realistic variation at the data layer so the model learns to generalise.

- **White noise** (`add_noise`) — simulates mic hiss and background static. Controlled by SNR (signal-to-noise ratio in dB) — higher SNR = cleaner signal.
- **Ambient mixing** (`add_ambient`) — blends in a real ambient recording (room tone, traffic, etc.) at a target SNR. More realistic than synthetic noise.
- **Pitch shift** (`pitch_shift`) — shifts frequency up/down by N semitones without changing speed. Simulates different voices or instruments.
- **Time stretch** (`time_stretch`) — speeds up or slows down without changing pitch. Simulates tempo variation.

All augmentations are composable — chain them in any order. Applied *before* feature extraction so the model trains on diverse inputs.

> **In practice:** Augmentation is how you deal with imbalanced or small datasets in any domain. In fraud detection, you synthesise rare fraud patterns to stop the model ignoring them. In medical imaging, you rotate and flip scans to multiply your training data. The principle is the same: inject realistic variation at the data layer so the model generalises instead of memorising.

→ [Why SNR matters](snr.md)

See: [pipeline/augment.py](../pipeline/augment.py)

---

## 3. Mel-Spectrograms

This is the core transform — it takes a raw waveform and produces a compact, CNN-ready tensor. A 33,075-sample clip becomes a `(1, 128, 65)` PyTorch tensor (one channel, 128 Mel frequency bands, 65 time frames) with mean≈0 and std≈1. Each step throws away information the model doesn't need and amplifies what it does.

The pipeline compresses the waveform into a `(128, 65)` perceptually-weighted frequency-over-time image:

1. **STFT** — break the waveform into overlapping windows, FFT each one → frequency bins × time frames
2. **Mel filterbank** — compress 1025 linear frequency bins into 128 Mel-scaled bands (fine detail at low frequencies, coarse at high — matching human hearing)
3. **Log-dB** — convert power to decibels so quiet sounds are visible alongside loud ones

The result looks like a greyscale image: a hum is a horizontal line, a whistle chirp is a rising diagonal, a clap is a vertical broadband burst. These visually distinct patterns are what the CNN will learn to classify.

> **In practice:** This is dimensionality reduction with domain knowledge baked in — like using business-informed feature engineering instead of dumping raw columns into a model. The Mel scale encodes what we know about human perception, just as a financial feature pipeline might encode what we know about market microstructure.

→ [Mel-spectrograms explained](mel-spectrograms.md) — STFT, Mel scale, log-dB, with C/JS/TS parallels and annotated output from our samples

See: [pipeline/features.py](../pipeline/features.py) — `stft()`, `mel_spectrogram()`, `to_log_db()`

---

## 4. Normalisation

Different recordings have different absolute volumes. A hum on a laptop mic vs. a studio condenser mic produces wildly different dB ranges — but they're the same sound. Per-spectrogram normalisation (mean=0, std=1) removes this bias so the model sees *relative patterns*, not absolute levels.

Edge case: a completely silent signal has std ≈ 0. Dividing by near-zero would explode, so the code returns all-zeros for silence — no features means no features.

> **In practice:** This is z-score / `StandardScaler` — the same normalisation you'd apply to any ML feature set. Stock prices and trading volumes differ by orders of magnitude; temperature and pressure readings do too. Without scaling, the model fixates on whichever feature has the biggest numbers.

See: [pipeline/features.py](../pipeline/features.py) — `normalise()`, `extract_features()`

---

## 5. Batch Processing & Dataset

The glue that turns a folder of `.wav` files into a training-ready tensor dataset. Runs every file through the full pipeline (ingest → augment → extract → save), producing 7 variants per source file and a JSON manifest linking each tensor to its source, label, and augmentation.

- **One command:** `python -m pipeline.batch data/raw data/processed` — walks `data/raw/{class}/` subfolders, produces `.pt` files and a `manifest.json`
- **Labels from folders** — the folder name is the class label (`data/raw/hum/` → label `"hum"`). File names don't matter — only the folder they're in.
- **7 augmentations per file** — clean + 2 noise levels + pitch up/down + slow/fast. 33 source recordings become 231 training samples.
- **Deterministic** — same seed produces identical tensors. Reproducible across machines.
- **Manifest** — JSON file mapping each tensor to its source, label, and augmentation. The `Dataset` class (Phase 2) uses this to load tensors without scanning the filesystem.
- **DataLoader-ready** — tensors are uniform `(1, 128, 65)`, so they stack directly into batches.

> **In practice:** This is the ETL step of any ML project. Raw data in various formats → standardised, augmented, labelled tensors ready for training. The manifest pattern (a metadata sidecar that describes the dataset) is how teams track data provenance — which version of the pipeline produced which training set, with what parameters. Without it, you're guessing what your model trained on.

→ [Batch processing explained](batch-processing.md) — augmentation strategy, manifest schema, DataLoader integration

See: [pipeline/batch.py](../pipeline/batch.py)

---

## 5a. End-to-End Integration

The demo script ties every module together — a single command that shows the full pipeline from raw audio to normalised tensor. Useful for verifying the pipeline works on new audio files and for understanding what each step produces.

`python -m pipeline.demo` loads a `.wav` file and runs it through:
1. **Ingest** — resample, trim, pad to 1.5s
2. **Augment** — generate all 7 variants (clean + noise + pitch + speed)
3. **Extract** — Mel-spectrogram → log-dB → normalise → `(1, 128, 65)` tensor

Prints shape, mean, and std for each variant — confirming normalisation holds across all augmentations.

> **In practice:** An end-to-end demo script is a smoke test for your pipeline. In production ML, this is the script you run after deploying a new version of the feature pipeline to verify it still produces sane output. If the shapes or stats change unexpectedly, something broke upstream.

See: [pipeline/demo.py](../pipeline/demo.py)

---

## 6. Dataset & DataLoader (Phase 2 — M7)

The bridge between the Phase 1 pipeline and the Phase 2 model. `Dataset` knows how to load one item (tensor + integer label) from the manifest. `DataLoader` wraps it and handles batching, shuffling, and parallel loading.

Key ideas:
- **`Dataset` contract** — implement `__len__` (total count) and `__getitem__` (load item at index). That's all PyTorch needs.
- **Label encoding** — `"hum"` → `1`, sorted alphabetically so the mapping is deterministic across machines.
- **Stratified split** — train/val/test split that preserves class proportions. With 77 samples per class, a random split might under-represent one class in test. Stratified prevents that.
- **Separation of concerns** — the model never touches disk. It only sees `(tensor, label)` pairs. Swap the dataset without touching the model.

→ [Dataset & DataLoader explained](dataset-dataloader.md) — the protocol, label encoding, stratified splits, and how it connects to the training loop

See: [model/dataset.py](../model/dataset.py) — `SignalDataset`, `make_label_map`, `load_splits`

---

## 7. CNN Architecture (Phase 2 — M8)

Three stacked conv blocks scan the spectrogram for frequency patterns, then Global Average
Pooling collapses the spatial dimensions into a fixed-length vector, and a small linear
head outputs 3 class logits.

- **Conv2d → BatchNorm2d → ReLU → MaxPool2d** — one block. Three blocks in sequence
  progressively extract higher-level features (edges → shapes → patterns).
- **Global Average Pooling** — collapses `(B, 64, H, W)` → `(B, 64)` by averaging each
  channel. Fewer parameters than Flatten; invariant to exact spatial position.
- **Linear → Dropout → Linear** — classification head. Dropout (p=0.3) prevents
  memorisation on our small dataset.
- **~25,700 parameters** — deliberately small. Right-sized for 3 classes and ~150
  training samples.

→ [CNN architecture explained](cnn-architecture.md) — Conv2d, BatchNorm, GAP, Dropout, with C/JS/TS callouts and parameter count breakdown

See: [model/cnn.py](../model/cnn.py) — `SoundClassifier`

---

## 8. Training Loop (Phase 2 — M9)

The cycle that makes the model learn: forward pass → loss → backprop → weight update.
Repeated for every batch, every epoch, with scheduling and early stopping to prevent overfitting.

- **CrossEntropyLoss** — combines log-softmax + NLL in one numerically stable step.
  Starts near `log(3) ≈ 1.1` (random) and should decrease.
- **Adam** — adaptive learning rate per parameter. Default starting point for most tasks.
- **ReduceLROnPlateau** — halves the learning rate when val loss stops improving for N epochs.
- **Early stopping** — ends training when validation loss hasn't improved for `patience` epochs.
  Saves the best checkpoint (not the final weights) for evaluation and inference.

→ [Training loop explained](training-loop.md) — loss, backprop, Adam, scheduling, early stopping, train vs eval mode

See: [model/train.py](../model/train.py) — `train_one_epoch`, `evaluate`, `train`, CLI

---

---

## 9. Evaluation (Phase 2 — M10)

After training, the best checkpoint is loaded and run against the held-out test set.
Precision, recall, and F1 per class reveal not just overall accuracy but *which* classes
are hard and *which* confusions are being made.

- **Classification report** — per-class precision, recall, F1. Generated by `python -m model.evaluate`.
- **Confusion matrix** — which classes the model confuses with which. Our run: clean diagonal, no errors.
- **Loss curves** — train vs val loss over epochs. Shows when learning happened, where LR reductions kicked in, and how large the overfitting gap is.
- **Phase 2 result** — 100% test accuracy (35/35). All three classes correct. Clap most confident (97.4%), hum least (58.8% — closest spectrally to whistle).

→ [Evaluation explained](evaluation.md) — precision/recall/F1, confusion matrix, loss curve analysis, inference confidence

See: [model/evaluate.py](../model/evaluate.py) — `run_evaluation`, CLI

---

## 10. Inference (Phase 2 — M11)

The end-to-end payoff: take a raw `.wav` file you've never trained on, run it through
the same pipeline as training, and get a class prediction with confidence.

```bash
bash demo.sh                        # → whistle (61.2% confidence)
bash demo.sh path/to/my_sound.wav   # → clap (97.4% confidence)
python -m model.predict --scan      # predict all .wav files in data/raw/
```

The pipeline consistency test: if `ingest` or `extract_features` behaved differently
at inference than at training time, the model would produce random outputs. It doesn't —
proving the preprocessing is deterministic end to end.

See: [model/predict.py](../model/predict.py) — `predict()`, `scan()`, CLI

---

## 11. Transfer Learning (Phase 3 — M14)

The Phase 2 CNN learned frequency-band patterns from hums, whistles, and claps.
Phase 3 reuses those same features for a new task: classifying **12 piano notes**.
Only the final classification layer is replaced (3 classes → 12) and retrained.

- **Frozen backbone** — conv blocks frozen, only the head trains. Fast convergence check.
- **Fine-tune** — everything trains end-to-end. Allows the backbone to adapt to piano timbre.
- **`--compare` flag** — runs both modes and prints a side-by-side accuracy summary.

→ [Transfer learning explained](transfer-learning.md) — what we froze and why, the head swap, frozen vs fine-tune comparison, business parallels

See: [model/transfer.py](../model/transfer.py) — `load_transfer_model`, `frozen_param_count`  
See: [model/transfer_train.py](../model/transfer_train.py) — training loop, `--compare`

---

## 12. Iowa Piano Data & Public Datasets (Phase 3 — M13)

Why we used the University of Iowa Musical Instrument Samples instead of recording our own, what the domain gap is, and why we switched from the original NSynth plan.

→ [Iowa piano data explained](iowa-piano-data.md) — public datasets, domain gap, NSynth vs Iowa, business parallel

---

## 13. Phase 3 Evaluation — Per-Note Results (Phase 3 — M16)

89.5% test accuracy on 12 chromatic notes. Confusion pattern is musically sensible: F4 is hardest (sits between two adjacent semitones), G4/Ab4 confuse each other. Analysis of Mel resolution limits and what CQT would fix.

→ [Phase 3 evaluation](evaluation-phase3.md) — per-note F1 table, semitone adjacency pattern, data vs model diagnosis

---

## 14. RNN Temporal Learning (Phase 4)

*Coming in Phase 4.* Recurrent layers learn how chord states change over time — the sequence that makes a I–IV–V–I progression different from V–IV–I–I.
