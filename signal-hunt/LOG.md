# Signal Hunt — Project Record

A deep learning study project — from raw audio to a piano teacher prototype.

## Contents

- [Learning Objectives](#learning-objectives)
- [Phase Progression](#phase-progression)
- [Phase 1 — Data Pipelines](#phase-1--data-pipelines)
- [Phase 2 — Sound Type Classification](#phase-2--sound-type-classification)
- [Phase 3 — Piano Note Classification](#phase-3--piano-note-classification)
- [Phase 4 — Chords & Progressions](#phase-4--chords--progressions)
- [Overall Lessons](#overall-lessons)

---

## Learning Objectives

| Objective | Where it lands |
|-----------|---------------|
| Data pipeline engineering | Phase 1 — ingest, augment, feature extraction at scale |
| Neural network training | Phase 2 — CNN, training loop, evaluation, inference |
| Transfer learning | Phase 3 — reuse Phase 2 backbone for a new 12-class task |
| Multi-label classification | Phase 4a — sigmoid/BCE, threshold selection, per-label F1 |
| Sequence modelling | Phase 4b — CNN-RNN hybrid, bidirectional GRU |

---

## Phase Progression

Each phase reuses and extends the model from the previous one — no throwaway code.

| Phase | Task | Classes | What it teaches |
|-------|------|---------|----------------|
| 1 | Feature extraction | — | Data pipelines, augmentation, tensors |
| 2 | Sound type | 3 | CNN basics, training loop, evaluation |
| 3 | Piano note | 12 | Transfer learning, fine-grained classification |
| 4a | Chord detection | 6 / 12-label | Multi-label, sigmoid/BCE, threshold selection |
| 4b | Chord progressions | 4 sequences | RNNs, CNN→RNN reshape, sequence classification |

---

## Phase 1 — Data Pipelines

**Goal:** Turn raw audio files into normalised PyTorch tensors ready for a CNN.

### What was built

| Module | What it does |
|--------|-------------|
| `pipeline/ingest.py` | Load any audio, resample to 22050 Hz mono, trim silence, pad/truncate to 1.5s |
| `pipeline/augment.py` | White noise, ambient mixing, pitch shift, time stretch — 7 variants per clip |
| `pipeline/features.py` | STFT → 128-band Mel-spectrogram → log-dB → z-score normalisation → `(1, 128, 65)` tensor |
| `pipeline/batch.py` | Walk a folder, produce augmented tensors + `manifest.json` |

**Result:** `python -m pipeline.batch data/raw data/processed` — 3 source files → 21 tensors.

### Key decisions

- Fixed-length tensors (pad/truncate) — simplifies batching
- Mel scale over linear STFT — perceptual weighting matches human hearing
- Per-spectrogram normalisation — removes mic/volume bias
- Seeded augmentation — reproducible datasets across machines
- Manifest pattern — JSON sidecar tracks data provenance

---

## Phase 2 — Sound Type Classification

**Goal:** Train a 3-class CNN (hum / whistle / clap) end-to-end.

### What was built

| Module | What it does |
|--------|-------------|
| `model/cnn.py` | `SoundClassifier` — 3 Conv2d blocks → GAP → Linear head. 25,699 parameters |
| `model/train.py` | CrossEntropyLoss + Adam + ReduceLROnPlateau + early stopping |
| `model/evaluate.py` | Classification report, confusion matrix, loss curves |
| `model/predict.py` | `bash demo.sh` — raw `.wav` → class + confidence |

### Results

- 33 recordings (11 per class) → 231 augmented tensors
- 161 train / 35 val / 35 test
- **Test accuracy: 100% (35/35)**

### Key decisions

- Global Average Pooling over Flatten — fewer parameters, less overfitting
- 25K params deliberately small — right-sized for 3 classes and ~150 training samples
- Labels from folder names — scales to any number of recordings
- Checkpoint saves `label_map` + config alongside weights — no separate config needed at inference

---

## Phase 3 — Piano Note Classification

**Goal:** Extend the Phase 2 CNN to classify 12 chromatic piano notes (C4–B4).

### Why piano (not voice)

Original plan required 240+ voice recordings. Problems: humans can't reliably hum the same pitch twice, labelling becomes the bottleneck, the model learns your voice as much as the pitch. Iowa piano dataset (University of Iowa, free educational use) solved all of this — 36 AIFF files, perfectly pitched, exact labels.

### What was built

| Module | What it does |
|--------|-------------|
| `model/transfer.py` | Load Phase 2 checkpoint, swap `Linear(32→3)` → `Linear(32→12)` |
| `model/transfer_train.py` | `--compare` flag runs frozen and fine-tune side-by-side |
| `model/predict.py` | `--mode note` flag — `bash demo_note.sh` → correct note |

### Results

| Mode | Val acc | Test acc |
|------|---------|---------|
| Frozen backbone | 36.8% | — |
| Fine-tune (all layers) | 76.3% | **89.5%** |
| From scratch | 78.9% | — |

Transfer reaches 50% ~10 epochs faster than scratch. Final accuracy is similar — voice-to-piano domain gap meant the backbone needed to adapt anyway. Transfer advantage is convergence speed, not final accuracy.

**Per-note:** F4 hardest (F1=0.40) — sits between adjacent semitones E4 and Gb4. 6 of 12 notes perfect F1=1.00. Confusion pattern is musically sensible.

---

## Phase 4 — Chords & Progressions

### The piano teacher goal

```
bash demo_chord.sh chord.wav --expected Cmaj

Chord:   Amin     ✗  (expected Cmaj)  (92.4% confidence)
Notes:   A4 ✓  C4 ✓  E4 ✓
Missing: G4  (Cmaj needs C4 + E4 + G4)
```

Two passes on the same clip:
1. **Name model (verdict):** "you played Amin, should be Cmaj"
2. **Notes model (correction):** "you have A4 and C4, missing E4"

### Phase 4a — Chord Detection

**Synthesis:** Mix 3 Iowa note WAVs per chord, normalise to peak 0.5 before summing to prevent clipping. 6 diatonic triads × 4 dynamic combos = 24 source clips → 168 augmented tensors.

| Module | What it does |
|--------|-------------|
| `scripts/synthesise_chords.py` | Mix Iowa WAVs into chord clips |
| `model/chord.py` | `ChordNameClassifier` (6-class) and `NoteSetClassifier` (12-label) |
| `model/chord_train.py` | `--mode name` and `--mode notes` |
| `model/chord_evaluate.py` | Confusion matrix, per-note F1, threshold sweep |
| `model/predict_chord.py` | Two-pass inference: verdict + missing notes |

**Results:**

| Model | Metric | Score |
|-------|--------|-------|
| Name (6-class, CrossEntropyLoss) | Test accuracy | **96.2%** |
| Notes (12-label, BCEWithLogitsLoss) | Exact-match | **73.1%** |
| Notes | Per-note F1 range | 0.82–1.00 |

Threshold 0.5 optimal. Precision high across all notes (0.80–1.00) — misses are false negatives not false positives. Right failure mode for a tutor.

### Phase 4b — Chord Progressions

**Architecture:** CNN backbone from Phase 4a extracts per-frame chord features. Mean over frequency axis collapses spatial dims. Bidirectional GRU reads the time sequence. Final hidden state → classification head.

```
(B, 1, 128, 291)  →  CNN  →  (B, 64, 16, 36)
                  →  mean(freq)  →  (B, 36, 64)
                  →  BiGRU  →  (B, 128)
                  →  Linear  →  (B, 4)
```

| Module | What it does |
|--------|-------------|
| `scripts/synthesise_progressions.py` | Concatenate chord clips + silence gaps into 6.75s progression clips |
| `scripts/batch_progressions.py` | Full-length pipeline (no 1.5s truncation) → `(1, 128, 291)` tensors |
| `model/progression.py` | `ProgressionClassifier` — CNN backbone + bidirectional GRU, 73,956 params |
| `model/progression_train.py` | Pad-collate for variable-T batches, gradient clipping |

**Results:** 66.7% accuracy on 4 progression classes (random = 25%). 16 source clips — data is the ceiling, not the architecture.

### Phase 4 milestone summary

| Milestone | Key result |
|-----------|------------|
| M18 — chord synthesis | 168 tensors, 6 chords, Cmaj spectrogram visually distinct |
| M19 — chord heads | Name 96% val_acc, notes 76% exact-match |
| M20 — evaluation | Threshold 0.5 optimal, D4/F4 weakest (2 chords each) |
| M21 — CNN-RNN | 66.7% on 4 progressions, data-limited |
| M22 — docs | `demo_chord.sh --expected Cmaj` → "Missing: G4" |

---

## Overall Lessons

**Data pipeline:**
- Mel-spectrograms over raw waveforms — frequency patterns the CNN needs to find are cleaner
- Per-spectrogram normalisation is non-negotiable — removes mic/volume bias
- Seeded, deterministic augmentation makes debugging possible

**Transfer learning:**
- Reusing a backbone across three tasks (voice types → piano notes → chords) worked throughout
- Domain gap (voice → piano) means the backbone adapts, but gives convergence speed not final accuracy
- Head-swapping is cheap; fine-tuning the whole network is almost always better than freezing

**Multi-label:**
- Exact-match is a harsh metric — per-label F1 gives a more useful picture
- High precision (rare false positives) is the right target for a tutor use case
- Threshold of 0.5 worked; in production you'd tune this on a validation set

**Sequence modelling:**
- The CNN→RNN reshape (`mean over freq → permute`) is the most common source of bugs
- Bidirectional GRU sees past and future context — better than unidirectional for fixed-length clips
- Gradient clipping is essential for RNN training stability

**Synthetic data:**
- Synthesising from existing recordings is powerful — no new recordings needed for Phases 4a/4b
- Domain gap from synthetic to real piano playing would need real recordings for production
