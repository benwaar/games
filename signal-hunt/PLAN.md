# Signal Hunt — Plan

## Learning Objectives

- **Data Pipeline Engineering for Production:** Ingest, clean, and transform unstructured real-world data streams into structured numerical features at scale.
- **Predictive Modeling:** Build custom neural networks in PyTorch to solve multi-class classification using industry-standard training, evaluation, and debugging techniques.
- **Fine-Grained Classification:** Extend a working classifier to handle higher-dimensional label spaces with transfer learning and curriculum strategies.
- **Multi-Label Classification:** Detect multiple simultaneous outputs (chord notes) — sigmoid/BCE instead of softmax/CE, threshold selection, exact-match vs per-label evaluation.
- **Sequence Learning:** Model temporal patterns across chord changes using RNNs, bridging from single-frame classification to sequence-over-time problems.

## Progression

The three classification phases build on each other deliberately:

| Phase | Task | Classes | What it teaches |
|-------|------|---------|----------------|
| 2 | Sound type | 3 (hum, whistle, clap) | CNN basics, training loops, evaluation |
| 3 | Note / pitch | ~12–24 (chromatic notes) | Transfer learning, larger label spaces, class imbalance |
| 4 | Chords & progressions | Multi-label ("C+E+G"), sequences ("Cmaj → Dmin → G7") | Multi-label classification, sigmoid/BCE, RNNs over chord frames |

Each phase reuses and extends the model from the previous one — no throwaway code.

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

## Phase 2: Sound Type Classification (3-class CNN) ✅

Train a CNN to classify spectrograms as **hum**, **whistle**, or **clap**.

**Business goal:** Learn the full model lifecycle: dataset construction, architecture design, training loop, evaluation, and inference.
**Business value:** Build custom classifiers tailored to specific business logic rather than relying on expensive third-party APIs.

### What was built

A complete classification pipeline from raw audio to inference:

| Module | What it does |
|--------|-------------|
| `model/dataset.py` | `SignalDataset` loads `(tensor, label)` pairs from the manifest. `load_splits` returns stratified 70/15/15 train/val/test splits with fixed seed. `make_label_map` encodes class names → integers (sorted alphabetically for determinism). |
| `model/cnn.py` | `SoundClassifier` — 3 Conv2d blocks (Conv → BatchNorm → ReLU → MaxPool) → Global Average Pooling → Linear(64→32) → Dropout(0.3) → Linear(32→3). 25,699 parameters. Input `(B, 1, 128, 65)` → output `(B, 3)` logits. |
| `model/config.py` | `TrainConfig` dataclass — all hyperparameters in one place (batch size, lr, epochs, dropout, patience, seed). |
| `model/train.py` | Full training loop — CrossEntropyLoss + Adam + ReduceLROnPlateau + early stopping. Saves best checkpoint (`output/best_model.pt`) and history (`output/history.json`). CLI: `python -m model.train --epochs 50`. |
| `model/evaluate.py` | Loads checkpoint, runs test set, prints classification report, saves confusion matrix and loss curves to `explainers/images/`. CLI: `python -m model.evaluate`. |
| `model/predict.py` | `predict(wav_path)` — raw `.wav` → class + confidence. Scan mode: `python -m model.predict --scan` identifies all `.wav` files dropped into `data/raw/`. |
| `demo.sh` | `bash demo.sh` — identifies `data/raw/unknown.wav`. `bash demo.sh file.wav` — any file. |

**Training result:**
- 33 source recordings (11 per class) → 231 augmented tensors (7 per recording)
- 161 train / 35 val / 35 test (stratified 70/15/15)
- Best checkpoint: epoch 41, val_loss=0.302, val_acc=97.1%
- **Test set: 100% accuracy (35/35) — clap 12/12, hum 11/11, whistle 12/12**
- 115 tests passing

**Decisions made:**
- Global Average Pooling over Flatten — fewer parameters, less overfitting on small dataset
- 25K params deliberately small — right-sized for 3 classes and ~150 training samples
- Labels from folder names (`data/raw/hum/`) not filenames — scales to any number of recordings
- Tensor filenames use source stem (not label) — avoids collisions with multiple files per class
- Checkpoint saves label_map + config alongside weights — inference needs no separate config file

**Docs written:** [cnn-architecture.md](explainers/cnn-architecture.md), [training-loop.md](explainers/training-loop.md), [evaluation.md](explainers/evaluation.md), [train-val-test-split.md](explainers/train-val-test-split.md), [dataset-dataloader.md](explainers/dataset-dataloader.md)

---

## Phase 3: Piano Note Classification

Extend the Phase 2 CNN to classify **which piano note** is being played — C4 through B4, one chromatic octave (12 classes). This is the same transfer learning and fine-grained classification goal as originally planned for humming/whistling, with a better data source.

**Business goal:** Learn transfer learning, handling larger label spaces, and the difference between coarse and fine-grained classification.
**Business value:** Start broad, then refine — the pattern for any ML system that needs to grow from a working baseline into something more specific.

### Why we switched from voice to piano

The original Phase 3 plan required recording 240+ voice clips (10 per note × 12 notes × 2 sound types). In practice:
- Humans can't reliably hum the same pitch twice — labelling becomes the bottleneck
- 240 clips for one octave, needing a reference pitch source for every one
- The model learns your voice as much as the pitch

Piano solves all of this:
- **NSynth** (Google/Magenta) is a free public dataset of 300K+ instrument notes, including clean acoustic grand piano recordings at every pitch with exact MIDI labels
- Every note is perfectly pitched — no labelling uncertainty
- Scales to any octave range without recording anything
- An electric piano is also available for recording real-world test clips — closing the domain gap at test time

The learning objectives (transfer learning, fine-grained classification, class imbalance) are identical. The domain is more tractable.

### The longer-term goal

Phase 3 builds the foundation for a piano teacher application:
- Phase 3: "what single note is this?" (pitch classification)
- Phase 4+: "what chord is this?" (multi-label)
- End goal: player plays a chord → model detects which notes were played → compare against expected → "you hit Ab, the note should be A"

This is a genuinely useful real-time feedback tool. Piano students learning chords is an unsolved problem at scale.

### Why piano spectrograms work

Piano notes have very clear harmonic structure on a Mel-spectrogram — a fundamental frequency with overtones at integer multiples. A C4 (261 Hz) looks different from a D4 (294 Hz): the fundamental peak is in a different Mel bin, and the harmonic ladder shifts with it. The CNN from Phase 2 already knows how to find frequency-band patterns. Retraining the head to recognise 12 pitch positions instead of 3 sound types is a natural extension.

### Prerequisites

- Phase 2 model complete ✅
- NSynth dataset (piano subset) downloaded — see M13

### Milestones

#### M13: Dataset preparation (~2 hrs)
- [x] Download NSynth piano subset (acoustic_grand_piano, C4–B4 = MIDI notes 60–71)
- [x] Filter and organise into `data/raw/notes/{note}/` (e.g. `C4/`, `Cs4/`, `D4/` ...)
- [x] Run Phase 1 batch pipeline to generate tensors
- [x] Spot-check: render a few spectrograms — can you see the pitch difference visually?
- [ ] Optional: record a few real piano clips (electric piano) and save to `data/raw/notes/{note}/` for later domain gap testing

**What was used:** University of Iowa Musical Instrument Samples (free, educational use) — 36 AIFF files (12 notes × pp/mf/ff), converted to WAV by `scripts/download_iowa_piano.py`. 252 tensors generated (36 × 7 augmentations). C4 spectrogram confirmed visually distinct harmonic ladder pattern.

**Gate:** ✅ 12 note classes, 3 clips each (21 augmented per class), tensors generated, labels correct.

#### M14: Transfer learning setup (~2 hrs)
- [x] `model/transfer.py` — load Phase 2 CNN checkpoint
- [x] Freeze early conv layers (feature extractor), replace classification head for 12 note classes
- [x] New dataset class that handles note labels
- [x] Experiment: frozen-early vs full-finetune — log both, compare convergence speed

**Gate:** Transfer model loads Phase 2 weights. Forward pass produces `(B, 12)`. Frozen layers don't update during backprop.

**What was built:** `model/transfer.py` — `load_transfer_model` loads Phase 2 checkpoint, swaps `Linear(32→3)` → `Linear(32→12)`, optionally freezes `conv_blocks` + `gap`. `model/transfer_train.py` — training loop with `--compare` flag to run both frozen and fine-tune and print side-by-side results. Existing `SignalDataset` / `load_splits` work unchanged — pointed at `data/processed/notes/`. 14 tests in `tests/test_transfer.py` covering output shape, head replacement, freeze/unfreeze, backbone weight preservation, and gradient flow. Explainer: [explainers/transfer-learning.md](explainers/transfer-learning.md).

**Training results (60 epochs, Iowa piano, 252 tensors, 12 classes):**

| Mode | Best val_acc | Best val_loss | vs random (8.3%) |
|------|-------------|--------------|-----------------|
| Frozen (head only) | 36.8% | 1.996 | 4.4× |
| Fine-tune (all layers) | **76.3%** | **0.907** | **9.2×** |

Frozen loss was still falling at epoch 60 — hit the epoch limit, not a ceiling. Fine-tune crossed frozen's best (36.8%) by epoch 16. The LR scheduler halved lr to 5e-4 at epoch 51; 76.3% arrived one epoch later. Gap shows the Phase 2 backbone features (voice timbre) needed to adapt to piano — partial transfer was real (4.4× random) but full adaptation was 2× better.

#### M15: Training on notes (~3 hrs)
- [x] Train with class-weighted CrossEntropyLoss
- [x] Curriculum strategy: start with well-separated notes (C4, E4, G4 — a major triad), add semitones
- [x] Compare: transfer learning vs training from scratch (expect transfer to converge faster)
- [x] Log per-note accuracy — expect C4 vs C#4 to confuse more than C4 vs F#4

**Gate:** Model above 50% accuracy on 12-class task (random = 8.3%). Nearby-note confusion visible in matrix. ✅

**What was built:**
- `model/evaluate.py` — confusion matrix figsize now scales with `n` classes; x-axis labels rotated 45°
- `model/train.py` — added `--processed-dir`, `--num-classes`, `--output-dir` CLI flags
- Scratch baseline: `python -m model.train --processed-dir data/processed/notes --num-classes 12 --output-dir output/scratch_notes`

**Results:**

| Mode | Best val_acc | Epoch to 50% | Test acc |
|------|------------|--------------|---------|
| Transfer fine-tune | 76.3% | ~epoch 24 | **89.5%** |
| Scratch (random init) | 78.9% | ~epoch 26 | — |

Transfer reaches 50% ~10 epochs faster. Final accuracy is similar — voice-to-piano domain gap meant the backbone needed to unlearn voice features anyway. Transfer advantage is timing, not final accuracy.

Per-note: F4 is hardest (F1=0.40) — sits between adjacent semitones E4 and Gb4. G4/Ab4 also confused (adjacent semitone pair). 6 of 12 notes are perfect F1=1.00. Confusion pattern is musically sensible.

Class-weighted loss and curriculum learning were not implemented — balanced data makes weighted loss a no-op; curriculum not needed at 76%+ accuracy. Both documented in [explainers/transfer-learning.md](explainers/transfer-learning.md).

#### M16: Evaluation & domain gap test (~2 hrs)
- [ ] Confusion matrix: are confusions musically sensible?
- [ ] If real piano clips recorded in M13: test on those — does NSynth training generalise?
- [ ] Error analysis: which notes are hardest? Data problem or model problem?

**Gate:** Evaluation report with per-note metrics. Confusion patterns make musical sense.

#### M17: Inference & documentation (~2 hrs)
- [ ] Update `model/predict.py` to support `--mode note`
- [ ] Explainer: transfer learning (what it is, why it works, when to use it)
- [ ] Explainer: NSynth as training data — why public datasets, what domain gap means
- [ ] Update docs, Phase 3 summary in PLAN.md

**Gate:** `python -m model.predict piano_clip.wav --mode note` → `"C4 (91% confidence)"`. Docs pass teaching test.

### What's tricky

1. **NSynth → real piano domain gap.** NSynth notes are clean, isolated, and perfectly pitched. Real piano has resonance, pedal sustain, and room acoustics. Test on real recordings; expect a drop in accuracy; augment with reverb and noise if it's too large.
2. **Semitone resolution.** C4 and C#4 differ by ~6% in frequency (261 Hz vs 277 Hz). The Mel spectrogram's frequency resolution at default settings may blur this. If accuracy stalls on adjacent notes, increase `n_mels` or switch to CQT (log-frequency bins aligned to musical pitch).
3. **NSynth clip length.** NSynth notes are 4 seconds. Our pipeline uses 1.5-second clips. Either truncate (use the attack/sustain), or re-tune the pipeline for longer clips.

### When to stop

| After | Stop if | Meaning |
|---|---|---|
| M13 | NSynth doesn't download or MIDI pitch labels are wrong | Check the dataset format — likely a parsing issue |
| M14 | Transfer learning shows no benefit over random init | Phase 2 features aren't pitch-relevant — may need to retrain from scratch on piano data |
| M15 | Accuracy stuck near random (8.3%) after tuning | Mel spectrograms may lack pitch resolution — try CQT or increase n_mels |
| M16 | Real piano test accuracy < 30% | Domain gap too large — add reverb/noise augmentation and retrain |

### Effort

~10 hours across 2–3 sessions.

---

## Phase 4: Chords & Chord Progressions

Extend the Phase 3 note classifier to detect **chords** (multiple notes played simultaneously) and then recognise **chord progressions** (sequences of chords over time). This is the payoff of the piano teacher goal.

**Business goal:** Learn multi-label classification and sequence modelling — the shift from "one label per input" to "multiple labels simultaneously" and then "ordered sequence of states over time."
**Business value:** Multi-label classification is everywhere: a product can belong to multiple categories, a customer can have multiple churn signals at once, an email can be both urgent and a complaint. Sequence modelling extends this to temporal patterns in clickstreams, transaction chains, and log event sequences.

### The two problems

**Part A — Chord detection (single clip):** a 1.5s clip contains a chord (e.g. C major = C4 + E4 + G4 playing at once). Classify which notes are present simultaneously. This is *multi-label*: the output is a set, not a single class.

**Part B — Chord progressions (longer clip):** a 5–10s clip contains a sequence of chords changing over time: "Cmaj → Amin → F → G". Classify the sequence. This is where the RNN earns its place.

### Why this progression

Phase 3 classifies one note in isolation. Real piano playing involves chords — multiple notes at once — and music is made of progressions of those chords. The step from Phase 3 to Phase 4 mirrors how a piano teacher app would actually work:

- Phase 3: "what note is this?" → pitch tuner
- Phase 4a: "what chord is this?" → chord recogniser
- Phase 4b: "what chord progression did you play?" → compare against the piece being learned

### Why multi-label first, then RNN

Single-chord detection teaches the new loss function and output format without temporal complexity. Once the model can reliably detect a chord in a clip, adding the RNN layer for progressions is a clean extension — the CNN detects what's present, the RNN tracks how it changes.

### Why RNN now (and not before)

The Phase 3 CNN classifies one snapshot in time. A chord progression has structure over multiple seconds — the model needs to learn that a G7 chord following a C major means something different from G7 following a Dmin. That temporal dependency is what RNNs are designed for.

### The piano teacher goal

> Player plays a chord → model detects which notes are present → compare against expected chord → "you hit Ab, the note should be A"
> Player plays a progression → model labels each chord → compare against the expected I-IV-V-I → "bar 2 should be F major, you played Fmin"

### Dataset strategy

**Chords:** synthesise from existing Iowa piano recordings. Mix two or three individual note clips together in the time domain — no new recordings needed. Start with common triads in C major: Cmaj (C+E+G), Dmin (D+F+A), Emin (E+G+B), Fmaj (F+A+C), G (G+B+D), Amin (A+C+E). That's 6 chord types from notes we already have.

**Progressions:** concatenate chord clips with short gaps or cross-fades. Synthetic sequences give exact labels for free. Validate with a few real recordings at the end.

### Prerequisites

- Phase 3 fine-tuned checkpoint (`output/transfer/finetune/best_model.pt`) ✅
- Iowa piano single-note recordings in `data/raw/notes/` ✅
- New synthesis script to mix notes into chords

---

### Milestones

#### M18: Chord dataset (~2 hrs)
- [ ] `scripts/synthesise_chords.py` — mix single-note WAVs to create chord clips
  - Start with diatonic triads in C major (6 chords): Cmaj, Dmin, Emin, Fmaj, Gmaj, Amin
  - Label format: note set `["C4","E4","G4"]` and chord name `"Cmaj"` in manifest
  - 3 source clips per note × pp/mf/ff → mix combinations → 7 augmentations → target ~150 clips per chord
- [ ] Run `pipeline.batch` on chord clips → `data/processed/chords/`
- [ ] Spot-check: spectrogram of Cmaj should show 3 harmonic ladders overlaid

**Gate:** 6 chord classes, ≥100 augmented tensors each. Manifest has both note-set and chord-name labels. Spectrograms visually distinguishable from single notes.

#### M19: Multi-label chord model (~2 hrs)
- [ ] `model/chord.py` — reuse Phase 3 backbone, replace head:
  - Output: `(B, 12)` — one logit per note (sigmoid, not softmax)
  - Loss: `BCEWithLogitsLoss` — treats each note as an independent binary prediction
  - Threshold: `pred = (sigmoid(logits) > 0.5)` — which notes are "on"
- [ ] Alternatively: chord-name classification head `(B, 6)` with `CrossEntropyLoss` — simpler, loses note-level detail
- [ ] Build both heads, document the tradeoff, pick one to train
- [ ] Metrics for multi-label: per-note F1, exact-match accuracy (all notes in chord correct)

**Gate:** Forward pass produces correct output shape. BCEWithLogitsLoss computes without NaN. Predict on a Cmaj clip → C4, E4, G4 flagged above threshold.

#### M20: Training on chords (~2.5 hrs)
- [ ] Train multi-label chord detector
- [ ] Compare: note-set prediction (12 binary outputs) vs chord-name classification (6 classes)
- [ ] Per-chord confusion: does Cmaj confuse with Amin? (share notes C and E) vs Gmaj? (share G)
- [ ] Threshold sensitivity: try 0.3, 0.5, 0.7 — what does precision/recall tradeoff look like?

**Gate:** Exact-match accuracy > 50% (random = 1/64 ≈ 1.6% for 6-note subsets). Confusions are musically sensible (shared notes → more confusion).

#### M21: Chord progressions + RNN (~3 hrs)
- [ ] `scripts/synthesise_progressions.py` — concatenate chord clips with gaps: Cmaj→Fmaj→G→Cmaj (I-IV-V-I), Amin→Fmaj→Cmaj→G (vi-IV-I-V), etc.
- [ ] `model/progression.py` — CNN-RNN hybrid:
  - Phase 3/4 CNN as feature extractor: `(B, 1, 128, T)` → `(B, 64, T')`
  - Collapse freq axis → `(B, T', 64)` time-step feature sequence
  - Bidirectional GRU: `(B, T', 64)` → `(B, T', hidden)`
  - Per-frame chord head: classify which chord is at each time step
- [ ] Start with progression classification (whole clip → one label like "I-IV-V-I") before per-step decoding

**Gate:** RNN forward pass produces `(B, num_progressions)` or `(B, T', num_chords)`. No gradient explosion. Progression labels distinguish order: I-IV-V ≠ V-IV-I.

#### M22: Documentation & project wrap-up (~2 hrs)
- [ ] Explainer: multi-label classification (sigmoid vs softmax, BCE vs CE, threshold selection, exact-match vs per-label F1)
- [ ] Explainer: RNNs (GRU vs LSTM, vanishing gradients, bidirectional, hidden states, the reshape from CNN to RNN)
- [ ] Explainer: chord progressions as sequence modelling (how music theory maps to ML sequence problems)
- [ ] Final project README: what Signal Hunt is, what it teaches, how to run everything end to end
- [ ] Phase 4 summary in PLAN.md
- [ ] Retrospective: what worked, what was harder than expected, what you'd do differently

**Gate:** Full project documentation passes the teaching test. All phases summarised. Someone can clone, run `setup.sh`, and work through all four phases.

### What's tricky

1. **Note mixing produces artefacts.** Simply adding two waveforms can clip if both are near-peak amplitude. Normalise each note clip before mixing; check the output spectrogram for clipping distortion.
2. **Multi-label vs chord-name tradeoff.** Predicting individual notes gives more interpretable errors (you can say "the model missed the E") but harder to train (12 binary tasks vs 1 6-class task). Chord-name is simpler and probably better for a first pass.
3. **Threshold sensitivity.** A 0.5 threshold on sigmoid outputs may miss softly-played notes (logit slightly below 0). Try lower thresholds and check recall; use the threshold that maximises F1 on the validation set.
4. **CNN→RNN reshape.** The 2D CNN output `(B, C, freq, time)` must become `(B, time, features)` for the RNN. Collapse the frequency dimension (average pooling or flatten), keep the time dimension. This reshape is the most common source of bugs in CNN-RNN hybrids.
5. **Progression labelling.** Synthetic progressions have exact chord-boundary labels. Real progressions have fuzzy transitions. For training, start purely synthetic.

### When to stop

| After | Stop if | Meaning |
|---|---|---|
| M18 | Chord spectrograms look identical | Note mixing not producing frequency-separable chords — check the mixing levels |
| M19 | BCELoss produces NaN | Numerical issue — add `eps` clipping or use `BCEWithLogitsLoss` (which handles it) |
| M20 | Exact-match accuracy < 20% after tuning | Multi-label too hard with synthesised data — fall back to chord-name classification |
| M21 | RNN gradients explode despite LSTM + clipping | Sequences too long — shorten clips, reduce progression length to 2 chords |

### Effort

~10 hours across 2–3 sessions.
