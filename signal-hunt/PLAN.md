# Signal Hunt — Plan

## Learning Objectives

- **Data Pipeline Engineering for Production:** Ingest, clean, and transform unstructured real-world data streams into structured numerical features at scale.
- **Predictive Modeling:** Build custom neural networks in PyTorch to solve multi-class classification using industry-standard training, evaluation, and debugging techniques.
- **Fine-Grained Classification:** Extend a working classifier to handle higher-dimensional label spaces with transfer learning and curriculum strategies.
- **Sequence Learning:** Model temporal patterns across multiple acoustic events using RNNs and attention, bridging from classification to sequence-to-sequence problems.

## Progression

The three classification phases build on each other deliberately:

| Phase | Task | Classes | What it teaches |
|-------|------|---------|----------------|
| 2 | Sound type | 3 (hum, whistle, clap) | CNN basics, training loops, evaluation |
| 3 | Note / pitch | ~12–24 (chromatic notes) | Transfer learning, larger label spaces, class imbalance |
| 4 | Sequences | Variable ("hum then clap") | RNNs, attention, sequence-to-sequence, temporal reasoning |

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

## Phase 3: Note & Pitch Classification

Extend the Phase 2 CNN to classify **which note** is being hummed or whistled. This jumps from 3 broad classes to ~12–24 fine-grained pitch classes (chromatic notes, 1–2 octaves).

**Business goal:** Learn transfer learning, handling larger label spaces, dealing with class imbalance, and the difference between coarse and fine-grained classification.
**Business value:** In business ML, you often start with broad categories then need finer resolution — customer segments → individual behaviours, document types → specific intents. Same skill.

### The problem

Sound type classification was easy because hums, whistles, and claps look completely different as spectrograms. Notes are harder — a hummed C4 and a hummed D4 have the same overall shape but different fundamental frequencies. The model needs to learn subtle pitch differences within the same sound type.

### Why this is a good next step

- Reuses the CNN from Phase 2 (transfer learning — freeze early layers, retrain the head)
- Forces you to deal with a larger label space (12–24 classes vs 3)
- Introduces class imbalance (some notes are easier to produce consistently)
- Teaches curriculum learning: train on easy notes first, add harder ones

### Prerequisites

- Phase 2 model working and evaluated
- New dataset: recordings of specific notes (C4, D4, E4, ... at minimum one octave)
- A reference pitch source (tuner app, piano) to label recordings accurately

### Milestones

#### M13: Note dataset collection (~2 hrs)
- [ ] Define label scheme: which notes, which octave range (C4–B4 = 12 classes is a good start)
- [ ] Record or source ~10 clips per note per sound type (hum + whistle = 240 clips minimum)
- [ ] Organise into `data/raw/notes/{note}_{type}/` folder structure
- [ ] Run Phase 1 batch pipeline to generate tensors
- [ ] Verify pitch labels with a frequency analysis spot-check

**Gate:** At least 8 notes with 10+ recordings each. Tensors generated. Manifest includes note labels.

#### M14: Transfer learning setup (~2 hrs)
- [ ] `model/transfer.py` — load Phase 2 CNN checkpoint
- [ ] Freeze early conv layers (feature extractor), replace classification head for N note classes
- [ ] New dataset class that handles note labels (extend or compose with Phase 2 dataset)
- [ ] Experiment: frozen-early vs full-finetune, log both

**Gate:** Transfer model loads Phase 2 weights. Forward pass produces `(B, num_notes)`. Frozen layers don't update during backprop.

#### M15: Training on notes (~3 hrs)
- [ ] Train with class-weighted CrossEntropyLoss (handles imbalanced note counts)
- [ ] Curriculum strategy: start with well-separated notes (C4, E4, G4 — a major triad), add semitones gradually
- [ ] Compare: transfer learning vs training from scratch (expect transfer to converge faster)
- [ ] Log per-note accuracy — expect nearby notes (C4 vs C#4) to confuse more than distant ones

**Gate:** Model above 50% accuracy on 12-class task (random = 8.3%). Nearby-note confusion visible in matrix.

#### M16: Pitch analysis & evaluation (~2 hrs)
- [ ] Confusion matrix: are confusions musically sensible? (C4↔C#4 more than C4↔F#4)
- [ ] Frequency analysis: does the CNN's first-layer filters show pitch-sensitive patterns?
- [ ] Error analysis: which notes are hardest? Is it a data problem or a model problem?
- [ ] Compare hum-note vs whistle-note accuracy (whistles have cleaner harmonics — expect better)

**Gate:** Evaluation report with per-note metrics. Confusion patterns make musical sense. Clear understanding of model limitations.

#### M17: Inference & documentation (~2 hrs)
- [ ] Update `model/predict.py` to support `--mode type` and `--mode note`
- [ ] Explainer: transfer learning (what it is, why it works, when to use it)
- [ ] Explainer: class imbalance and curriculum learning
- [ ] Update all docs (explainers README, python-concepts, libraries, project README)
- [ ] Phase 3 summary in PLAN.md

**Gate:** `python -m model.predict song.wav --mode note` → `"C4 (78% confidence)"`. Docs pass teaching test.

### What's tricky

1. **Labelling accuracy.** If your "C4" recording is actually a B3, the model learns garbage. Need a reference pitch source and ideally a frequency-check script.
2. **Semitone confusion.** Adjacent notes (C4 vs C#4) differ by ~6% in frequency. The Mel spectrogram's frequency resolution may blur this. May need to increase `n_mels` or use a different feature (CQT, which has log-frequency bins aligned to musical notes).
3. **Recording consistency.** Humans can't hum a perfect C4 every time. Pitch drift within a recording adds noise. May need to accept "close enough" labels or use pitch detection (e.g. CREPE) to auto-label.
4. **Class count.** 12 classes with small data is hard. Start with a subset (C, E, G = 3 notes) to validate the approach, then expand.

### When to stop

| After | Stop if | Meaning |
|---|---|---|
| M13 | Can't reliably produce/label 8+ distinct notes | Task is too hard with current recording setup |
| M14 | Transfer learning shows no benefit over random init | Phase 2 features aren't note-relevant — rethink features |
| M15 | Accuracy stuck near random after tuning | Mel spectrograms may lack pitch resolution — try CQT |
| M16 | Confusions are random (not musically sensible) | Model isn't learning pitch — fundamental feature problem |

### Effort

~11 hours across 2–3 sessions.

---

## Phase 4: Sequence Recognition

Detect **ordered patterns** of sounds: "hum then clap", "whistle-whistle-clap", or even simple melodies. This is where the RNN finally earns its place.

**Business goal:** Learn sequence modelling — RNNs, attention, variable-length inputs, and the shift from "what is this?" to "what happened in what order?"
**Business value:** Sequence recognition is everywhere in business: user clickstreams, transaction patterns, log event sequences, customer journey modelling. The audio domain makes it tangible and visual.

### The problem

Phases 2 and 3 classify **single, isolated sounds** (one 1.5s clip → one label). Phase 4 classifies **sequences of sounds over time**: a 5-second clip might contain "hum, pause, clap, clap" and the model needs to output that ordered sequence.

This is fundamentally different:
- Input length is variable (sequences have different numbers of events)
- Output is a sequence of labels, not a single label
- The model needs to learn temporal ordering, not just "what sounds are present"

### Why RNN now (and not before)

The CNN from Phase 2 already extracts per-time-step features from spectrograms. The RNN sits on top:
- CNN processes the spectrogram spatially (frequency patterns)
- RNN processes the CNN output temporally (how patterns change over time)

We delayed the RNN because it wasn't needed for Phases 2–3 (single-event classification). Adding it here, where temporal reasoning is actually required, means we can clearly see what it contributes.

### Prerequisites

- Phase 2 CNN working (feature extractor)
- Phase 3 nice-to-have but not required (Phase 4 can use sound-type labels from Phase 2)
- New dataset: recordings of sound sequences (longer clips, 3–10 seconds)
- Sequence labelling scheme

### Milestones

#### M18: Sequence dataset (~2.5 hrs)
- [ ] Define sequence vocabulary: what patterns to recognise
  - Start simple: 2-event sequences ("hum-clap", "whistle-hum", etc.) = 9 patterns (3×3)
  - Stretch: 3-event sequences = 27 patterns
- [ ] Two approaches (pick one, document why):
  - **Synthetic:** Concatenate existing single-event clips with random gaps → known labels for free
  - **Recorded:** Record actual sequences → more realistic but labelling is manual
- [ ] Longer tensors: extend pipeline to handle 3–10s clips (more time frames)
- [ ] Sequence label format: list of `(event_type, approximate_time)` or just ordered list

**Gate:** At least 6 distinct sequence patterns with 10+ examples each. Tensors are longer than Phase 2. Labels encode order.

#### M19: Sequence model architecture (~3 hrs)
- [ ] `model/sequence.py` — CNN-RNN hybrid:
  - Phase 2 CNN as frozen feature extractor (or fine-tunable)
  - CNN output: `(batch, features, time_steps)` — a feature vector per time slice
  - Bidirectional GRU/LSTM on the time-step sequence
  - Two possible heads (pick one, explain tradeoff):
    - **Sequence classification:** RNN final hidden state → linear → one label per sequence (simpler)
    - **Sequence-to-sequence:** RNN per-step output → CTC loss or attention → label per event (harder, more powerful)
- [ ] Start with sequence classification (the whole clip → one pattern label like "hum-clap")

**Gate:** Forward pass produces `(B, num_patterns)` logits. RNN hidden states are reasonable (no vanishing/exploding gradients).

#### M20: Training on sequences (~3 hrs)
- [ ] Train the sequence model end-to-end
- [ ] Compare: frozen CNN + trained RNN vs fine-tuning both
- [ ] Attention visualisation: which time steps does the model attend to? (should align with event boundaries)
- [ ] If using sequence classification: does the model learn order? ("hum-clap" ≠ "clap-hum")

**Gate:** Model above chance on held-out sequences. Correctly distinguishes order (not just set of sounds).

#### M21: Attention & temporal analysis (~2 hrs)
- [ ] Visualise attention weights or RNN hidden states over time
- [ ] Overlay on spectrogram: does the model "look at" the right parts?
- [ ] Error analysis: which sequences confuse the model? Similar starts? Similar ends?
- [ ] GRU vs LSTM comparison (if time permits — document which and why)

**Gate:** Attention/activation plots show temporal structure. Model demonstrably uses order, not just presence.

#### M22: Stretch — CTC or seq2seq (~3 hrs, optional)
- [ ] If sequence classification works well, try true sequence-to-sequence:
  - CTC (Connectionist Temporal Classification) — no alignment needed, learns from unaligned labels
  - Or simple attention-based decoder
- [ ] This is stretch goal territory — only if Phases 2–3 went smoothly and there's energy left
- [ ] The learning value is high: CTC and seq2seq are used in speech recognition, OCR, and translation

**Gate:** Model produces variable-length output sequences. Can decode "hum-pause-clap-clap" from audio.

#### M23: Documentation & project wrap-up (~2 hrs)
- [ ] Explainer: RNNs (GRU vs LSTM, vanishing gradients, bidirectional, hidden states)
- [ ] Explainer: sequence modelling (classification vs seq2seq, CTC, attention)
- [ ] Explainer: the CNN-RNN hybrid (why combine, how data flows, reshape trick)
- [ ] Final project README: what Signal Hunt is, what it teaches, how to run everything
- [ ] Phase 4 summary in PLAN.md
- [ ] Retrospective: what worked, what was harder than expected, what you'd do differently

**Gate:** Full project documentation passes the teaching test. All phases summarised. Someone can clone, run `setup.sh`, and work through all four phases.

### What's tricky

1. **Variable-length inputs.** Sequences have different durations. Options: pad to max length (wasteful), use packed sequences (PyTorch supports this but it's fiddly), or bucket by length.
2. **CNN→RNN reshape.** The 2D CNN output `(batch, channels, freq, time)` needs to become `(batch, time, features)` for the RNN. Collapse the frequency axis, keep time. This is the same reshape from Phase 2's plan but now it actually matters.
3. **Order sensitivity.** A bag-of-sounds model (just detecting which sounds are present) will score well on accuracy but fail at the actual task. Need evaluation that specifically tests order: "hum-clap" vs "clap-hum" must be different predictions.
4. **Vanishing gradients.** Long sequences + RNN = gradients can vanish during backprop through time. LSTM/GRU help, but gradient clipping and careful initialisation are still needed.
5. **Synthetic vs real.** Synthetic sequences (concatenated clips) may not capture how real sequences sound (transitions, overlapping sounds, natural rhythm). Start synthetic, validate with a few real recordings.

### When to stop

| After | Stop if | Meaning |
|---|---|---|
| M18 | Can't produce distinguishable sequence patterns | Sequence task too ambiguous — simplify patterns |
| M19 | RNN gradients explode/vanish despite LSTM + clipping | Architecture needs rethinking — try simpler sequences first |
| M20 | Model can't distinguish order ("hum-clap" = "clap-hum") | RNN isn't learning temporal structure — debug hidden states |
| M22 | CTC/seq2seq doesn't converge | This is stretch — the learning happened. Document what you tried and why it's hard |

### Effort

~15 hours across 3–4 sessions (~12 hrs for M18–M21 core, ~3 hrs for M22 stretch + M23 docs).
