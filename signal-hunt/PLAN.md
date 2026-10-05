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

## Phase 2: Sound Type Classification (3-class CNN)

Train a CNN to classify spectrograms as **hum**, **whistle**, or **clap**. This is the simplest classification task — three classes with very different spectral signatures.

**Business goal:** Learn the full model lifecycle: dataset construction, architecture design, training loop, evaluation, and inference.
**Business value:** Build custom classifiers tailored to specific business logic rather than relying on expensive third-party APIs.

### The problem

Each sound type has a distinct spectral fingerprint: hums are low-frequency sustained tones, whistles are high-frequency narrow bands, claps are broadband transients. A CNN should learn these spatial patterns from Mel spectrograms without needing temporal modelling — the shapes alone are enough for this task.

### Why CNN only (no RNN yet)

A common mistake is overbuilding. For 3-class sound type classification, the spectral shape alone is highly discriminative — you don't need to model how the sound changes over time. Starting with a pure CNN:
- Simpler to debug (fewer moving parts)
- Faster to train (no sequence unrolling)
- Establishes a strong baseline to compare against when RNN is added in Phase 4
- Teaches CNN fundamentals without the distraction of RNN complexity

### What to build

1. **Dataset + DataLoader** — load tensors, map to labels, train/val/test split
2. **CNN classifier** — 2D conv blocks → global average pooling → linear head
3. **Training loop** — loss, backprop, validation, checkpointing
4. **Evaluation** — accuracy, confusion matrix, overfitting analysis
5. **Inference script** — raw audio → prediction

### Milestones

#### M7: Dataset & DataLoader (~2 hrs)
- [x] `model/dataset.py` — PyTorch `Dataset` that loads `.pt` tensors from Phase 1
- [x] Labels from folder structure (`data/raw/hum/`, `data/raw/whistle/`, `data/raw/clap/`) or manifest
- [x] Stratified train/val/test split (70/15/15) with fixed seed
- [x] Label encoding: string → integer mapping, stored in dataset metadata

**Gate:** `DataLoader` iterates `(tensor, label)` batches. Shapes `(B, 1, 128, 65)` and `(B,)`. Split is reproducible.

**Data requirement:** Need at least 10 recordings per class (× 7 augmentations = 70 tensors per class, 210 total). If we don't have enough, M7 includes recording or sourcing more clips.

#### M8: CNN architecture (~2 hrs)
- [x] `model/cnn.py` — the classifier:
  - 3 Conv2d blocks: Conv → BatchNorm → ReLU → MaxPool
  - Global Average Pooling (not flatten — reduces parameters, prevents overfitting)
  - Linear → Dropout → Linear → 3-class output
- [x] Input: `(batch, 1, 128, 65)` → Output: `(batch, 3)`
- [x] Parameter count logged (target: <500K for this task)

**Gate:** Forward pass on dummy batch produces `(B, 3)` logits. No NaNs. Softmax sums to 1. Parameter count reasonable.

#### M9: Training loop (~3 hrs)
- [x] `model/train.py` — complete training script:
  - `CrossEntropyLoss` + `Adam` optimizer
  - Per-epoch: train loss, val loss, val accuracy
  - Learning rate scheduler (ReduceLROnPlateau)
  - Early stopping on val loss (patience=5)
  - Save best checkpoint + training history JSON
- [x] `model/config.py` — hyperparameters as a dataclass (batch size, lr, epochs, etc.)
- [x] CLI: `python -m model.train --epochs 50 --lr 0.001`

**Gate:** Loss decreases. Val accuracy above 50% (random baseline = 33%). Training history saved.

#### M10: Evaluation & analysis (~2 hrs)
- [ ] `model/evaluate.py` — load best checkpoint, run on held-out test set:
  - Accuracy, precision, recall, F1 per class
  - Confusion matrix plot (saved to `explainers/images/`)
  - Train vs val loss curves plot
  - 3 correct + 3 incorrect predictions shown with spectrograms
- [ ] Analysis: which class is hardest? Which augmentations help/hurt?

**Gate:** Evaluation report generated. Model meaningfully above random on test data. Confusion matrix shows the model learned real differences.

#### M11: Inference & end-to-end (~1.5 hrs)
- [ ] `model/predict.py` — takes a raw `.wav` file, runs the full pipeline, outputs prediction:
  - Load audio → ingest → features → model → softmax → top class + confidence
  - CLI: `python -m model.predict data/raw/sample.wav`
- [ ] End-to-end test: record a new sound, run prediction, verify it works

**Gate:** `python -m model.predict some_file.wav` → `"hum (92.3% confidence)"`. All tests green.

#### M12: Documentation & consolidation (~1.5 hrs)
- [ ] Explainer: CNN architecture (how conv layers extract spatial features from spectrograms)
- [ ] Explainer: training loops (loss, backprop, optimisers, learning rate schedules)
- [ ] Update explainers README, python-concepts, libraries
- [ ] Update project README with model commands
- [ ] Phase 2 summary in PLAN.md (replace checkboxes with what-was-built)

**Gate:** Docs pass the teaching test. Someone with C/JS/TS background can follow the full path.

### What's tricky

1. **Dataset size.** 3 classes × 10 recordings × 7 augmentations = 210 samples. Tight, but workable for a 3-class CNN. Watch for overfitting hard.
2. **Class balance.** If one sound type has more samples, the model will be biased toward it. Stratified splits and class-weighted loss help.
3. **Overfitting.** Small dataset + CNN = memorisation risk. Defences: dropout, augmentation, early stopping, global average pooling (fewer params than flatten).
4. **Evaluation honesty.** With 210 samples, test set is ~30 items. Metrics will be noisy. Don't over-interpret small differences.

### When to stop

| After | Stop if | Meaning |
|---|---|---|
| M7 | <5 recordings per class | Need more data before training is meaningful |
| M8 | Forward pass produces NaNs or all-zeros | Architecture bug — fix before training |
| M9 | Loss doesn't decrease after 20 epochs | Hyperparameter or architecture issue — simplify |
| M10 | No better than random (33%) | Features, labels, or architecture need rethinking |

### Effort

~12 hours across 2–3 sessions.

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
