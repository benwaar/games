# Signal Hunt — Explainers

Project-specific explainers for Signal Hunt. Shared concepts (CNNs, training loops, Mel-spectrograms, transfer learning, etc.) live in the programme-level explainers folder:

→ **[../../explainers/](../../explainers/README.md)** — all shared explainers

The most relevant shared explainers for this project:

→ [Mel-spectrograms](../../explainers/mel-spectrograms.md) — STFT, Mel scale, log-dB, with C/JS/TS parallels
→ [CNN Architecture](../../explainers/cnn-architecture.md) — Conv2d, BatchNorm, GAP, Dropout, parameter count
→ [Training Loop](../../explainers/training-loop.md) — loss, backprop, Adam, scheduling, early stopping
→ [Transfer Learning](../../explainers/transfer-learning.md) — what we froze and why, frozen vs fine-tune
→ [Evaluation](../../explainers/evaluation.md) — precision, recall, F1, confusion matrix, loss curves
→ [Train / Val / Test Split](../../explainers/train-val-test-split.md) — why three sets, stratified splits
→ [Dataset & DataLoader](../../explainers/dataset-dataloader.md) — the protocol, label encoding
→ [Batch Processing](../../explainers/batch-processing.md) — augmentation strategy, manifest schema
→ [Libraries](../../explainers/libraries.md) — PyTorch, librosa, soundfile, what each dependency does
→ [Python Concepts](../../explainers/python-concepts.md) — patterns used throughout this codebase
→ [Data Collection](../../explainers/data-collection.md) — why real recordings, how to add more
→ [Reading Waveforms & Spectrograms](../../explainers/reading-plots.md) — annotated examples
→ [SNR](../../explainers/snr.md) — signal-to-noise ratio and why it matters for augmentation

---

## Signal Hunt — Project-specific explainers

### Iowa Piano Data (Phase 3 — M13)

Why the University of Iowa dataset, what domain gap means, why we switched from NSynth.

→ [Iowa piano data](iowa-piano-data.md)

---

### Phase 3 Evaluation — Per-Note Results (Phase 3 — M16)

89.5% test accuracy on 12 chromatic notes. F4 hardest (semitone adjacency). Mel resolution analysis.

→ [Phase 3 evaluation](evaluation-phase3.md)

---

### Chord Synthesis — Mixing Notes in the Time Domain (Phase 4 — M18)

Mixing Iowa WAVs into chord clips. Clipping prevention, dynamic combinations, spectrogram comparison.

→ [Chord synthesis](chord-synthesis.md)

---

### The Two-Model Feedback Design (Phase 4 — M19)

Name model (verdict) + note-set model (correction). Same clip, two passes. Mirrors real teaching.

→ [Two-model feedback](two-model-feedback.md)

---

### Multi-Label Evaluation (Phase 4 — M20)

Exact-match vs per-label F1, threshold selection, why precision and recall behave differently when multiple labels are simultaneously active. Phase 4 results: D4/F4 weakest (appear in only 2 chords), precision high across all notes (false negatives, not false positives).

→ [Multi-label evaluation](multi-label-evaluation.md)

---

### RNNs and the CNN→RNN Reshape (Phase 4 — M21)

How GRUs work, why bidirectional matters, and the `(B,64,16,T) → mean → permute → GRU` reshape that connects the CNN to the RNN. The Signal Hunt progression classifier: 36 time steps, 73,956 params, 66.7% on 4 progression classes.

→ [RNN sequence modelling](rnn-sequence-modelling.md)

---

### RNN Temporal Learning (Phase 4 — coming)

*Coming in Phase 4.* Recurrent layers for chord progressions over time.
