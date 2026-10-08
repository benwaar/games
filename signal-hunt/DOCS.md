# Signal Hunt — Docs

Full command reference, project structure, and data setup.

## Contents

- [Setup](#setup)
- [Data pipelines](#data-pipelines)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Tests](#tests)
- [Project structure](#project-structure)

---

## Setup

```bash
bash setup.sh              # create .venv, install deps
source .venv/bin/activate
python hello_audio.py      # smoke test
```

---

## Data pipelines

### Phase 1–2: sound type tensors

```bash
python -m pipeline.batch data/raw data/processed
```

### Phase 3: piano note tensors

```bash
python scripts/download_iowa_piano.py              # fetch Iowa WAVs
python -m pipeline.batch data/raw/notes data/processed/notes
```

### Phase 4a: chord tensors

```bash
python scripts/synthesise_chords.py               # mix notes into chord clips
python -m pipeline.batch data/raw/chords data/processed/chords
```

### Phase 4b: progression tensors

```bash
python scripts/synthesise_progressions.py         # build 6.75s progression clips
python scripts/batch_progressions.py              # full-length pipeline → (1, 128, 291)
```

---

## Training

```bash
# Phase 2 — sound type classifier
python -m model.train --epochs 50

# Phase 3 — note classifier
python -m model.transfer_train --epochs 60
python -m model.transfer_train --epochs 60 --compare   # frozen vs fine-tune

# Phase 4a — chord classifiers
python -m model.chord_train --mode name --epochs 80
python -m model.chord_train --mode notes --epochs 80
python -m model.chord_train --mode name --mode notes   # run both

# Phase 4b — progression CNN-RNN
python -m model.progression_train --epochs 100
```

---

## Inference

```bash
# Sound type (Phase 2)
bash demo.sh                            # data/raw/unknown.wav
bash demo.sh path/to/sound.wav

# Piano note (Phase 3)
bash demo_note.sh                       # data/raw/unknown_note.wav
python -m model.predict --mode note file.wav --verbose

# Chord — two-pass pipeline (Phase 4a)
bash demo_chord.sh                      # data/raw/unknown_chord.wav
bash demo_chord.sh chord.wav --expected Cmaj     # show what's missing
python -m model.predict_chord chord.wav --verbose
```

---

## Evaluation

```bash
# Phase 2/3 — confusion matrix + loss curves
python -m model.evaluate

# Phase 4a — chord name model
python -m model.chord_evaluate --mode name

# Phase 4a — note-set model + threshold sweep
python -m model.chord_evaluate --mode notes --thresholds 0.3 0.5 0.7
```

---

## Tests

```bash
python -m pytest tests/ -v
```

148 tests across pipeline, model, transfer learning, chord detection, and prediction modules.

---

## Project structure

```
pipeline/
  ingest.py            — load, resample, trim, pad/truncate audio
  augment.py           — noise, ambient, pitch shift, time stretch
  features.py          — STFT → Mel-spectrogram → log-dB → normalise → tensor
  batch.py             — folder → augmented tensors + manifest

model/
  cnn.py               — SoundClassifier (25K params, 3-class CNN)
  dataset.py           — SignalDataset, make_label_map, load_splits
  config.py            — TrainConfig, TransferConfig
  train.py             — Phase 2/3 training loop
  evaluate.py          — metrics, confusion matrix, loss curves
  predict.py           — raw .wav → class + confidence (--mode sound|note)
  transfer.py          — load Phase 2 backbone, swap head for N classes
  transfer_train.py    — Phase 3 training loop, --compare flag
  chord.py             — ChordNameClassifier (6-class) + NoteSetClassifier (12-label)
  chord_train.py       — Phase 4a training (--mode name|notes)
  chord_evaluate.py    — confusion matrix, per-note F1, threshold sweep
  predict_chord.py     — two-pass chord inference: verdict + missing notes
  progression.py       — CNN-RNN hybrid for chord progression classification
  progression_train.py — Phase 4b training loop

scripts/
  download_iowa_piano.py      — fetch Iowa piano samples, convert AIFF → WAV
  synthesise_chords.py        — mix notes into 6 diatonic chord clips
  synthesise_progressions.py  — concatenate chord clips into progression sequences
  batch_progressions.py       — full-length pipeline for progression tensors

tests/                 — 148 unit tests
explainers/            — project-specific knowledge docs

data/raw/              — source .wav files
  hum/, whistle/, clap/       — 11 each (Phase 2, tracked)
  notes/{C4..B4}/             — Iowa piano, 3 per note (Phase 3, gitignored)
  chords/{Cmaj..Amin}/        — synthesised, 4 per chord (Phase 4a, gitignored)
  progressions/{I-IV-V-I..}/  — synthesised, 4 per label (Phase 4b, gitignored)

data/processed/        — generated .pt tensors (gitignored)
output/                — model checkpoints and training history (gitignored)
```
