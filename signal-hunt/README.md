# Signal Hunt

A deep learning project — from raw audio to a trained classifier, built phase by phase.

Turn raw audio into clean Mel-spectrogram tensors, train a CNN to classify them, extend it with transfer learning, detect chords with multi-label classification, and model chord progressions with a CNN-RNN hybrid.

## What this builds

### Phase 1: Data Pipelines & Feature Extraction
- **Ingestion wrapper** — load or record 1–2s audio clips
- **Augmentation engine** — noise injection, pitch shift, time stretch
- **Feature extraction** — STFT → Mel-spectrogram → normalised PyTorch tensor
- **Batch pipeline** — process folders of audio into `.pt` dataset files

### Phase 2: Sound Type Classification
- **3-class CNN** — classifies spectrograms as hum, whistle, or clap (25K parameters)
- **Training loop** — loss, backprop, validation, early stopping, checkpoints
- **Evaluation** — accuracy, confusion matrix, loss curves (100% test accuracy)
- **Inference** — `bash demo.sh` to identify any `.wav` file

### Phase 3: Piano Note Classification
- **Transfer learning** — Phase 2 backbone adapted from 3 sound types to 12 chromatic notes (C4–B4)
- **Iowa piano dataset** — 36 University of Iowa recordings → 252 augmented tensors
- **Fine-tuned classifier** — 89.5% test accuracy on 12-class note identification
- **Inference** — `bash demo_note.sh` or `python -m model.predict --mode note`

### Phase 4a: Chord Detection
- **Chord synthesis** — mix individual Iowa note WAVs into 6 diatonic triads
- **Two-model pipeline** — name model (verdict: "you played Amin") + note-set model (correction: "missing E4")
- **Name classifier** — 96.2% test accuracy (6-class, CrossEntropyLoss)
- **Note-set classifier** — 73.1% exact-match, per-note F1 0.82–1.00 (12-label, BCEWithLogitsLoss)
- **Inference** — `bash demo_chord.sh chord.wav --expected Cmaj`

### Phase 4b: Chord Progressions
- **Progression synthesis** — concatenate chord clips into 6.75s sequences (I-IV-V-I etc.)
- **CNN-RNN hybrid** — Phase 4 CNN backbone + bidirectional GRU, 73K parameters
- **Sequence classifier** — 66.7% accuracy on 4 progression classes (random = 25%)

## Learn

Project-specific explainers: [signal-hunt/explainers/](explainers/README.md)  
Shared concept explainers: [explainers/](../explainers/README.md)

## Quick start

```bash
bash setup.sh              # install Python, venv, deps
source .venv/bin/activate
python hello_audio.py      # verify everything works
```

## Usage

### Run the full pipeline

```bash
# Phase 1–2: sound type tensors
python -m pipeline.batch data/raw data/processed

# Phase 3: piano note tensors
python -m pipeline.batch data/raw/notes data/processed/notes

# Phase 4a: chord tensors
python -m pipeline.batch data/raw/chords data/processed/chords

# Phase 4b: progression tensors (full-length, no truncation)
python scripts/batch_progressions.py
```

### Identify a sound type (Phase 2)

```bash
bash demo.sh                        # identify data/raw/unknown.wav
bash demo.sh path/to/my_sound.wav
```

### Identify a piano note (Phase 3)

```bash
bash demo_note.sh                             # identify data/raw/unknown_note.wav
python -m model.predict --mode note file.wav
```

### Identify a chord (Phase 4a)

```bash
bash demo_chord.sh                              # identify data/raw/unknown_chord.wav
bash demo_chord.sh chord.wav --expected Cmaj    # show what's missing
python -m model.predict_chord chord.wav --verbose
```

### Train

```bash
# Phase 2 — sound type classifier
python -m model.train --epochs 50

# Phase 3 — note classifier (transfer learning)
python -m model.transfer_train --epochs 60
python -m model.transfer_train --epochs 60 --compare   # frozen vs fine-tune

# Phase 4a — chord classifiers
python -m model.chord_train --mode name --epochs 80
python -m model.chord_train --mode notes --epochs 80
python -m model.chord_train --mode name --mode notes   # both

# Phase 4b — progression CNN-RNN
python -m model.progression_train --epochs 100
```

### Synthesise data

```bash
python scripts/download_iowa_piano.py          # fetch Iowa piano WAVs
python scripts/synthesise_chords.py            # mix notes into chord clips
python scripts/synthesise_progressions.py      # concatenate chords into progressions
```

### Evaluate

```bash
python -m model.evaluate                       # Phase 2/3 confusion matrix + loss curves
python -m model.chord_evaluate --mode name     # chord-name confusion matrix
python -m model.chord_evaluate --mode notes    # per-note F1 + threshold sweep
```

### Run tests

```bash
python -m pytest tests/ -v
```

148 tests across pipeline, model, transfer learning, chord detection, and prediction modules.

### Project structure

```
pipeline/
  ingest.py            — load, resample, trim, pad/truncate audio
  augment.py           — noise, ambient, pitch shift, time stretch
  features.py          — STFT → Mel-spectrogram → log-dB → normalise → tensor
  batch.py             — folder → augmented tensors + manifest
model/
  dataset.py           — SignalDataset, make_label_map, load_splits
  cnn.py               — SoundClassifier (25K params, 3-class CNN)
  config.py            — TrainConfig, TransferConfig hyperparameter dataclasses
  train.py             — Phase 2/3 training loop
  evaluate.py          — metrics, confusion matrix, loss curves
  predict.py           — raw .wav → class + confidence (--mode sound|note)
  transfer.py          — load Phase 2 backbone, swap head for N classes
  transfer_train.py    — Phase 3 training loop, --compare flag
  chord.py             — ChordNameClassifier (6-class) and NoteSetClassifier (12-label)
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
tests/                 — 148 unit tests across all modules
data/raw/              — source .wav files (gitignored except hum/whistle/clap)
  hum/, whistle/, clap/       — 11 each (Phase 2)
  notes/{C4..B4}/             — 3 per note, Iowa piano (Phase 3)
  chords/{Cmaj..Amin}/        — 4 per chord, synthesised (Phase 4a, gitignored)
  progressions/{I-IV-V-I..}/  — 4 per label, synthesised (Phase 4b, gitignored)
data/processed/        — generated .pt tensors (gitignored)
output/                — checkpoints and history (gitignored)
explainers/            — project-specific knowledge docs
```
