# Signal Hunt

A deep learning project — from raw audio to a trained classifier, built phase by phase.

Turn raw audio into clean Mel-spectrogram tensors, train a CNN to classify them, extend it with transfer learning, and build toward chord and progression detection.

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

### Phase 4: Chords & Progressions (next)
- **Chord synthesis** — mix individual note WAVs into diatonic triads
- **Multi-label classification** — detect which notes are present simultaneously (sigmoid/BCE)
- **Sequence modelling** — CNN-RNN hybrid for chord progressions over time

## Learn

Step-by-step notes on each part of the pipeline — what it does and why: [explainers/](explainers/README.md)

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
```

Takes every `.wav` in the source folder, runs it through ingest → augment (7 variants) → feature extraction, and saves `.pt` tensors + a `manifest.json`. Output tensors are `(1, 128, 65)` — one channel, 128 Mel bands, 65 time frames.

### Classify a sound type (Phase 2)

```bash
bash demo.sh                        # identify data/raw/unknown.wav
bash demo.sh path/to/my_sound.wav   # identify any recording
```

### Classify a piano note (Phase 3)

```bash
bash demo_note.sh                           # identify data/raw/unknown_note.wav
python -m model.predict --mode note file.wav  # any piano recording
```

### Predict (general)

```bash
# Single file — sound type (default) or note
python -m model.predict path/to/recording.wav
python -m model.predict path/to/recording.wav --mode note --verbose

# Scan all .wav files in data/raw/
python -m model.predict --scan
```

### Train

```bash
# Phase 2 — sound type classifier
python -m model.train --epochs 50

# Phase 3 — note classifier (transfer learning)
python -m model.transfer_train --epochs 60
python -m model.transfer_train --epochs 60 --compare   # frozen vs fine-tune side-by-side

# Phase 3 — note classifier (from scratch, for comparison)
python -m model.train --processed-dir data/processed/notes --num-classes 12 --output-dir output/scratch_notes
```

### Evaluate

```bash
python -m model.evaluate
```

Loads the best checkpoint, runs it on the held-out test set, prints a classification report, and saves a confusion matrix and loss curves to `explainers/images/`.

### Run tests

```bash
python -m pytest tests/ -v
```

129 tests across pipeline, model, transfer learning, and prediction modules.

### Project structure

```
pipeline/
  ingest.py          — load, resample, trim, pad/truncate audio
  augment.py         — noise, ambient, pitch shift, time stretch
  features.py        — STFT → Mel-spectrogram → log-dB → normalise → tensor
  batch.py           — folder → augmented tensors + manifest
model/
  dataset.py         — SignalDataset, make_label_map, load_splits
  cnn.py             — SoundClassifier (25K params, 3-class CNN)
  config.py          — TrainConfig hyperparameter dataclass
  train.py           — training loop, checkpointing, early stopping
  evaluate.py        — metrics, confusion matrix, loss curves
  predict.py         — raw .wav → class + confidence (--mode sound|note)
  transfer.py        — load Phase 2 backbone, swap head for N classes
  transfer_train.py  — transfer learning training loop, --compare flag
scripts/
  download_iowa_piano.py  — fetch Iowa piano samples, convert AIFF → WAV
tests/               — 129 unit tests across all modules
data/raw/            — source .wav files
  hum/, whistle/, clap/   — 11 each (Phase 2)
  notes/{C4..B4}/         — 3 per note, Iowa piano (Phase 3)
data/processed/      — generated .pt tensors (gitignored)
output/              — checkpoints and history (gitignored)
explainers/          — step-by-step notes on every concept
```
