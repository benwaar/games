# Signal Hunt

Signal Hunt is a deep learning project — from raw audio to a trained classifier.

Turn raw audio (hums, whistles, claps) into clean Mel-spectrogram tensors, then train a hybrid CNN-RNN to classify them.

## What this builds

### Phase 1: Data Pipelines & Feature Extraction
- **Ingestion wrapper** — load or record 1–2s audio clips
- **Augmentation engine** — noise injection, pitch shift, time stretch
- **Feature extraction** — STFT → Mel-spectrogram → normalised PyTorch tensor
- **Batch pipeline** — process folders of audio into `.pt` dataset files

### Phase 2: Core Model Architecture & Training
- **Hybrid CNN-RNN model** — CNN extracts frequency shapes, RNN learns temporal sequences
- **Training loop** — loss, backprop, validation, early stopping, checkpoints
- **Evaluation** — accuracy, confusion matrix, overfitting analysis

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
python -m pipeline.batch data/raw data/processed
```

Takes every `.wav` in `data/raw/`, runs it through ingest → augment (7 variants) → feature extraction, and saves `.pt` tensors + a `manifest.json` to `data/processed/`. Output tensors are `(1, 128, 65)` — one channel, 128 Mel bands, 65 time frames.

### Run tests

```bash
python -m pytest tests/ -v
```

### Project structure

```
pipeline/
  ingest.py      — load, resample, trim, pad/truncate audio
  augment.py     — noise, ambient, pitch shift, time stretch
  features.py    — STFT → Mel-spectrogram → log-dB → normalise → tensor
  batch.py       — folder → augmented tensors + manifest
tests/             — unit tests for each module
data/raw/          — source .wav files (hum, whistle, clap)
data/processed/    — generated .pt tensors (gitignored)
explainers/        — step-by-step notes on each part of the pipeline
```
