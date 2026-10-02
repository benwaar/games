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

## Quick start

```bash
bash setup.sh              # install Python, venv, deps
source .venv/bin/activate
python hello_audio.py      # M1: verify everything works
```
