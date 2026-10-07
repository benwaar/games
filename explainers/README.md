# Explainers

Shared learning docs covering concepts used across all projects in this programme. Each file explains *why* a technique exists, how it works, and where it appears in business problems — not just what it does.

Project-specific notes (dataset details, per-experiment results) live inside each project folder.

---

## Python & PyTorch foundations

→ [Python Concepts](python-concepts.md) — tuples, type hints, dataclasses, dunder methods, comprehensions, fixtures — patterns used throughout this codebase, with C/JS/TS callouts

→ [Key Libraries](libraries.md) — NumPy, librosa, PyTorch (`nn`, `optim`, `DataLoader`), sklearn — what each does, when to reach for it, checkpoint save/load, `requires_grad`

---

## Data pipelines

→ [Data Collection](data-collection.md) — why real recordings, labelling strategy, how to add more

→ [Batch Processing](batch-processing.md) — augmentation strategy, manifest schema, DataLoader integration

→ [Mel-Spectrograms](mel-spectrograms.md) — STFT → Mel scale → log-dB → tensor, with annotated output and C/JS/TS parallels

→ [Signal-to-Noise Ratio](snr.md) — what SNR is, why it matters for augmentation, how to control it

→ [Reading Waveforms & Spectrograms](reading-plots.md) — how to read the two core audio visualisations

---

## Model training

→ [Train / Val / Test Split](train-val-test-split.md) — why three sets, stratification, the no-peeking rule

→ [Dataset & DataLoader](dataset-dataloader.md) — the PyTorch Dataset protocol, label encoding, batching

→ [Training Loop](training-loop.md) — forward pass, loss, backprop, Adam, scheduling, early stopping

→ [Evaluation](evaluation.md) — precision, recall, F1, confusion matrix, loss curves

---

## Architectures

→ [CNN Architecture](cnn-architecture.md) — Conv2d, BatchNorm, ReLU, MaxPool, Global Average Pooling, parameter count

→ [Transfer Learning](transfer-learning.md) — frozen backbone, head swap, fine-tune vs frozen, scratch vs transfer comparison, what actually happened in Signal Hunt Phase 3

→ [Reinforcement Learning](reinforcement-learning.md) — MDP, Q-learning, TD learning, DQN (replay buffer, target network, ε-greedy), curriculum training, distillation, self-play — from Utala KAOS 9

---

## Where each explainer came from

| Explainer | Project | Phase |
|-----------|---------|-------|
| Python Concepts | Signal Hunt | Phase 1–3 |
| Libraries | Signal Hunt | Phase 1–3 |
| Data Collection | Signal Hunt | Phase 1 |
| Batch Processing | Signal Hunt | Phase 1 |
| Mel-Spectrograms | Signal Hunt | Phase 1 |
| SNR | Signal Hunt | Phase 1 |
| Reading Plots | Signal Hunt | Phase 1 |
| Train/Val/Test | Signal Hunt | Phase 2 |
| Dataset & DataLoader | Signal Hunt | Phase 2 |
| Training Loop | Signal Hunt | Phase 2 |
| Evaluation | Signal Hunt | Phase 2 |
| CNN Architecture | Signal Hunt | Phase 2 |
| Transfer Learning | Signal Hunt | Phase 3 |
| Reinforcement Learning | Utala KAOS 9 | Phases 1–5 |
