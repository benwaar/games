# Signal Hunt — Principles

Carried forward as the project progresses.

---

## Learning first, speed second
This is a study project. When there's a choice between the fast way and the way that teaches more, pick the one that teaches. Cut corners on polish, not understanding.

## Document as you learn
When a concept clicks — why Mel-spectrograms work, what a stride does in a conv layer, why normalisation matters — write it down in `explainers/`. Short, plain-language notes for future-you. If you can't explain it simply, you don't understand it yet.

## Mel-spectrograms over raw waveforms
Raw audio is high-dimensional and noisy. Mel-spectrograms compress frequency into perceptually meaningful bands — what a CNN needs to spot patterns humans hear.

## Normalise everything
Mean=0, std=1 per spectrogram. Without this, different mic gains and recording volumes make the same sound look completely different to the model.

## Augment before you model
Real-world variance (noise, pitch, speed) should be injected at the data layer, not handled by model complexity. A simple model on good data beats a complex model on clean-room data.

## Fixed-length tensors
Pad short clips, truncate long ones to a standard frame count. Variable-length inputs complicate batching and model architecture for no gain at this stage.

## File-first, mic-second
The core pipeline runs on audio files. Mic recording is a convenience wrapper, not the critical path. Don't let PortAudio issues block progress.

## Python 3.12 for this project
Sunder pinned to 3.11 because of pywasm constraints. Signal Hunt has no such dependency — use current stable.
