# Explainers

Brief notes on each step of the pipeline — what it does and why. Detailed explainers linked where the concept needs more depth.

---

## 1. Audio Loading & Resampling

Raw audio comes in many formats, sample rates, and channel layouts. The ingestion step normalises all of that:

- **Resample to 22050 Hz** — a standard rate that captures frequencies up to ~11 kHz (Nyquist). Human speech and most musical content sit well below this. Higher rates waste compute; lower rates lose detail.
- **Mono conversion** — stereo channels carry spatial info we don't need. Averaging to mono halves the data and keeps the model focused on *what* sounds are present, not *where*.
- **Silence trimming** — leading/trailing silence varies by recording. Trimming it means the model sees signal, not dead air.
- **Fixed-length output** — pad short clips with zeros, truncate long ones. Consistent tensor shapes simplify batching and model architecture.

See: [pipeline/ingest.py](../pipeline/ingest.py)

---

## 2. Augmentation

A model trained on clean recordings fails on real-world audio. Augmentation injects realistic variation at the data layer so the model learns to generalise.

- **White noise** (`add_noise`) — simulates mic hiss and background static. Controlled by SNR (signal-to-noise ratio in dB) — higher SNR = cleaner signal.
- **Ambient mixing** (`add_ambient`) — blends in a real ambient recording (room tone, traffic, etc.) at a target SNR. More realistic than synthetic noise.
- **Pitch shift** (`pitch_shift`) — shifts frequency up/down by N semitones without changing speed. Simulates different voices or instruments.
- **Time stretch** (`time_stretch`) — speeds up or slows down without changing pitch. Simulates tempo variation.

All augmentations are composable — chain them in any order. Applied *before* feature extraction so the model trains on diverse inputs.

→ [Why SNR matters](snr.md)

See: [pipeline/augment.py](../pipeline/augment.py)

---

## 3. Mel-Spectrograms

*Coming in M4.* Raw waveforms are high-dimensional and hard for models to learn from. Mel-spectrograms compress frequency into perceptually meaningful bands — matching how humans hear.

→ [Mel-spectrograms explained](mel-spectrograms.md) *(to write in M4)*

---

## 4. Normalisation

*Coming in M4.* Different recordings have different volumes. Without normalisation (mean=0, std=1), the same sound at different gains looks completely different to the model.

---

## 5. CNN Feature Extraction

*Coming in M8.* Convolutional layers scan the spectrogram for local frequency patterns — the shapes that distinguish a hum from a clap.

---

## 6. RNN Temporal Learning

*Coming in M9.* Recurrent layers learn how features change over time — the sequence that makes a whistle different from a sustained hum.
