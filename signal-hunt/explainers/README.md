# Explainers

Brief notes on each step of the pipeline — what it does and why. Detailed explainers linked where the concept needs more depth.

→ [Key Libraries](libraries.md) — what each dependency does and when you'd reach for it
→ [Python Concepts](python-concepts.md) — tuples, type hints, fixtures, and other patterns used in this codebase

---

## 1. Audio Loading & Resampling

Raw audio comes in many formats, sample rates, and channel layouts. The ingestion step normalises all of that:

- **Resample to 22050 Hz** — a standard rate that captures frequencies up to ~11 kHz (Nyquist). Human speech and most musical content sit well below this. Higher rates waste compute; lower rates lose detail.
  - https://share.google/ZVNmS9fvoroRms70g 
- **Mono conversion** — stereo channels carry spatial info we don't need. Averaging to mono halves the data and keeps the model focused on *what* sounds are present, not *where*.
- **Silence trimming** — leading/trailing silence varies by recording. Trimming it means the model sees signal, not dead air.
- **Fixed-length output** — pad short clips with zeros, truncate long ones. Consistent tensor shapes simplify batching and model architecture.

> **In practice:** This is the same problem as normalising data from different sources. Financial tick data arrives at different rates per exchange; IoT sensors report at different intervals; customer event logs have different schemas. The ingestion step — resample to a common rate, trim noise, enforce a fixed shape — is the universal first step in any ML pipeline.

See: [pipeline/ingest.py](../pipeline/ingest.py)

---

## 1a. Reading Waveforms & Spectrograms

The two core visualisations for understanding audio. A waveform shows amplitude over time (when things happen); a spectrogram shows frequency over time with brightness for intensity (what frequencies are present). If you can read these, you can debug every step of the pipeline.

→ [How to read waveforms and spectrograms](reading-plots.md) — with annotated examples from our pipeline output

---

## 2. Augmentation

A model trained on clean recordings fails on real-world audio. Augmentation injects realistic variation at the data layer so the model learns to generalise.

- **White noise** (`add_noise`) — simulates mic hiss and background static. Controlled by SNR (signal-to-noise ratio in dB) — higher SNR = cleaner signal.
- **Ambient mixing** (`add_ambient`) — blends in a real ambient recording (room tone, traffic, etc.) at a target SNR. More realistic than synthetic noise.
- **Pitch shift** (`pitch_shift`) — shifts frequency up/down by N semitones without changing speed. Simulates different voices or instruments.
- **Time stretch** (`time_stretch`) — speeds up or slows down without changing pitch. Simulates tempo variation.

All augmentations are composable — chain them in any order. Applied *before* feature extraction so the model trains on diverse inputs.

> **In practice:** Augmentation is how you deal with imbalanced or small datasets in any domain. In fraud detection, you synthesise rare fraud patterns to stop the model ignoring them. In medical imaging, you rotate and flip scans to multiply your training data. The principle is the same: inject realistic variation at the data layer so the model generalises instead of memorising.

→ [Why SNR matters](snr.md)

See: [pipeline/augment.py](../pipeline/augment.py)

---

## 3. Mel-Spectrograms

This is the core transform — it takes a raw waveform and produces a compact, CNN-ready tensor. A 33,075-sample clip becomes a `(1, 128, 65)` PyTorch tensor (one channel, 128 Mel frequency bands, 65 time frames) with mean≈0 and std≈1. Each step throws away information the model doesn't need and amplifies what it does.

The pipeline compresses the waveform into a `(128, 65)` perceptually-weighted frequency-over-time image:

1. **STFT** — break the waveform into overlapping windows, FFT each one → frequency bins × time frames
2. **Mel filterbank** — compress 1025 linear frequency bins into 128 Mel-scaled bands (fine detail at low frequencies, coarse at high — matching human hearing)
3. **Log-dB** — convert power to decibels so quiet sounds are visible alongside loud ones

The result looks like a greyscale image: a hum is a horizontal line, a whistle chirp is a rising diagonal, a clap is a vertical broadband burst. These visually distinct patterns are what the CNN will learn to classify.

> **In practice:** This is dimensionality reduction with domain knowledge baked in — like using business-informed feature engineering instead of dumping raw columns into a model. The Mel scale encodes what we know about human perception, just as a financial feature pipeline might encode what we know about market microstructure.

→ [Mel-spectrograms explained](mel-spectrograms.md) — STFT, Mel scale, log-dB, with C/JS/TS parallels and annotated output from our samples

See: [pipeline/features.py](../pipeline/features.py) — `stft()`, `mel_spectrogram()`, `to_log_db()`

---

## 4. Normalisation

Different recordings have different absolute volumes. A hum on a laptop mic vs. a studio condenser mic produces wildly different dB ranges — but they're the same sound. Per-spectrogram normalisation (mean=0, std=1) removes this bias so the model sees *relative patterns*, not absolute levels.

Edge case: a completely silent signal has std ≈ 0. Dividing by near-zero would explode, so the code returns all-zeros for silence — no features means no features.

> **In practice:** This is z-score / `StandardScaler` — the same normalisation you'd apply to any ML feature set. Stock prices and trading volumes differ by orders of magnitude; temperature and pressure readings do too. Without scaling, the model fixates on whichever feature has the biggest numbers.

See: [pipeline/features.py](../pipeline/features.py) — `normalise()`, `extract_features()`

---

## 5. CNN Feature Extraction

*Coming in M8.* Convolutional layers scan the spectrogram for local frequency patterns — the shapes that distinguish a hum from a clap.

---

## 6. RNN Temporal Learning

*Coming in M9.* Recurrent layers learn how features change over time — the sequence that makes a whistle different from a sustained hum.
