# M4: Feature Extraction

Build `pipeline/features.py` — STFT, Mel-spectrogram, log-dB conversion, normalisation. Output a PyTorch tensor ready for a CNN.

## What to build

- `mel_spectrogram(signal, sr, n_mels=128)` — compute Mel-spectrogram in log-dB scale
- `normalise(spectrogram)` — zero mean, unit variance per spectrogram
- `to_tensor(spectrogram)` — wrap as PyTorch tensor with channel dim `(1, n_mels, time_frames)`

## Gate

Unit tests verify:
- Tensor shape `(1, n_mels, time_frames)`
- Mean ≈ 0, std ≈ 1 after normalisation
- Spectrogram plot saved for visual sanity check
