"""Feature extraction — STFT, Mel-spectrogram, log-dB, normalisation."""

import librosa
import numpy as np
import torch

DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512
DEFAULT_N_MELS = 128


def stft(signal: np.ndarray, n_fft: int = DEFAULT_N_FFT, hop_length: int = DEFAULT_HOP_LENGTH) -> np.ndarray:
    S = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    return np.abs(S).astype(np.float32)


def mel_spectrogram(
    signal: np.ndarray,
    sr: int,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    n_mels: int = DEFAULT_N_MELS,
) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
    )
    return S.astype(np.float32)


def to_log_db(spectrogram: np.ndarray, ref: float = 1.0, amin: float = 1e-10) -> np.ndarray:
    return librosa.power_to_db(spectrogram, ref=ref, amin=amin).astype(np.float32)


def normalise(spectrogram: np.ndarray) -> np.ndarray:
    mean = spectrogram.mean()
    std = spectrogram.std()
    if std < 1e-8:
        return np.zeros_like(spectrogram)
    return ((spectrogram - mean) / std).astype(np.float32)


def pad_or_truncate_frames(spectrogram: np.ndarray, target_frames: int) -> np.ndarray:
    _, current_frames = spectrogram.shape
    if current_frames >= target_frames:
        return spectrogram[:, :target_frames]
    padding = np.zeros(
        (spectrogram.shape[0], target_frames - current_frames),
        dtype=spectrogram.dtype,
    )
    return np.concatenate([spectrogram, padding], axis=1)


def to_tensor(spectrogram: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(spectrogram).unsqueeze(0)


def extract_features(
    signal: np.ndarray,
    sr: int,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    n_mels: int = DEFAULT_N_MELS,
    target_frames: int | None = None,
) -> torch.Tensor:
    mel = mel_spectrogram(signal, sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    db = to_log_db(mel)
    if target_frames is not None:
        db = pad_or_truncate_frames(db, target_frames)
    normed = normalise(db)
    return to_tensor(normed)
