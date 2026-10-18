"""Audio ingestion — load, resample, trim, and normalise audio files."""

import librosa
import numpy as np
from pathlib import Path

DEFAULT_SR = 22050
DEFAULT_DURATION = 1.5


def load_audio(path: str | Path, target_sr: int = DEFAULT_SR) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(str(path), sr=target_sr, mono=True)
    return y.astype(np.float32), sr


def trim_silence(signal: np.ndarray, top_db: float = 20.0) -> np.ndarray:
    trimmed, _ = librosa.effects.trim(signal, top_db=top_db)
    return trimmed


def pad_or_truncate(signal: np.ndarray, target_length: int) -> np.ndarray:
    if len(signal) >= target_length:
        return signal[:target_length]
    padding = np.zeros(target_length - len(signal), dtype=signal.dtype)
    return np.concatenate([signal, padding])


def ingest(
    path: str | Path,
    target_sr: int = DEFAULT_SR,
    duration: float = DEFAULT_DURATION,
    trim: bool = True,
    top_db: float = 20.0,
) -> tuple[np.ndarray, int]:
    y, sr = load_audio(path, target_sr)
    if trim:
        y = trim_silence(y, top_db)
    target_length = int(sr * duration)
    y = pad_or_truncate(y, target_length)
    return y, sr
