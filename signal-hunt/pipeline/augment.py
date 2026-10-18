"""Audio augmentation — composable transforms for training data diversity."""

import librosa
import numpy as np


def add_noise(signal: np.ndarray, snr_db: float = 20.0, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.standard_normal(len(signal)).astype(signal.dtype) * np.sqrt(noise_power)
    return signal + noise


def add_ambient(
    signal: np.ndarray,
    ambient: np.ndarray,
    snr_db: float = 15.0,
) -> np.ndarray:
    if len(ambient) < len(signal):
        repeats = int(np.ceil(len(signal) / len(ambient)))
        ambient = np.tile(ambient, repeats)
    ambient = ambient[: len(signal)].astype(signal.dtype)

    signal_power = np.mean(signal ** 2)
    ambient_power = np.mean(ambient ** 2)
    if ambient_power < 1e-10:
        return signal
    scale = np.sqrt(signal_power / (ambient_power * 10 ** (snr_db / 10)))
    return signal + ambient * scale


def pitch_shift(signal: np.ndarray, sr: int, n_steps: float = 2.0) -> np.ndarray:
    shifted = librosa.effects.pitch_shift(y=signal, sr=sr, n_steps=n_steps)
    return shifted.astype(signal.dtype)


def time_stretch(signal: np.ndarray, rate: float = 1.2) -> np.ndarray:
    stretched = librosa.effects.time_stretch(y=signal, rate=rate)
    return stretched.astype(signal.dtype)
