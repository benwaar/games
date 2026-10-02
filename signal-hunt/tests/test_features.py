"""Tests for pipeline.features — STFT, Mel-spectrogram, normalisation, tensor output."""

import numpy as np
import pytest
import torch

from pipeline.features import (
    stft,
    mel_spectrogram,
    to_log_db,
    normalise,
    pad_or_truncate_frames,
    to_tensor,
    extract_features,
)

SR = 22050
DURATION = 1.5


@pytest.fixture
def sine_signal() -> np.ndarray:
    t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.fixture
def silent_signal() -> np.ndarray:
    return np.zeros(int(SR * DURATION), dtype=np.float32)


class TestSTFT:
    def test_output_shape(self, sine_signal):
        S = stft(sine_signal)
        assert S.shape[0] == 1025  # n_fft/2 + 1
        assert S.shape[1] > 0

    def test_output_dtype(self, sine_signal):
        S = stft(sine_signal)
        assert S.dtype == np.float32

    def test_magnitudes_non_negative(self, sine_signal):
        S = stft(sine_signal)
        assert np.all(S >= 0)


class TestMelSpectrogram:
    def test_output_shape(self, sine_signal):
        S = mel_spectrogram(sine_signal, SR)
        assert S.shape[0] == 128  # default n_mels
        assert S.shape[1] > 0

    def test_custom_n_mels(self, sine_signal):
        S = mel_spectrogram(sine_signal, SR, n_mels=64)
        assert S.shape[0] == 64

    def test_output_dtype(self, sine_signal):
        S = mel_spectrogram(sine_signal, SR)
        assert S.dtype == np.float32


class TestToLogDb:
    def test_output_dtype(self, sine_signal):
        S = mel_spectrogram(sine_signal, SR)
        db = to_log_db(S)
        assert db.dtype == np.float32

    def test_output_shape_preserved(self, sine_signal):
        S = mel_spectrogram(sine_signal, SR)
        db = to_log_db(S)
        assert db.shape == S.shape

    def test_values_in_reasonable_range(self, sine_signal):
        S = mel_spectrogram(sine_signal, SR)
        db = to_log_db(S)
        assert db.max() < 80  # Mel bins concentrate energy, so dB can exceed 0
        assert db.min() >= -100  # amin=1e-10 floors at -100 dB


class TestNormalise:
    def test_mean_near_zero(self, sine_signal):
        S = mel_spectrogram(sine_signal, SR)
        db = to_log_db(S)
        normed = normalise(db)
        assert abs(normed.mean()) < 1e-5

    def test_std_near_one(self, sine_signal):
        S = mel_spectrogram(sine_signal, SR)
        db = to_log_db(S)
        normed = normalise(db)
        assert abs(normed.std() - 1.0) < 1e-5

    def test_silent_signal_returns_zeros(self, silent_signal):
        S = mel_spectrogram(silent_signal, SR)
        db = to_log_db(S)
        normed = normalise(db)
        assert np.all(normed == 0)

    def test_output_dtype(self, sine_signal):
        S = mel_spectrogram(sine_signal, SR)
        normed = normalise(to_log_db(S))
        assert normed.dtype == np.float32


class TestPadOrTruncateFrames:
    def test_truncate(self):
        spec = np.ones((128, 100), dtype=np.float32)
        result = pad_or_truncate_frames(spec, 50)
        assert result.shape == (128, 50)

    def test_pad(self):
        spec = np.ones((128, 30), dtype=np.float32)
        result = pad_or_truncate_frames(spec, 50)
        assert result.shape == (128, 50)

    def test_exact_length_unchanged(self):
        spec = np.ones((128, 50), dtype=np.float32)
        result = pad_or_truncate_frames(spec, 50)
        assert result.shape == (128, 50)
        np.testing.assert_array_equal(result, spec)

    def test_padding_uses_min_value(self):
        spec = np.full((4, 3), 5.0, dtype=np.float32)
        result = pad_or_truncate_frames(spec, 6)
        assert np.all(result[:, 3:] == 5.0)


class TestToTensor:
    def test_output_is_torch_tensor(self, sine_signal):
        S = mel_spectrogram(sine_signal, SR)
        t = to_tensor(S)
        assert isinstance(t, torch.Tensor)

    def test_channel_dimension_added(self, sine_signal):
        S = mel_spectrogram(sine_signal, SR)
        t = to_tensor(S)
        assert t.dim() == 3
        assert t.shape[0] == 1
        assert t.shape[1] == S.shape[0]
        assert t.shape[2] == S.shape[1]


class TestExtractFeatures:
    def test_output_shape(self, sine_signal):
        t = extract_features(sine_signal, SR)
        assert t.dim() == 3
        assert t.shape[0] == 1       # channel
        assert t.shape[1] == 128     # n_mels

    def test_with_target_frames(self, sine_signal):
        t = extract_features(sine_signal, SR, target_frames=65)
        assert t.shape == (1, 128, 65)

    def test_normalisation_applied(self, sine_signal):
        t = extract_features(sine_signal, SR)
        assert abs(t.mean().item()) < 1e-5
        assert abs(t.std().item() - 1.0) < 1e-4

    def test_output_dtype(self, sine_signal):
        t = extract_features(sine_signal, SR)
        assert t.dtype == torch.float32

    def test_silent_input(self, silent_signal):
        t = extract_features(silent_signal, SR)
        assert torch.all(t == 0)
