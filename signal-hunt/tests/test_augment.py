"""Tests for pipeline.augment — noise, ambient, pitch shift, time stretch."""

import numpy as np
import pytest

from pipeline.augment import add_noise, add_ambient, pitch_shift, time_stretch

SR = 22050


@pytest.fixture
def sine_signal() -> np.ndarray:
    t = np.linspace(0, 1.0, SR, endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


class TestAddNoise:
    def test_output_shape_matches_input(self, sine_signal):
        result = add_noise(sine_signal, snr_db=20)
        assert result.shape == sine_signal.shape

    def test_output_differs_from_input(self, sine_signal):
        result = add_noise(sine_signal, snr_db=20, rng=np.random.default_rng(0))
        assert not np.array_equal(result, sine_signal)

    def test_deterministic_with_rng(self, sine_signal):
        a = add_noise(sine_signal, snr_db=20, rng=np.random.default_rng(42))
        b = add_noise(sine_signal, snr_db=20, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(a, b)

    def test_higher_snr_means_less_noise(self, sine_signal):
        loud = add_noise(sine_signal, snr_db=5, rng=np.random.default_rng(0))
        quiet = add_noise(sine_signal, snr_db=40, rng=np.random.default_rng(0))
        diff_loud = np.mean((loud - sine_signal) ** 2)
        diff_quiet = np.mean((quiet - sine_signal) ** 2)
        assert diff_loud > diff_quiet


class TestAddAmbient:
    def test_output_shape_matches_input(self, sine_signal):
        ambient = np.random.default_rng(0).standard_normal(SR).astype(np.float32) * 0.1
        result = add_ambient(sine_signal, ambient, snr_db=15)
        assert result.shape == sine_signal.shape

    def test_short_ambient_gets_tiled(self, sine_signal):
        short_ambient = np.ones(100, dtype=np.float32) * 0.1
        result = add_ambient(sine_signal, short_ambient, snr_db=15)
        assert result.shape == sine_signal.shape

    def test_silent_ambient_returns_original(self, sine_signal):
        silent = np.zeros(SR, dtype=np.float32)
        result = add_ambient(sine_signal, silent, snr_db=15)
        np.testing.assert_array_equal(result, sine_signal)


class TestPitchShift:
    def test_output_shape_matches_input(self, sine_signal):
        result = pitch_shift(sine_signal, sr=SR, n_steps=2.0)
        assert result.shape == sine_signal.shape

    def test_output_dtype(self, sine_signal):
        result = pitch_shift(sine_signal, sr=SR, n_steps=-3.0)
        assert result.dtype == np.float32


class TestTimeStretch:
    def test_faster_produces_shorter(self, sine_signal):
        result = time_stretch(sine_signal, rate=1.5)
        assert len(result) < len(sine_signal)

    def test_slower_produces_longer(self, sine_signal):
        result = time_stretch(sine_signal, rate=0.8)
        assert len(result) > len(sine_signal)

    def test_output_dtype(self, sine_signal):
        result = time_stretch(sine_signal, rate=1.2)
        assert result.dtype == np.float32
