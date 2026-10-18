"""Tests for pipeline.ingest — load, trim, pad/truncate."""

import numpy as np
import pytest

from pipeline.ingest import load_audio, trim_silence, pad_or_truncate, ingest, DEFAULT_SR


class TestLoadAudio:
    def test_returns_mono_float32(self, audio_fixtures):
        y, sr = load_audio(audio_fixtures / "mono_44100.wav")
        assert y.ndim == 1
        assert y.dtype == np.float32

    def test_resamples_to_target(self, audio_fixtures):
        y, sr = load_audio(audio_fixtures / "mono_44100.wav", target_sr=DEFAULT_SR)
        assert sr == DEFAULT_SR

    def test_stereo_to_mono(self, audio_fixtures):
        y, sr = load_audio(audio_fixtures / "stereo_44100.wav")
        assert y.ndim == 1


class TestTrimSilence:
    def test_removes_leading_trailing_silence(self, audio_fixtures):
        y, _ = load_audio(audio_fixtures / "silence_padded.wav")
        trimmed = trim_silence(y, top_db=20)
        assert len(trimmed) < len(y)
        assert len(trimmed) > 0


class TestPadOrTruncate:
    def test_truncates_long_signal(self):
        signal = np.ones(10000, dtype=np.float32)
        result = pad_or_truncate(signal, 5000)
        assert len(result) == 5000

    def test_pads_short_signal(self):
        signal = np.ones(1000, dtype=np.float32)
        result = pad_or_truncate(signal, 5000)
        assert len(result) == 5000
        assert result[999] == 1.0
        assert result[1000] == 0.0

    def test_exact_length_unchanged(self):
        signal = np.ones(5000, dtype=np.float32)
        result = pad_or_truncate(signal, 5000)
        assert len(result) == 5000
        np.testing.assert_array_equal(result, signal)


class TestIngest:
    def test_output_shape_matches_duration(self, audio_fixtures):
        y, sr = ingest(audio_fixtures / "mono_44100.wav", duration=1.5)
        expected = int(DEFAULT_SR * 1.5)
        assert len(y) == expected
        assert sr == DEFAULT_SR

    def test_dtype_float32(self, audio_fixtures):
        y, _ = ingest(audio_fixtures / "mono_44100.wav")
        assert y.dtype == np.float32
