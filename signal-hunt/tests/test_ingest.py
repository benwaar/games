"""Tests for pipeline.ingest — load, trim, pad/truncate."""

import numpy as np
import pytest
import soundfile as sf
from pathlib import Path

from pipeline.ingest import load_audio, trim_silence, pad_or_truncate, ingest, DEFAULT_SR

FIXTURES = Path("tests/fixtures")


@pytest.fixture(autouse=True, scope="module")
def create_fixtures():
    FIXTURES.mkdir(parents=True, exist_ok=True)

    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    mono = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(str(FIXTURES / "mono_44100.wav"), mono, sr)

    stereo = np.column_stack([mono, mono * 0.8])
    sf.write(str(FIXTURES / "stereo_44100.wav"), stereo, sr)

    quiet = np.zeros(sr, dtype=np.float32)
    loud_tone = (0.8 * np.sin(2 * np.pi * 440 * t[:int(sr * 0.5)])).astype(np.float32)
    padded = np.concatenate([quiet[:int(sr * 0.2)], loud_tone, quiet[:int(sr * 0.2)]])
    sf.write(str(FIXTURES / "silence_padded.wav"), padded, sr)


class TestLoadAudio:
    def test_returns_mono_float32(self):
        y, sr = load_audio(FIXTURES / "mono_44100.wav")
        assert y.ndim == 1
        assert y.dtype == np.float32

    def test_resamples_to_target(self):
        y, sr = load_audio(FIXTURES / "mono_44100.wav", target_sr=DEFAULT_SR)
        assert sr == DEFAULT_SR

    def test_stereo_to_mono(self):
        y, sr = load_audio(FIXTURES / "stereo_44100.wav")
        assert y.ndim == 1


class TestTrimSilence:
    def test_removes_leading_trailing_silence(self):
        y, _ = load_audio(FIXTURES / "silence_padded.wav")
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
    def test_output_shape_matches_duration(self):
        y, sr = ingest(FIXTURES / "mono_44100.wav", duration=1.5)
        expected = int(DEFAULT_SR * 1.5)
        assert len(y) == expected
        assert sr == DEFAULT_SR

    def test_dtype_float32(self):
        y, _ = ingest(FIXTURES / "mono_44100.wav")
        assert y.dtype == np.float32
