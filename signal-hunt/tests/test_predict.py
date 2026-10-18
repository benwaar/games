"""Tests for model.predict — predict function and output shape."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from model.predict import predict

SR = 22050
DURATION = 1.5
_REAL_CHECKPOINT = Path("output/best_model.pt")


@pytest.fixture
def hum_wav(tmp_path):
    """Synthetic hum — sine at 440 Hz."""
    t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)
    signal = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = tmp_path / "hum.wav"
    sf.write(path, signal, SR)
    return path


class TestPredict:
    def test_returns_dict_with_expected_keys(self, hum_wav, fake_phase2_checkpoint):
        result = predict(hum_wav, fake_phase2_checkpoint)
        assert "class" in result
        assert "confidence" in result
        assert "scores" in result

    def test_class_is_valid(self, hum_wav, fake_phase2_checkpoint):
        result = predict(hum_wav, fake_phase2_checkpoint)
        assert result["class"] in {"clap", "hum", "whistle"}

    def test_confidence_in_range(self, hum_wav, fake_phase2_checkpoint):
        result = predict(hum_wav, fake_phase2_checkpoint)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_scores_sum_to_one(self, hum_wav, fake_phase2_checkpoint):
        result = predict(hum_wav, fake_phase2_checkpoint)
        assert abs(sum(result["scores"].values()) - 1.0) < 1e-4

    def test_scores_has_all_classes(self, hum_wav, fake_phase2_checkpoint):
        result = predict(hum_wav, fake_phase2_checkpoint)
        assert set(result["scores"].keys()) == {"clap", "hum", "whistle"}

    def test_top_class_matches_max_score(self, hum_wav, fake_phase2_checkpoint):
        result = predict(hum_wav, fake_phase2_checkpoint)
        assert result["class"] == max(result["scores"], key=result["scores"].__getitem__)

    @pytest.mark.skipif(
        not _REAL_CHECKPOINT.exists(),
        reason="requires trained model at output/best_model.pt",
    )
    def test_sine_predicts_hum(self, hum_wav):
        """A pure 440 Hz sine should look like a hum spectrogram."""
        result = predict(hum_wav, _REAL_CHECKPOINT)
        assert result["class"] == "hum"
