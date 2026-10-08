"""Tests for model/progression.py — CNN-RNN chord progression classifier."""

import torch
import pytest

from model.cnn import SoundClassifier
from model.progression import ProgressionClassifier, ProgressionConfig, load_progression_model


def _make_backbone() -> SoundClassifier:
    return SoundClassifier(num_classes=6, dropout=0.0)


class TestProgressionClassifier:
    def test_output_shape(self):
        model = ProgressionClassifier(
            backbone=_make_backbone(),
            num_progressions=4,
            gru_hidden=32,
        )
        x = torch.randn(2, 1, 128, 291)
        out = model(x)
        assert out.shape == (2, 4)

    def test_variable_time_dim(self):
        model = ProgressionClassifier(
            backbone=_make_backbone(),
            num_progressions=4,
            gru_hidden=32,
        )
        model.eval()
        for t in [65, 150, 291, 400]:
            out = model(torch.randn(1, 1, 128, t))
            assert out.shape == (1, 4), f"Failed for T={t}"

    def test_num_progressions_respected(self):
        for n in [2, 4, 8]:
            model = ProgressionClassifier(
                backbone=_make_backbone(),
                num_progressions=n,
                gru_hidden=32,
            )
            out = model(torch.randn(1, 1, 128, 100))
            assert out.shape == (1, n)

    def test_no_nan_in_output(self):
        model = ProgressionClassifier(
            backbone=_make_backbone(),
            num_progressions=4,
            gru_hidden=64,
        )
        out = model(torch.randn(3, 1, 128, 291))
        assert not out.isnan().any()

    def test_trainable_params_nonzero(self):
        model = ProgressionClassifier(
            backbone=_make_backbone(),
            num_progressions=4,
            gru_hidden=64,
        )
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable > 0

    def test_bidirectional_doubles_hidden(self):
        hidden = 32
        model = ProgressionClassifier(
            backbone=_make_backbone(),
            num_progressions=4,
            gru_hidden=hidden,
        )
        # head input = hidden * 2 (bidirectional)
        assert model.head.in_features == hidden * 2


@pytest.mark.skipif(
    not __import__("pathlib").Path("output/chords/name/best_model.pt").exists(),
    reason="Phase 4 chord checkpoint not found",
)
def test_load_progression_model_shape():
    config = ProgressionConfig()
    model = load_progression_model(config)
    x = torch.randn(2, 1, 128, 291)
    assert model(x).shape == (2, config.num_progressions)
