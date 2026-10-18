"""Tests for model.cnn — SoundClassifier forward pass, shapes, params, softmax."""

import torch
import pytest
from model.cnn import SoundClassifier

BATCH = 4
INPUT = (BATCH, 1, 128, 65)


@pytest.fixture
def model():
    m = SoundClassifier()
    m.eval()
    return m


@pytest.fixture
def dummy_batch():
    return torch.randn(*INPUT)


class TestForwardPass:
    def test_output_shape(self, model, dummy_batch):
        out = model(dummy_batch)
        assert out.shape == (BATCH, 3)

    def test_no_nans(self, model, dummy_batch):
        out = model(dummy_batch)
        assert not torch.isnan(out).any()

    def test_outputs_vary_across_batch(self, model):
        # Different inputs should produce different outputs
        x = torch.randn(*INPUT)
        out = model(x)
        assert not torch.all(out[0] == out[1])

    def test_single_item_batch(self, model):
        x = torch.randn(1, 1, 128, 65)
        out = model(x)
        assert out.shape == (1, 3)


class TestParameterCount:
    def test_under_500k(self, model):
        assert model.num_parameters() < 500_000

    def test_parameters_logged(self, model, capsys):
        n = model.num_parameters()
        print(f"{n:,} trainable parameters")
        captured = capsys.readouterr()
        assert "parameters" in captured.out


class TestSoftmax:
    def test_softmax_sums_to_one(self, model, dummy_batch):
        out = model(dummy_batch)
        probs = torch.softmax(out, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(BATCH), atol=1e-5)

    def test_softmax_all_positive(self, model, dummy_batch):
        out = model(dummy_batch)
        probs = torch.softmax(out, dim=1)
        assert (probs > 0).all()


class TestTrainVsEval:
    def test_dropout_active_in_train_mode(self):
        model = SoundClassifier(dropout=0.9)
        model.train()
        x = torch.ones(*INPUT)
        out1 = model(x)
        out2 = model(x)
        # With high dropout, two forward passes should differ during training
        assert not torch.allclose(out1, out2)

    def test_dropout_inactive_in_eval_mode(self, model):
        x = torch.ones(*INPUT)
        out1 = model(x)
        out2 = model(x)
        assert torch.allclose(out1, out2)
