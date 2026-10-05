"""Tests for model.train — train_one_epoch, evaluate, full training run."""

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model.config import TrainConfig
from model.cnn import SoundClassifier
from model.train import evaluate, train, train_one_epoch


def _make_loader(n: int = 32, num_classes: int = 3) -> DataLoader:
    """Synthetic DataLoader — random tensors with balanced labels."""
    x = torch.randn(n, 1, 128, 65)
    y = torch.arange(n) % num_classes
    return DataLoader(TensorDataset(x, y), batch_size=8)


@pytest.fixture
def model():
    return SoundClassifier()


@pytest.fixture
def criterion():
    return nn.CrossEntropyLoss()


@pytest.fixture
def optimiser(model):
    return torch.optim.Adam(model.parameters(), lr=1e-3)


class TestTrainOneEpoch:
    def test_returns_loss_and_accuracy(self, model, criterion, optimiser):
        loss, acc = train_one_epoch(model, _make_loader(), criterion, optimiser)
        assert isinstance(loss, float)
        assert isinstance(acc, float)

    def test_loss_is_positive(self, model, criterion, optimiser):
        loss, _ = train_one_epoch(model, _make_loader(), criterion, optimiser)
        assert loss > 0

    def test_accuracy_in_range(self, model, criterion, optimiser):
        _, acc = train_one_epoch(model, _make_loader(), criterion, optimiser)
        assert 0.0 <= acc <= 1.0

    def test_weights_change_after_step(self, model, criterion, optimiser):
        before = model.classifier[-1].weight.data.clone()
        train_one_epoch(model, _make_loader(), criterion, optimiser)
        after = model.classifier[-1].weight.data
        assert not torch.allclose(before, after)


class TestEvaluate:
    def test_returns_loss_and_accuracy(self, model, criterion):
        loss, acc = evaluate(model, _make_loader(), criterion)
        assert isinstance(loss, float) and isinstance(acc, float)

    def test_deterministic(self, model, criterion):
        loader = _make_loader(n=16)
        loss1, acc1 = evaluate(model, loader, criterion)
        loss2, acc2 = evaluate(model, loader, criterion)
        assert loss1 == loss2 and acc1 == acc2

    def test_model_stays_in_eval_mode(self, model, criterion):
        model.eval()
        evaluate(model, _make_loader(), criterion)
        assert not model.training


class TestFullTrainingRun:
    def test_checkpoint_written(self, tmp_path):
        config = TrainConfig(epochs=2, processed_dir=Path("data/processed"),
                             output_dir=tmp_path, seed=42)
        train(config)
        assert (tmp_path / "best_model.pt").exists()

    def test_history_written(self, tmp_path):
        config = TrainConfig(epochs=2, processed_dir=Path("data/processed"),
                             output_dir=tmp_path, seed=42)
        train(config)
        history = json.loads((tmp_path / "history.json").read_text())
        assert len(history) == 2
        assert "train_loss" in history[0]
        assert "val_loss" in history[0]
        assert "val_acc" in history[0]

    def test_checkpoint_contains_label_map(self, tmp_path):
        config = TrainConfig(epochs=1, processed_dir=Path("data/processed"),
                             output_dir=tmp_path, seed=42)
        train(config)
        ckpt = torch.load(tmp_path / "best_model.pt", weights_only=False)
        assert "label_map" in ckpt
        assert set(ckpt["label_map"].keys()) == {"clap", "hum", "whistle"}

    def test_early_stopping(self, tmp_path):
        # patience=1 means stop after 1 epoch with no improvement
        config = TrainConfig(epochs=20, patience=1,
                             processed_dir=Path("data/processed"),
                             output_dir=tmp_path, seed=42)
        result = train(config)
        # Should stop well before 20 epochs
        assert len(result["history"]) < 20
