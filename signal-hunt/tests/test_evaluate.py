"""Tests for model.evaluate — checkpoint loading, metrics, plots."""

import json
from pathlib import Path

import pytest
import torch

from model.config import TrainConfig
from model.evaluate import (
    collect_predictions,
    load_checkpoint,
    plot_confusion_matrix,
    plot_loss_curves,
    print_metrics,
    run_evaluation,
)
from model.train import train


@pytest.fixture(scope="module")
def trained_output(tmp_path_factory, synthetic_processed_dir):
    """Run a short training session once for all tests in this module."""
    out = tmp_path_factory.mktemp("output")
    config = TrainConfig(epochs=3, processed_dir=synthetic_processed_dir, output_dir=out, seed=42)
    train(config)
    return out


@pytest.fixture(scope="module")
def checkpoint_path(trained_output):
    return trained_output / "best_model.pt"


class TestLoadCheckpoint:
    def test_loads_model(self, checkpoint_path):
        model, label_map, cfg = load_checkpoint(checkpoint_path)
        assert model is not None

    def test_label_map_has_three_classes(self, checkpoint_path):
        _, label_map, _ = load_checkpoint(checkpoint_path)
        assert set(label_map.keys()) == {"clap", "hum", "whistle"}

    def test_model_in_eval_mode(self, checkpoint_path):
        model, _, _ = load_checkpoint(checkpoint_path)
        assert not model.training


class TestCollectPredictions:
    def test_lengths_match(self, checkpoint_path, synthetic_processed_dir):
        from model.dataset import load_splits
        from torch.utils.data import DataLoader
        model, _, cfg = load_checkpoint(checkpoint_path)
        _, _, test_ds = load_splits(synthetic_processed_dir, seed=cfg["seed"])
        loader = DataLoader(test_ds, batch_size=32)
        preds, labels, _, _ = collect_predictions(model, loader)
        assert len(preds) == len(labels) == len(test_ds)

    def test_predictions_in_range(self, checkpoint_path, synthetic_processed_dir):
        from model.dataset import load_splits
        from torch.utils.data import DataLoader
        model, _, cfg = load_checkpoint(checkpoint_path)
        _, _, test_ds = load_splits(synthetic_processed_dir, seed=cfg["seed"])
        loader = DataLoader(test_ds, batch_size=32)
        preds, _, _, _ = collect_predictions(model, loader)
        assert all(0 <= p <= 2 for p in preds)

    def test_loss_none_without_criterion(self, checkpoint_path, synthetic_processed_dir):
        from model.dataset import load_splits
        from torch.utils.data import DataLoader
        model, _, cfg = load_checkpoint(checkpoint_path)
        _, _, test_ds = load_splits(synthetic_processed_dir, seed=cfg["seed"])
        loader = DataLoader(test_ds, batch_size=32)
        _, _, loss, acc = collect_predictions(model, loader)
        assert loss is None
        assert isinstance(acc, float) and 0.0 <= acc <= 1.0

    def test_loss_and_acc_returned_with_criterion(self, checkpoint_path, synthetic_processed_dir):
        import torch.nn as nn
        from model.dataset import load_splits
        from torch.utils.data import DataLoader
        model, _, cfg = load_checkpoint(checkpoint_path)
        _, _, test_ds = load_splits(synthetic_processed_dir, seed=cfg["seed"])
        loader = DataLoader(test_ds, batch_size=32)
        _, _, loss, acc = collect_predictions(model, loader, nn.CrossEntropyLoss())
        assert isinstance(loss, float) and loss > 0
        assert isinstance(acc, float) and 0.0 <= acc <= 1.0


class TestPrintMetrics:
    def test_report_file_written(self, tmp_path, checkpoint_path):
        _, label_map, _ = load_checkpoint(checkpoint_path)
        print_metrics([0, 1, 2], [0, 1, 2], label_map, tmp_path)
        assert (tmp_path / "eval_report.txt").exists()

    def test_report_contains_class_names(self, tmp_path, checkpoint_path):
        _, label_map, _ = load_checkpoint(checkpoint_path)
        print_metrics([0, 1, 2], [0, 1, 2], label_map, tmp_path)
        report = (tmp_path / "eval_report.txt").read_text()
        assert "clap" in report and "hum" in report and "whistle" in report


class TestPlots:
    def test_confusion_matrix_saved(self, tmp_path, checkpoint_path):
        _, label_map, _ = load_checkpoint(checkpoint_path)
        plot_confusion_matrix([0, 1, 2, 0], [0, 1, 2, 1], label_map, tmp_path)
        assert (tmp_path / "confusion_matrix.png").exists()

    def test_loss_curves_saved(self, tmp_path):
        history = [{"epoch": i, "train_loss": 1.0 - i * 0.1,
                    "val_loss": 1.0 - i * 0.08} for i in range(5)]
        (tmp_path / "history.json").write_text(json.dumps(history))
        plot_loss_curves(tmp_path / "history.json", tmp_path)
        assert (tmp_path / "loss_curves.png").exists()


class TestRunEvaluation:
    def test_returns_summary(self, tmp_path, trained_output, synthetic_processed_dir):
        result = run_evaluation(
            checkpoint_path=trained_output / "best_model.pt",
            processed_dir=synthetic_processed_dir,
            images_dir=tmp_path / "images",
            output_dir=trained_output,
        )
        assert "test_acc" in result
        assert 0.0 <= result["test_acc"] <= 1.0

    def test_all_output_files_written(self, tmp_path, trained_output, synthetic_processed_dir):
        images = tmp_path / "images"
        run_evaluation(
            checkpoint_path=trained_output / "best_model.pt",
            processed_dir=synthetic_processed_dir,
            images_dir=images,
            output_dir=trained_output,
        )
        assert (trained_output / "eval_report.txt").exists()
        assert (images / "confusion_matrix.png").exists()
        assert (images / "loss_curves.png").exists()
