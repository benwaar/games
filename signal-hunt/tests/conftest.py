"""Shared fixtures for signal-hunt tests."""

import json

import pytest
import torch

from model.cnn import SoundClassifier


_PHASE2_LABEL_MAP = {"clap": 0, "hum": 1, "whistle": 2}


@pytest.fixture(scope="session")
def synthetic_processed_dir(tmp_path_factory):
    """
    Minimal processed dataset: 3 classes × 7 tensors + manifest.json.

    Used by test_train and test_evaluate so those tests don't depend on
    data/processed/ existing on disk.
    """
    base = tmp_path_factory.mktemp("processed")
    records = []
    for label in ["clap", "hum", "whistle"]:
        for i in range(7):
            name = f"{label}_{i}_clean.pt"
            torch.save(torch.randn(1, 128, 65), base / name)
            records.append({
                "file": name,
                "source": f"{label}_{i}.wav",
                "label": label,
                "augmentation": "clean",
                "shape": [1, 128, 65],
            })
    (base / "manifest.json").write_text(json.dumps(records))
    return base


@pytest.fixture(scope="session")
def fake_phase2_checkpoint(tmp_path_factory):
    """
    Minimal Phase 2 checkpoint for tests that don't need a real trained model.

    Used by test_predict so those tests don't depend on output/best_model.pt.
    """
    model = SoundClassifier(num_classes=3, dropout=0.3)
    path = tmp_path_factory.mktemp("ckpt") / "best_model.pt"
    torch.save({
        "model_state": model.state_dict(),
        "label_map": _PHASE2_LABEL_MAP,
        "config": {"num_classes": 3, "dropout": 0.3, "seed": 42},
        "epoch": 1,
        "val_loss": 1.0,
        "val_acc": 0.5,
    }, path)
    return path
