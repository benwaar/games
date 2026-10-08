"""Shared fixtures for signal-hunt tests."""

import json

import numpy as np
import pytest
import soundfile as sf
import torch

from model.cnn import SoundClassifier


_PHASE2_LABEL_MAP = {"clap": 0, "hum": 1, "whistle": 2}


@pytest.fixture(scope="session")
def audio_fixtures(tmp_path_factory):
    """
    Minimal WAV files for pipeline.ingest tests.
    Generated fresh each session — not committed to the repo.
    """
    d = tmp_path_factory.mktemp("audio_fixtures")
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    mono = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(str(d / "mono_44100.wav"), mono, sr)

    stereo = np.column_stack([mono, mono * 0.8])
    sf.write(str(d / "stereo_44100.wav"), stereo, sr)

    quiet = np.zeros(sr, dtype=np.float32)
    loud = (0.8 * np.sin(2 * np.pi * 440 * t[:int(sr * 0.5)])).astype(np.float32)
    padded = np.concatenate([quiet[:int(sr * 0.2)], loud, quiet[:int(sr * 0.2)]])
    sf.write(str(d / "silence_padded.wav"), padded, sr)

    return d


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
