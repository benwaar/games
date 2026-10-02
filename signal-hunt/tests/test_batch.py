"""Tests for pipeline.batch — batch processing, manifest, DataLoader integration."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader, TensorDataset

from pipeline.batch import (
    default_augmentations,
    apply_augmentation,
    process_file,
    process_folder,
)

SR = 22050
DURATION = 1.5


@pytest.fixture
def audio_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)
    for name, freq in [("hum", 440), ("whistle", 880)]:
        signal = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        sf.write(raw / f"{name}.wav", signal, SR)
    return raw


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    return tmp_path / "processed"


class TestDefaultAugmentations:
    def test_returns_list(self):
        augs = default_augmentations()
        assert isinstance(augs, list)
        assert len(augs) == 7

    def test_first_is_clean(self):
        augs = default_augmentations()
        assert augs[0]["name"] == "clean"

    def test_all_have_names(self):
        for aug in default_augmentations():
            assert "name" in aug


class TestApplyAugmentation:
    def test_clean_returns_copy(self):
        signal = np.ones(100, dtype=np.float32)
        rng = np.random.default_rng(0)
        result = apply_augmentation(signal, SR, {"name": "clean"}, rng)
        np.testing.assert_array_equal(result, signal)
        assert result is not signal

    def test_noise_changes_signal(self):
        t = np.linspace(0, 0.1, int(SR * 0.1), endpoint=False)
        signal = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        rng = np.random.default_rng(0)
        aug = {"name": "noise_low", "fn": "add_noise", "kwargs": {"snr_db": 20.0}}
        result = apply_augmentation(signal, SR, aug, rng)
        assert not np.array_equal(result, signal)
        assert len(result) == len(signal)


class TestProcessFile:
    def test_produces_tensors(self, audio_dir, output_dir):
        output_dir.mkdir()
        records = process_file(audio_dir / "hum.wav", output_dir)
        assert len(records) == 7  # one per augmentation
        for record in records:
            pt_path = output_dir / record["file"]
            assert pt_path.exists()
            tensor = torch.load(pt_path, weights_only=True)
            assert tensor.shape == (1, 128, 65)

    def test_records_have_metadata(self, audio_dir, output_dir):
        output_dir.mkdir()
        records = process_file(audio_dir / "hum.wav", output_dir)
        for record in records:
            assert record["label"] == "hum"
            assert record["source"] == "hum.wav"
            assert "augmentation" in record
            assert record["shape"] == [1, 128, 65]

    def test_deterministic_with_same_seed(self, audio_dir, output_dir):
        out1 = output_dir / "run1"
        out2 = output_dir / "run2"
        out1.mkdir(parents=True)
        out2.mkdir(parents=True)
        r1 = process_file(audio_dir / "hum.wav", out1, seed=42)
        r2 = process_file(audio_dir / "hum.wav", out2, seed=42)
        for a, b in zip(r1, r2):
            t1 = torch.load(out1 / a["file"], weights_only=True)
            t2 = torch.load(out2 / b["file"], weights_only=True)
            torch.testing.assert_close(t1, t2)


class TestProcessFolder:
    def test_processes_all_files(self, audio_dir, output_dir):
        manifest = process_folder(audio_dir, output_dir)
        assert len(manifest) == 14  # 2 files × 7 augmentations

    def test_writes_manifest(self, audio_dir, output_dir):
        process_folder(audio_dir, output_dir)
        manifest_path = output_dir / "manifest.json"
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text())
        assert len(data) == 14

    def test_raises_on_empty_dir(self, tmp_path, output_dir):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            process_folder(empty, output_dir)

    def test_dataloader_can_iterate(self, audio_dir, output_dir):
        manifest = process_folder(audio_dir, output_dir)
        tensors = [torch.load(output_dir / r["file"], weights_only=True) for r in manifest]
        stacked = torch.cat(tensors, dim=0)  # (14, 128, 65)
        labels = torch.zeros(len(manifest), dtype=torch.long)
        dataset = TensorDataset(stacked, labels)
        loader = DataLoader(dataset, batch_size=4)
        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape == (4, 128, 65)
        assert batch_y.shape == (4,)
