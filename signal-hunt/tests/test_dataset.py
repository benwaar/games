"""Tests for model.dataset — SignalDataset, make_label_map, load_splits."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from model.dataset import SignalDataset, make_label_map


@pytest.fixture
def manifest_dir(tmp_path: Path) -> Path:
    """Three classes, 2 tensors each — minimal dataset for testing."""
    processed = tmp_path / "processed"
    processed.mkdir()
    records = []
    for label in ["clap", "hum", "whistle"]:
        for i in range(2):
            name = f"{label}_{i}_clean.pt"
            tensor = torch.zeros(1, 128, 65)
            torch.save(tensor, processed / name)
            records.append({"file": name, "label": label, "source": f"{label}_{i}.wav", "augmentation": "clean", "shape": [1, 128, 65]})
    (processed / "manifest.json").write_text(json.dumps(records))
    return processed


@pytest.fixture
def label_map() -> dict[str, int]:
    return make_label_map(["clap", "hum", "whistle"])


@pytest.fixture
def dataset(manifest_dir, label_map) -> SignalDataset:
    manifest = json.loads((manifest_dir / "manifest.json").read_text())
    return SignalDataset(manifest, manifest_dir, label_map)


class TestMakeLabelMap:
    def test_sorted_alphabetically(self):
        m = make_label_map(["whistle", "hum", "clap"])
        assert m == {"clap": 0, "hum": 1, "whistle": 2}

    def test_deterministic_regardless_of_input_order(self):
        assert make_label_map(["whistle", "hum", "clap"]) == make_label_map(["clap", "hum", "whistle"])

    def test_deduplicates(self):
        m = make_label_map(["hum", "hum", "clap"])
        assert len(m) == 2


class TestSignalDataset:
    def test_len(self, dataset):
        assert len(dataset) == 6

    def test_getitem_tensor_shape(self, dataset):
        tensor, _ = dataset[0]
        assert tensor.shape == (1, 128, 65)

    def test_getitem_label_is_int(self, dataset):
        _, label = dataset[0]
        assert isinstance(label, int)

    def test_label_in_range(self, dataset):
        for i in range(len(dataset)):
            _, label = dataset[i]
            assert 0 <= label <= 2

    def test_labels_property(self, dataset):
        labels = dataset.labels
        assert set(labels) == {"clap", "hum", "whistle"}
