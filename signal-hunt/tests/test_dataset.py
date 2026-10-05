"""Tests for model.dataset — SignalDataset, make_label_map, load_splits."""

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from model.dataset import SignalDataset, load_splits, make_label_map

CLASSES = ["clap", "hum", "whistle"]


def _make_processed_dir(tmp_path: Path, per_class: int) -> Path:
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    records = []
    for label in CLASSES:
        for i in range(per_class):
            name = f"{label}_{i}_clean.pt"
            torch.save(torch.zeros(1, 128, 65), processed / name)
            records.append({
                "file": name, "label": label,
                "source": f"{label}_{i}.wav", "augmentation": "clean",
                "shape": [1, 128, 65],
            })
    (processed / "manifest.json").write_text(json.dumps(records))
    return processed


@pytest.fixture
def small_dir(tmp_path):
    """2 per class — for basic Dataset tests."""
    return _make_processed_dir(tmp_path / "small", 2)


@pytest.fixture
def split_dir(tmp_path):
    """20 per class — enough for stratified 70/15/15 split (val/test each get 3+ per class)."""
    return _make_processed_dir(tmp_path / "split", 20)


@pytest.fixture
def label_map():
    return make_label_map(CLASSES)


@pytest.fixture
def dataset(small_dir, label_map):
    records = json.loads((small_dir / "manifest.json").read_text())
    return SignalDataset(records, small_dir, label_map)


class TestMakeLabelMap:
    def test_sorted_alphabetically(self):
        assert make_label_map(["whistle", "hum", "clap"]) == {"clap": 0, "hum": 1, "whistle": 2}

    def test_deterministic_regardless_of_input_order(self):
        assert make_label_map(["whistle", "hum", "clap"]) == make_label_map(["clap", "hum", "whistle"])

    def test_deduplicates(self):
        assert len(make_label_map(["hum", "hum", "clap"])) == 2


class TestSignalDataset:
    def test_len(self, dataset):
        assert len(dataset) == 6  # 2 per class × 3 classes

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
        assert set(dataset.labels) == {"clap", "hum", "whistle"}


class TestLoadSplits:
    def test_returns_three_datasets(self, split_dir):
        train, val, test = load_splits(split_dir)
        assert all(isinstance(d, SignalDataset) for d in (train, val, test))

    def test_splits_cover_all_records(self, split_dir):
        train, val, test = load_splits(split_dir)
        assert len(train) + len(val) + len(test) == 60  # 20 per class × 3

    def test_each_split_has_all_classes(self, split_dir):
        train, val, test = load_splits(split_dir)
        for split in (train, val, test):
            assert set(split.labels) == {"clap", "hum", "whistle"}

    def test_reproducible_with_same_seed(self, split_dir):
        train_a, _, _ = load_splits(split_dir, seed=42)
        train_b, _, _ = load_splits(split_dir, seed=42)
        assert [r["file"] for r in train_a.records] == [r["file"] for r in train_b.records]

    def test_different_seeds_give_different_splits(self, split_dir):
        train_a, _, _ = load_splits(split_dir, seed=42)
        train_b, _, _ = load_splits(split_dir, seed=99)
        assert [r["file"] for r in train_a.records] != [r["file"] for r in train_b.records]


class TestDataLoaderIntegration:
    def test_batch_shapes(self, split_dir):
        train, _, _ = load_splits(split_dir)
        loader = DataLoader(train, batch_size=8, shuffle=False)
        tensors, labels = next(iter(loader))
        assert tensors.shape == (8, 1, 128, 65)
        assert labels.shape == (8,)

    def test_labels_are_integers_in_range(self, split_dir):
        train, _, _ = load_splits(split_dir)
        loader = DataLoader(train, batch_size=len(train), shuffle=False)
        _, labels = next(iter(loader))
        assert labels.dtype == torch.int64
        assert labels.min() >= 0
        assert labels.max() <= 2

    def test_full_iteration_no_errors(self, split_dir):
        train, val, test = load_splits(split_dir)
        for dataset in (train, val, test):
            loader = DataLoader(dataset, batch_size=4, shuffle=True)
            assert sum(1 for _ in loader) > 0
