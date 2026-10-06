"""Dataset and label utilities for Phase 2 sound-type classification."""

import json
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class SignalDataset(Dataset):
    """Loads (tensor, label) pairs from a processed manifest."""

    def __init__(
        self,
        records: list[dict],
        processed_dir: Path,
        label_map: dict[str, int],
    ) -> None:
        self.records = records
        self.processed_dir = Path(processed_dir)
        self.label_map = label_map

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        record = self.records[idx]
        tensor = torch.load(self.processed_dir / record["file"], weights_only=True)
        label = self.label_map[record["label"]]
        return tensor, label

    @property
    def labels(self) -> list[str]:
        return [r["label"] for r in self.records]


def make_label_map(labels: list[str]) -> dict[str, int]:
    """Map class names to integers, sorted alphabetically for determinism."""
    return {label: i for i, label in enumerate(sorted(set(labels)))}


def load_splits(
    processed_dir: Path,
    manifest_path: Path | None = None,
    seed: int = 42,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> tuple[SignalDataset, SignalDataset, SignalDataset]:
    """Load manifest and return stratified (train, val, test) datasets."""
    processed_dir = Path(processed_dir)
    if manifest_path is None:
        manifest_path = processed_dir / "manifest.json"

    records = json.loads(Path(manifest_path).read_text())
    label_map = make_label_map([r["label"] for r in records])
    stratify = [r["label"] for r in records]

    rest_size = val_size + test_size
    train_records, rest_records = train_test_split(
        records, test_size=rest_size, random_state=seed, stratify=stratify
    )
    val_records, test_records = train_test_split(
        rest_records,
        test_size=test_size / rest_size,
        random_state=seed,
        stratify=[r["label"] for r in rest_records],
    )

    return (
        SignalDataset(train_records, processed_dir, label_map),
        SignalDataset(val_records, processed_dir, label_map),
        SignalDataset(test_records, processed_dir, label_map),
    )
