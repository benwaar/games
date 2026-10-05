"""Dataset and label utilities for Phase 2 sound-type classification."""

import json
from pathlib import Path

import torch
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
