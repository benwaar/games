# Next: Phase 2 — M7: Dataset & DataLoader

Data is ready. Research done. Run `/study` to build.

## State

- 33 raw recordings committed (`data/raw/{hum,whistle,clap}/`, 11 per class)
- 231 processed tensors in `data/processed/` (33 × 7 augmentations, shape `(1, 128, 65)`)
- Labels come from folder name — manifest has `"label": "hum"` etc.
- `scikit-learn` added to `requirements.txt` (needed for stratified split)
- 60 tests passing

## Concepts researched (read before building)

→ [Dataset & DataLoader explainer](explainers/dataset-dataloader.md)
→ [Dunder protocol in python-concepts.md](explainers/python-concepts.md)
→ [torch.utils.data in libraries.md](explainers/libraries.md)

## M7: What to build

**`model/dataset.py`** — a PyTorch `Dataset` class:

```python
class SignalDataset(Dataset):
    def __init__(self, records, processed_dir, label_map): ...
    def __len__(self): ...
    def __getitem__(self, idx): ...  # returns (tensor, int_label)
```

Plus a factory function:
```python
def load_splits(processed_dir, manifest_path, seed=42) -> tuple[SignalDataset, SignalDataset, SignalDataset]:
    # reads manifest, builds label_map, stratified 70/15/15 split
    # returns train_dataset, val_dataset, test_dataset
```

**Gate:** `DataLoader(train_dataset, batch_size=32)` iterates `(B, 1, 128, 65)` + `(B,)` batches.
Split is reproducible. Each split has proportional class representation.

## File to create

- `model/__init__.py` (empty)
- `model/dataset.py`
- `tests/test_dataset.py`
