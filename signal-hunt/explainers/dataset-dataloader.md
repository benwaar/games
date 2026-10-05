# Dataset & DataLoader

How PyTorch thinks about data — and why the abstraction matters.

---

## The problem this solves

A model needs data in batches. Batches need to be:
- **Shuffled** — so the model doesn't learn the order, just the patterns
- **Split** — into train, validation, and test sets
- **Labelled** — each tensor paired with an integer class
- **Loaded efficiently** — without blowing up RAM on large datasets

You could write all of that from scratch. PyTorch gives you two classes that handle it:

- **`Dataset`** — knows how to load one item
- **`DataLoader`** — wraps a `Dataset` and handles batching, shuffling, and parallel loading

---

## Dataset — the one-item contract

`Dataset` is an abstract base class. You subclass it and implement two methods:

```python
class SignalDataset(Dataset):
    def __len__(self):
        return 231  # total number of items

    def __getitem__(self, idx):
        return tensor, label  # one item at index idx
```

That's the entire contract. PyTorch doesn't care what's inside — it just calls `__len__` to know the size and `__getitem__` to get any item by index.

> **Coming from C:** This is a vtable — a struct with function pointers. `__len__` is `size()` and `__getitem__` is `get(i)`. Any struct that implements those two is a valid "array-like" that PyTorch can drive. The double underscores mark it as a Python protocol — the language-level equivalent of an interface.

> **Coming from JS/TS:** This is like implementing an interface with `length` and `[Symbol.iterator]` so your object works with `for...of` and spread. `__len__` is `.length`; `__getitem__` is indexed access `arr[i]`. Python uses dunder methods (double-underscore) where TS uses symbols or known property names.

---

## What our Dataset does

We load tensors from `data/processed/` using the manifest as an index:

```python
class SignalDataset(Dataset):
    def __init__(self, records: list[dict], processed_dir: Path, label_map: dict[str, int]):
        self.records = records          # list of manifest entries
        self.processed_dir = processed_dir
        self.label_map = label_map      # {"clap": 0, "hum": 1, "whistle": 2}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        tensor = torch.load(self.processed_dir / record["file"], weights_only=True)
        label = self.label_map[record["label"]]
        return tensor, label
```

The manifest (from `pipeline.batch`) already has `file` and `label` for each tensor, so loading is just a lookup + file read.

---

## Label encoding

The model outputs numbers, not strings. "hum" → `1`, "whistle" → `2`, "clap" → `0`. This mapping is the **label map** — a dict stored with the dataset so inference can reverse it.

```python
def make_label_map(labels: list[str]) -> dict[str, int]:
    # Sorted so the mapping is deterministic across machines
    return {label: i for i, label in enumerate(sorted(set(labels)))}

# {"clap": 0, "hum": 1, "whistle": 2}
```

Why sort? If you build the map from a random iteration order, "hum" might be `0` on your machine and `1` on the next. Sort → deterministic → reproducible predictions.

> **In practice:** Label encoding is how you turn any categorical column into something a model can process. Customer tiers ("bronze", "silver", "gold"), document types ("invoice", "receipt", "contract"), transaction categories — same pattern. The map is always stored alongside the model so you can decode predictions back to human-readable labels at inference time.

---

## Stratified split

A dataset of 231 samples split 70/15/15 gives roughly 162 train / 35 val / 34 test. The key word is *stratified* — each split has the same proportion of each class.

**Why not just random?** With 77 samples per class (11 recordings × 7 augmentations):
- A random split might give you 60 hum / 50 whistle / 52 clap in train
- An unlucky split might put 8 clap recordings in test — then overfit one class, underrepresent another, and your validation metrics mislead you

Stratified split ensures each class appears in train/val/test at the same rate it appears overall.

```python
from sklearn.model_selection import train_test_split

# First split: train vs rest
train_records, rest = train_test_split(
    records, test_size=0.30, random_state=42, stratify=labels
)
# Second split: val vs test (50/50 of the remaining 30%)
val_records, test_records = train_test_split(
    rest, test_size=0.50, random_state=42, stratify=[r["label"] for r in rest]
)
```

`stratify=labels` is the key argument — it tells sklearn to preserve class proportions in each split.

> **Coming from C/JS:** This is like partitioning a sorted array proportionally — except you're preserving the *distribution* of a categorical column, not a numeric range. Libraries like sklearn's `train_test_split` handle the edge cases (what if a class has only 3 samples? stratify still works as long as each class has at least 2).

---

## DataLoader — batching and shuffling

`DataLoader` wraps a `Dataset` and handles the mechanics of batching:

```python
from torch.utils.data import DataLoader

loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)

for batch_tensors, batch_labels in loader:
    # batch_tensors: (32, 1, 128, 65)
    # batch_labels:  (32,)
    loss = model(batch_tensors, batch_labels)
    loss.backward()
```

Key parameters:
- **`batch_size`** — how many items per batch. Larger = more stable gradients, more RAM.
- **`shuffle=True`** — randomise order each epoch (train set only; val/test should not shuffle)
- **`num_workers`** — parallel data loading. `0` = main process; `2` or `4` = background workers that pre-fetch batches while the GPU trains. On macOS with MPS, start with `0` and raise if loading is the bottleneck.

> **Coming from C:** `DataLoader` is a producer in a producer-consumer queue. The workers are producer threads that load and pre-process batches; your training loop is the consumer. The `batch_size` is the queue item size. Setting `num_workers > 0` spawns OS-level subprocesses (not threads) in Python to avoid the GIL.

> **Coming from JS/TS:** `DataLoader` is an async iterable — `for await (const batch of loader)`. The workers are like Node's worker_threads loading data off-thread. `shuffle` is like calling `Array.sort(() => Math.random() - 0.5)` between each pass (but proper Fisher-Yates, not JS's broken version).

---

## How it fits in the pipeline

```
data/raw/{class}/*.wav
    ↓ pipeline.batch (Phase 1)
data/processed/{name}.pt + manifest.json
    ↓ SignalDataset (M7)
Dataset(train) / Dataset(val) / Dataset(test)
    ↓ DataLoader
(batch_tensors, batch_labels)  ← shape (B, 1, 128, 65) + (B,)
    ↓ CNN (M8)
logits (B, 3)
    ↓ CrossEntropyLoss + Adam (M9)
trained model
```

The `Dataset`/`DataLoader` layer is the bridge between the Phase 1 pipeline and the Phase 2 model. It's the only place that knows about file paths, manifests, and label encoding — the model never touches disk.

---

## Separation of concerns

This is the most important design principle here: **the model never loads data**. It only sees tensors and labels. The `Dataset` handles file I/O; the `DataLoader` handles batching.

This matters for two reasons:
1. **Swap data without touching the model.** Add more recordings? Update the manifest and re-run `pipeline.batch`. The model code is unchanged.
2. **Test independently.** You can write tests for the dataset (does it load the right shapes? are labels correct?) without involving the model at all.

> **In practice:** This is the repository/service pattern from backend engineering. The model is a service that processes inputs and produces outputs. The dataset is a repository that retrieves data. They talk via a defined contract (tensor shape + integer label). The separation makes both testable and replaceable independently.
