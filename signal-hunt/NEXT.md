# M7 Plan — Dataset & DataLoader

**Goal:** Build `model/dataset.py` — a PyTorch `Dataset` that loads tensors from `data/processed/`
using the manifest, encodes labels as integers, and exposes a factory that returns stratified
train/val/test splits ready for a `DataLoader`.

## Prerequisites

- `data/processed/` exists with 231 tensors and `manifest.json` (run `python -m pipeline.batch data/raw data/processed` if missing)
- `scikit-learn` in `requirements.txt` ✅ (added in research step)
- venv active: `source .venv/bin/activate`

---

## Steps

### Step 1 — scaffold `model/` package
- [ ] Create `model/__init__.py` (empty)
- [ ] Verify `python -c "import model"` works

**Test:** `python -c "import model"` exits 0.
**Commit:** `feat(signal-hunt): scaffold model package`

---

### Step 2 — `SignalDataset` class
- [ ] Create `model/dataset.py`
- [ ] Implement `SignalDataset(Dataset)` with `__init__`, `__len__`, `__getitem__`
  - `__init__` takes `records: list[dict]`, `processed_dir: Path`, `label_map: dict[str, int]`
  - `__getitem__` returns `(tensor, int_label)` — tensor shape `(1, 128, 65)`
- [ ] Test: `dataset[0]` returns correct shape and label type

**Test:** `assert dataset[0][0].shape == (1, 128, 65)` and `isinstance(dataset[0][1], int)`
**Commit:** `feat(signal-hunt): SignalDataset — loads tensors from manifest`

---

### Step 3 — label encoding
- [ ] `make_label_map(labels: list[str]) -> dict[str, int]` — sorted alphabetical, deterministic
- [ ] `{"clap": 0, "hum": 1, "whistle": 2}` is the expected output for our 3 classes
- [ ] Test: same input always produces same map regardless of input order

**Test:** `make_label_map(["whistle", "hum", "clap"]) == make_label_map(["clap", "hum", "whistle"])`
**Commit:** included in Step 2 commit (it's a helper, not a separate concept)

---

### Step 4 — stratified splits
- [ ] `load_splits(processed_dir, manifest_path, seed=42)` factory function
  - Reads `manifest.json`
  - Builds label map
  - Stratified 70/15/15 split using `sklearn.model_selection.train_test_split`
  - Returns `(train_dataset, val_dataset, test_dataset)`
- [ ] Test: split sizes are approximately 162 / 35 / 34 for 231 samples
- [ ] Test: each split contains all 3 classes
- [ ] Test: same seed → same split (reproducibility)

**Test:** run `load_splits` twice with same seed, check `len(train)` matches both times.
**Commit:** `feat(signal-hunt): load_splits — stratified train/val/test factory`

---

### Step 5 — DataLoader integration test
- [ ] Verify `DataLoader(train_dataset, batch_size=32, shuffle=True)` iterates without error
- [ ] First batch shape: `(32, 1, 128, 65)` tensors, `(32,)` labels
- [ ] Labels are integers in range `[0, 2]`

**Test:** one full iteration through the train loader — no errors, correct shapes.
**Commit:** part of test file for Step 4

---

### Step 6 — tick PLAN.md checkboxes
- [ ] Mark M7 items `[x]` in `PLAN.md`
- [ ] Update `NEXT.md` to point at M8

**Commit:** `docs(signal-hunt): tick M7 checkboxes`

---

## Files to create

```
model/__init__.py       # empty
model/dataset.py        # SignalDataset, make_label_map, load_splits
tests/test_dataset.py   # tests for all of the above
```

## Gate (from PLAN.md)

`DataLoader` iterates `(tensor, label)` batches. Shapes `(B, 1, 128, 65)` and `(B,)`. Split is reproducible with fixed seed. Each split has all 3 classes.
