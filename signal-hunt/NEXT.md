# M7 Plan — Dataset & DataLoader

**Goal:** Build `model/dataset.py` — a PyTorch `Dataset` that loads tensors from `data/processed/`
using the manifest, encodes labels as integers, and exposes a factory that returns stratified
train/val/test splits ready for a `DataLoader`.

## Prerequisites

- `data/processed/` exists with 231 tensors and `manifest.json` (run `python -m pipeline.batch data/raw data/processed` if missing)
- `scikit-learn` in `requirements.txt` ✅ (added in research step)
- venv active: `source .venv/bin/activate`

---

## Loop: Plan → Implement → Test → Document → Commit → Tick → Next

Each step is not done until ALL boxes are ticked — including the doc steps.

---

### Step 1 — scaffold `model/` package

- [ ] Create `model/__init__.py` (empty)
- [ ] Verify `python -c "import model"` works

**Test:** `python -c "import model"` exits 0.

**Docs:**
- [ ] Nothing to document — no new concepts introduced

**Commit:** `feat(signal-hunt): scaffold model package`

---

### Step 2 — `SignalDataset` class + label encoding

- [ ] Create `model/dataset.py`
- [ ] Implement `SignalDataset(Dataset)` with `__init__`, `__len__`, `__getitem__`
  - `__init__` takes `records: list[dict]`, `processed_dir: Path`, `label_map: dict[str, int]`
  - `__getitem__` returns `(tensor, int_label)` — tensor shape `(1, 128, 65)`
- [ ] `make_label_map(labels: list[str]) -> dict[str, int]` — sorted alphabetical, deterministic
- [ ] Tests in `tests/test_dataset.py`

**Test:** `dataset[0][0].shape == (1, 128, 65)`, `isinstance(dataset[0][1], int)`.
`make_label_map(["whistle", "hum", "clap"]) == make_label_map(["clap", "hum", "whistle"])`.

**Docs:**
- [ ] `dataset-dataloader.md` already covers these concepts ✅ — verify code matches the explainer, update if it diverges
- [ ] `python-concepts.md` dunder methods already added ✅

**Commit:** `feat(signal-hunt): SignalDataset and make_label_map`

---

### Step 3 — stratified splits

- [ ] `load_splits(processed_dir, manifest_path, seed=42)` factory function
  - Reads `manifest.json`, builds label map, stratified 70/15/15 split
  - Returns `(train_dataset, val_dataset, test_dataset)`
- [ ] Tests: split sizes ~162/35/34, each split has all 3 classes, same seed → same split

**Test:** run `load_splits` twice with same seed, check `len(train)` matches both times.

**Docs:**
- [ ] `dataset-dataloader.md` already covers stratified splits ✅ — verify accuracy against implementation
- [ ] `explainers/README.md` section 6 — update with actual function signature once written

**Commit:** `feat(signal-hunt): load_splits — stratified train/val/test factory`

---

### Step 4 — DataLoader integration

- [ ] Verify `DataLoader(train_dataset, batch_size=32, shuffle=True)` iterates without error
- [ ] First batch shape: `(32, 1, 128, 65)` tensors, `(32,)` labels
- [ ] Labels are integers in `[0, 2]`

**Test:** one full iteration through the train loader — no errors, correct shapes.

**Docs:**
- [ ] `dataset-dataloader.md` — add a "Putting it together" code block showing actual usage with `load_splits` + `DataLoader`
- [ ] `explainers/README.md` section 6 — confirm the "See:" link points to the real file

**Commit:** included in Step 3 commit

---

### Step 5 — close out M7

- [ ] Mark M7 checkboxes `[x]` in `PLAN.md`
- [ ] Update `NEXT.md` to point at M8 with its own step plan

**Commit:** `docs(signal-hunt): tick M7 checkboxes, point NEXT at M8`

---

## Files to create

```
model/__init__.py       # empty
model/dataset.py        # SignalDataset, make_label_map, load_splits
tests/test_dataset.py   # tests for all of the above
```

## Gate (from PLAN.md)

`DataLoader` iterates `(tensor, label)` batches. Shapes `(B, 1, 128, 65)` and `(B,)`.
Split reproducible with fixed seed. Each split has all 3 classes.
