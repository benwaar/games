# Next: Phase 2 — M7: Dataset & DataLoader

Data is ready. Run `/study` to continue.

## State

- 33 raw recordings committed (`data/raw/{hum,whistle,clap}/`, 11 per class)
- 231 processed tensors in `data/processed/` (33 × 7 augmentations, shape `(1, 128, 65)`)
- Labels come from folder name, not filename
- 60 tests passing

## Start here: M7

Build `model/dataset.py` — a PyTorch `Dataset` that loads tensors from `data/processed/`
and maps them to integer labels using the manifest.

Gate: `DataLoader` iterates `(B, 1, 128, 65)` + `(B,)` batches. Stratified 70/15/15 split.
Reproducible with fixed seed.
