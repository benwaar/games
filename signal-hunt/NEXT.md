# M9 Plan — Training Loop

**Goal:** Build `model/train.py` — a complete training script that trains `SoundClassifier`
on the Phase 2 dataset, logs loss/accuracy per epoch, checkpoints the best model,
and stops early if validation loss stops improving.

## Prerequisites

- M7 ✅ — `load_splits` and `DataLoader` working
- M8 ✅ — `SoundClassifier` forward pass verified
- venv active: `source .venv/bin/activate`
- `data/processed/` with 231 tensors (run `python -m pipeline.batch data/raw data/processed` if missing)

## Research to do first

Before writing any code, read and document:
- `CrossEntropyLoss` — what it computes, why it combines softmax + NLL in one step
- `Adam` optimiser — adaptive learning rates, why it's the default starting point
- `ReduceLROnPlateau` — what it does when val loss plateaus
- Early stopping — patience, what "best model" means, why we save a checkpoint not just the final weights

Use `/research` at the start of the session to write these to the training explainer before building.

---

## Loop: Plan → Implement → Test → Document → Commit → Tick → Next

---

### Step 1 — `model/config.py` — hyperparameters dataclass

- [ ] `TrainConfig` dataclass: `batch_size`, `lr`, `epochs`, `dropout`, `seed`, `patience`
- [ ] Sensible defaults: batch_size=32, lr=1e-3, epochs=50, dropout=0.3, seed=42, patience=5

**Test:** `TrainConfig()` instantiates with defaults. Override one field works.

**Docs:**
- [ ] Note in training explainer why hyperparams live in a config object (reproducibility, CLI override)

**Commit:** `feat(signal-hunt): TrainConfig hyperparameter dataclass`

---

### Step 2 — training loop core

- [ ] `model/train.py` — `train_one_epoch(model, loader, criterion, optimiser)` → `(train_loss, train_acc)`
- [ ] `evaluate(model, loader, criterion)` → `(val_loss, val_acc)`
- [ ] Both return float values, no side effects

**Test:** one epoch on the real dataset completes without error. Loss is a positive float. Accuracy in [0, 1].

**Docs:**
- [ ] Training explainer: loss function, backprop step (`loss.backward()`, `optimiser.step()`, `optimiser.zero_grad()`)

**Commit:** `feat(signal-hunt): train_one_epoch and evaluate functions`

---

### Step 3 — full training script with checkpointing

- [ ] `train(config)` — outer loop over epochs:
  - Train + evaluate each epoch
  - Log: `epoch | train_loss | val_loss | val_acc`
  - `ReduceLROnPlateau` on val loss
  - Early stopping (patience=5)
  - Save best checkpoint to `output/best_model.pt` (model weights + label_map + config)
  - Save training history to `output/history.json`
- [ ] CLI: `python -m model.train --epochs 50 --lr 0.001`

**Test:** training run of 3 epochs completes, checkpoint and history files are written.

**Docs:**
- [ ] Training explainer: learning rate scheduling, early stopping, checkpointing — why save best not last

**Commit:** `feat(signal-hunt): full training loop with checkpointing and early stopping`

---

### Step 4 — close out M9

- [ ] Mark M9 checkboxes `[x]` in `PLAN.md`
- [ ] Update `NEXT.md` to point at M10

**Commit:** `docs(signal-hunt): tick M9 checkboxes, point NEXT at M10`

---

## Files to create

```
model/config.py         # TrainConfig dataclass
model/train.py          # train_one_epoch, evaluate, train, CLI
tests/test_train.py     # short training run, checkpoint written, history saved
explainers/training-loop.md
```

## Gate (from PLAN.md)

Loss decreases over epochs. Val accuracy above 50% (random baseline = 33%). Training history saved to `output/history.json`. Best checkpoint saved to `output/best_model.pt`.
