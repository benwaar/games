# M14 Plan — Transfer Learning Setup

**Goal:** Load the Phase 2 `SoundClassifier` checkpoint, freeze the convolutional feature extractor,
replace the 3-class head with a 12-class head, and confirm the model trains correctly on note data.

## Prerequisites

- M13 ✅ — 252 tensors in `data/processed/notes/`, 12 classes
- Phase 2 checkpoint: `output/best_model.pt` (if missing, run `python -m model.train --epochs 50`)
- venv active: `source .venv/bin/activate`

## What transfer learning means here

The Phase 2 CNN learned to distinguish hums, whistles, and claps by their spectral shape.
The early conv layers learned general frequency-pattern detectors — edges, bands, transients.
Those patterns are also useful for distinguishing piano notes (different harmonic positions).

We reuse that learning:
1. Load Phase 2 weights into the CNN
2. Freeze the conv blocks (don't update during training)
3. Replace the final Linear(32→3) with Linear(32→12)
4. Train only the new head — fast, needs less data

---

## Loop: Plan → Implement → Test → Document → Commit → Tick → Next

---

### Step 1 — `model/transfer.py` — load and adapt

- [ ] `TransferClassifier` — loads Phase 2 checkpoint, freezes conv_blocks, replaces head
- [ ] `freeze_conv_blocks(model)` — sets `requires_grad=False` on all conv_blocks parameters
- [ ] Forward pass: `(B, 1, 128, 65)` → `(B, 12)` logits

**Test:** frozen layers have zero gradient after a backward pass. New head has gradients.

**Docs:**
- [ ] New explainer: `explainers/transfer-learning.md` — what transfer learning is,
  why it works, frozen vs unfrozen, C/JS/TS callouts
- [ ] `explainers/libraries.md` — `requires_grad`, `param.grad`

**Commit:** `feat(signal-hunt): TransferClassifier — freeze conv, replace head for 12 notes`

---

### Step 2 — note dataset class

- [ ] `NoteDataset` (or reuse `SignalDataset` with `load_splits` pointed at `data/processed/notes/`)
- [ ] Verify `load_splits("data/processed/notes")` returns correct label map: 12 classes

**Test:** DataLoader yields `(B, 1, 128, 65)` + `(B,)` labels in range `[0, 11]`. Label map is alphabetical.

**Docs:**
- [ ] Note in `dataset-dataloader.md`: same Dataset works for any manifest — no code change needed

**Commit:** `feat(signal-hunt): note dataset — load_splits pointed at notes folder`

---

### Step 3 — compare frozen vs full finetune

- [ ] Train `TransferClassifier` (frozen conv) for 20 epochs, log val accuracy
- [ ] Train same architecture from scratch for 20 epochs, log val accuracy
- [ ] Train with all layers unfrozen (full finetune) for 20 epochs, log val accuracy
- [ ] Plot: which converges faster? Which reaches higher accuracy?

**Test:** frozen-head run completes without NaN loss. Val accuracy above random (8.3%) after 5 epochs.

**Docs:**
- [ ] `transfer-learning.md` — add findings: frozen vs scratch vs full finetune on this data

**Commit:** `feat(signal-hunt): transfer learning experiment — frozen vs scratch vs full finetune`

---

### Step 4 — close out M14

- [ ] Tick M14 checkboxes in `PLAN.md`
- [ ] Update `NEXT.md` to point at M15

**Commit:** `docs(signal-hunt): tick M14 checkboxes, point NEXT at M15`

---

## Stop conditions

| Situation | Action |
|-----------|--------|
| Frozen transfer doesn't beat random after 10 epochs | Phase 2 features may not generalise to piano — unfreeze all layers and retrain from scratch |
| Val loss goes to NaN | Learning rate too high for frozen training — reduce to 1e-4 |
| Label map has fewer than 12 classes | Check `data/processed/notes/manifest.json` — some notes may have failed to process |
