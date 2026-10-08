# M20 — Training on Chords

**Goal:** Deep evaluation of both chord classifiers — per-chord confusion, threshold sensitivity for the note-set model, and a final decision on which head to carry into Phase 4b (progressions).

## Status going in

M19 complete:
- `model/chord.py` — `ChordNameClassifier` (6-class) and `NoteSetClassifier` (12-label)
- `model/chord_train.py` — `--mode name` and `--mode notes`
- `output/chords/name/best_model.pt` — 96% val_acc
- `output/chords/notes/best_model.pt` — 76% exact-match

Both models trained 80 epochs. Notes mode was still improving at epoch 80.

---

## Steps

### Step 1 — Evaluate chord-name model

```bash
python -m model.chord_evaluate --mode name
```

- Confusion matrix: does Cmaj confuse with Amin? (share C4, E4) vs Gmaj? (share G4)
- Per-chord F1
- Save confusion matrix to `explainers/images/chord_confusion_name.png`

**Gate:** Confusion pattern is musically sensible — shared-note chords confuse more than unrelated ones.

**Commit:** `feat(signal-hunt): chord_evaluate.py — per-chord metrics and confusion matrix`

---

### Step 2 — Evaluate note-set model + threshold sensitivity

```bash
python -m model.chord_evaluate --mode notes --thresholds 0.3 0.5 0.7
```

- Exact-match accuracy at each threshold
- Per-note F1 (which notes does the model miss most?)
- Save to `explainers/images/chord_confusion_notes.png`

**Gate:** Threshold sweep shows precision/recall tradeoff. Best threshold identified.

**Commit:** `feat(signal-hunt): chord_evaluate.py --mode notes — threshold sweep, per-note F1`

---

### Step 3 — Retrain notes model longer if needed

If exact-match was still improving at epoch 80, try 120 epochs:

```bash
python -m model.chord_train --mode notes --epochs 120
```

**Gate:** Exact-match > 80% or plateaued.

---

### Step 4 — Close M20

- Tick M20 in PLAN.md, add results summary
- Rewrite NEXT for M21

**Commit:** `docs(signal-hunt): tick M20, NEXT → M21`

---

## Stop conditions

| Situation | Action |
|-----------|--------|
| Name model confusions are random (not musical) | Check label map — ensure chord folder names loaded correctly |
| Notes model stuck below 50% at all thresholds | Try lower lr (1e-4) or more augmentation |
| Per-note F1 shows one note always missed | Check `_NOTE_INDEX` mapping in chord_train.py — may be an index mismatch |
