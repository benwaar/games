# M19 — Multi-Label Chord Model

**Goal:** Build a chord detection head on top of the Phase 3 backbone that predicts which notes are present in a chord clip.

## Status going in

M18 complete:
- `scripts/synthesise_chords.py` — 24 source chord clips from Iowa piano notes
- `data/raw/chords/{Cmaj,Dmin,Emin,Fmaj,Gmaj,Amin}/` — 4 clips per chord
- `data/processed/chords/` — 168 tensors (28 per chord × 6 chords)
- `output/transfer/finetune/best_model.pt` — Phase 3 checkpoint (12-class note classifier, 89.5% test acc)

Phase 4 goal: chord detection (multi-label) → chord progressions (CNN-RNN). M19 is the model layer.

---

## The two approaches

**Option A — Note-set head (multi-label):**
- Output: `(B, 12)` — one sigmoid logit per note
- Loss: `BCEWithLogitsLoss`
- Prediction: `(sigmoid(logits) > 0.5)` — which notes are "on"
- Richer: tells you exactly which notes are present
- Harder: 12 binary tasks vs 1 multi-class task

**Option B — Chord-name head (multi-class):**
- Output: `(B, 6)` — one softmax logit per chord name
- Loss: `CrossEntropyLoss`
- Simpler: same setup as Phase 2/3
- Loses note-level detail

Build both, document the tradeoff, train Option B first (faster to get running), then Option A.

---

## Steps

### Step 1 — `model/chord.py`

Two head variants on the Phase 3 backbone:

```python
# ChordNameClassifier — Option B (start here)
# Load Phase 3 backbone, swap Linear(32→12) → Linear(32→6)
# CrossEntropyLoss, same training loop as Phase 3

# NoteSetClassifier — Option A
# Load Phase 3 backbone, swap Linear(32→12) → Linear(32→12) with sigmoid
# BCEWithLogitsLoss, threshold at 0.5
```

**Gate:** Both models load Phase 3 weights. Forward passes produce correct output shapes. BCEWithLogitsLoss computes without NaN.

**Commit:** `feat(signal-hunt): chord.py — ChordNameClassifier and NoteSetClassifier heads`

---

### Step 2 — Train chord-name classifier (Option B)

```bash
python -m model.chord_train --mode name --epochs 80
```

**Gate:** Validation accuracy > 50% (random = 16.7%). Loss decreasing. No NaN.

**Commit:** `feat(signal-hunt): chord_train.py — train chord-name classifier`

---

### Step 3 — Train note-set classifier (Option A)

```bash
python -m model.chord_train --mode notes --epochs 80
```

Metrics: per-note F1, exact-match accuracy (all 3 notes in chord correct).

**Gate:** Exact-match accuracy > 50% (random ≈ 1.6% for 3-of-12 subsets). Predict on Cmaj → C4, E4, G4 flagged above threshold.

**Commit:** `feat(signal-hunt): chord_train.py --mode notes — multi-label note-set classifier`

---

### Step 4 — Close M19

- Tick M19 in PLAN.md, add results summary
- Rewrite NEXT for M20

**Commit:** `docs(signal-hunt): tick M19, NEXT → M20`

---

## Stop conditions

| Situation | Action |
|-----------|--------|
| BCELoss produces NaN | Switch to `BCEWithLogitsLoss` (already planned), add eps clipping |
| Option B stuck near random (16.7%) | Check label loading — confirm chord folder names match expected keys |
| Option A exact-match < 20% after tuning | Fall back to chord-name classification only, document why |
