# M21 — Chord Progressions + RNN

**Goal:** Synthesise chord progression clips and build a CNN-RNN hybrid that classifies sequences of chords over time.

## Status going in

M20 complete:
- `output/chords/name/best_model.pt` — 96.2% test accuracy (chord-name, 6-class)
- `output/chords/notes/best_model.pt` — 73.1% exact-match / F1 0.89 avg (note-set, 12-label)
- `model/chord_evaluate.py` — full evaluation tooling for both modes
- Both models ready for the two-pass tutor pipeline

Phase 4b goal: model temporal sequences — "Cmaj → Fmaj → Gmaj → Cmaj" (I-IV-V-I).

---

## The two-part architecture

**CNN (already built):** Extracts per-frame chord features. Takes `(B, 1, 128, T)` spectrogram → `(B, 64, T')` feature maps.

**GRU on top:** Collapses the frequency axis, then passes the time-step sequence through a bidirectional GRU to classify the progression as a whole.

```
(B, 1, 128, T)
    ↓  CNN conv_blocks
(B, 64, H, T')
    ↓  mean over freq axis
(B, T', 64)
    ↓  bidirectional GRU
(B, hidden*2)
    ↓  linear head
(B, num_progressions)
```

Start with whole-clip classification (one label per progression) before per-step decoding.

---

## Steps

### Step 1 — `scripts/synthesise_progressions.py`

Concatenate chord clips with short gaps to produce 4-chord progression clips.

Target progressions (start with 4, can add more):

| Label | Progression | Roman numerals |
|-------|------------|----------------|
| I-IV-V-I | Cmaj→Fmaj→Gmaj→Cmaj | Tonic-subdominant-dominant-tonic |
| vi-IV-I-V | Amin→Fmaj→Cmaj→Gmaj | Common pop progression |
| I-V-vi-IV | Cmaj→Gmaj→Amin→Fmaj | Another pop staple |
| ii-V-I | Dmin→Gmaj→Cmaj | Jazz cadence |

Mix dynamics: use pp/mf/ff combos for each chord within a progression. Target ≥8 source clips per label before augmentation → ≥56 tensors per label.

**Gate:** 4 progression labels, ≥50 tensors each. Spectrogram shows clear chord-boundary transitions.

**Commit:** `feat(signal-hunt): synthesise_progressions.py — 4-chord progression clips`

---

### Step 2 — `model/progression.py`

CNN-RNN hybrid:
- Load Phase 4 chord-name backbone (conv_blocks + gap) as feature extractor
- Replace gap with adaptive avg pool over freq axis only → preserve time axis
- Bidirectional GRU (hidden=64, 1 layer)
- Linear head → num_progressions logits
- CrossEntropyLoss (whole-clip classification to start)

**Gate:** Forward pass with `(B, 1, 128, T)` input produces `(B, num_progressions)`. No gradient explosion.

**Commit:** `feat(signal-hunt): progression.py — CNN-RNN hybrid for chord progressions`

---

### Step 3 — Train and evaluate

```bash
python -m model.progression_train --epochs 100
```

**Gate:** Test accuracy > 40% (random = 25% for 4 classes). Verify: I-IV-V-I ≠ V-I-IV-I (order matters).

**Commit:** `feat(signal-hunt): progression_train.py — train CNN-RNN on chord progressions`

---

### Step 4 — Close M21

- Tick M21 in PLAN.md, add results summary
- Rewrite NEXT for M22

**Commit:** `docs(signal-hunt): tick M21, NEXT → M22`

---

## Stop conditions

| Situation | Action |
|-----------|--------|
| Progression spectrograms look like noise | Check gap length between chords — may need silence trimming |
| RNN gradients explode | Add gradient clipping (`torch.nn.utils.clip_grad_norm_`) |
| Accuracy stuck near random (25%) after tuning | Shorten progressions to 2 chords, reduce complexity |
| I-IV-V-I and V-I-IV-I not distinguished | Check label encoding — progression order must be in the label |
