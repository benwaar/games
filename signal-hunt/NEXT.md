# M18 — Chord Dataset

**Goal:** Synthesise chord clips from the Iowa piano recordings and run them through the pipeline to produce augmented tensors for Phase 4 chord training.

## Status going in

Phase 3 complete:
- `data/raw/notes/{note}/iowa_{pp,mf,ff}_{note}.wav` — 36 WAVs, 12 notes × 3 dynamics
- `pipeline/batch.py` — handles arbitrary `data/raw/{label}/` folders
- `output/transfer/finetune/best_model.pt` — Phase 3 checkpoint (12-class note classifier, 89.5% test acc)

Phase 4 goal: chord detection (multi-label) → chord progressions (CNN-RNN). M18 is the data layer.

---

## What a chord is here

A chord = 2+ notes played simultaneously. Starting with the 6 diatonic triads in C major — built entirely from notes we already have:

| Chord | Notes |
|-------|-------|
| Cmaj | C4 + E4 + G4 |
| Dmin | D4 + F4 + A4 |
| Emin | E4 + G4 + B4 |
| Fmaj | F4 + A4 + C4 |
| Gmaj | G4 + B4 + D4 |
| Amin | A4 + C4 + E4 |

Synthesised by mixing individual note WAVs in the time domain — no new recordings needed.

---

## Steps

### Step 1 — `scripts/synthesise_chords.py`

Mix single-note WAVs into chord clips. For each chord:
1. Load the three note WAVs (vary dynamics: pp+pp+pp, mf+mf+mf, ff+ff+ff, pp+mf+ff)
2. Normalise each to the same peak amplitude before mixing — avoids clipping
3. Sum the signals, renormalise the result
4. Save to `data/raw/chords/{chord_name}/`

Target: **≥10 source clips per chord** before augmentation.

**Gate:** Cmaj spectrogram shows 3 harmonic ladders overlaid. Waveform doesn't clip.

**Commit:** `feat(signal-hunt): synthesise_chords.py — mix Iowa notes into diatonic triads`

---

### Step 2 — Run pipeline, verify tensors

```bash
python -m pipeline.batch data/raw/chords data/processed/chords
```

**Gate:** 6 chord labels, ≥70 tensors per chord (10 source × 7 aug). Cmaj spectrogram visually distinct from single C4.

**Commit:** `feat(signal-hunt): chord tensors — 6 diatonic triads, 7 augmentations each`

---

### Step 3 — Close M18

- Tick M18 in PLAN.md, add results summary
- Rewrite NEXT for M19

**Commit:** `docs(signal-hunt): tick M18, NEXT → M19`

---

## Stop conditions

| Situation | Action |
|-----------|--------|
| Output waveform clips | Normalise each note to peak 0.5 before mixing |
| Spectrogram looks like single note | Check mix levels — verify all three notes audible |
| Fewer than 6 labels in manifest | Check folder names: `data/raw/chords/{Cmaj,Dmin,...}/` |
