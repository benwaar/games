# M18 — Chord Dataset

**Goal:** Synthesise chord clips from existing Iowa piano recordings and run them through the Phase 1 pipeline to produce augmented tensors for Phase 4 chord training.

## Status going in

Phase 3 is complete. We have:
- `data/raw/notes/{note}/iowa_{pp,mf,ff}_{note}.wav` — 36 WAV files, 12 notes × 3 dynamics
- `pipeline/batch.py` — already handles arbitrary `data/raw/{label}/` folders
- `output/transfer/finetune/best_model.pt` — Phase 3 checkpoint (12-class note classifier)

Phase 4 goal: chord detection (multi-label) → chord progressions (CNN-RNN). M18 is the data layer.

## What a chord is here

A **chord** = 2 or more notes played simultaneously. We start with the 6 diatonic triads in C major — built entirely from the 12 notes we already have:

| Chord | Notes | Type |
|-------|-------|------|
| Cmaj | C4 + E4 + G4 | Major triad |
| Dmin | D4 + F4 + A4 | Minor triad |
| Emin | E4 + G4 + B4 | Minor triad |
| Fmaj | F4 + A4 + C4 | Major triad |
| Gmaj | G4 + B4 + D4 | Major triad |
| Amin | A4 + C4 + E4 | Minor triad |

Each chord is synthesised by **mixing the individual note WAVs** in the time domain — no new recordings needed.

---

## Loop: Plan → Implement → Test → Commit → Tick → Next

---

### Step 1 — `scripts/synthesise_chords.py`

Mix single-note WAVs to create chord clips. For each chord:
1. Load the three note WAVs (use mf/ff/pp for variety)
2. Normalise each to the same peak amplitude before mixing (avoids clipping)
3. Mix (sum) the signals, renormalise the result
4. Save to `data/raw/chords/{chord_name}/` (e.g. `data/raw/chords/Cmaj/`)

Generate multiple combinations per chord:
- pp+pp+pp, mf+mf+mf, ff+ff+ff (uniform dynamics)
- pp+mf+ff (mixed — one note louder than others, common in real playing)
- That gives ~4–5 clips per chord × 3 chords each = sufficient variety

Target: **≥10 source clips per chord** before augmentation.

**Test:** Load and play back a synthesised Cmaj. Its spectrogram should show 3 harmonic ladders (C4, E4, G4) overlaid.

**Docs:** Note in `explainers/iowa-piano-data.md` that chords are synthesised from the same Iowa source files.

**Commit:** `feat(signal-hunt): synthesise_chords.py — mix Iowa notes into diatonic triads`

---

### Step 2 — Run pipeline, verify tensors

```bash
python -m pipeline.batch data/raw/chords data/processed/chords
```

Expect: 6 chord labels, ≥70 tensors per chord (10 source × 7 augmentations).

Check `data/processed/chords/manifest.json` — labels should be the chord names (`Cmaj`, `Dmin`, etc.).

**Spot-check:** render a Cmaj spectrogram — should look meaningfully different from a single C4 spectrogram.

**Test:** `assert len(manifest) >= 420` (6 chords × 10 clips × 7 aug). All label values are valid chord names.

**Commit:** `feat(signal-hunt): generate chord tensors — 6 diatonic triads, 7 augmentations each`

---

### Step 3 — Tick M18, point NEXT at M19

- Tick M18 checkboxes in PLAN.md, add results summary
- Rewrite NEXT.md for M19

**Commit:** `docs(signal-hunt): tick M18 checkboxes, NEXT → M19`

---

## Stop conditions

| Situation | Action |
|-----------|--------|
| Synthesised chords clip (amplitude > 1.0) | Normalise each note before mixing — check peak amplitude after load |
| Spectrogram looks identical to single note | Mixing levels wrong — verify all three notes are audible |
| Pipeline produces fewer than 6 distinct labels | Check folder structure: `data/raw/chords/{Cmaj,Dmin,...}/` |

---

## Scope check

3 steps, ~2 hours. The pipeline already exists — this is a script to generate input data.
