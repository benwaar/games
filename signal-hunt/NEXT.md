# Phase 3 — Note & Pitch Classification

Phase 2 complete ✅. Next: extend the classifier to recognise **which note** is being
hummed or whistled (C4, D4, E4, ... up to one octave = 12 classes).

## Before starting

Read the Phase 3 plan in [PLAN.md](PLAN.md) — M13 through M17.

Phase 3 introduces:
- Transfer learning (freeze Phase 2 CNN, replace classification head)
- Larger label space (12 classes vs 3)
- Class imbalance (some notes easier to produce consistently)
- Curriculum learning (start with C/E/G, add semitones gradually)

## Data requirement (do this before coding)

Need ~10 recordings per note per sound type. Start with one octave of hummed notes
(C4–B4 = 12 classes). Use a tuner app or piano as a reference pitch.

Folder structure:
```
data/raw/notes/
  C4_hum/   ← 10 recordings of C4 hummed
  D4_hum/
  ...
  C4_whistle/
  ...
```

Run `/study` to start M13.
