# Signal Hunt

A deep learning project — from raw audio to a piano teacher prototype, built phase by phase.

## What it builds

| Phase | Task | Result |
|-------|------|--------|
| 1 | Data pipeline — ingest, augment, Mel-spectrograms | `(1, 128, 65)` tensors, 7 augmentations per clip |
| 2 | Sound type CNN (hum / whistle / clap) | 100% test accuracy |
| 3 | Piano note classifier, transfer learning | 89.5% test accuracy, 12 classes |
| 4a | Chord detection — name + note-set models | 96.2% / 73.1% exact-match |
| 4b | Chord progression CNN-RNN | 66.7% on 4 classes |

The end result: drop in a chord recording and get piano tutor feedback.

```bash
bash demo_chord.sh chord.wav --expected Cmaj
# Chord:   Amin  ✗  (expected Cmaj)
# Notes:   A4 ✓  C4 ✓  E4 ✓
# Missing: G4
```

## Quick start

```bash
bash setup.sh
source .venv/bin/activate
python hello_audio.py       # verify everything works
```

## Run each phase

```bash
# Phase 2 — classify sound type
bash demo.sh path/to/sound.wav

# Phase 3 — classify piano note
bash demo_note.sh path/to/note.wav

# Phase 4a — identify chord (two-pass: verdict + correction)
bash demo_chord.sh path/to/chord.wav --expected Cmaj
```

## Train

```bash
python -m model.train                              # Phase 2
python -m model.transfer_train --compare           # Phase 3
python -m model.chord_train --mode name --mode notes  # Phase 4a
python -m model.progression_train                  # Phase 4b
```

## Learn

- Project record (what was built, decisions, results): [PLAN.md](PLAN.md)
- Project-specific explainers: [explainers/](explainers/README.md)
- Shared concept explainers: [../explainers/](../explainers/README.md)
- Full command reference and project structure: [DOCS.md](DOCS.md)
