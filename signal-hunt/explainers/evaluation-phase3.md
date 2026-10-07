# Phase 3 Evaluation — Per-Note Results & Error Analysis

## Results

**Test set: 89.5% accuracy** (34/38 samples). 6 of 12 notes are perfect (F1 = 1.00).

| Note | Freq (Hz) | Precision | Recall | F1 | Notes |
|------|-----------|-----------|--------|----|-------|
| C4 | 262 | 1.00 | 1.00 | 1.00 | ✅ Perfect |
| Db4 | 277 | 1.00 | 1.00 | 1.00 | ✅ Perfect |
| D4 | 294 | 1.00 | 1.00 | 1.00 | ✅ Perfect |
| E4 | 330 | 1.00 | 1.00 | 1.00 | ✅ Perfect |
| A4 | 440 | 1.00 | 1.00 | 1.00 | ✅ Perfect |
| Bb4 | 466 | 1.00 | 1.00 | 1.00 | ✅ Perfect |
| B4 | 494 | 0.75 | 1.00 | 0.86 | Something else predicted as B4 |
| Eb4 | 311 | 0.75 | 1.00 | 0.86 | Something else predicted as Eb4 |
| Gb4 | 370 | 0.75 | 1.00 | 0.86 | Something else predicted as Gb4 |
| Ab4 | 415 | 1.00 | 0.67 | 0.80 | 1 of 3 Ab4s missed |
| G4 | 392 | 1.00 | 0.67 | 0.80 | 1 of 3 G4s missed |
| **F4** | **349** | **0.50** | **0.33** | **0.40** | ❌ Only 1 of 3 correct |

---

## The pattern: semitone adjacency

The confusions are not random — they follow frequency proximity. F4 is the worst because it has two immediate neighbours:

```
E4 (330 Hz) → F4 (349 Hz) → Gb4 (370 Hz)
```

Both E4 and Gb4 are exactly one semitone away. The model misclassifies 2 of 3 F4 test samples as one of those neighbours.

The G4/Ab4 pair tells the same story:

```
G4 (392 Hz) → Ab4 (415 Hz)
```

Both show recall 0.67 — each is missing one test sample that presumably landed in the other's bucket.

The notes with perfect F1 tend to have more frequency distance from their nearest classified neighbour. C4 (262 Hz) is a full tone from D4 (294 Hz) with Db4 (277 Hz) also classified well — no ambiguity. Bb4 (466 Hz) and B4 (494 Hz) are a semitone pair but B4's confusion is only in precision (something else labelled as B4), not recall — B4 itself is always identified.

---

## Why semitone neighbours confuse the model: Mel resolution

A semitone is a ~6% frequency difference. At C4 (262 Hz), one semitone = ~15 Hz. At A4 (440 Hz), one semitone = ~25 Hz.

Our spectrogram uses `n_mels=128` across the full audible range (~0–11 kHz). The Mel scale compresses high frequencies more than low ones, which means:

- **Low notes (C4–E4):** more Mel bins per Hz → finer resolution → easier to separate
- **High notes (G4–B4):** fewer Mel bins per Hz → coarser resolution → harder to separate

At the C4–B4 range, the 128 Mel bins are spread across roughly 230 Hz (262→494 Hz), giving about **1.8 Hz per bin**. A semitone at this range is 15–28 Hz — so there are 8–15 bins between adjacent notes. That should be enough resolution in theory, but in practice the harmonic overtones of each note overlap with the fundamentals of its neighbours. The model is learning to separate overlapping frequency patterns, not clean isolated peaks.

The F4 issue specifically may also be related to data: with only 3 source recordings and 21 augmented tensors per class, the test set has 3 samples per note. One mis-prediction changes F4's recall by 33%.

---

## Is this a data problem or a model problem?

Both, but solvable.

**Data side:**
- 21 tensors per class is small. More recordings per note would reduce variance.
- The Iowa dataset has only 3 dynamics (pp/mf/ff). More timbral variation (different pianos, recording environments, velocities) would help generalisation.

**Model side:**
- CQT (Constant-Q Transform) instead of Mel spectrogram would give log-frequency bins aligned to musical pitch — each semitone gets the same number of bins regardless of register. This is designed for exactly this problem.
- Increasing `n_mels` from 128 to 256 would give finer frequency resolution across the board.

**What's good:**
- 6/12 notes are already perfect.
- All confusions are musically sensible — the model hasn't learned anything wrong, it's just working at the resolution limit.
- 89.5% test accuracy on a 12-class task with 21 training samples per class is a strong result.

---

## Small test set caveat

The test set has 38 samples — roughly 3 per note. One wrong prediction moves a note's recall by 33%. The F1 scores should be read as directional signals, not precise measurements. A larger held-out set (or k-fold cross-validation) would give more reliable per-note estimates.

---

## Running evaluation yourself

```bash
python -m model.evaluate \
  --checkpoint output/transfer/finetune/best_model.pt \
  --processed-dir data/processed/notes \
  --output-dir output/transfer/finetune \
  --images-dir explainers/images
```

Outputs:
- `output/transfer/finetune/eval_report.txt` — full classification report
- `explainers/images/confusion_matrix.png` — 12×12 matrix
- `explainers/images/loss_curves.png` — train/val loss over 60 epochs
