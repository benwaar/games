# M16 — Evaluation & Domain Gap Test

**Goal:** Analyse the confusion matrix, understand which confusions are musically sensible, and document the evaluation findings. Optionally test on real piano clips if recorded.

## Status going in

M15 is done:
- Fine-tune test accuracy: **89.5%** (34/38 samples)
- Confusion matrix saved to `explainers/images/confusion_matrix.png`
- Per-note report saved to `output/transfer/finetune/eval_report.txt`
- F4 is the hardest note (F1=0.40): adjacent-semitone confusion with E4/Gb4
- G4/Ab4 also confused (adjacent semitone pair)
- 6/12 notes perfect F1=1.00

Gate from PLAN.md: confusion patterns make musical sense ✅ (already confirmed). Real piano test is the optional stretch goal.

---

## Loop: Plan → Implement → Test → Commit → Tick → Next

---

### Step 1 — Error analysis write-up

The main work of M16 is interpreting what the confusion matrix shows and documenting it. The evaluation already ran in M15. Now explain the patterns.

**Questions to answer:**
- Which notes confuse the model, and is the pattern musically sensible?
- Is F4's struggle a data problem (only 3 raw clips, one test sample) or a model problem (Mel resolution)?
- What would fix it — more data, higher `n_mels`, or CQT?

**New explainer: `explainers/evaluation-phase3.md`**
- Confusion matrix walkthrough: which cells are non-zero, why
- Semitone adjacency as the pattern: frequency proximity → spectrogram proximity → model confusion
- F4 specifically: sitting between E4 (330 Hz) and Gb4 (370 Hz), both at semitone distance
- Small test set caveat: 38 samples, 3 per note — one wrong prediction changes F1 by 33%
- Mel resolution limit: default `n_mels=128` represents ~86 Hz per bin at the C4/B4 range. C4→C#4 is 15 Hz. That's smaller than one bin — the model is working at or near the resolution limit.
- What would help: more data per note, or CQT (Constant-Q Transform) — log-frequency bins aligned to musical pitch

**Commit:** `docs(signal-hunt): M16 per-note error analysis and evaluation explainer`

---

### Step 2 — Real piano clips (optional)

If you have an electric piano, record ~3 clips per note (C4–B4, ~2 seconds each) and save to `data/raw/notes/{note}/my_*.wav`. Then run:

```bash
python -m pipeline.batch data/raw/notes data/processed/notes_real --source-dir notes
python -m model.evaluate \
  --checkpoint output/transfer/finetune/best_model.pt \
  --processed-dir data/processed/notes_real \
  --output-dir output/eval_real \
  --images-dir explainers/images
```

**What to expect:** accuracy drop from 89.5%. Iowa piano is clean studio recordings. Real piano has sustain, room acoustic, possibly slight tuning differences. The gap size tells you how much augmentation (reverb, noise) the model needs.

**If you don't have a piano:** skip. The M16 gate ("evaluation report with per-note metrics") is already met from M15.

**Commit (if done):** `feat(signal-hunt): domain gap test — real piano vs Iowa training data`

---

### Step 3 — Tick M16, point NEXT at M17

- Tick M16 checkboxes in PLAN.md, add results summary
- Rewrite NEXT.md for M17 (inference `--mode note`, docs consolidation)

**Commit:** `docs(signal-hunt): tick M16 checkboxes, NEXT → M17`

---

## Stop conditions

| Situation | Action |
|-----------|--------|
| Real piano accuracy < 30% | Domain gap too large — add reverb augmentation and retrain before M17 |
| Real piano accuracy > 70% | Model generalises well — proceed to M17 without augmentation changes |
| No piano available | Skip Step 2 entirely — gate is already met |

---

## Scope check

2 steps (Step 2 is optional). Estimated ~1–2 hours. The heavy lifting (training, evaluation) is already done — M16 is mostly interpretation and documentation.
