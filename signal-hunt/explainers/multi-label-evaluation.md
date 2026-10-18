# Multi-Label Evaluation

Single-label classification has one right answer per input — the model either gets it or it doesn't. Multi-label classification can have multiple right answers simultaneously. That changes what the metrics mean and how you choose a decision threshold.

---

## The difference

**Single-label (Phase 2/3):** Each clip is exactly one class. Accuracy = fraction correct. Confusion matrix shows which classes get swapped.

**Multi-label (Phase 4 notes model):** Each clip has a *set* of active labels. A Cmaj chord has C4, E4, and G4 simultaneously on. The model outputs a probability for each of the 12 notes independently, and you threshold each one separately.

```python
# Single-label: pick the highest logit
pred = logits.argmax(dim=1)

# Multi-label: threshold each independently
pred = (torch.sigmoid(logits) > 0.5)   # shape (B, 12), each True/False
```

---

## Exact-match accuracy

The strictest metric: a prediction is only correct if *every* label matches exactly.

```python
correct = (preds == targets).all(dim=1)   # all 12 notes must match
exact_match = correct.float().mean()
```

For a 3-note chord, the model must get all 3 notes right (and all 9 non-notes right). Getting 2 of 3 correct counts as 0 — the same as getting 0 right.

**This is harsh.** It's a useful headline number but understates the model's usefulness. For the piano tutor, a model that finds 2 of 3 notes and points at the specific missing note is still actionable feedback.

---

## Per-label F1

Treats each note as a separate binary classification problem and computes F1 independently.

```
F1 for note C4 = 2 × (precision_C4 × recall_C4) / (precision_C4 + recall_C4)
```

Where:
- **Precision:** of all clips the model said C4 was on, what fraction actually had C4?
- **Recall:** of all clips that actually had C4, what fraction did the model detect?

Then average across all notes (macro F1) for an overall score.

**Why this is more informative than exact-match:** A model that reliably detects 6 of 7 notes at F1=0.95 per note is very useful even if its exact-match score is low. Per-label F1 tells you which notes are hard and which are easy — that's diagnostic information the piano tutor can use.

---

## Phase 4 results

| Note | F1 | Appears in chords |
|------|----|------------------|
| B4 | 1.00 | Emin, Gmaj |
| E4 | 0.96 | Cmaj, Emin, Amin |
| G4 | 0.89 | Cmaj, Emin, Gmaj |
| A4 | 0.88 | Dmin, Fmaj, Amin |
| C4 | 0.87 | Cmaj, Fmaj, Amin |
| D4 | 0.82 | Dmin, Gmaj |
| F4 | 0.82 | Dmin, Fmaj |

D4 and F4 are weakest — both appear in only 2 chords, so they have less training signal than notes that appear in 3. This is a data problem, not a model problem. More chord types containing D4 and F4 would close the gap.

Precision is high across all notes (0.80–1.00): when the model says a note is on, it's almost always right. The errors are false negatives — occasionally missing a note — rather than false positives — incorrectly flagging one. For the tutor, this is the better failure mode: a missed note is less confusing than a wrongly flagged one.

---

## Threshold selection

The sigmoid output is a probability in (0, 1). The threshold controls the precision/recall tradeoff:

| Threshold | Effect |
|-----------|--------|
| Low (0.3) | More notes flagged → higher recall, lower precision. More false positives — notes flagged that aren't there |
| Default (0.5) | Balanced. Optimal for this dataset |
| High (0.7) | Fewer notes flagged → higher precision, lower recall. Fewer false positives but more missed notes |

**Threshold sweep results (Phase 4, per-note macro F1):**

| Threshold | Exact-match | Per-note F1 |
|-----------|------------|-------------|
| 0.3 | 34.6% | 0.500 |
| **0.5** | **73.1%** | **0.520** |
| 0.7 | 38.5% | 0.470 |

0.5 wins on both metrics here. On a larger or noisier dataset you'd tune this on a validation set — the threshold that maximises F1 on validation is the one to use at test time.

---

## The evaluation hierarchy

For multi-label classification, read metrics in this order:

1. **Per-label recall** — does the model find the notes that are there? (tutor needs high recall for the correction role)
2. **Per-label precision** — when it flags a note, is it right? (false positives confuse students)
3. **Per-label F1** — the harmonic mean, overall health per note
4. **Exact-match** — all correct simultaneously; useful for tracking training progress but harsh as a final verdict

> **In practice:** Multi-label evaluation is common in any ML task where inputs can belong to multiple categories simultaneously — a support ticket can be both urgent and a billing issue, a product can be in both "electronics" and "home office." Exact-match is the strictest production standard; per-label F1 is the right metric for diagnosing where the model is weak.

See: [model/chord_evaluate.py](../model/chord_evaluate.py) — threshold sweep, per-note F1, confusion analysis  
See: [../../explainers/evaluation.md](../../explainers/evaluation.md) — single-label evaluation (precision, recall, F1, confusion matrix)
