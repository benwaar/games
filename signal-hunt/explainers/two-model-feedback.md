# The Two-Model Feedback Design

Phase 4 produces two chord classifiers that work together — not alternatives, but a pipeline. Understanding why requires understanding what each model is good at and what a piano student actually needs to hear.

---

## The problem with one model

A single classifier has to pick a lane:

- **Chord-name classifier** ("is this Cmaj or Amin?") is simple and accurate, but when it's wrong all it can say is "wrong chord." The student knows that already.
- **Note-set classifier** ("which notes are on?") is more powerful but harder to train. Its accuracy metric — exact-match, all notes correct — is harsh. Getting 2 of 3 notes right is still useful; exact-match calls it a failure.

Neither on its own gives a student everything they need.

---

## The two-pass pipeline

Both models run on the same clip:

```
student plays chord
        ↓
   name model → "you played A minor, should be C major"    ← verdict
        ↓
  notes model → "you have A4 and C4, missing E4"           ← correction
```

**Pass 1 — verdict:** The name model (6-class, CrossEntropyLoss) gives the high-confidence chord label. 96% val_acc. Fast, reliable. Tells the student *what* they got wrong.

**Pass 2 — correction:** The note-set model (12-label, BCEWithLogitsLoss) identifies which of the 12 chromatic notes are present. 76% exact-match — but on misses it typically identifies 2 of 3 notes correctly, pointing directly at the one to fix. Tells the student *why*.

---

## Why this mirrors real teaching

A piano teacher doesn't just say "wrong" and move on. They say:

> "That's A minor, not C major. You've got two of the three — your thumb on C and your little finger on A are fine, but your middle finger is landing on E flat instead of E natural."

The name model is the teacher's verdict. The note-set model is the teacher pointing at the finger.

---

## Why exact-match understates the notes model

Exact-match accuracy requires all three notes correct. If the model predicts C4 and G4 but misses E4, that's a 0 — even though it correctly identified 2 of 3 notes and pointed at the specific gap.

For the tutor use case, per-note recall is the more meaningful metric: "of the notes that should be on, how many did the model detect?" A model that consistently finds 2 of 3 is still useful — it narrows the correction from "wrong chord" to "you're missing this specific note."

M20 evaluation looks at per-note F1 to get the fuller picture.

---

## The tradeoff in numbers (after M19 training)

| Model | Metric | Score | Random baseline |
|-------|--------|-------|----------------|
| Name (6-class) | val accuracy | 96% | 16.7% |
| Notes (12-label) | exact-match | 76% | ~1.6% |

The name model's higher number isn't because it's more capable — it's because chord-name classification is a simpler task (one of 6 labels) compared to getting all 3 notes correct simultaneously. The notes model is doing harder work.

---

## Business parallel

This pattern — coarse label first, fine-grained detail second — appears across many ML products:

- **Fraud detection:** "this transaction is suspicious" (binary) → "these three features are anomalous" (explanation)
- **Content moderation:** "this post violates policy" (classifier) → "here's the specific rule and the phrase that triggered it" (explainer model)
- **Medical imaging:** "abnormality detected" (detection model) → "here's the region and probable type" (segmentation model)

In each case the coarse classifier sets the threshold for action; the fine-grained model provides the actionable detail. Neither alone is as useful as both together.

See: [model/chord.py](../model/chord.py) — both head implementations  
See: [model/chord_train.py](../model/chord_train.py) — `--mode name` and `--mode notes`
