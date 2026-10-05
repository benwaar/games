# M10 Plan — Evaluation & Analysis

**Goal:** Load the best checkpoint from M9, run it on the held-out test set, and produce
a full evaluation report — accuracy, per-class precision/recall/F1, confusion matrix plot,
and loss curves — to understand what the model learned and where it fails.

## Prerequisites

- M9 ✅ — `output/best_model.pt` and `output/history.json` exist
  (run `python -m model.train --epochs 50` first if missing)
- venv active: `source .venv/bin/activate`

## Research to do first

Before writing any code, document:
- Confusion matrix — what it shows, how to read it
- Precision, recall, F1 — what each measures and when each matters
- Train vs val loss curves — what overfitting looks like visually

Write these to a new `explainers/evaluation.md` before building.

---

## Loop: Plan → Implement → Test → Document → Commit → Tick → Next

---

### Step 1 — load checkpoint and run test set

- [ ] `model/evaluate.py` — `evaluate_checkpoint(checkpoint_path, processed_dir)`:
  - Load checkpoint (model weights + label_map + config)
  - Run `load_splits` with same seed to get the test split
  - Collect all predictions and true labels from the test set
  - Returns `(preds, labels, label_map)`

**Test:** returns arrays of correct length, all values in `[0, num_classes-1]`.

**Docs:**
- [ ] `evaluation.md` — written before coding (research step)
- [ ] `explainers/README.md` section — add evaluation step

**Commit:** `feat(signal-hunt): evaluate_checkpoint — load and run test set`

---

### Step 2 — metrics

- [ ] `print_metrics(preds, labels, label_map)` — accuracy, per-class precision/recall/F1
  using `sklearn.metrics.classification_report`
- [ ] Save report to `output/eval_report.txt`

**Test:** report file written, contains class names, contains "accuracy".

**Docs:**
- [ ] `evaluation.md` — explain what the metrics mean for our 3-class task

**Commit:** `feat(signal-hunt): classification report`

---

### Step 3 — plots

- [ ] Confusion matrix heatmap → `explainers/images/confusion_matrix.png`
- [ ] Train vs val loss curves → `explainers/images/loss_curves.png`
- [ ] Both saved as images (not just shown), linked from `evaluation.md`

**Test:** both image files exist after running.

**Docs:**
- [ ] `evaluation.md` — embed images, explain what to look for in each

**Commit:** `feat(signal-hunt): confusion matrix and loss curve plots`

---

### Step 4 — CLI

- [ ] `python -m model.evaluate` — runs all of the above, prints report, saves plots
- [ ] Accepts `--checkpoint output/best_model.pt` and `--processed-dir data/processed`

**Test:** CLI runs end-to-end, all output files present.

**Docs:**
- [ ] `README.md` — add evaluation command to usage section

**Commit:** `feat(signal-hunt): evaluate CLI`

---

### Step 5 — close out M10

- [ ] Mark M10 checkboxes `[x]` in `PLAN.md`
- [ ] Update `NEXT.md` to point at M11

**Commit:** `docs(signal-hunt): tick M10 checkboxes, point NEXT at M11`

---

## Files to create

```
model/evaluate.py               # evaluate_checkpoint, print_metrics, CLI
tests/test_evaluate.py          # checkpoint loads, metrics computed, files written
explainers/evaluation.md        # confusion matrix, precision/recall/F1, loss curves
explainers/images/              # confusion_matrix.png, loss_curves.png
```

## Gate (from PLAN.md)

Evaluation report generated. Model meaningfully above random (>33%) on test data.
Confusion matrix shows the model learned real differences between classes.
