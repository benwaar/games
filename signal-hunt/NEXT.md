# M15 — Training on Notes: Evaluation & Scratch Baseline

**Goal:** Evaluate the fine-tuned model's per-note accuracy, run a from-scratch baseline for comparison, and document what the numbers show.

## Status going in

M14 is done. We have:
- `output/transfer/finetune/best_model.pt` — 76.3% val accuracy, 60 epochs, fine-tuned from Phase 2 weights
- `output/transfer/frozen/best_model.pt` — 36.8%, head-only training
- `data/processed/notes/` — 252 tensors, 12 balanced classes (21 per class)
- `model/evaluate.py` — already accepts `--checkpoint` and `--processed-dir`; just needs the confusion matrix plot fixed for 12 classes (currently hardcoded figsize for 3 classes)

Gate from PLAN.md: model above 50% ✅ (already 76.3%). Per-note confusion matrix showing nearby-note confusions is the deliverable.

## What we're skipping from the original M15 spec and why

**Class-weighted CrossEntropyLoss** — data is perfectly balanced (21 per class, by design of the Iowa dataset). No imbalance to correct. Useful concept to explain ("when would you use it?") but no reason to implement here.

**Curriculum strategy** (C4/E4/G4 → add semitones) — at 76.3% accuracy, adding complexity doesn't improve learning outcomes and is non-trivial to wire up correctly (requires filtered dataset splits per curriculum stage). Document the concept, not the implementation.

Both will be covered briefly in explainer updates at the end.

---

## Loop: Plan → Implement → Test → Commit → Tick → Next

---

### Step 1 — Fix evaluate.py for 12 classes, run it

`run_evaluation` is already generic (reads `num_classes` from the checkpoint). The only problem is the confusion matrix plot: `figsize=(5, 4)` was designed for 3 classes — 12 will have overlapping axis labels. Fix: scale figsize with `num_classes`, rotate tick labels.

**Changes to `model/evaluate.py`:**
- Scale figure: `figsize = (max(5, n * 0.8), max(4, n * 0.7))` where `n = len(class_names)`
- Rotate x-axis labels: `ax.set_xticklabels(class_names, rotation=45, ha="right")`
- Keep everything else — it already works for any label set

**Run:**
```bash
python -m model.evaluate \
  --checkpoint output/transfer/finetune/best_model.pt \
  --processed-dir data/processed/notes \
  --output-dir output/transfer/finetune \
  --images-dir explainers/images
```

**Test:** `eval_report.txt` has 12 rows (one per note). Confusion matrix is readable. Per-note F1 scores visible.

**Watch for:** Which notes have low recall? Adjacent semitones (C4/Db4, F4/Gb4, G4/Ab4) are the expected trouble pairs.

**Docs:** None yet — save notes from the report output for Step 3.

**Commit:** `fix(signal-hunt): scale evaluate.py confusion matrix for n classes`

---

### Step 2 — Add CLI flags to train.py, run scratch baseline on notes

`model/train.py` works for any dataset via `TrainConfig`, but the CLI only exposes `--epochs`, `--lr`, `--batch-size`, `--patience`, `--seed`. There's no way to point it at notes data from the command line without editing code.

**Changes to `model/train.py` main():**
- Add `--processed-dir` (default `data/processed`)
- Add `--num-classes` (default `3`)
- Wire both into `TrainConfig`

**Run scratch training:**
```bash
python -m model.train \
  --processed-dir data/processed/notes \
  --num-classes 12 \
  --epochs 60 \
  --patience 8
```

Save output to `output/scratch_notes/` — add `--output-dir` flag too while we're there.

**Compare to fine-tune:**

| Mode | Best val_acc | Epochs to 50% |
|------|-------------|--------------|
| Fine-tune (Phase 2 weights) | 76.3% | ~20 epochs |
| Scratch (random init) | ? | ? |

The learning point: transfer converges faster and/or reaches higher accuracy on the same data.

**Test:** Scratch run completes. `output/scratch_notes/best_model.pt` exists. Val accuracy beats random (8.3%) after 10 epochs.

**Docs:** Note the comparison in transfer-learning.md.

**Commit:** `feat(signal-hunt): add --processed-dir / --num-classes / --output-dir to train.py CLI`  
**Commit:** `docs(signal-hunt): scratch vs transfer comparison, M15 per-note results`

---

### Step 3 — Document findings, tick M15

With the evaluation report and scratch comparison in hand, update the docs and close out the milestone.

**Updates:**
- `explainers/transfer-learning.md` — add "scratch vs transfer" comparison table with actual numbers from Step 2
- Add a short section on class-weighted loss: what it is, when you'd use it (imbalanced data), why we didn't need it here
- Add a short section on curriculum learning: the concept, when it helps (hard-to-distinguish classes with limited data), why we didn't need it at 76.3%
- `PLAN.md` — tick M15 checkboxes, add results summary
- `NEXT.md` — rewrite for M16

**Commit:** `docs(signal-hunt): tick M15, per-note analysis + curriculum/weighted-loss explainers`

---

## Stop conditions

| Situation | Action |
|-----------|--------|
| Scratch accuracy matches fine-tune | Transfer learning didn't help — likely Phase 2 features weren't relevant. Document and move on; the negative result is still a finding. |
| Confusion matrix shows no semitone confusions | Model has learned clean pitch separation — good result. Document and go to M16. |
| Confusion matrix shows all confusions are semitone pairs | Expected — Mel resolution may blur nearby pitches. Note as motivation for CQT in "What's tricky" section. |
| train.py CLI changes break existing tests | Fix before committing. `test_train.py` tests the `train()` function directly — CLI changes shouldn't affect it, but check. |

---

## Scope check

3 steps. Estimated ~2 hours. M15 was planned for 3 hours but some items were cut (curriculum, weighted loss implementation). On track.

After M15, M16 is: confusion matrix analysis, domain gap test (real piano clips if recorded), error analysis.
