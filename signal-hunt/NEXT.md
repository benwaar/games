# M22 — Documentation & Project Wrap-Up

**Goal:** Write the final explainers, complete the Phase 4 summary in PLAN.md, and leave the project in a state someone can clone and run end-to-end.

## Status going in

M21 complete. The full Signal Hunt pipeline is built:
- Phase 1: data pipelines (ingest, augment, features, batch)
- Phase 2: 3-class CNN (hum/whistle/clap, 100% test accuracy)
- Phase 3: 12-class note classifier (transfer learning, 89.5% test accuracy)
- Phase 4a: chord classifiers — name (96.2%) and note-set (73.1% exact-match)
- Phase 4b: CNN-RNN progression classifier (66.7% val, 4 classes, data-limited)

M22 is documentation only — no new code.

---

## Steps

### Step 1 — Explainer: RNNs and the CNN→RNN reshape

Write `explainers/rnn-sequence-modelling.md`:
- GRU vs LSTM (gating mechanisms, when to use each)
- Why vanishing gradients matter in sequences and what gating fixes
- Bidirectional: seeing past AND future context
- The CNN→RNN reshape: `(B, C, freq, time)` → mean over freq → `(B, time, features)`
- Why this reshape is the most common source of bugs in CNN-RNN hybrids
- Business parallel: any sequence classification (clickstreams, log events, transaction chains)

Add entry to `signal-hunt/explainers/README.md`.

---

### Step 2 — Explainer: chord progressions as sequence modelling

Write `explainers/chord-progressions.md`:
- What a progression is (ordered sequence of chords, not a bag)
- Why order matters: I-IV-V-I ≠ V-IV-I-I (different musical meaning)
- How synthetic progressions give exact labels for free
- The gap between 66.7% (16 clips) and what more data would do
- What per-step decoding would add (label each chord, not just the whole sequence)
- Link to the two-model tutor design

Add entry to `signal-hunt/explainers/README.md`.

---

### Step 3 — Update README with Phase 4 and full usage

Update `signal-hunt/README.md`:
- Add Phase 4a and 4b to "What this builds"
- Add progression training and pipeline commands
- Update project structure (progression.py, progression_train.py, batch_progressions.py, synthesise_progressions.py)

---

### Step 4 — Phase 4 summary in PLAN.md

Add a Phase 4 summary section (like Phase 2/3 have) with milestones table and lessons.

---

### Step 5 — Close M22

- Tick all M22 items in PLAN.md
- Update memory

**Commit:** `docs(signal-hunt): M22 — RNN explainers, progression explainer, Phase 4 summary`

---

## Note on multi-label explainer

`explainers/multi-label-evaluation.md` was written as part of M20 — M22 originally planned to add this, but it's already done.
