# M11 Plan — Inference Script

**Goal:** Build `model/predict.py` — a script that takes a raw `.wav` file,
runs it through the full pipeline (ingest → features → model), and outputs
a prediction with confidence score.

## Prerequisites

- M10 ✅ — `output/best_model.pt` exists and is evaluated
- venv active: `source .venv/bin/activate`

---

## Loop: Plan → Implement → Test → Document → Commit → Tick → Next

---

### Step 1 — `predict` function

- [ ] `model/predict.py` — `predict(wav_path, checkpoint_path)`:
  - Load checkpoint (model + label_map + config)
  - Run `ingest` → `extract_features` on the wav file
  - Forward pass → softmax → top class + confidence
  - Returns `{"class": "hum", "confidence": 0.923, "all_scores": {...}}`

**Test:** `predict("data/raw/hum/h-1.wav", "output/best_model.pt")` returns
a dict with `"class"` and `"confidence"` keys, confidence in `[0, 1]`.

**Docs:**
- [ ] Update `evaluation.md` — add a note that inference uses the same
  pipeline as training (same ingest + features) so there's no train/test skew

**Commit:** `feat(signal-hunt): predict function — wav → class + confidence`

---

### Step 2 — CLI

- [ ] `python -m model.predict data/raw/hum/h-1.wav` → `"hum (92.3% confidence)"`
- [ ] `--checkpoint` flag to specify a different model
- [ ] Prints all class scores if `--verbose`

**Test:** CLI runs end-to-end on a known file, prints expected class.

**Docs:**
- [ ] `README.md` — add inference command to usage section

**Commit:** `feat(signal-hunt): predict CLI`

---

### Step 3 — end-to-end test with a new recording

- [ ] Record or use an existing clip not in the training set
- [ ] Run `python -m model.predict <new_file.wav>` and verify it predicts correctly
- [ ] Document the result in `evaluation.md` (what file, what prediction, what confidence)

**Docs:**
- [ ] `evaluation.md` — end-to-end inference section with real example

**Commit:** `docs(signal-hunt): end-to-end inference result`

---

### Step 4 — close out M11

- [ ] Mark M11 checkboxes `[x]` in `PLAN.md`
- [ ] Update `NEXT.md` to point at M12 (documentation & consolidation)

**Commit:** `docs(signal-hunt): tick M11 checkboxes, point NEXT at M12`

---

## Files to create

```
model/predict.py        # predict(), CLI
tests/test_predict.py   # predict returns correct shape/types, CLI runs
```

## Gate (from PLAN.md)

`python -m model.predict some_file.wav` → `"hum (92.3% confidence)"`. All tests green.
