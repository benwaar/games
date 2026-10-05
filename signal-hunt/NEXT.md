# M8 Plan — CNN Architecture

**Goal:** Build `model/cnn.py` — a 3-class CNN classifier that takes `(B, 1, 128, 65)` spectrograms
and outputs `(B, 3)` logits. Parameter count under 500K.

## Prerequisites

- M7 complete ✅ — `load_splits` and `DataLoader` working
- venv active: `source .venv/bin/activate`

## Research to do first

Before writing any code, read:
- What is a Conv2d layer? (kernel, stride, padding — how it scans the spectrogram)
- What is BatchNorm2d? (why normalise between layers)
- What is Global Average Pooling vs Flatten? (why GAP reduces overfitting)
- What is Dropout? (why it helps on small datasets)

Use `/research` or `/study` at the start of the session to document these before building.

---

## Loop: Plan → Implement → Test → Document → Commit → Tick → Next

---

### Step 1 — forward pass skeleton

- [ ] Create `model/cnn.py`
- [ ] `SoundClassifier(nn.Module)` with `__init__` and `forward`
- [ ] Architecture: 3× (Conv2d → BatchNorm2d → ReLU → MaxPool2d) → Global Average Pooling → Linear → Dropout → Linear
- [ ] Input `(B, 1, 128, 65)` → Output `(B, 3)` logits

**Test:** dummy batch `torch.zeros(4, 1, 128, 65)` → output shape `(4, 3)`, no NaNs, no errors.

**Docs:**
- [ ] New explainer: `explainers/cnn-architecture.md` — what each layer does, C/JS callouts
- [ ] `explainers/README.md` section 7 — update from stub to real description + See: link
- [ ] `explainers/libraries.md` — `torch.nn` layers used

**Commit:** `feat(signal-hunt): SoundClassifier CNN — forward pass`

---

### Step 2 — parameter count check

- [ ] Log total trainable parameters (target: <500K)
- [ ] `python -c "from model.cnn import SoundClassifier; m = SoundClassifier(); print(sum(p.numel() for p in m.parameters() if p.requires_grad), 'params')`
- [ ] If over 500K: reduce channels in conv layers

**Test:** parameter count printed and under 500K.

**Docs:**
- [ ] Add parameter count to `cnn-architecture.md` with explanation of why we target <500K for this task

**Commit:** included in Step 1 commit if under budget; separate fix commit if channels need adjusting

---

### Step 3 — softmax sanity check

- [ ] `torch.softmax(output, dim=1).sum(dim=1)` ≈ 1.0 for all items in batch
- [ ] No all-zero outputs, no all-identical outputs for different random inputs

**Test:** assert softmax sums to 1 within tolerance, assert outputs vary across batch.

**Docs:**
- [ ] `cnn-architecture.md` — explain why we check softmax (catch dead neurons, weight init issues)

**Commit:** included in Step 1 commit

---

### Step 4 — close out M8

- [ ] Mark M8 checkboxes `[x]` in `PLAN.md`
- [ ] Update `NEXT.md` to point at M9

**Commit:** `docs(signal-hunt): tick M8 checkboxes, point NEXT at M9`

---

## Files to create

```
model/cnn.py            # SoundClassifier
tests/test_cnn.py       # forward pass, shapes, parameter count, softmax
explainers/cnn-architecture.md
```

## Gate (from PLAN.md)

Forward pass on dummy batch produces `(B, 3)` logits. No NaNs. Softmax sums to 1. Parameter count ≤500K.
