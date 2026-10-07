# Transfer Learning

## What is it?

Transfer learning means taking a model trained on one task and reusing it for a different (but related) task — instead of starting from scratch.

In Signal Hunt, the Phase 2 CNN learned to tell apart **hums, whistles, and claps**. The Phase 3 CNN uses those same learned features to tell apart **12 piano notes**.

---

## Why does this work?

The Phase 2 conv blocks learned to detect **frequency-band patterns** in Mel spectrograms. Things like:
- Where on the frequency axis the signal is brightest
- How quickly the brightness rises and falls (onset/attack)
- Whether there are harmonic overtones above the fundamental

Those are exactly the features that distinguish piano pitches. A C4 (261 Hz) and a D4 (294 Hz) have their fundamental in different Mel bins — the CNN's first conv layer already knows how to find frequency peaks. We just need to teach the final layer what to *call* them.

**Analogy in software:** Imagine you've built a component that parses and validates JSON. You're now asked to validate YAML. You don't rewrite the validator from scratch — you keep the parsing logic and swap out the schema rules at the end. The conv blocks are the parsing logic. The classifier head is the schema rules.

---

## What we froze and why

The `SoundClassifier` has two parts:

```
conv_blocks  →  gap  →  classifier
```

| Part | What it learned | Frozen? |
|------|----------------|---------|
| `conv_blocks` (3 Conv2d blocks) | Frequency-band pattern detection | ✅ Yes (initially) |
| `gap` (Global Average Pooling) | Aggregation — no learned weights | — |
| `classifier` head | Which pattern = which class | ❌ No — replaced + retrained |

Freezing means: during backpropagation, gradients stop at the frozen layers. Their weights do not update. Only the new head learns.

### Why freeze first?

With 252 training tensors (21 per note class), the backbone has far more knowledge than the new head. If you let everything train at once from the start, the random new head can send large gradients into the backbone and **destabilise** the good features it already has.

Freeze-then-finetune is the standard recipe:
1. Freeze backbone. Train just the head for a few epochs (fast, stable).
2. Unfreeze everything. Fine-tune end-to-end at a low learning rate (slow, but lets the backbone adapt to piano timbre).

We do both in one command: `python -m model.transfer_train --compare`.

---

## The head swap

Original Phase 2 head:
```python
nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 3)
```

Phase 3 head (only the last layer changes):
```python
nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 12)
```

The `64` comes from Global Average Pooling — it's the number of channels in the last conv block. That doesn't change. Only the number of output classes changes: 3 → 12.

The first three layers of the head are carried over **with their weights** — the 64→32 projection already learned a useful compression of the frequency features.

---

## Running it

```bash
# Default: frozen backbone, train head only
python -m model.transfer_train

# Full fine-tune (unfreeze everything)
python -m model.transfer_train --no-freeze

# Run both and compare
python -m model.transfer_train --compare
```

Checkpoints saved to:
- `output/transfer/frozen/best_model.pt`
- `output/transfer/finetune/best_model.pt`

---

## What actually happened

We ran both modes for 60 epochs on the Iowa piano dataset (252 tensors, 12 classes).

| Mode | Val accuracy | vs random (8.3%) |
|------|-------------|-----------------|
| Frozen (head only) | **36.8%** | 4.4× |
| Fine-tune (all layers) | **76.3%** | 9.2× |

### Frozen: slow climb, never converged

Loss was still falling at epoch 60 — the run hit the epoch limit, not a ceiling. The head improved steadily but slowly (15.8% at epoch 1 → 36.8% at epoch 43/59) and never got a learning rate reduction (the scheduler didn't trigger because loss kept improving, just barely).

**Why it underperformed the prediction of 50–80%:**

The Phase 2 backbone learned voice features: smooth tonal hums, short chirpy whistles, broadband clap bursts. Piano is different enough that those features don't map cleanly to pitch. A hum at 261 Hz and a whistle at 880 Hz activate the backbone differently from a piano C4 vs C5 — same pitches, completely different timbre. The head could only work with what the frozen backbone gave it, and what it gave it was calibrated for the wrong source domain.

With more epochs, frozen would eventually converge — but probably around 40–50%, not 75%.

### Fine-tune: fast divergence from frozen, decisive LR kick

Fine-tune crossed frozen's best accuracy (36.8%) by epoch 16. By epoch 26 it was at 60.5%. The backbone adapted to piano's sharp attack and harmonic ladder within the first 15 epochs.

The curve was noisy (val accuracy bouncing 50%–76% in the final phase) — expected with only ~38 validation samples. Loss is the more reliable signal: 2.4 → 0.9 over 60 epochs.

The **learning rate halving at epoch 51** (5e-4 from 1e-3) was decisive. The model had plateaued around 71–74% for several epochs; the smaller steps let it settle into a better minimum. 76.3% arrived at epoch 52, one epoch after the LR drop.

### What the gap tells you

Both modes started from the same Phase 2 weights. Both beat random (8.3%) easily — the backbone isn't useless, it does know something about frequency patterns. But the 2× accuracy gap between frozen and fine-tune shows the backbone needed to adapt its *representation* of frequency, not just reuse it as-is.

This is the key insight about transfer learning: it's not binary. The question isn't "do the features transfer?" but "how much adaptation does the target domain need?" Here, partial transfer (frozen 36.8%) was real but limited. Full adaptation (fine-tune 76.3%) was much better with the same data.

### The standard recipe in practice

What we did here is actually the common pattern reversed for speed. The standard freeze-then-finetune recipe would be:
1. Freeze backbone, train head until converged (~head stabilised)
2. Unfreeze everything, fine-tune at a low lr

We ran them as separate experiments to compare. In a production setting, you'd do them sequentially — warm up the head first, then unlock the backbone — to get fine-tune accuracy with more stable early training.

---

## Why not just train from scratch? (We tried it)

We ran `model/train.py` from random initialisation on the same notes dataset for 60 epochs.

| Mode | Best val_acc | Epoch to reach 50% |
|------|-------------|-------------------|
| Transfer fine-tune | 76.3% | ~epoch 24 |
| Scratch (random init) | **78.9%** | ~epoch 26 |

**Scratch matched fine-tune.** At 60 epochs on this dataset, both converge to similar accuracy (~76–79%). Transfer is not a clear winner on the final number.

**Where transfer did help: early convergence.** Fine-tune was at 47.4% by epoch 16. Scratch was at 28.9%. Transfer got to usable accuracy roughly 10 epochs faster. If training is expensive or you need quick iterations, that advantage is real.

**Why scratch caught up:** The source domain (voice: hum/whistle/clap) is quite different from the target (piano pitch). The fine-tune backbone had to unlearn voice features and relearn piano features anyway — it just started from a closer-to-useful initial state. Given enough epochs, a randomly initialised backbone learns the same pitch features from scratch.

**When transfer wins more decisively:**
- Source and target domains are close (e.g. acoustic piano → electric piano, not voice → piano)
- Target dataset is very small (the backbone provides regularisation that scratch can't match)
- Pre-trained model was trained on a very large dataset (ImageNet, language model corpora)

In all three of those situations, the pretrained features transfer more directly and scratch can't compete within the same number of epochs. Here, with 252 balanced samples and a domain gap, the advantage was timing — not final accuracy.

---

## Per-note results (test set, fine-tune model)

89.5% test accuracy (34/38 samples). 6 of 12 notes are perfect (F1 = 1.00).

| Note | Precision | Recall | F1 | Freq (Hz) |
|------|-----------|--------|-----|-----------|
| C4 | 1.00 | 1.00 | 1.00 | 262 |
| Db4 | 1.00 | 1.00 | 1.00 | 277 |
| D4 | 1.00 | 1.00 | 1.00 | 294 |
| E4 | 1.00 | 1.00 | 1.00 | 330 |
| A4 | 1.00 | 1.00 | 1.00 | 440 |
| Bb4 | 1.00 | 1.00 | 1.00 | 466 |
| B4 | 0.75 | 1.00 | 0.86 | 494 |
| Eb4 | 0.75 | 1.00 | 0.86 | 311 |
| Gb4 | 0.75 | 1.00 | 0.86 | 370 |
| Ab4 | 1.00 | 0.67 | 0.80 | 415 |
| G4 | 1.00 | 0.67 | 0.80 | 392 |
| **F4** | **0.50** | **0.33** | **0.40** | **349** |

**The confusions are musically sensible.** F4 is the hardest note — it sits between E4 (330 Hz) and Gb4 (370 Hz), both adjacent semitones. The model gets only 1 of 3 F4 test samples right; the others are mis-classified as its neighbours. G4/Ab4 are also an adjacent semitone pair — both show recall 0.67.

Eb4 and B4 and Gb4 all show precision < 1.00 — something else is being misclassified as them. That something is most likely the adjacent-semitone neighbour in each case.

**Key pattern:** notes with no semitone neighbours at the edges of the scale (C4, D4, A4, Bb4) tend to be cleaner. Notes surrounded by two semitone neighbours (F4 between E4 and Gb4) are the hardest. This is exactly the Mel resolution limit described in PLAN.md "What's tricky" — a semitone is ~6% frequency difference, which is subtle at default `n_mels=128`.

---

## Two M15 techniques we didn't implement (and why)

### Class-weighted CrossEntropyLoss

Standard `CrossEntropyLoss` treats every class equally. Class-weighted loss applies a per-class multiplier to the loss — misclassifying a rare class hurts more than misclassifying a common one.

```python
weights = torch.tensor([w0, w1, ..., w11])  # inverse of class frequency
criterion = nn.CrossEntropyLoss(weight=weights)
```

**Why we skipped it:** our dataset is perfectly balanced — 21 samples per note by design (3 recordings × 7 augmentations, every note). With equal class counts, all weights would be 1.0, so it has no effect. It's the right tool for imbalanced data (e.g. if you had 50 C4 samples and only 5 F4 samples — which is realistic for a self-recorded dataset).

### Curriculum learning

The idea: start training on the easy cases (C4, E4, G4 — a major triad, well-separated in frequency), then add the hard cases (Db4, Eb4, Gb4 — semitones adjacent to notes you already know).

**Why we skipped it:** at 76.3–89.5% accuracy, the model is not struggling. Curriculum helps when the model can't learn at all with the full label set — too many similar classes, too little data. We didn't hit that. If accuracy had stalled at 20–30%, curriculum would be the next thing to try.

- **`requires_grad = False`** — tells PyTorch not to compute or store gradients for a parameter. Without a gradient, the optimiser has nothing to update, so the weight stays frozen.
- **`model.parameters()`** — returns all parameters in the model. Passing only `[p for p in model.parameters() if p.requires_grad]` to the optimiser ensures frozen layers are excluded from the update step.
- **`torch.load` + `model.load_state_dict`** — restores saved weights into a freshly constructed model. The architecture must match the saved weights — which is why we build the Phase 2 model first, load its weights, then swap the head.

---

## Business parallel

Transfer learning is routine in production ML:

- **Computer vision:** A model trained on ImageNet (1M images, 1000 classes) is used as a backbone for medical imaging classifiers trained on hundreds of samples.
- **NLP:** GPT-style language models are pretrained on the entire internet, then fine-tuned for specific tasks (sentiment analysis, contract review, customer support).
- **Recommendation systems:** Item embeddings trained on click data transfer to cold-start scenarios for new products.

The principle is always the same: generic features (edges, textures, grammar, co-occurrence) are expensive to learn. Domain-specific decisions (is this a cat or a tumour?) are cheaper once the generic features are in place.
