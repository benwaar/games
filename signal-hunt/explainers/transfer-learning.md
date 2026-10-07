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

## What to expect

| Mode | Convergence | Final accuracy |
|------|-------------|----------------|
| Frozen (head only) | Fast — typically < 20 epochs | Good — backbone features transfer well |
| Fine-tune (all layers) | Slower — needs more epochs | Often better — backbone adapts to piano |

With 252 samples across 12 classes (21 per class), expect:
- Random baseline: **8.3%** (1/12)
- Frozen head: likely **50–80%** — piano notes have clear frequency separation
- Fine-tune: potentially higher, especially for adjacent semitones (C4 vs C#4)

---

## Why not just train from scratch?

You can — and it's a useful experiment (`model/train.py` works fine with note data). But:

- **Less data per class.** 21 samples per note vs 77 per sound type in Phase 2. The backbone already has shape knowledge; starting fresh forces the conv layers to relearn it.
- **Slower convergence.** Transfer typically reaches useful accuracy 2–3× faster.
- **Better generalisation on small datasets.** The pretrained features act as a regulariser.

The `--compare` flag in `transfer_train.py` logs both runs so you can see the difference yourself.

---

## Python concepts used here

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
