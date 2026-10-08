# RNNs and the CNN→RNN Reshape

A CNN classifies one snapshot. A recurrent neural network classifies a *sequence* — it reads time steps one by one and maintains a hidden state that carries information from earlier steps to later ones. This document explains how GRUs work, why the bidirectional setup matters, and how to reshape a CNN's output into a sequence the RNN can read.

---

## The problem CNNs can't solve alone

A Mel-spectrogram of a single chord is a snapshot — one `(128, 65)` image. A CNN can classify it. But a chord *progression* is a sequence: Cmaj → Fmaj → Gmaj → Cmaj, changing over time. The CNN sees patterns within one frame; the RNN learns what patterns across frames mean.

```
CNN:  "this frame looks like a Cmaj chord"
RNN:  "Cmaj, then Fmaj, then Gmaj, then Cmaj — that's a I-IV-V-I progression"
```

---

## What a GRU does

A GRU (Gated Recurrent Unit) processes one time step at a time and maintains a *hidden state* `h` that carries information forward:

```python
h_t = GRU(x_t, h_{t-1})
```

At each step:
- `x_t` is the current input (the CNN features at time step t)
- `h_{t-1}` is the hidden state from the previous step
- `h_t` is the new hidden state — a compressed representation of everything seen so far

After reading all T time steps, `h_T` is used to classify the whole sequence.

**Why not a vanilla RNN?** Vanilla RNNs suffer from *vanishing gradients* — the gradient signal shrinks as it propagates back through many time steps, so the model forgets what happened early in the sequence. GRUs (and LSTMs) solve this with *gating mechanisms* that control what to remember and what to forget:

- **Update gate:** how much of the previous hidden state to keep
- **Reset gate:** how much of the previous hidden state to expose when computing the new candidate

These gates are learned, not hand-tuned. The model learns what's worth remembering for this specific task.

**GRU vs LSTM:** Both solve vanishing gradients. LSTMs have an extra cell state and three gates (input, forget, output) — more expressive but more parameters. GRUs have two gates — faster and often matches LSTM on shorter sequences. For chord progressions (~36 time steps), GRU is the right choice.

---

## Bidirectional

A standard RNN reads left-to-right. A *bidirectional* RNN runs two GRUs in parallel — one forward (left-to-right) and one backward (right-to-left) — and concatenates their hidden states:

```
h_forward  = GRU_forward(x_1, x_2, ..., x_T)      # sees past context
h_backward = GRU_backward(x_T, x_{T-1}, ..., x_1)  # sees future context
h_final    = concat(h_forward, h_backward)           # sees both
```

For chord progressions this matters: the model can use both "what came before this chord" AND "what came after" to decide what progression it is. The final hidden state (both directions) is what gets passed to the classification head.

---

## The CNN→RNN reshape

This is where most bugs happen. The CNN produces a 4D tensor; the GRU expects a 3D sequence.

**CNN output:** `(B, channels, freq, time)` — batch, 64 feature maps, 16 freq bins, 36 time steps (after 3× MaxPool2d reducing both spatial dimensions).

**GRU input:** `(B, time, features)` — batch, sequence length, feature size.

The reshape:

```python
features = conv_blocks(x)          # (B, 64, 16, 36)
features = features.mean(dim=2)    # (B, 64, 36)   — mean over freq axis
features = features.permute(0,2,1) # (B, 36, 64)   — swap to (batch, time, features)
out, _ = gru(features)             # (B, 36, hidden*2)
last = out[:, -1, :]               # (B, hidden*2) — last time step
```

**Why mean over freq?** You need to collapse the frequency dimension to get a per-time-step feature vector. Average pooling preserves energy information across freq bins. Flatten would work too but produces a much larger feature vector (16×64=1024 vs 64) — harder to train with a small dataset.

**Why use the last time step?** After reading the whole sequence, `out[:, -1, :]` contains the GRU's summary of everything it saw. With a bidirectional GRU, the last step of the forward pass has seen all previous steps; combining with the backward pass (which started from the end) gives a summary with full context.

---

## Signal Hunt implementation

```
(B, 1, 128, 291)     ← 6.75s progression spectrogram
        ↓  3× Conv2d + MaxPool2d
(B, 64, 16, 36)      ← 64 feature maps, 16 freq bins, 36 time steps
        ↓  mean(dim=2)
(B, 36, 64)          ← 36 time steps, each a 64-dim feature vector
        ↓  bidirectional GRU (hidden=64)
(B, 36, 128)         ← 128 = 64 forward + 64 backward
        ↓  last time step + dropout
(B, 128)
        ↓  Linear
(B, 4)               ← one logit per progression label
```

Total: 73,956 trainable parameters. The CNN backbone (25,798 params) was loaded from the Phase 4 chord-name checkpoint and fine-tuned end-to-end alongside the GRU.

**Gradient clipping** (`max_norm=1.0`) prevents the gradients from exploding during RNN backpropagation — a common issue with recurrent networks on short datasets where the loss surface is steep.

---

## Results

66.7% accuracy on 4 progression classes (random = 25%). Limited by dataset size (16 source clips total, 3 val samples) — not architecture.

The model distinguished I-IV-V-I from vi-IV-I-V from I-V-vi-IV from ii-V-I-I using only the spectrogram. Order matters in the labels — the same chords in a different order are a different class.

---

## Business parallel

The CNN→RNN pattern appears wherever you need to classify structured sequences:

- **Clickstream analysis:** CNN extracts features from each page view, RNN classifies the journey ("browsed → searched → abandoned" vs "browsed → added to cart → purchased")
- **Log anomaly detection:** CNN extracts features per log line, RNN classifies the sequence of events as normal or anomalous
- **Transaction chains:** per-transaction features from a dense layer, sequence classified by a GRU as fraud or legitimate

In each case: extract features per step (CNN or MLP), then model temporal structure (RNN).

> **Sequence length matters.** RNNs work best when the temporal pattern is long enough to be meaningful. A 2-step sequence adds little over a CNN. The chord progressions here (4 chords × ~9 time steps each = 36 steps) are at the minimum useful length.

See: [model/progression.py](../model/progression.py) — `ProgressionClassifier`, the CNN→RNN reshape  
See: [model/progression_train.py](../model/progression_train.py) — gradient clipping, pad-collate
