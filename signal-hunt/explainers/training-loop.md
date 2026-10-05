# Training Loop

How a neural network learns — loss functions, backpropagation, optimisers,
learning rate scheduling, and early stopping.

---

## The learning cycle (one step)

Every training step does four things:

```python
optimiser.zero_grad()        # 1. clear gradients from the previous step
logits = model(x)            # 2. forward pass — compute predictions
loss = criterion(logits, y)  # 3. compute how wrong the predictions are
loss.backward()              # 4. backprop — compute gradients
optimiser.step()             # 5. update weights in the direction that reduces loss
```

Repeat this for every batch. Repeat for every epoch. That's the training loop.

> **Coming from C:** This is gradient descent implemented as a loop. `loss.backward()`
> traverses the computation graph in reverse (chain rule) and fills `.grad` on every
> parameter tensor. `optimiser.step()` applies `param -= lr * param.grad` (or a
> more sophisticated update). `zero_grad()` resets `.grad` to zero — if you forget
> it, gradients accumulate across batches and the update is wrong.

> **Coming from JS/TS:** PyTorch builds a computation graph as you run the forward
> pass — every operation is recorded. `loss.backward()` walks that graph in reverse
> to compute derivatives. It's like an automatic chain-rule calculator wired into
> every tensor operation. `optimiser.step()` is the final "apply the update" call.

---

## CrossEntropyLoss

`nn.CrossEntropyLoss` is the standard loss for multi-class classification. It takes
**raw logits** (not softmax probabilities) and integer class labels:

```python
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, labels)
# logits: (B, num_classes) — raw model output
# labels: (B,)             — integer class indices 0..num_classes-1
```

Internally it applies `log_softmax` then negative log-likelihood. Combining them in
one step is more numerically stable than doing `softmax → log → NLL` separately.

**What the loss means:**
- Loss near 0 → model is very confident and correct
- Loss near `log(num_classes)` ≈ 1.1 → model is at random chance (3 classes: log(3) ≈ 1.099)
- Loss above that → model is confidently wrong

Watch for loss starting around 1.1 (random) and decreasing — that's the model learning.

> **In practice:** Cross-entropy is to classification what mean-squared error is to
> regression. It penalises confident wrong predictions exponentially harder than
> uncertain ones — a model that says "definitely class A" when the answer is class B
> gets a much larger gradient than one that says "maybe class A".

---

## Adam optimiser

`torch.optim.Adam` is the default starting point for most deep learning tasks.
It adapts the learning rate per-parameter based on first and second moment estimates
of the gradients.

```python
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
```

Why Adam over plain SGD?
- SGD uses the same learning rate for every parameter. Rarely seen parameters get
  the same update as frequently-seen ones — inefficient.
- Adam tracks how often each parameter has been updated and scales accordingly.
  Rarely-updated parameters get larger steps; frequently-updated ones get smaller.
- It also smooths noisy gradients (moment 1) and normalises by gradient magnitude
  (moment 2), making it more stable on varied architectures.

For a small dataset like ours, Adam with `lr=1e-3` is a safe starting point. If
training is unstable, lower to `1e-4`.

---

## Learning rate scheduling — ReduceLROnPlateau

After many epochs, the loss may stop improving — the model is near a local minimum
but the learning rate is too large to descend further. `ReduceLROnPlateau` halves
the learning rate when validation loss stops improving for N epochs:

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, mode="min", factor=0.5, patience=3
)
# After each epoch:
scheduler.step(val_loss)
```

- **`mode="min"`** — we're minimising val loss (use `"max"` for accuracy)
- **`factor=0.5`** — multiply lr by 0.5 on plateau
- **`patience=3`** — wait 3 epochs with no improvement before reducing

> **In practice:** Learning rate scheduling is like adjusting step size when
> navigating terrain. Large steps are efficient on open ground; near the minimum
> (a valley floor), large steps cause you to overshoot back and forth. Reducing
> the step size as you converge is standard practice.

---

## Early stopping

Training for too many epochs overfits — the model memorises training data and
performs worse on validation. Early stopping ends training when val loss hasn't
improved for `patience` epochs:

```python
best_val_loss = float("inf")
epochs_without_improvement = 0

for epoch in range(max_epochs):
    val_loss = evaluate(...)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model)       # save the best weights seen so far
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break  # stop training
```

**Why save the best checkpoint, not the final weights?**

The final weights are from after the model started overfitting. The best checkpoint
is from the epoch where generalisation was strongest. We restore the best checkpoint
before evaluation and inference.

> **Coming from C/JS/TS:** Early stopping is a convergence guard. Like a retry loop
> with a maximum attempts counter — but instead of "did it succeed?", you ask "did
> it improve?" If N attempts pass without improvement, abort and take the best result
> seen so far.

---

## Epoch vs batch vs step

These three terms mean different things:

| Term | Meaning |
|------|---------|
| **Step** | One forward + backward pass on one batch |
| **Epoch** | One full pass through the entire training set |
| **Batch** | A subset of training data processed in one step |

With 162 training samples and batch size 32, one epoch = 6 steps
(162 / 32 = 5.06, rounds up to 6 batches). After 50 epochs, the model has
seen each training sample 50 times.

---

## Training vs evaluation mode

PyTorch models have two modes:

```python
model.train()   # dropout active, BatchNorm uses batch statistics
model.eval()    # dropout off, BatchNorm uses running statistics
```

**Always switch modes.** During the validation loop, `model.eval()` ensures:
- Dropout doesn't zero random activations (you'd get different outputs every call)
- BatchNorm uses stable running statistics instead of noisy per-batch stats

Wrap evaluation in `torch.no_grad()` too — it skips gradient computation entirely,
saving memory and time:

```python
with torch.no_grad():
    val_loss, val_acc = evaluate(model, val_loader, criterion)
```

---

## What "loss decreases" looks like

A healthy training run:
```
Epoch  1 | train_loss=1.098 | val_loss=1.095 | val_acc=0.36   ← near random (log(3)=1.099)
Epoch  5 | train_loss=0.821 | val_loss=0.876 | val_acc=0.52
Epoch 10 | train_loss=0.543 | val_loss=0.612 | val_acc=0.71
Epoch 20 | train_loss=0.312 | val_loss=0.481 | val_acc=0.82
Epoch 30 | train_loss=0.198 | val_loss=0.522 | val_acc=0.80   ← val loss rising = overfitting
→ early stop, restore epoch 24 checkpoint
```

Warning signs:
- Train loss decreasing but val loss increasing → overfitting. Dropout and early stopping help.
- Both losses stuck near 1.1 after 10 epochs → model not learning. Check learning rate, data.
- Loss goes to NaN → learning rate too high, or a bug in the pipeline.
