# Train / Val / Test Split

Why you can't measure how well a model works using the same data it learned from —
and why you need three sets, not two.

---

## The core problem: you can't test what you've seen

Imagine teaching someone to pass a driving test by making them memorise the exact
questions on the test paper. They'd score 100%. They also wouldn't know how to drive.

A neural network has the same failure mode. Given enough parameters and training time,
it can memorise every training example — including noise, quirks of your microphone,
and the specific pitch wobble in your hum recordings. That's **overfitting**: perfect
performance on seen data, poor performance on anything new.

To detect this, you need data the model has never seen. That's the test set.

---

## Why three sets, not two

The obvious solution is to split your data into two parts: train and test. Train on
one, evaluate on the other. But there's a subtlety.

During training, we make decisions based on the validation loss:
- Learning rate scheduler reduces lr when val loss stops improving
- Early stopping halts training when val loss hasn't improved for N epochs
- You might also tune hyperparameters (batch size, dropout) based on val performance

Every time you use val loss to make a decision, **the val set leaks indirectly into
training**. The model hasn't seen the val data directly, but the training process has
been shaped by it. If you then evaluate on the same val set, you're measuring partly
how well you tuned to it.

The test set fixes this. It's locked away and never touched until the very end —
not for scheduling, not for hyperparameter tuning, not for early stopping. It sees
the final model exactly once, as a measure of real-world generalisation.

```
data/processed/
    ↓ load_splits(seed=42)
┌──────────────┬─────────────┬─────────────┐
│  TRAIN (70%) │  VAL (15%)  │  TEST (15%) │
│  161 samples │  35 samples │  35 samples │
│              │             │             │
│ Model sees   │ Used for LR │ Locked away │
│ these and    │ scheduling  │ until final │
│ updates      │ + early     │ evaluation  │
│ weights      │ stopping    │ only        │
└──────────────┴─────────────┴─────────────┘
```

> **In practice:** This maps directly to software testing.
> - **Train** = your test suite. You write code against it, you fix bugs it catches.
> - **Val** = your CI environment. You check against it during development; it gives
>   you feedback, so you tune to it over time.
> - **Test** = a real user on a production build. They've never seen your code while
>   you were writing it. Their experience is the only honest measure of quality.
>
> The same logic applies: if you optimise against CI forever, CI stops being an honest
> signal. You need a held-out production test.

---

## Stratified split — why random isn't enough

A plain random split might give you 60 hums and 50 claps in train, 5 claps in val.
The model learns clap well but val never tests it properly.

**Stratified** split ensures every split contains the same proportion of each class:

```
Original:  77 hum / 77 whistle / 77 clap  (231 total)

Train:     54 hum / 54 whistle / 53 clap  (161 total — 70%)
Val:       11 hum / 12 whistle / 12 clap   (35 total — 15%)
Test:      12 hum / 11 whistle / 12 clap   (35 total — 15%)
```

Each split mirrors the original class balance. Val and test can measure all three
classes. `sklearn.model_selection.train_test_split(stratify=labels)` handles this.

---

## Fixed seed — reproducibility

```python
load_splits(processed_dir, seed=42)
```

The seed fixes the random state for the split. Same seed → same exact split every run,
on every machine. This matters for:

- **Comparing runs** — if the split changes between experiments, you can't tell whether
  a score improvement came from a better model or an easier test set
- **Reproducing results** — someone cloning the repo gets the same split, so they can
  verify the results

> **Coming from C/JS/TS:** This is like a fixed random seed for test data generation.
> In Jest you'd use `Math.random = () => 0.42` for determinism. PyTorch uses
> `torch.manual_seed` / `random_state=42` in sklearn. Same principle.

---

## The "no peeking" rule in practice

In this project:
1. `load_splits` divides the manifest into three lists
2. Training only ever touches `train_loader` and `val_loader`
3. `test_loader` is only constructed and evaluated in `model/evaluate.py`
4. The test result is reported once — never used to tune anything

If you find a test accuracy you don't like and retrain with different hyperparameters
to improve it, you've broken the no-peeking rule. The test set is now implicitly part
of your development loop and no longer measures generalisation.

For production ML, the test set is often called the **held-out set** or
**evaluation set**, and access to it is controlled to prevent this leakage.
