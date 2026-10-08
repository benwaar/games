# When Deep Learning Helps — The Goldilocks Zone

One of the clearest findings in this project: DQN failed on the original rules and succeeded on Variant A. Understanding why is more useful than the result itself — it gives you a framework for deciding when to reach for deep learning.

---

## The result

| Model | Rules | vs Heuristic | Params |
|-------|-------|-------------|--------|
| TD-Linear (42 features) | Original | **42%** | ~50 |
| DQN v1 | Original | 31% | 34,518 |
| TD-Linear NoInt (27 features) | Original | **47.5%** | 27 |
| DQN v2 (Variant A) | Variant A | **47% final / 53% peak** | 39,135 |

DQN with 700× more parameters was 11 percentage points *worse* than the linear model on the same game. Then the rules changed, and DQN became necessary.

---

## Why DQN failed on original rules

The original game's win condition was largely determined by a linear combination of features:

```
win probability ≈ w₁ × (my_squares - opp_squares) + w₂ × (my_weapons - opp_weapons) + ...
```

This is exactly what a linear model (`Q(s,a) = w^T × φ`) is designed to learn. The DQN was trying to find a nonlinear decision boundary where none existed. With:

- **Short episodes** (~15–20 steps) — few transitions to learn from per game
- **Sparse terminal reward** (+1/-1/0 only at the end) — credit assignment across all steps
- **Linear patterns** — the extra capacity of a neural network adds noise, not signal

The DQN converged to a worse policy than the linear model. More capacity made things worse.

---

## Why Variant A changed the answer

Variant A adds one rule: **after each dogfight, the winner chooses which contested square to fight next**.

This sounds like a small change. It creates a fundamentally new class of decision:

> "Fight this square because I will complete MY row **AND** I have a power advantage here."

Both conditions must be true simultaneously. A linear model computes a weighted sum of features. It cannot represent "feature A AND feature B" — that requires multiplying them, which is a nonlinear operation.

```
# Linear model: can learn this
value = w1 * completes_my_row + w2 * power_advantage

# But NOT this (requires conjunction)
value = f(completes_my_row AND power_advantage)
```

A DQN with ReLU activations can represent conjunctions. Under Variant A, this nonlinear capacity became necessary — and DQN recovered to 53% peak.

**The signal:** TD-Linear dropped from 47.5% (original rules) to 35.5% under Variant A. The linear model hit a wall because the game now required something it fundamentally cannot do. DQN recovered above the linear ceiling.

---

## The interaction-feature trap

During Phase 2.3, a set of "interaction features" was added to the linear model:

- `strong_move_when_winning` — high power AND material advantage
- `aggressive_when_behind` — weapon play AND material deficit
- Several others combining two game concepts

**Result: performance dropped from 44.5% to 39%.** Then removing them raised it to 47.5%.

Why did they hurt? Individual features like `power_advantage` and `material_lead` had stable, interpretable learned weights. Combining them into `strong_move_when_winning` introduced collinearity — the combined feature partially overlaps with both originals, diluting the signal from each. The linear model already knew to play power-strong moves when winning; the interaction feature told it nothing new but added noise to the gradient.

> **The lesson:** In linear models, clean individual features beat clever interaction features. Interactions exist for neural networks to discover — that's literally what the nonlinear activation functions are for.

---

## The line-formation negative-weight discovery

During Phase 3 feature ablation, a bug was found: the 2-in-a-row threat features were hardcoded to 0.0 — they were always zero, contributing nothing. After fixing the bug, the model was retrained.

**The learned weights for 2-in-a-row threat features were negative.**

The agent learned to *avoid* forming visible 2-in-a-row lines. This seems counterintuitive — shouldn't completing a line be good? But the Heuristic has an explicit +30 bonus for blocking any 2-in-a-row. Telegraphing a line to a Heuristic opponent guarantees a blocking response. The optimal strategy against this opponent is to *not* show lines.

The agent discovered this counter-strategy purely from game outcomes — no explicit rule was written. It ran games, got feedback, and converged to a policy that models human strategic thinking: don't telegraph your intent.

---

## The diagnostic framework

When should you reach for deep learning?

| Signal | What it suggests |
|--------|-----------------|
| Linear model plateaus and DL doesn't improve | Patterns are linear — add more features, not layers |
| Linear model plateaus and DL beats it | Nonlinear interactions present — DL is justified |
| DL overfits badly on small data | Problem is data-limited, not model-limited |
| Short episodes + sparse rewards + DL worse than linear | Capacity is hurting, not helping |
| DL improvement is large (>5pp) | Nonlinear interactions are load-bearing |

In this project: TD-Linear plateaued at 47.5% on original rules. DQN matched it under Variant A then exceeded it. The gap between linear ceiling and DQN peak widened as the game complexity grew.

> **In practice:** In production ML, this diagnostic matters. A gradient-boosted tree with 100 features often outperforms a neural network on tabular data because tabular patterns are frequently linear or piecewise-linear. Reach for deep learning when you have spatial or sequential structure, or when you know the signal requires nonlinear feature combinations. Otherwise start with a linear model — it's faster to train, easier to debug, and its weights are interpretable.

See: [agent-progression.md](agent-progression.md) for full results across all phases.
