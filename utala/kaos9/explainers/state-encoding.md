# State Encoding — The 80-Dimensional Feature Vector

Every agent decision starts from the same place: a fixed-length vector of numbers that describe the current game state. This document walks through all 80 dimensions, explains the design choices, and shows how the encoding evolved across phases.

---

## Why a fixed vector

The game state is a complex object — a 3×3 grid of cards, resource piles, a phase flag, weapon histories. Neural networks and linear models both need a flat numeric input. The feature vector is the translation layer between game state and learnable representation.

A fixed-length vector also means the network architecture never changes between phases or game modes. The same 80-dimensional input works during deployment and during dogfights — the phase indicator features tell the model where it is.

---

## Evolution: 53-dim (Phase 2) → 80-dim (Phase 4)

| Phase | Dimensions | What changed |
|-------|-----------|-------------|
| Phase 2 | 53 | Occupancy-only per-square (3 values per square: empty/mine/theirs), basic resource counts |
| Phase 4 | 80 | Per-square expanded to 6 features (added power, face-down flags), Variant A context, deck awareness |

The jump from 53 to 80 happened because Variant A (choosable dogfight order) introduced decisions where knowing the power distribution across squares mattered — a linear model with only occupancy couldn't represent "I'm strong in this square AND it completes my row."

---

## The full 80 dimensions

### Per-square features (54 = 9 squares × 6)

For each of the 9 grid positions:

| Offset | Feature | Range | Notes |
|--------|---------|-------|-------|
| +0 | My occupancy | {0, 1} | 1 if I have a Rocketman here |
| +1 | Opponent occupancy | {0, 1} | 1 if opponent has a Rocketman here |
| +2 | My power | [0, 1] | Card value / 10 (0 if unoccupied) |
| +3 | Opponent visible power | [0, 1] | Card value / 10; **0 if face-down** |
| +4 | My face-down flag | {0, 1} | 1 if my piece here is face-down |
| +5 | Opponent face-down flag | {0, 1} | 1 if opponent's piece here is face-down |

**The information asymmetry:** Feature +2 always contains the true power of my piece — the agent always knows its own face-down values. Feature +3 is suppressed to 0 when the opponent's piece is face-down. This is the information boundary that makes bluffing possible: the opponent's face-down flag (feature +5) signals that a piece exists with unknown power. The agent learns from game outcomes what this uncertainty means strategically.

### Resource counts (6)

| Feature | What it is | Normalised by |
|---------|-----------|---------------|
| My rocketmen remaining | How many placement cards I have left | Max (9) |
| Opponent rocketmen remaining | Same for opponent | Max (9) |
| My weapons remaining | Dual-purpose cards (A,K,Q,J) left | Max (4) |
| Opponent weapons remaining | Same for opponent | Max (4) |
| My Kaos cards remaining | Cards left in my Kaos deck | Max (13) |
| Opponent Kaos cards remaining | Same for opponent | Max (13) |

### Material balance (3)

Difference signals rather than absolute counts — the model can learn "I have more pieces" without separately learning the scale of each player's total:

- Rocketmen differential (mine − theirs), normalised
- Weapons differential
- Kaos differential

### Board control (6)

| Feature | What it captures |
|---------|-----------------|
| My controlled squares | Squares only I occupy |
| Opponent controlled squares | Squares only opponent occupies |
| Contested squares | Squares both occupy |
| Empty squares | Squares neither occupies |
| My 2-in-a-row threat count | Lines where I have 2 and opponent has 0 |
| Opponent 2-in-a-row threat count | Same for opponent |

The 2-in-a-row threat features were central to the interaction-feature finding (see [when-deep-learning-helps.md](when-deep-learning-helps.md)): once the hardcoded-zero bug was fixed, the TD-Linear model learned **negative weights** for forming visible lines — because the Heuristic's explicit +30 blocking bonus means telegraphing a 2-in-a-row is counterproductive. The agent discovered the counter-strategy from game outcomes alone.

### Phase and turn (3)

| Feature | Range | Notes |
|---------|-------|-------|
| Deployment phase flag | {0, 1} | 1 during placement |
| Dogfight phase flag | {0, 1} | 1 during combat |
| Turn number (normalised) | [0, 1] | turn / max_turns |

### Variant A context (3)

Only meaningful in dogfight phase under Variant A rules:

| Feature | What it captures |
|---------|-----------------|
| Awaiting dogfight choice | 1 when winner must select next square |
| Remaining contested count | How many dogfights still to resolve |
| Has joker | 1 if this player holds the joker (tie-break token) |

### Deck awareness (4)

| Feature | What it captures |
|---------|-----------------|
| My high-card ratio | Fraction of my Kaos deck that is ≥ 8 |
| Opponent high-card ratio | Same for opponent |
| My expected Kaos value | Mean of remaining cards in my Kaos deck |
| Opponent expected Kaos value | Same for opponent |

The Kaos deck is finite and tracked (visible discard pile). These features turn the randomness of Kaos resolution into a probability management signal — the agent can learn to commit to dogfights when it has a high expected Kaos value and the opponent's deck is depleted.

### Bias (1)

A constant 1.0. Standard in linear models to allow the decision boundary to shift away from the origin. Neural networks don't need it (bias terms are in each layer), but keeping it in the shared feature vector means both model types use identical inputs.

---

## Design principles

**Normalise everything to [0, 1] or [-1, 1].** Raw card values (2–10), deck counts (0–13), and turn numbers all have different scales. Without normalisation, the model pays disproportionate attention to whichever features have the largest absolute values.

**Symmetric where possible.** My features and opponent features use the same encoding. The model can learn "I have more weapons than they do" as a general pattern rather than needing separate weights for each player.

**Information boundary is explicit.** The face-down suppression (feature +3 = 0 when face-down, feature +5 = 1) makes the information boundary a first-class feature rather than an implicit absence. The model can learn the *meaning* of face-down, not just treat it as missing data.

> **In practice:** Feature engineering is often the most important part of an ML project. The 80 features here encode decades of game design intuition — what matters in a tactical game — into a form the model can use. In fraud detection this is account age, transaction velocity, and device fingerprint. In document classification it's n-grams, metadata, and structural features. The model can only learn patterns from features you give it.

See: [src/utala/learning/dqn_features.py](../src/utala/learning/dqn_features.py) — `DQNFeatureExtractor`  
See: [src/utala/learning/state_action_features.py](../src/utala/learning/state_action_features.py) — Phase 2 (53-dim) feature vector
