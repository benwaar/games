# utala: kaos 9 — AI Handoff for Flutter (Phase 5)

This folder contains everything needed to complete the AI work in the Flutter app.
It was produced by the Python research codebase after Phases 5.1–5.3.

## Files here

| File | What it is |
|------|------------|
| `tiny_nn_f32.json` | Distilled TinyNN model weights, float32 precision |

---

## What's done, what's left

### Done (Python research side)
- **5.1** — DQN v2 trained to consistent >50% vs Heuristic (Run 4 / `dqn_v2d`)
- **5.2** — TinyNN distilled from DQN teacher: **80 → 32 → 95**, 5,727 params, **54% vs Heuristic**

### Your job (Flutter side)
- **5.3** — Port Variant A rules to the Flutter engine (choosable dogfight order)
- **5.4** — Implement TinyNN forward pass + 80-dim feature extractor in Dart, wire up as Hard-tier AI

---

## The model

### Architecture

```
input:   80-dim state feature vector  (Float32List)
hidden:  32 units, ReLU
output:  95 logits  (one per action)

h      = relu(W1 @ x + b1)    // [32]
logits = W2 @ h + b2           // [95]
```

Select action: mask illegal actions to `-infinity`, return `argmax`.

### JSON format (`tiny_nn_f32.json`)

```json
{
  "metadata": { "model": "TinyImitationNN", "state_dim": 80, "hidden_dim": 32,
                "action_dim": 95, "n_params": 5727, "vs_heuristic": 0.54, ... },
  "layers": [
    { "name": "fc1.weight", "shape": [32, 80], "data": [[...], ...] },
    { "name": "fc1.bias",   "shape": [32],     "data": [...] },
    { "name": "fc2.weight", "shape": [95, 32], "data": [[...], ...] },
    { "name": "fc2.bias",   "shape": [95],     "data": [...] }
  ]
}
```

Weights follow PyTorch convention: `fc1.weight[i]` is the weight vector for output unit `i`,
so `h[i] = dot(fc1.weight[i], x) + fc1.bias[i]`.

### Dart forward pass (pseudocode)

```dart
List<double> forward(List<double> state80) {
  // Layer 1: [32 x 80] @ [80] + [32] -> relu
  final h = List<double>.filled(32, 0.0);
  for (int i = 0; i < 32; i++) {
    double sum = fc1Bias[i];
    for (int j = 0; j < 80; j++) sum += fc1Weight[i][j] * state80[j];
    h[i] = sum > 0 ? sum : 0.0;  // relu
  }
  // Layer 2: [95 x 32] @ [32] + [95]
  final logits = List<double>.filled(95, 0.0);
  for (int i = 0; i < 95; i++) {
    double sum = fc2Bias[i];
    for (int j = 0; j < 32; j++) sum += fc2Weight[i][j] * h[j];
    logits[i] = sum;
  }
  return logits;
}

int selectAction(List<double> state80, List<int> legalActions) {
  final logits = forward(state80);
  int best = legalActions[0];
  double bestScore = logits[legalActions[0]];
  for (final a in legalActions) {
    if (logits[a] > bestScore) { bestScore = logits[a]; best = a; }
  }
  return best;
}
```

---

## 80-dim state feature vector

Features must be extracted in **exactly this order**. Any difference from the Python
`DQNFeatureExtractor` will silently degrade performance.

### Group 1 — Per-square (54 features)

For each square in row-major order: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2).
Six features per square = 54 total.

| # | Feature | Value |
|---|---------|-------|
| sq*6+0 | has_mine | 1.0 if player's rocketman is on this square, else 0.0 |
| sq*6+1 | has_opp | 1.0 if opponent's rocketman is on this square, else 0.0 |
| sq*6+2 | my_power | `my_rocketman.power / 10.0` if present, else 0.0 |
| sq*6+3 | opp_power_visible | `opp_rocketman.power / 10.0` if present AND face-up, else 0.0 |
| sq*6+4 | my_face_down | 1.0 if my rocketman is present and face-down, else 0.0 |
| sq*6+5 | opp_face_down | 1.0 if opp rocketman is present and face-down, else 0.0 |

`sq` = square index in row-major order (0–8). So square (1,1) [center] is sq=4, features at indices 24–29.

### Group 2 — Resources (6 features, indices 54–59)

| Index | Feature | Value |
|-------|---------|-------|
| 54 | my_rocketmen | rocketmen remaining in hand / 9.0 |
| 55 | my_weapons | weapons remaining in hand / 4.0 |
| 56 | my_kaos | kaos cards remaining in deck / 13.0 |
| 57 | opp_rocketmen | opponent rocketmen in hand / 9.0 |
| 58 | opp_weapons | opponent weapons in hand / 4.0 |
| 59 | opp_kaos | opponent kaos cards remaining / 13.0 |

`remaining_kaos = 13 - len(kaos_discard)`. Kaos deck is 13 cards (values 1–13).

### Group 3 — Material balance (3 features, indices 60–62)

| Index | Feature | Value |
|-------|---------|-------|
| 60 | rm_advantage | `(my_rm - opp_rm + 9) / 18.0` |
| 61 | wp_advantage | `(my_wp - opp_wp + 4) / 8.0` |
| 62 | kaos_advantage | `(my_kaos_rem - opp_kaos_rem + 13) / 26.0` |

### Group 4 — Board control (6 features, indices 63–68)

| Index | Feature | Value |
|-------|---------|-------|
| 63 | my_controlled | squares where I am sole occupant / 9.0 |
| 64 | opp_controlled | squares where opponent is sole occupant / 9.0 |
| 65 | contested | squares where both players present / 9.0 |
| 66 | empty | squares with no rocketmen / 9.0 |
| 67 | my_2_in_row | 1.0 if I control ≥2 squares in any row, column, or diagonal |
| 68 | opp_2_in_row | 1.0 if opponent controls ≥2 squares in any line |

Lines to check for 2-in-a-row: all 3 rows, all 3 columns, both diagonals (8 lines total).
A square is "controlled" by a player if it has their rocketman and no opponent rocketman
(i.e. `controller == player`, not contested).

### Group 5 — Phase + turn (3 features, indices 69–71)

| Index | Feature | Value |
|-------|---------|-------|
| 69 | phase_placement | 1.0 if placement phase, else 0.0 |
| 70 | phase_dogfight | 1.0 if dogfight phase, else 0.0 |
| 71 | turn_normalized | `min(turn_number / 50.0, 1.0)` |

### Group 6 — Variant A context (3 features, indices 72–74)

| Index | Feature | Value |
|-------|---------|-------|
| 72 | awaiting_choice | 1.0 if `state.awaitingDogfightChoice`, else 0.0 |
| 73 | remaining_contested | `state.remainingContested.length / 9.0` |
| 74 | has_joker | 1.0 if this player holds the joker, else 0.0 |

### Group 7 — Deck awareness (4 features, indices 75–78)

Computed for the agent's player first, then opponent.

| Index | Feature | Value |
|-------|---------|-------|
| 75 | my_high_kaos_ratio | fraction of my remaining kaos cards with value ≥ 8 |
| 76 | my_kaos_expected | mean value of my remaining kaos cards / 13.0 |
| 77 | opp_high_kaos_ratio | same for opponent |
| 78 | opp_kaos_expected | same for opponent |

`remaining = {1..13} minus discard`. If all 13 have been discarded, treat as full deck.

### Group 8 — Bias (1 feature, index 79)

| Index | Feature | Value |
|-------|---------|-------|
| 79 | bias | always 1.0 |

---

## 95-action space

Matches `ActionSpace` in both Python and Dart. Must be identical.

| Range | Type | Encoding |
|-------|------|----------|
| 0–80 | PLACE_ROCKETMAN | `(power - 2) * 9 + row * 3 + col` — power 2–10, positions row-major |
| 81 | PLAY_WEAPON | card_index = 0 |
| 82 | PLAY_WEAPON | card_index = 1 |
| 83 | PLAY_WEAPON | card_index = 2 |
| 84 | PLAY_WEAPON | card_index = 3 |
| 85 | PASS | — |
| 86–94 | CHOOSE_DOGFIGHT | `86 + row * 3 + col` — square index in row-major order |

CHOOSE_DOGFIGHT quick reference:

| Action | Square |
|--------|--------|
| 86 | (0,0) top-left |
| 87 | (0,1) top-center |
| 88 | (0,2) top-right |
| 89 | (1,0) mid-left |
| 90 | (1,1) **center** |
| 91 | (1,2) mid-right |
| 92 | (2,0) bot-left |
| 93 | (2,1) bot-center |
| 94 | (2,2) bot-right |

---

## 5.3 — Variant A rules implementation plan

Six files to change in `lib/`. No existing stubs — green-field.

### 1. `models/game_state.dart`

Add three fields:
```dart
final bool awaitingDogfightChoice;    // winner must pick next square
final String? dogfightChoicePlayerId; // which player chooses
final List<int> remainingContested;   // square indices not yet fought
```
Extend `copyWith()` for all three. `remainingContested` shrinks by one each time a dogfight
completes. Square index = `row * 3 + col`.

### 2. `ai/action_space.dart`

- Add sealed class `ChooseDogfightAction(int squareIndex)`
- Extend `decode(int index)`: indices 86–94 → `ChooseDogfightAction(squareIndex: index - 86)`
- Update `getLegalMask()`: during `awaitingDogfightChoice`, only CHOOSE_DOGFIGHT actions
  for squares in `remainingContested` are legal; all other types are illegal
- Change `size` constant: `86 → 95`

### 3. `providers/game_provider.dart`

- `_startDogfightPhase()` — fight center (index 4) first if contested, else first in canonical
  order `[4,1,5,7,3,0,2,8,6]`. Initialise `remainingContested` = all contested minus that first
  square. Set `awaitingDogfightChoice = false`.
- Add `selectDogfightSquare(int squareIndex)` — apply a CHOOSE_DOGFIGHT action:
  set `currentDogfightSquare = squareIndex`, remove from `remainingContested`,
  clear `awaitingDogfightChoice`, advance to `DogfightStep.reveal`.
- `nextDogfight()` — after a dogfight completes: if `remainingContested` is empty, end phase;
  otherwise set `awaitingDogfightChoice = true`, `dogfightChoicePlayerId = winnerId`.

### 4. `ai/heuristic_agent.dart`

Add `selectDogfightChoice(GameState state, String playerId) → int squareIndex`:
- Score each square in `remainingContested` using `_evaluateDogfightImportance`
- If leading: pick square where player is strongest (consolidate)
- If behind: pick square that would block opponent's 3-in-a-row threat
- Return square index of best choice

### 5. `ai/ai_controller.dart`

Add a branch for `awaitingDogfightChoice && state.dogfightChoicePlayerId == aiPlayerId`:
- Call `heuristicAgent.selectDogfightChoice(state, aiPlayerId)`
- Apply via `gameProvider.selectDogfightSquare(squareIndex)`
- Same ~600ms scheduling delay as weapon plays

### 6. `widgets/dogfight_overlay.dart`

When `awaitingDogfightChoice == true`:
- Human player: show a grid overlay of remaining contested squares, each tappable.
  Tap calls `gameProvider.selectDogfightSquare(index)`.
- AI player: show brief "AI is choosing next square..." message.
  AI controller fires on its own timer.

### Pass criteria for 5.3
- Heuristic vs Heuristic game completes without errors
- Human player sees the choice prompt after winning a dogfight
- AI picks a square and play continues
- `ActionSpace.size == 95`

---

## 5.4 — Ship the distilled agent

### New agent: `DistilledNnAgent`

Implement in `lib/ai/distilled_nn_agent.dart`:
1. Load `assets/models/tiny_nn_f32.json` at startup, parse weight matrices
2. Implement `forward(state80)` — two linear layers + ReLU (see pseudocode above)
3. Implement `selectPlacement` and `selectDogfightAction` using the 80-dim feature
   extractor and argmax over legal actions
4. For CHOOSE_DOGFIGHT actions (indices 86–94): handle in a new
   `selectDogfightChoice(state, playerId)` method, same pattern as Heuristic

### New feature extractor: `DqnStateFeatures`

Implement in `lib/ai/dqn_state_features.dart`:
- Produces exactly the 80-dim vector specified above
- Spot-check: for a known starting state, compare output to the Python extractor
  (run Python's `DQNFeatureExtractor.extract()` on the same state and compare values)

### Difficulty tier update

| Tier | Agent | Expected strength |
|------|-------|------------------|
| Easy | `RandomAgent` | ~20% vs Heuristic |
| Medium | `TdLinearAgent` (ε=0.3) | ~30% |
| Hard | `DistilledNnAgent` | **~54%** |
| Expert | `HeuristicAgent` | ~55% |

### Asset
Bundle `tiny_nn_f32.json` as a Flutter asset. It is ~59 KB as float32 JSON — fine for
a bundled asset. Add to `pubspec.yaml` under `assets:`.

### Testing
Verify `DistilledNnAgent` always returns a legal action. Win-rate verification
is done by running `scripts/eval/` in the Python project against the same weights.
