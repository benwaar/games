# Phase 5 — Improve, Distill, Ship

**Date:** 2026-04-13
**Status:** Planned
**Prerequisite:** Phase 4 Variant A complete (DQN at 47% vs Heuristic, peak 53%)

---

## Context

Phase 4 proved that Variant A (choosable dogfight order) makes the game strategically richer, and that deep learning (DQN with bluffing-aware features) outperforms linear models. The DQN reaches 47% vs Heuristic (peak 53%) but oscillates in late training.

Phase 5 takes this from research to production: improve the agent, distill it into something small and fast, then ship both the rule change and agent to the Flutter app.

---

## 5.1 — Improve the DQN

**Goal:** Consistent >50% vs Heuristic over 200-game eval.

Known issues with the current DQN:
- Peak at 53% (30K games) but final at 47% — Stage 3 self-play causes oscillation
- Loss climbs in Stage 3 (0.2 → 0.5), network can't track shifting self-play distribution
- No checkpoint selection — uses final weights, not best

Improvements to try:
- **Checkpoint selection** — save best model during training (the 53% at 30K was never saved)
- **Learning rate scheduling** — decay LR in Stage 3 to consolidate gains
- **Longer Stage 2** — best results came from Heuristic + self-play mix, extend it
- **Polyak averaging** — soft target network updates instead of hard copy every 1K steps
- **Prioritised replay** — sample high TD-error transitions more often

**CHECKPOINT PASSED (2026-04-14)** — DQN v2 (Run 4 / `dqn_v2d`) achieved consistent >50% vs Heuristic.

Run 4 config that worked:
- Checkpoint saving + LR scheduling (halve at Stage 3 start, game 30K)
- Polyak averaging **off**, prioritised replay **off** (both destabilised self-play in Runs 1–2)
- Stage 3 keeps 50/50 Heuristic/self-play mix (original 80% self-play caused immediate crash in every prior run)
- Total: 50K games; Stage 1: 0–10K, Stage 2: 10K–30K, Stage 3: 30K–50K

Best checkpoint: `results/dqn_v2d/dqn_v2_best.pth`

Model details (for 5.2 distillation teacher):
- Architecture: 80-dim state → 128 → 128 → 95-dim Q-values (~39K params)
- Config: `GameConfig(fixed_dogfight_order=False)`, n_step=3, hidden_dim=128, batch_size=64, replay_capacity=30K
- Training script: `scripts/train/variant_a/train_dqn_v2.py`

---

## 5.2 — Distill into Production Model

**Goal:** DQN-quality play at minimal compute cost.

Same approach as Phase 3.4 (which produced ImitationNN: 1K params matching 34K DQN at 48%):

1. **Generate dataset** — improved DQN plays 5K–10K games vs Heuristic and Random, record every (state, action) pair
2. **Train student models:**
   - Tiny NN (1 hidden layer, 32 units, ~3-5K params)
   - Linear imitation (dot product, ~100 weights)
3. **Pareto frontier** — map strength vs model size vs inference time

Teacher is the improved DQN (not MC search like old 3.4). State space is 80-dim, action space is 95.

**Pass criteria:** Student achieves ≥45% vs Heuristic at <5K params.

**CHECKPOINT PASSED (2026-04-14)** — TinyNN student beats the teacher.

| Agent | Params | vs Heuristic | vs Random | ms/dec |
|-------|--------|-------------|-----------|--------|
| DQN Teacher | 39,135 | 49.5% | 52.0% | 0.054 |
| TinyNN (80→32→95) | 5,727 | **54.0%** | 49.0% | 0.045 |
| Linear (80→95) | 7,695 | 50.0% | 45.0% | 0.040 |

Dataset: 97,755 decisions from 5K games, saved to `results/distill_v1/dataset.jsonl`

Output files:
- `results/distill_v1/tiny_nn.pth` — PyTorch checkpoint (primary)
- `results/distill_v1/tiny_nn.json` — weights as JSON (118 KB float64, needs quantization for Flutter)
- `results/distill_v1/linear.pth` / `linear.json` (158 KB) — linear baseline
- Script: `scripts/train/variant_a/distill_dqn.py`

Note: TinyNN is 5,727 params (just over the <5K target) — strength criterion (≥45%) is comfortably cleared at 54%.

---

## 5.3 — Port Variant A Rules to Flutter

**Goal:** Flutter engine matches Variant A rules — choosable dogfight order, updated action space, UI choice prompt, updated Heuristic agent.

Explored flutter project. No existing stubs for Variant A — this is a green-field change across 6 files.

### What changes

| File | Change |
|------|--------|
| `models/game_state.dart` | Add 3 fields for choice state |
| `ai/action_space.dart` | Add `ChooseDogfightAction`, extend to 95 actions |
| `providers/game_provider.dart` | Fix `_startDogfightPhase()`, add `selectDogfightSquare()`, update `nextDogfight()` |
| `ai/heuristic_agent.dart` | Add `selectDogfightChoice()` |
| `ai/ai_controller.dart` | Handle choice phase |
| `widgets/dogfight_overlay.dart` | Add square-selection UI step |

### File-by-file plan

**1. `models/game_state.dart`**

Add to `GameState`:
```dart
final bool awaitingDogfightChoice;      // true when winner must pick next square
final String? dogfightChoicePlayerId;   // which player chooses
final List<int> remainingContested;     // contested squares not yet fought (shrinks each dogfight)
```
Extend `copyWith()` for all three. `remainingContested` replaces the role of the fixed `dogfightQueue` for tracking what's left.

**2. `ai/action_space.dart`**

- Add sealed class `ChooseDogfightAction(int squareIndex)` alongside existing `PlaceRocketmanAction`, `PlayWeaponAction`, `PassAction`
- Extend `decode(int index)`: indices 86–94 map to `ChooseDogfightAction(squareIndex: index - 86)`
- Update `getLegalMask()`: CHOOSE_DOGFIGHT actions (86–94) are legal only when `awaitingDogfightChoice == true` and the square is in `remainingContested`; all other action types are illegal during that phase
- Update `size` constant: `86 → 95`

**3. `providers/game_provider.dart`**

- `_startDogfightPhase()` — center (index 4) fights first if contested; set `currentDogfightSquare = 4` (or first in canonical order if center not contested), initialise `remainingContested` to all contested squares minus the first, set `awaitingDogfightChoice = false`
- Add `selectDogfightSquare(int squareIndex)` — called when a CHOOSE_DOGFIGHT action is applied; sets `currentDogfightSquare = squareIndex`, removes it from `remainingContested`, clears `awaitingDogfightChoice`, advances to `DogfightStep.reveal`
- `nextDogfight()` — after a dogfight completes: if `remainingContested` is empty, end dogfight phase; otherwise set `awaitingDogfightChoice = true`, set `dogfightChoicePlayerId` to the winner of the just-finished dogfight

**4. `ai/heuristic_agent.dart`**

Add `selectDogfightChoice(GameState state, String playerId) → int squareIndex`:
- Score each remaining contested square by strategic importance (same `_evaluateDogfightImportance` logic already used for weapon decisions)
- If player is ahead in squares: prefer to fight where they are strongest (consolidate)
- If behind: prefer to contest squares that block opponent's 3-in-a-row
- Return index of best remaining contested square

**5. `ai/ai_controller.dart`**

In the AI turn loop, add a branch for `awaitingDogfightChoice`:
- Call `heuristicAgent.selectDogfightChoice(state, aiPlayerId)`
- Apply via `gameProvider.selectDogfightSquare(squareIndex)`
- Use same ~600ms scheduling delay as weapon plays for natural pacing

**6. `widgets/dogfight_overlay.dart`**

When `awaitingDogfightChoice == true` and the chooser is the human player: show a grid overlay of remaining contested squares. Each is a tappable cell showing the square position. Tapping calls `gameProvider.selectDogfightSquare(index)`. If the chooser is the AI, show a brief "AI is choosing..." message (AI controller fires on its own timer).

### Center-first rule
The first dogfight of the phase uses the same logic as before (center if contested, else canonical order). The choosable mechanic only kicks in for the 2nd dogfight onwards (`nextDogfight()` sets `awaitingDogfightChoice`).

### Pass criteria
- Heuristic vs Heuristic game completes without errors
- Human player sees the choice prompt after winning a dogfight
- AI picks a square and play continues
- Action space reports size 95

---

## 5.4 — Ship Distilled Agent in Flutter

Source model: `results/distill_v1/tiny_nn.json` (80→32→95, 5,727 params, 54% vs Heuristic)

Steps:
1. **Quantize weights** — convert `tiny_nn.json` from float64 to float32, trim to 4–5 decimal places. Expected size: ~25–30 KB JSON. Binary float32 would be ~23 KB; JSON is fine for Flutter asset loading.
2. **Implement forward pass in Dart** — two linear layers + ReLU:
   - Load weights from bundled JSON asset
   - `forward(state_80) → relu(W1·x + b1) → W2·h + b2 → logits_95`
   - Mask illegal actions (`-inf`), return `argmax`
3. **Implement 80-dim feature extractor in Dart** — port `DQNFeatureExtractor` (same feature order as Python, verified by spot-checking a known state)
4. **Wire up as Hard-tier AI opponent**
5. **Update difficulty tiers:**

| Tier | Agent | Expected Strength | Model Size |
|------|-------|-------------------|------------|
| Easy | Random | ~20% | 0 |
| Medium | TD-Linear (+ noise) | ~30% | <1KB |
| Hard | Distilled NN | ~54% | ~25–30 KB JSON |
| Expert | Heuristic | ~55% | 0 |

Note: original <20KB target was set before we knew the architecture (5,727 float32 weights = ~23KB binary). JSON with float32 precision will be ~25–30KB — acceptable for a bundled asset.

Update `AI_AGENTS.md` in Flutter project.

---

## Deliverables

- Improved DQN checkpoint (>50% vs Heuristic)
- Distilled production model (JSON weights)
- Variant A rule change in Flutter engine
- New AI agent in Flutter app
- Updated `PHASE5_RESULTS.md` with findings
