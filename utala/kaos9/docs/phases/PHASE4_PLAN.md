# Phase 4 — Game Evolution & Future Investigations

**Date:** 2026-04-09 (updated)
**Status:** Planned — Phase 3 Complete

---

## Step 0: Guardrails (DO THIS FIRST)

Phase 4 modifies game rules — the riskiest type of change. Before touching the engine, set up guardrails.

### 0.1: `GameConfig` dataclass

Add to `src/utala/state.py` — centralise all rule parameters.

```python
@dataclass(frozen=True)
class GameConfig:
    rocketman_powers: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9, 10)
    face_down_powers: frozenset[int] = frozenset({2, 3, 9, 10})
    kaos_deck_values: tuple[int, ...] = tuple(range(1, 14))
    fixed_dogfight_order: bool = True
    dogfight_priority: tuple[tuple[int,int], ...] = (
        (1,1), (0,1),(1,0),(1,2),(2,1), (0,0),(0,2),(2,0),(2,2)
    )
    three_in_row_wins: bool = True
    use_point_scoring: bool = False
    square_weights: dict | None = None
    simultaneous_placement: bool = False
```

Wire through `src/utala/engine.py` — replace all hardcoded values:
- `range(2, 11)` → `self.config.rocketman_powers`
- `range(1, 14)` → `self.config.kaos_deck_values`
- `[2, 3, 9, 10]` → `self.config.face_down_powers`
- dogfight order → `self.config.dogfight_priority`

Also `src/utala/actions.py` — `ActionSpace` takes config. Keep mask size fixed per config.

**All 89 existing tests must pass unchanged** — defaults match current values.

### 0.2: `InvariantChecker` class

New `tests/invariants.py` — checks 12 critical invariants after every state transition:

1. Rocketman conservation (hand + board = config count per player)
2. Kaos conservation (deck + discard = config count per player)
3. Action space size fixed per config
4. Legal mask length matches action space size
5. Phase consistency
6. No double occupation (same player, same square)
7. Game termination (always ends)
8. Face-down reveal timing (only at dogfight start)
9. Turn alternation
10. Contested squares valid
11. Joker alternation
12. 3-in-a-row checked after every dogfight

### 0.3: Smoke runner + Makefile

```
python -m tests.run_invariants --games 50
python -m tests.run_invariants --games 50 --config variant_a
```

```makefile
check: lint typecheck test       # ~8s, before every commit
full:  check invariants          # ~10s, before pushing
```

### 0.4: Harness integration

Optional `invariant_checker` param on `Harness.run_game()`. Phase 4 evaluation runs always enable it.

### 0.5 (optional): Property-based tests with hypothesis

- Any random game terminates (for any seed)
- All invariants hold throughout (for any seed)
- Same seed + same actions = same outcome (determinism)

---

## Current State

Research through Phase 3 has established:

- **TD-Linear NoInt (47.5% vs Heuristic)** is the best learning agent (27 features, <0.25ms)
- **All fast models converge to ~48%** — a performance ceiling for current features
- **MC search is NOT on the Pareto frontier** — slower AND weaker than TD-Linear
- **Fewer features > more features:** Removing interaction features was the best improvement
- **Spatial tactics dominate:** 3-in-a-row positioning is the primary intelligence layer
- **Hidden information matters for search** but not for feature-based agents

Phase 4 focuses on **making a better game** using everything we've learned, plus optional deeper investigations.

---

## Already Completed (Phases 1–3)

- ~~CartPole DQN validation~~ — Phase 3.2 (PASS, episode 319)
- ~~Interpretability study~~ — Phase 3.3 (weight analysis, ablation, minimal intelligence)
- ~~Imitation learning~~ — Phase 3.4 (student beat teacher, Pareto frontier)
- ~~Deck awareness / card counting~~ — Phase 3.2 (no benefit)
- ~~Model export~~ — Phase 3.4 (JSON + ONNX in `models/mobile/`)
- ~~More DQN training~~ — Not recommended (validated as unnecessary)

---

## Option 1: Game Balance Tuning — Make a Better Game

**Goal:** Adjust game rules to create a richer strategy-vs-luck balance

The current game is dominated by spatial tactics (3-in-a-row). Kaos cards and probabilistic reasoning barely matter for AI agents. The game works, but it could be *more interesting*.

### What the Research Tells Us

The AI research identified two specific weaknesses:

1. **Fixed dogfight order makes placement dominant.** Since resolution order is always center → edges → corners, optimal placement is solvable by position alone. The dogfight phase is an afterthought.

2. **Kaos cards don't matter enough.** Deck awareness features (card counting, expected value, variance) provided zero benefit (Phase 3.2). The 13-card deck is too large for individual draws to shift strategy, and rocketman power (2–10, range 8) dominates Kaos values.

Additionally:
- Hidden information *does* matter: MC-Perfect beats MC-Fair 72-28% (Phase 3.2). More hidden info = more interesting decisions.
- DQN and complex models failed — but they might succeed if the game had deeper decision trees or required belief tracking.

### Proposed Rule Variants to Test

#### Strategic improvements (Priority 1)

Variants A and B change *how players make decisions* — they add new strategic layers to the game without altering the underlying math. These come first because they address the core finding that the current game is solvable by spatial tactics alone.

---

#### Variant A: Choosable Dogfight Order

**Change:** Winner of previous dogfight chooses which square to resolve next. First dogfight still starts at center.

**Why:** This alone transforms the game. Instead of optimizing placement for a known resolution sequence, players must think about:
- Which fights to win to control the *order* of remaining fights
- Multi-step planning: "If I win here, I choose to fight *there* next"
- Cascading consequences: winning a fight gives strategic advantage beyond the square itself

**What it tests:** Does adding strategic depth to the dogfight phase make placement less dominant? Does MC or DQN benefit from the longer planning horizon?

**Engine changes:** Small — modify dogfight resolution loop to accept player choice instead of fixed order.

---

#### Variant B: Bluffing & Hidden Information

Bluffing is a first-class placement tactic. An advanced player doesn't just place strong cards in strong positions — they place a 2 in the center to bait the opponent into a costly contest, or hide a 10 on a corner to build an unexpected 3-in-a-row. The goal is agents that reason about *what the opponent thinks they placed*, not just *where to place*.

##### Rule change

Test three tiers of hidden information, expanding outward from the current extremes:

| Tier | Face-down | Hidden set | What changes |
|------|-----------|------------|--------------|
| 4 (baseline) | 4 of 9 | {2, 3, 9, 10} | Current rules — only extremes hidden |
| 5 | 5 of 9 | {2, 3, 8, 9, 10} | Adds the 8 — "probably strong" range gets murkier |
| 6 | 6 of 9 | {2, 3, 4, 8, 9, 10} | Adds the 4 — mid-low cards also uncertain |

Only 5, 6, 7 are always visible. Each tier hides one more card from the middle inward, increasing uncertainty gradually without eliminating the information gradient. The mix of known and unknown cards is what makes bluffing work — if everything is hidden, there's nothing to bluff against.

**Engine changes:** Small — `GameConfig.face_down_powers` accepts any frozenset. Test all three tiers. Reveal logic unchanged (showdown at dogfight start).

##### Research hypothesis

Phase 3.2 showed hidden info gives MC +12% vs Heuristic, but no agent currently *exploits* hidden info through deliberate deception. Current agents either ignore it (Heuristic places strong cards in strong positions — a readable pattern) or sample it fairly (MC-Fair). None ask "what will my opponent *infer* from where I placed this?"

**What we're testing:**
- Can an agent learn to vary its placement pattern to become less predictable?
- Does a belief-aware agent (one that models what the opponent thinks is face-down) outperform a naive one?
- Does deceptive placement (low cards in high-value positions) create measurable downstream advantages (opponent wastes weapons/Kaos on weak targets)?
- Does increased uncertainty finally create a game where complex models (DQN, belief networks) outperform linear?

##### Bluffing-specific evaluation

Standard Phase 1 checkpoint + agent re-evaluation, plus:

1. **Placement predictability** — measure card-power vs position-value correlation across games. A bluffing agent should have lower correlation than the Heuristic (which always puts strong cards in strong spots).
2. **Deception payoff** — track games where a player placed a low card in a contested high-value position: did the opponent commit weapons? Did the bluffer win more squares elsewhere as a result?
3. **Opponent modelling gap** — compare an agent with a belief model of opponent face-down cards against one without. The gap measures whether reasoning about hidden info is worth the complexity.
4. **Pattern exploitation** — pit a belief-aware agent against the Heuristic (predictable placement). Does it learn to read the Heuristic's "strong card = strong position" pattern and exploit it?

---

#### Probability tweaks (Priority 2)

Variants C and D change *the odds* — they adjust deck sizes and power ranges to shift the balance between deterministic strength and probabilistic outcomes. These are tested after A and B because they're tuning knobs, not new strategic dimensions. They may combine well with the strategic improvements above.

---

#### Variant C: Smaller Kaos Deck

**Change:** Kaos decks are 7 cards (1–7) instead of 13 cards (1–13).

**Why:** Each draw is ~14% of the remaining deck instead of ~8%. Card counting becomes twice as impactful. After 3 dogfights, you know 3 of 7 cards — nearly half the deck is visible. Deck awareness features that failed in Phase 3.2 might actually work here.

**What it tests:** Does making randomness more trackable create space for probabilistic reasoning? Does deck awareness (Phase 3.2 features) now provide a measurable advantage?

**Engine changes:** Small — modify Kaos deck initialization.

---

#### Variant D: Compressed Rocketman Power

**Change:** Rocketmen are 4–8 (range 4) instead of 2–10 (range 8). Keep Kaos deck at 1–13 (or 1–7 if combined with Variant B).

**Why:** Currently a 10 vs 2 matchup has an 8-point head start — Kaos rarely flips that. With 4–8, the max power gap is 4 points, so Kaos cards more frequently decide outcomes. Combat becomes less "bigger number wins" and more "who manages their Kaos deck better."

**What it tests:** Does reducing deterministic advantage in combat make probabilistic reasoning valuable?

**Engine changes:** Moderate — modify rocketman creation, adjust hidden card rules (which cards are face-down?).

---

### Evaluation for Each Variant

Re-run the full agent toolkit:
1. **Phase 1 checkpoint:** Does skill still beat luck? Smooth strength curve?
2. **Heuristic vs Random:** Is the game still worth studying?
3. **Linear Value training:** Does the same simple model work, or does it need more?
4. **MC-Fair evaluation:** Does search still dominate?
5. **Deck awareness re-test:** Do the Phase 3.2 features now provide a measurable advantage?
6. **DQN re-test:** Does the game now benefit from non-linear models?

**Success criteria for a "better game":**
- Deck awareness features show ≥ +3–5pp improvement (Phase 3.2 failed this)
- More diverse agent rankings (not just "position is everything")
- Hidden information creates meaningful decisions (not just noise)
- Skill still matters — stronger agents still beat weaker ones

