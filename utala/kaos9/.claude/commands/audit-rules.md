# Audit Game Rules

Systematically compare the game rules (source of truth) against the Python implementation, find discrepancies, fix them, add tests, and verify.

**Focus area:** $ARGUMENTS

## Step 1: Read the Rules

Read `utala-kaos-9.md` in its entirety. This is the authoritative source of truth for all game mechanics. Pay attention to:
- Phase 1 (Placement) rules
- Phase 2 (Dogfight) resolution steps
- Weapon system (Rocket/Flare, dual-purpose, sacrifice mechanic)
- Kaos resolution (draw, total power, ties)
- Win conditions (immediate 3-in-a-row, most squares, draw)
- Joker token (equal-power tie-breaking, passing)
- Face-down placement (cards 2, 3, 9, 10)
- v1.9: Choosable dogfight order (winner picks next square, center first)

If a focus area was provided above, pay special attention to those sections but still read the full rules for context.

## Step 2: Read the Implementation

Read these files to understand the current code:
- `src/utala/engine.py` -- all game logic (placement, dogfight flow, weapon resolution, Kaos resolution, win checking, choosable dogfight order)
- `src/utala/state.py` -- state shape, GameConfig, GameState, Player, Rocketman, DogfightContext
- `src/utala/actions.py` -- ActionType enum (PLACE_ROCKETMAN, PLAY_WEAPON, PASS, CHOOSE_DOGFIGHT), ActionSpace, legal action masking
- `src/utala/agents/heuristic_agent.py` -- hand-crafted agent (placement strategy, dogfight choice, weapon use)
- `src/utala/agents/human_agent.py` -- human text input agent (display, action formatting)
- `scripts/demo/demo_human.py` -- interactive human vs AI demo (game loop, dogfight display, save/load)

If a focus area was provided, you may skim files unrelated to that area, but still review the full `engine.py`.

## Step 3: Compare Rules to Implementation

Go through each rule section and verify the implementation matches. Mark each item: **PASS** (matches rules), **BUG** (contradicts rules), **MISSING** (not implemented), or **PARTIAL** (partially correct).

### Placement Rules
- [ ] Players alternate turns
- [ ] Cards 2, 3, 9, 10 placed face-down; cards 4-8 face-up
- [ ] A square may hold one card from each player (contested)
- [ ] Cannot place on a square you already occupy
- [ ] Phase transitions to dogfight when all 9 squares are filled (all rocketmen placed)
- [ ] Rocketman removed from player's hand after placement

### Dogfight Order (v1.9: Choosable)
- [ ] Center square (1,1) always resolves first if contested
- [ ] After each dogfight, the winner chooses the next contested square
- [ ] If both rocketmen eliminated (no winner), joker holder chooses
- [ ] Only contested squares enter the queue
- [ ] If center is not contested, Player 1 chooses the first square
- [ ] CHOOSE_DOGFIGHT actions correctly masked to remaining contested squares only
- [ ] GameConfig(fixed_dogfight_order=True) falls back to canonical order: center -> edges -> corners

### Dogfight Step 1: Turn Order
- [ ] Face-down cards revealed (showdown) before resolution
- [ ] Lower power = underdog, acts first
- [ ] Equal power: joker holder acts first
- [ ] Joker passes to opponent after equal-power use

### Dogfight Step 2: Weapon Exchange
- [ ] Round 1 (underdog acts): underdog may play Rocket or pass
- [ ] Round 2 (other responds): if attacked, may play Flare or pass; if underdog passed, may play Rocket or pass
- [ ] Round 3 (underdog counters): only if other attacked in Round 2; underdog may play Flare or pass
- [ ] Both pass with no weapons -> skip to Kaos Resolution
- [ ] Weapon type determined by context (first attacker = Rocket, response to attack = Flare)
- [ ] Weapons removed from hand after use
- [ ] Turn switches correctly between underdog and other at each round

### Dogfight Step 3: Weapon Outcomes
- [ ] Undefended Rocket: attacker draws 1 Kaos card; >= 7 = HIT (target removed, dogfight ends); < 7 = MISS (continue to Kaos)
- [ ] Rocket vs Flare: both cancel, proceed to Kaos Resolution
- [ ] Kaos card consumed from attacker's deck and added to discard

### Dogfight Step 4: Kaos Resolution
- [ ] Each player draws 1 Kaos card from their own deck
- [ ] Total Power = Rocketman power + Kaos value
- [ ] Higher total wins; loser's Rocketman removed
- [ ] Tie: both Rocketmen removed
- [ ] Kaos cards consumed and added to discard piles
- [ ] If deck empty, reshuffle discard pile

### Win Conditions
- [ ] After EACH dogfight: check 3-in-a-row (immediate victory)
- [ ] After all dogfights: most squares wins
- [ ] Equal squares: draw

### Human Demo Display (demo_human.py)
- [ ] Dogfight choice shows position names and power matchups
- [ ] Weapon interaction display correctly identifies Rocket vs Flare from full 3-action sequence
- [ ] Undefended rocket shows hit/miss with Kaos draw
- [ ] Rocket vs Flare shows cancel message then Kaos resolution
- [ ] Kaos resolution shows both players' draws and total power

## Step 4: Read Existing Tests

Read tests to understand coverage:
- `tests/` directory for existing test files
- `tests/invariants.py` for game invariant checks
- Check which rules are tested and which discrepancies from Step 3 lack coverage
- Note the test patterns used (how engines are created, how games are simulated)

## Step 5: Plan Fixes

Enter plan mode. Present findings as:

### Discrepancies Found

For each issue:
| # | Rule | Implementation | Impact | Fix (file + change) |
|---|------|---------------|--------|---------------------|

### New Tests Needed

For each discrepancy, describe:
- Test name (descriptive of the rule being verified)
- Setup (how to configure engine/state)
- Key assertions

### Files to Modify
- **Engine**: `src/utala/engine.py`
- **State**: `src/utala/state.py`
- **Actions**: `src/utala/actions.py`
- **Agents**: files in `src/utala/agents/`
- **Demo**: `scripts/demo/demo_human.py`
- **Tests**: files in `tests/`

**Wait for user approval before proceeding to Step 6.**

## Step 6: Implement Fixes

After approval:
1. Fix game logic in `src/utala/engine.py`
2. Fix any state/action issues in `src/utala/state.py`, `src/utala/actions.py`
3. Fix agent behaviour in `src/utala/agents/`
4. Fix display in `scripts/demo/demo_human.py`
5. Keep changes minimal and targeted -- do not refactor unrelated code

## Step 7: Add Tests

Add tests following existing patterns:
- Use deterministic seeds for reproducible game outcomes
- Test names should describe the rule being verified
- Each bug fix gets at least one regression test

## Step 8: Verify

Run and confirm all pass:
```
python -m pytest tests/
python tests/run_invariants.py
```

If anything fails, fix and re-run. Do not leave broken tests.

Report summary: discrepancies found, fixed, tests added, final pass/fail status.
