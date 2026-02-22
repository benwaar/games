# Utala KAOS 9: AI Study Plan

## Project Ethos

This project is about:
- understanding algorithms by building them
- seeing how different approaches *think*
- learning by stripping problems down to their **nuts and bolts**

Everything is text-based, inspectable, and hackable.

---

![Human gameplay screenshot](screenshot.png)

---

## Quick Start

```bash
# Setup
./setup.sh              # Creates Python 3.11 venv and installs dependencies

# Play the game yourself
./run.sh                # Play as human vs AI (default)

# Watch AI vs AI demos
./run.sh demo.py        # Quick tournament view
./run.sh demo_heuristic.py      # Verbose gameplay
./run.sh demo_montecarlo.py     # Monte Carlo analysis

# Run tests
python run_tests.py     # 79 tests, 94% coverage
```

---

## Core goal

Build one game engine and one evaluation harness, then plug in different
decision-making systems and compare them.

We care about:
- win / draw / loss vs fixed opponents
- robustness under randomness
- interpretability
- simplicity vs performance

For stochastic games, “solved” means:
- strong, measurable baselines
- clarity about where skill matters
- predictable behaviour under self-play

---

# Phase 1 — Baselines and instrumentation (no learning)

**Question:**  
_Do I understand the game well enough to measure anything at all?_

### Scope
- Implement the full game rules in Python (canonical for the study)
- Ensure determinism:
  - explicit RNG
  - versioned replay format
- Build an evaluation harness:
  - self-play
  - cross-play
  - tournament metrics

**Note:**  
All randomness lives in the **engine**, never in agents.  
Agents observe sampled state; they do not own RNG.

### Agents
Implement several **fully readable** agents:
- random legal
- simple heuristics
- rollout-based evaluator (Monte Carlo)

No learning.  
No gradients.  
No frameworks.

**Note:**  
Rollouts simulate *legal futures only*.  
If an agent proposes an illegal action, it is a bug in the agent, not the engine.

### Deliverables
- canonical Python engine
- replay format v1 (seed + action list)
- baseline agents ordered by strength
- bulk evaluation results

---

## Checkpoint — is the game worth studying?

This checkpoint exists **before any learning is added**.

**Pass if:**
- stronger baseline agents consistently outperform weaker ones
- weaker agents still win sometimes
- different strategies produce different outcomes
- randomness affects *close* matches, not everything

**Fail if:**
- outcomes are dominated by luck
- trivial heuristics dominate
- play collapses into forced lines

If this checkpoint fails, **change or abandon the game**.

Only if it passes do we proceed.

**Note:**  
Learning is *not* used to rescue a weak game.  
This checkpoint protects time, not pride.

---


# Phase 2 — Learning without frameworks

**Question:**  
_Can learning improve play in ways I can still understand?_

### Scope
- Implement learning systems by hand:
  - associative memory / nearest neighbour
  - simple linear scorers with manual updates
- Train only on internally generated data
- Keep all intermediate state readable

### Constraints
- fixed state encoding
- fixed action enumeration
- illegal moves are masked by the engine

**Note:**  
The action space is **fixed and fully enumerated**.  
Illegal actions are masked, never removed.  
If a learning method requires a variable action set, it is rejected.

**Note:**  
Models only **propose actions**.  
They never apply actions, mutate state, or validate legality.

## Simulation Metrics to Report in phase 2

- **Win rate vs. skill gap:** Strong→medium→weak should form a smooth curve, not a cliff.  
- **First-player advantage:** Keep near 52–56% unless intentionally asymmetric.  
- **Weapon spend profile:** Avg. use + attack/defense split; always-spend or always-hoard is a red flag.  
- **Rocket hit value:** 7+ (7/13 ≈ 53.85%); if rockets are almost always right/wrong, choice collapses.  
- **Tie rate & wipeouts:** Too many “both removed” results make endgames swingy/empty.  
- **Comeback rate:** Underdogs should recover sometimes—but through skill, not randomness.  

## Common Simulation Pitfalls

- **Symmetry trap:** Identical AI strategies can mask a dominant line.  
- **Reward hacking:** Optimizing a proxy metric instead of win probability.  
- **Limited state awareness:** Undervaluing discards lowers the apparent skill ceiling.  
- **Policy overfitting:** Agents beat each other but lose to simple human heuristics.


### Deliverables
- at least one learning agent that beats Phase 1 heuristics
- readable model state (tables, weights, examples)
- evaluation vs baseline agents
