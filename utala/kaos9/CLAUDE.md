# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**utala: kaos 9** — an AI research project studying a 2-player tactical grid combat card game. The goal is to build a game engine, evaluation harness, and progressively sophisticated agents to study skill expression, risk management, and learning algorithms.

**This research produces generic, reusable game AI.** The agents and learning systems will be designed for integration into production game implementations. Design decisions should consider:
- Agent performance and portability
- Clean, documented interfaces using open formats
- Behavior quality suitable for player-facing AI opponents
- Platform-agnostic output (serializable state, actions, and learned models)

The project follows a phased research plan (see README.md). The current focus is **Phase 5: Improve, Distill, Ship** — improving the DQN to consistent >50% vs Heuristic, distilling it into a tiny production model, and porting Variant A rules and the distilled agent to the Flutter game app.

**Phases completed:**
- Phase 1: Engine, replay, baselines (Random, Heuristic, MC). Checkpoint PASS: Heuristic 65% vs Random, clear skill gradient.
- Phase 2: TD-Linear with hand-built gradient updates. Plateau ~47% vs Heuristic on original rules.
- Phase 3: DQN (bluffing-aware 80-dim state, 95-action space). ImitationNN distillation (1K params, 48% vs Heuristic).
- Phase 4: Variant A rules (choosable dogfight order, action space 86→95). DQN retrained: 47% final / 53% peak vs Heuristic. Checkpoint PASS: deep learning beats linear.

## Tech Stack & Constraints

- **Language:** Python (canonical for the study; agents designed for portability to other platforms)
- **Frameworks:** NumPy throughout. PyTorch used from Phase 3 onwards for DQN and imitation learning. Gymnasium used for environment wrapping. ONNX for model export.
- **Text-only research environment:** No graphics or UI in this Python research codebase. Everything must be text-based, inspectable, and hackable.
- **Open formats:** All outputs (state, actions, replays, learned models) use simple, documented formats for easy integration into game implementations. Model weights exported as JSON for the Flutter app.

## Architecture Rules

These are hard constraints that must not be violated:

- **All randomness lives in the engine, never in agents.** Agents observe sampled state; they do not own RNG.
- **Agents only propose actions.** They never apply actions, mutate state, or validate legality.
- **The action space is fixed and fully enumerated.** Illegal actions are masked by the engine, never removed. If a learning method requires a variable action set, it is rejected.
- **State encoding and action enumeration are fixed** across all agents and phases.
- **Determinism is required:** explicit RNG seeds, versioned replay format (seed + action list).

## Game Rules Reference

Full rules are in `utala-kaos-9.md`. Key structural points:

- **3×3 grid**, 2 players, each with 9 rocketmen (numbered 2–10)
- **Phase 1 (Placement):** Alternate placing rocketmen; squares can be contested (both players present)
- **Phase 2 (Dogfights):** Resolve contested squares one at a time. **Variant A (current canon):** center fights first if contested, then each dogfight winner chooses the next square. Each dogfight has: simultaneous rocket/flare commit → rocket/flare interaction → Kaos resolution
- **Kaos decks:** Personal 9-card decks (1–9), finite and trackable via visible discard
- **Win condition:** 3-in-a-row, or most squares controlled
- **Level 2** adds hidden deployment (10 and 2 placed face-down)
- **Experimental mechanics (E1, E2)** explore Kaos deck distribution variants and tunable hidden information count

## State and Action Space (current, Variant A)

- **State encoding:** 80-dim bluffing-aware feature vector (per-square power values, face-down flags, deck state, game phase)
- **Action space:** 95 actions fixed and fully enumerated (86 base + 9 CHOOSE_DOGFIGHT actions for square selection)
- These are **fixed** across all agents and phases — do not change without updating all agents and the engine

## Agents (current)

| Agent | Type | Strength |
|-------|------|----------|
| Random | Baseline | ~20% vs Heuristic |
| Heuristic | Rule-based | reference (55% vs Random) |
| TD-Linear | Hand-built TD | ~35–47% vs Heuristic |
| DQN | PyTorch, 39K params | 47% final / 53% peak vs Heuristic |
| ImitationNN | Distilled, ~1–5K params | Phase 5 target: ≥45% vs Heuristic |

## Phase 5 Deliverables

- Improved DQN checkpoint (consistent >50% vs Heuristic)
- Distilled production model (JSON weights, ≤5K params)
- Variant A rule changes ported to Flutter engine
- New AI agent shipped in Flutter app
- `PHASE5_RESULTS.md` with findings
