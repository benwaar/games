# Phase 3.4 — Compress Search into Evaluation (Production)

**Date:** 2026-04-07
**Status:** Planned
**Prerequisite:** Phase 3.3 complete (understanding what matters)

---

## The Phase 3 Question

> **Search (MC) is strong but expensive. Learning (Linear) is cheap but weaker. Can we distill search knowledge into a learned model — getting the best of both?**

Phase 3.3 answered *what matters*. Phase 3.4 is the **distillation** step: train a fast model on MC's decisions and see if we can replace online thinking with offline learned intuition.

---

## Context

**From Phase 3.3 we will know:**

- Which features matter and which are redundant
- The minimal intelligence required to play well
- Whether the game can be learned from raw input or requires engineered features

**This phase applies those findings:**

- Use the minimal/best feature set identified in 3.3
- Train a student model to mimic MC-Fast's action choices
- If it works: MC-quality play at linear-model speed — the best of both worlds

**Why this matters for mobile:**

- MC search requires simulation every move (~100ms, expensive)
- A learned model = single forward pass (<2ms, instant)
- If imitation learning works, we close the gap between Linear Value (42%) and Heuristic (58%)

---

## Goals

1. Train a student model from MC-Fast demonstrations
2. Map the Pareto frontier: strength vs compute
3. Produce the final production-ready model(s) for mobile

---

## Plan (1–2 Weeks)

### 1. Imitation Learning — Train Student from MC-Fast

Generate a dataset of (state, action) pairs from MC-Fast games, then train a student model to predict MC's choices.

**Dataset generation:**
- MC-Fast plays 5,000–10,000 games vs Heuristic and Random
- Record every (state_features, chosen_action) pair
- Use MC-Fair (information sets enabled) — not perfect-info MC

**Student models to train:**
- **Linear imitation** — same architecture as Linear Value, but trained on MC actions instead of TD learning
- **Tiny neural net** — 2-layer, 32-unit hidden (5–15KB ONNX)
- Use minimal feature set from Phase 3.3 if ablation identifies one

**Note:** Based on Phase 3.2 findings, do NOT include deck awareness features. Spatial/tactical features only.

**Scripts:** `generate_mc_dataset.py`, `train_imitation.py`

---

### 2. Search vs Evaluation Pareto Frontier

Map the tradeoff between thinking time and play strength:

| Method | Strength | Speed | Compute |
|---|---|---|---|
| MC-10 (Fair) | ~54% vs Heuristic | Slow | High |
| MC-1 (Fair) | ? | Medium | Medium |
| Imitation (tiny NN) | ? | <2ms | Low |
| Imitation (linear) | ? | <1ms | Minimal |
| Linear Value (TD) | ~42% vs Heuristic | <1ms | Minimal |
| Heuristic | Baseline | <1ms | Minimal |

**Goal:** Find the best strength-per-millisecond agent.

**Script:** `pareto_frontier.py`

---

### 3. Mobile AI Production Package

Using results from Phases 3.3 and 3.4, produce the final mobile-ready agent set:

**Difficulty tiers (refined):**

| Tier | Agent | Expected Strength | Model Size | Inference |
|---|---|---|---|---|
| Easy | Random | ~20% | 0 | <1ms |
| Medium | Linear Value (+ noise) | ~30% | <1KB | <1ms |
| Hard | Best imitation model | ~50%? | <15KB | <2ms |
| Expert | Heuristic | ~58% | 0 | <1ms |

If imitation learning fails to beat Linear Value, fall back to:

| Tier | Agent | Strength |
|---|---|---|
| Easy | Random | ~20% |
| Medium | Linear Value + 30% epsilon | ~30% |
| Hard | Linear Value (pure greedy) | ~42% |
| Expert | Heuristic | ~58% |

**Deliverable:** Final model exports (JSON for linear, ONNX for tiny NN if applicable)

---

## Success Criteria

- Imitation model achieves ≥ Heuristic-level performance (≥50% vs Heuristic)
- Significant speedup vs MC (no rollouts at inference time)
- Clear Pareto frontier showing strength-compute tradeoff
- Production-ready model package for mobile integration

---

## What This Phase Does NOT Include

- Interpretability / weight analysis (done in Phase 3.3)
- Deck awareness features (Phase 3.2 showed no benefit)
- Alternative RL algorithms (future work if needed)
- Level 2 hidden information / opponent modeling (future work)
- Human play study, transfer learning, multi-agent training (future work)

---

## Deliverables

- `generate_mc_dataset.py` — MC demonstration dataset generator
- `train_imitation.py` — imitation learning training script
- `pareto_frontier.py` — strength vs compute comparison
- `models/imitation_*.json` or `models/imitation_*.onnx` — trained models
- `PHASE3.4_RESULTS.md` — analysis and Pareto frontier
- Updated `MOBILE_GAME_AGENTS.md` — final difficulty tier recommendation

---

## Expected Outcomes

### If imitation learning succeeds:
- Student model approximates MC strength at linear-model cost
- New "Hard" tier fills the gap between Linear Value (42%) and Heuristic (58%)
- Confirms that search knowledge can be distilled into evaluation
- Key finding: **runtime thinking can be replaced by offline learned intuition**

### If imitation learning fails:
- MC's decisions are too context-dependent to compress into a static model
- Linear Value + Heuristic remain the best production agents
- Finding: the game's complexity sits in a narrow band where search helps but can't be easily approximated
- Still valuable: confirms the 4-tier difficulty system with existing agents

---

## Future Work (Deferred)

These remain optional investigations, not part of the core plan:

- **Level 2 hidden information** — test if hidden deployment benefits DL
- **Multi-agent / self-play** — population-based training
- **Transfer learning** — apply to other grid games
- **Human play study** — ground AI performance in human context
- **Shaped rewards** — test if dense rewards help DQN
- **Alternative algorithms** — Actor-Critic, MCTS, evolutionary strategies
- **Neural network deck awareness** — Phase 3.2 suggests marginal gain at best
