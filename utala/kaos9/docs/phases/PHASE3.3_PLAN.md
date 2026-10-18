# Phase 3.3 — What Matters and Why (Understanding)

**Date:** 2026-04-07
**Status:** Planned
**Prerequisite:** Phase 3.2 complete

---

## The Phase 3 Question

> **Search (MC) is strong but expensive. Learning (Linear) is cheap but weaker. Can we distill search knowledge into a learned model — getting the best of both?**

Phase 3.3 is the **understanding** step: before we try to compress MC into a fast model (Phase 3.4), we need to know what features actually matter and what the minimal representation looks like. There's no point training an imitation model on 43 features if only 10 of them drive decisions.

---

## Context

**What we know from Phases 1–3.2:**

- Linear Value (42% vs Heuristic) is the strongest learning agent
- MC-Fast (54% vs Heuristic) is the strongest overall but compute-heavy (~100ms/move)
- Two attempts to add complexity failed (Phase 2.5 enhanced features, Phase 3.1 DQN)
- Deck awareness (card counting) provided no benefit to a linear model
- The game is dominated by spatial tactics (3-in-a-row positioning)

**What we need to know before distillation (Phase 3.4):**

- Which features actually drive decisions? (So the student model uses the right inputs)
- What is the minimal feature set? (Smaller = faster + easier to learn from MC)
- Can learning discover tactics from raw input, or must features be engineered? (Determines student architecture)

---

## Goals

1. Understand what the Linear Value agent learned (interpretability)
2. Find the minimal feature set required for competent play
3. Determine whether the game's tactics can be auto-discovered

---

## Plan (1–2 Weeks)

### 1. Weight Analysis & Interpretability

Analyze the trained Linear Value agent's weights:

- Rank features by absolute weight magnitude
- Identify positive correlations (good for winning) vs negative (bad)
- Compare learned weights to Heuristic's hand-coded rules
- Find surprising patterns (features that matter more/less than expected)

**Script:** `analyze_linear_weights.py`

**Output:** Feature importance ranking, comparison table vs Heuristic logic

---

### 2. Feature Importance via Ablation

Systematically remove features to measure impact:

- Remove one feature at a time → measure win rate drop
- Remove feature *categories* (material, control, position, threats)
- Identify critical vs redundant features
- Find the minimal feature set that retains ~40% vs Heuristic

**Script:** `feature_importance.py`

**Output:** Feature ranking by impact, minimal feature set

---

### 3. Minimal Intelligence Experiment

Test how far we can strip down and still play:

- **Heuristic with only 3-in-a-row features** — is pattern recognition enough?
- **Reduced feature linear model** — train with only top-N features
- **1-step lookahead agent** — evaluate immediate outcomes only

Find the **collapse point**: where does performance drop sharply?

**Script:** `minimal_intelligence.py`

**Output:** Performance vs feature count curve, collapse threshold

---

### 4. Representation Test

Can learning discover 3-in-a-row without being told?

- **Model A:** Raw board input only (grid occupancy, no engineered features)
- **Model B:** Engineered features (material, control, threats — current)

Train both with identical hyperparameters. Compare performance.

**Note:** Based on Phase 3.2 deck awareness results, we already know a linear model struggles with extra features. If Model A fails badly, it confirms that **feature engineering is essential** for linear models in this game. A neural network *might* discover 3-in-a-row from raw input, but given DQN's Phase 3.1 failure, this is unlikely to be productive — include as a brief experiment, not a deep investigation.

**Script:** `representation_test.py`

**Output:** Raw vs engineered feature comparison

---

## Success Criteria

- Clear feature importance ranking (which features matter, which don't)
- Minimal feature set identified (fewest features for ~40% vs Heuristic)
- Understanding of whether the game requires *given* tactical knowledge or can *discover* it
- Comparison between learned strategy and hand-coded Heuristic rules

---

## What This Phase Does NOT Include

- Imitation learning (moved to Phase 3.4)
- Search vs evaluation Pareto frontier (moved to Phase 3.4)
- Mobile AI features / difficulty tuning (moved to Phase 3.4)
- Neural network deck awareness extraction (Phase 3.2 showed marginal gain unlikely)
- Curriculum learning, multi-agent training, transfer learning (future work)

---

## Deliverables

- `analyze_linear_weights.py` — weight analysis + Heuristic comparison
- `feature_importance.py` — ablation study
- `minimal_intelligence.py` — stripped-down agent experiments
- `representation_test.py` — raw vs engineered features
- `PHASE3.3_RESULTS.md` — analysis document with findings

---

## Expected Outcomes

### If features are highly redundant:
- Many features can be removed without loss
- A tiny model (5-10 features) suffices
- Mobile deployment even simpler

### If features are all important:
- Current 43-feature set is well-designed
- Feature engineering was critical to success
- Validates Phase 2 design decisions

### Either way:
- Deep understanding of what makes the game tick
- Publication-quality findings about game complexity
- Clear guidance for mobile AI implementation
