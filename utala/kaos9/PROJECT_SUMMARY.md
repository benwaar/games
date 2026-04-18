# utala: kaos 9 - AI Research Project Summary

**Date:** 2026-04-08 (updated)
**Status:** Phase 3 Complete — Phase 4 Planned

---

## Project Overview

**utala: kaos 9** is a 2-player tactical grid combat card game. This research project built a game engine, evaluation harness, and progressively sophisticated AI agents to study learning algorithms and tactical decision-making.

**Key Finding:** For tactical grid games of this complexity, **linear function approximation with hand-crafted features** is optimal. Removing harmful features matters more than adding clever ones. Search knowledge can be distilled — but TD learning already captures it without a teacher.

---

## Deliverables

### ✅ Game Engine
- Complete Python implementation of utala: kaos 9 rules
- 3×3 grid placement and dogfight mechanics
- Kaos deck tracking and resolution
- Deterministic replay system (seed + action list)

### ✅ Evaluation Harness
- Self-play and cross-play evaluation
- Balanced matchups (alternating first player)
- Statistical analysis and reporting
- Tournament infrastructure

### ✅ AI Agents (7 Implementations)

| Agent | Type | Performance vs Heuristic | Notes |
|-------|------|-------------------------|-------|
| **Random** | Baseline | ~20% | Uniform random legal actions |
| **Heuristic** | Rule-based | N/A (baseline) | Hand-coded tactical rules |
| **MC-Fast** | Monte Carlo (10 rollouts) | 44.5% | Search-based, compute-heavy |
| **k-NN** | Memory-based | 33% | 1K game memory, k=5 |
| **TD-Linear (NoInt)** | TD Learning | **47.5%** | **Production agent** (27 features) |
| **Imitation-NN** | Imitation Learning | 48.0% | Tiny NN, 1,089 params |
| **Imitation-Linear** | Imitation Learning | 47.0% | Linear, trained on MC decisions |
| **Linear Value** | TD Learning | 42% | Original 32-feature version |
| **Policy Network** | REINFORCE | 18% | Failed (sparse rewards) |
| **DQN** | Deep Q-Learning | 31% | Failed (unnecessary complexity) |

### ✅ Production-Ready Agent

**TD-Linear NoInt (Phase 3.3)**
- **Performance:** 47.5% win rate vs Heuristic (best seed: 49.5%)
- **Algorithm:** Temporal Difference (TD) learning
- **Features:** 27 hand-crafted tactical features (removed 5 harmful interaction features)
- **Training:** 5K games (~2 minutes)
- **Parameters:** 27 weights (<1KB)
- **Inference:** <0.25ms per decision

**Ready for mobile game integration:**
- Fast inference (<0.25ms per action)
- Small model size (<1KB)
- Deterministic behavior
- No external dependencies
- Platform-agnostic

---

## Research Phases

### Phase 1: Baselines (Complete)
- Random agent: uniform random legal moves
- Heuristic agent: hand-coded tactical rules
  - Material advantage
  - Control advantage
  - Position value
- Established 70% win rate of Heuristic vs Random

### Phase 2: Learning Algorithms (Complete)

**Phase 2.2: k-Nearest Neighbors (Memory-Based Learning)**
- 1K game memory, k=5 neighbors
- Result: 33% vs Heuristic
- Insight: Memory alone insufficient

**Phase 2.3: Linear Value (TD Learning)**
- 43 hand-crafted features
- TD(0) learning, α=0.01
- Result: **42% vs Heuristic** ✅
- Insight: Linear approximation sufficient

**Phase 2.4: Policy Network (REINFORCE)**
- 285K parameters, policy gradient
- Result: 18% vs Heuristic ❌
- Insight: Sparse rewards hurt policy gradients

**Phase 2.5: Enhanced Features (Failed Experiment)**
- 49 enhanced features (line detection, blocking, forking)
- Result: 22.5% vs Heuristic ❌
- Insight: Feature engineering made it worse

### Phase 3: Deep Learning & Validation (Complete)

**Phase 3.1: Deep Q-Network (DQN)**
- 34,518 parameters (3-layer network)
- Experience replay + target networks
- Curriculum learning (3 stages)
- Result: 31% vs Heuristic ❌
- Insight: Neural networks don't help

**Phase 3.2: Information Integrity & Deck Awareness (Complete)**

Three investigations completed:

1. **Information integrity audit** — Confirmed learning agents never see hidden info. MC agents *were* using perfect information (face-down cards, Kaos deck order). Fixed: `use_information_sets=True` now default. Impact: MC-Perfect beats MC-Fair 72-28% head-to-head; prior MC benchmarks were inflated by ~12%.

2. **DQN sanity check** — CartPole-v1 solved at episode 319 (avg reward 275.6). Confirms DQN implementation is correct; utala's poor DQN results are a genuine finding about the game.

3. **Deck awareness (card counting)** — Added 10 features tracking Kaos deck composition (high/low ratios, expected value, variance, strength). Result: **hypothesis rejected**. DeckAware agent was 19pp worse vs Heuristic, neutral vs Random, marginal 53.5% head-to-head. The game is dominated by spatial tactics; probabilistic reasoning doesn't help a linear model.

**Phase 3.3: What Matters and Why (Complete)**

Four investigations completed:

1. **Weight analysis** — Top 5 features (material, control, opp count, my squares, turn) hold 50% of total weight. 6 features near-zero. Model highly concentrated.

2. **Feature ablation** — Removing interaction features improves by +6.5pp. Top-5 features match full 32-feature model. Interaction features are actively harmful.

3. **Minimal intelligence** — Single feature (material_advantage) gets 32% vs Heuristic. No sharp cliff — performance degrades gradually. Best model: 27 features without interaction = **47.5% vs Heuristic**.

4. **Representation test** — Raw board (27 grid occupancy features) gets 29.5% vs 44.5% for engineered features. Feature engineering is essential for linear models.

**Bonus finding:** Line formation features were hardcoded to 0.0 (TODO never implemented). When fixed, agent learned *negative* weights — forming visible lines is counterproductive against Heuristic's blocking logic.

**Phase 3.4: Compress Search into Evaluation (Complete)**

1. **Imitation learning** — Trained linear and tiny NN models on 8,228 MC-Fast decisions. Linear: 79.4% pairwise accuracy. NN (32-unit hidden, 1,089 params): 48.2%.

2. **Student beat teacher** — Imitation models (47-48% vs Heuristic) outperform MC-Fast (44.5%). MC with 10 rollouts has high variance; averaging over decisions smooths noise.

3. **Pareto frontier** — MC-Fast is NOT on the frontier (slower AND weaker than fast models). All fast models (TD-Linear, Imitation-Linear, Imitation-NN) converge to 47-48% at <0.25ms.

4. **Key conclusion** — Distillation works but is unnecessary. TD learning already captures everything MC search discovers. **Performance ceiling at ~48%** for current features.

---

## Key Research Findings

### 1. Linear Approximation Is Optimal for This Game

**Evidence:**
- Linear Value (42%) > Enhanced Features (22.5%)
- Linear Value (42%) > DQN (31%)
- Linear Value (42%) > Policy Network (18%)

**Why Linear Works:**
- Game tactics captured by linear feature combinations
- Material + control + position ≈ win probability
- 43 features sufficient to represent state value
- Fast learning (5K games vs 100K+ for DQN)

### 2. Complexity Doesn't Always Help

**Pattern of Failures:**
- Phase 2.5: Add 6 features → -19.5 percentage points
- Phase 3.1: Add 34K parameters → -11 percentage points

**Root Causes:**
- Problem not complex enough to warrant deep learning
- Sparse rewards hurt credit assignment
- Hand-crafted features already optimal
- More parameters = more overfitting risk

### 3. When NOT to Use Deep Learning

**utala: kaos 9 characteristics:**
- Small state space (53 features)
- Short episodes (15-20 actions)
- Sparse rewards (only at game end)
- Tactical patterns are linear

**Compare to successful DL domains:**
- Atari: Complex visual patterns, millions of samples
- Go/Chess: Huge state space, long horizons
- Continuous control: High-dimensional sensors

**Lesson:** Try simple methods first. Use DL only if simple methods fail.

### 4. Feature Engineering > Architecture Engineering

**Phase 2 success factors:**
- Good features: material, control, position, threats
- Simple algorithm: TD(0) with linear weights
- Fast learning: 5K games, 2 minutes

**Phase 3 failure factors:**
- Same features, complex architecture
- Slower learning: 20K games, 3 minutes
- Worse performance

**Conclusion:** Time spent on features beats time spent on architecture.

### 5. Spatial Tactics Dominate — Probabilistic Reasoning Doesn't Help

**Phase 3.2 deck awareness experiment:**
- Added 10 features tracking Kaos deck composition (high/low card ratios, expected value, variance)
- DeckAware agent: -19pp vs Heuristic, +0.5% vs Random, 53.5% head-to-head
- Card counting signal is either too weak or too nuanced for linear features

**Conclusion:** The game has one primary layer of intelligence: 3-in-a-row positioning. Probabilistic reasoning about deck composition is not worth the complexity.

### 6. Hidden Information Matters (For Search Agents)

**Phase 3.2 information integrity audit:**
- MC agents were using perfect information (seeing face-down cards and Kaos deck order)
- MC-Perfect beats MC-Fair 72-28% head-to-head
- Against Heuristic: perfect info worth +12%
- Learning agents were always clean (features never included hidden info)

**Conclusion:** Hidden information is a significant factor in this game, but only matters for agents that access raw state (MC). Feature-based agents are naturally information-set compliant.

### 7. Fewer, Cleaner Features Beat More Features

**Phase 3.3 ablation study:**
- Removing 5 interaction features improves by +6.5pp (from 44.5% to 47.5%)
- Top 5 features match the full 32-feature model
- Every experiment in this project confirms: feature reduction > feature addition
- Phase 2.5 (+6 features = -19.5pp), Phase 3.2 (+10 deck features = -19pp), Phase 3.3 (-5 interaction features = +6.5pp)

**Conclusion:** For this game, the optimal approach is aggressive feature pruning, not feature engineering.

### 8. Search Knowledge Can Be Distilled — But TD Already Has It

**Phase 3.4 imitation learning:**
- Trained fast models on MC-Fast's decisions (8,228 examples from 500 games)
- Student beat teacher: imitation models (47-48%) > MC-Fast (44.5%)
- But TD-Linear (47.5%) already matches imitation without needing MC as a teacher
- All fast models converge to ~48% — a performance ceiling for current features

**Conclusion:** The game's decision space is simple enough that TD learning with good features captures everything MC search finds. Runtime thinking can be replaced by offline learning, but it happens automatically.

---

## Technical Specifications

### State Representation
- **53 features** per state
- Categories: material (18), control (9), position (9), threats (7)
- Normalized to [-1, 1] range
- Extracted for current player's perspective

### Action Space
- **86 discrete actions** (fixed enumeration)
- Placement: 9 positions
- Rocket commit: 77 choices (number × target)
- Illegal actions masked by engine
- Agents never see variable action sets

### Learning Infrastructure
- TD learning with linear function approximation
- Experience replay buffer (20K capacity)
- PyTorch neural networks (for Phase 3)
- Curriculum learning framework
- Evaluation harness with statistical analysis

### Model Formats
- **Linear Value:** JSON (weights vector)
- **DQN:** PyTorch .pth + JSON metadata
- **Replay:** Seed + action sequence
- All formats portable and human-readable

---

## Performance Comparison

| Metric | Random | Heuristic | MC-Fast | TD-Linear NoInt | Imitation-NN | Linear Value | DQN |
|--------|--------|-----------|---------|-----------------|--------------|--------------|-----|
| **vs Heuristic** | ~20% | N/A | 44.5% | **47.5%** | 48.0% | 42% | 31% |
| **Training Time** | — | — | — | 5K games | 500 MC games* | 5K games | 20K games |
| **Parameters** | 0 | 0 | 0 | 27 | 1,089 | ~50 | 34K |
| **Inference Speed** | <0.01ms | <0.05ms | 485ms | <0.25ms | <0.25ms | <1ms | ~3ms |

*Imitation-NN requires 67 min MC dataset generation + 3s training. TD-Linear takes ~2 min total.

**Pareto frontier:** Random → TD-Linear NoInt / Imitation models (all ~47-48% at <0.25ms)
**MC-Fast is NOT on the frontier** — slower AND weaker than fast models.
**Best for mobile:** TD-Linear NoInt (47.5%, <0.25ms, <1KB, no dependencies)

---

## Code Repository Structure

```
utala-kaos-9-ai/
├── src/utala/
│   ├── engine.py        # Game engine core
│   ├── state.py         # Game state representation
│   ├── actions.py       # Action space and masking
│   ├── agents/          # All agent implementations
│   ├── learning/        # Feature extraction, training infra
│   ├── deep_learning/   # DQN, neural networks
│   └── evaluation/      # Harness for running games
├── scripts/
│   ├── train/           # Training scripts (train_*.py)
│   ├── eval/            # Evaluation scripts
│   ├── analysis/        # Phase 3.3/3.4 analysis
│   ├── export/          # Model export (JSON, ONNX)
│   └── demo/            # Demo/play scripts
├── tests/               # Unit tests (89 tests)
├── models/              # Trained agent checkpoints
│   └── mobile/          # Exported models for mobile
├── docs/
│   ├── phases/          # Phase plans and results (2–4)
│   └── mobile/          # Mobile integration guides
├── results/             # Training output data
└── PROJECT_SUMMARY.md   # This file
```

---

## For Mobile Game Integration

### Using the Linear Value Agent

**Model File:** `models/LinearValue-v1-checkpoint-5000_20260324_090050.json`

**Loading (any platform):**
```python
# Python example
from src.utala.agents.linear_value_agent import LinearValueAgent
agent = LinearValueAgent(name="AI")
agent.load("models/LinearValue-v1-checkpoint-5000_20260324_090050.json")

# JSON structure:
{
  "name": "LinearValue-v1-checkpoint-5000",
  "agent_type": "linear_value",
  "version": "1.0",
  "weights": [0.123, -0.456, ...],  # 43 weights
  "feature_names": [...],
  "training_games": 5000,
  "win_rate_vs_heuristic": 0.42
}
```

**Integration Steps:**
1. Extract 53 features from game state (see `src/utala/learning/features.py`)
2. Load 43 weights from JSON
3. Compute Q(s,a) = weights · features for each legal action
4. Select action with highest Q-value
5. Optionally add epsilon-greedy exploration for variety

**Difficulty Levels (Final — Phase 3.4):**
- **Easy:** Random (~20%)
- **Medium:** TD-Linear + 30% epsilon (~30%)
- **Hard:** TD-Linear greedy (~48%)
- **Expert:** Heuristic (~58%)

---

## Research Contributions

### Scientific Value
1. **Characterized problem complexity:** When linear suffices vs when DL helps
2. **Negative results:** Documented two failed enhancement attempts
3. **Methodology:** Reproducible research pipeline
4. **Open source:** All code and data available

### Practical Value
1. **Production agent:** Fast, small, effective AI opponent
2. **Portable format:** JSON weights, no dependencies
3. **Interpretable:** Can analyze learned weights
4. **Extensible:** Framework ready for other games

### Lessons for Game AI
1. Start simple (linear before neural)
2. Feature engineering matters most
3. Measure problem complexity first
4. Sparse rewards hurt learning
5. Know when NOT to use deep learning

---

## Lessons Learned

### What Worked
- ✅ Hand-crafted features for tactical patterns
- ✅ Linear function approximation with TD learning
- ✅ Fast training (5K games sufficient)
- ✅ Simple, interpretable models
- ✅ Systematic evaluation methodology

### What Didn't Work
- ❌ Enhanced feature engineering (made it worse)
- ❌ Policy gradients (sparse rewards)
- ❌ Deep Q-Networks (unnecessary complexity)
- ❌ Self-play curriculum (didn't help convergence)

### Key Insights
1. **Simplicity beats complexity** when problem permits
2. **Features > architecture** for structured games
3. **Sample efficiency** crucial for practical training
4. **Know your problem** before choosing algorithms
5. **Negative results** are valuable research findings

---

## Project Statistics

- **Duration:** 4 weeks (March–April 2026)
- **Lines of Code:** ~7,000 Python
- **Agents Implemented:** 9 (Random, Heuristic, MC variants, k-NN, Linear Value, TD-Linear NoInt, Imitation-Linear, Imitation-NN, Policy, DQN)
- **Total Training Games:** ~80K across all experiments + 500 MC dataset games
- **Documentation:** 4,000+ lines across 15+ analysis documents
- **Best Agent:** TD-Linear NoInt (47.5% vs Heuristic, 27 parameters, <1KB)
- **Performance Ceiling:** ~48% vs Heuristic with current features (all methods converge)

---

## References

### Game Rules
- `utala-kaos-9.md` - Complete game rules and mechanics

### Technical Documentation (in `docs/phases/`)
- `PHASE2_COMPLETE.md` - Learning algorithm implementations
- `PHASE2.5_ANALYSIS.md` - Failed enhancement attempt
- `PHASE3.1_ANALYSIS.md` - Deep learning experiment
- `PHASE3.2.md` - Information integrity, DQN validation, deck awareness
- `PHASE3.3_RESULTS.md` - Weight analysis, ablation, minimal intelligence results
- `PHASE3.4_RESULTS.md` - Imitation learning, Pareto frontier, Phase 3 conclusion
- `PHASE4_PLAN_NEXT.md` - Future work (game balance tuning, guardrails plan)

### Code Entry Points (in `scripts/`)
- `scripts/train/train_linear_agent.py` - Train Linear Value agent
- `scripts/train/train_imitation.py` - Phase 3.4 imitation learning + Pareto frontier
- `scripts/analysis/analyze_linear_weights.py` - Phase 3.3 weight analysis
- `scripts/analysis/feature_importance.py` - Phase 3.3 feature ablation
- `scripts/analysis/minimal_intelligence.py` - Phase 3.3 minimal models + raw board test
- `scripts/export/export_models.py` - Export models for mobile (JSON + ONNX)
- `scripts/eval/eval_info_sets.py` - MC information sets comparison (Phase 3.2)
- `src/utala/evaluation/harness.py` - Evaluation framework

---

## Conclusion

**utala: kaos 9 AI research is complete through Phase 3, delivering a production-ready agent and comprehensive understanding of game intelligence.**

Key findings:
- **Linear function approximation is optimal** for this game's complexity (47.5% vs Heuristic)
- **Fewer features > more features:** Removing interaction features was the single best improvement (+6.5pp)
- **Complexity doesn't help:** Enhanced features (-19.5pp), DQN (-11pp), deck awareness (-19pp) all made it worse
- **Spatial tactics dominate:** 3-in-a-row positioning is the primary driver of play strength
- **Search knowledge can be distilled** — but TD learning already captures it without a teacher
- **Performance ceiling at ~48%:** All fast models (TD, imitation) converge regardless of training method
- **Hidden information matters for search:** MC-Perfect beats MC-Fair 72-28%, but learning agents are naturally fair

Phase 4 (game balance tuning and rule variants) is the next frontier.

---

**Project Status:** Phase 3 Complete — Phase 4 Planned
**Best Agent:** TD-Linear NoInt (47.5% vs Heuristic, 27 params, <1KB, <0.25ms)
**Performance Ceiling:** ~48% vs Heuristic (all methods converge)
**Next Steps:** See `PHASE4_PLAN.md` (game balance tuning, rule variants)
