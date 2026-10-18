# Mobile Game AI Agents - Difficulty Ranking

**For utala: kaos 9 Mobile Game Integration**
**Updated:** 2026-04-08 (post Phase 3 complete)

---

## Agent Difficulty Ladder (Easiest → Hardest)

### Level 1: Random Agent (Tutorial/Beginner)
**Win Rate:** ~20% vs Heuristic | **Model:** `src/utala/agents/random_agent.py`

**Strength:** Weakest possible opponent
**Play Style:** Completely random legal moves
**Best For:**
- First-time players learning the rules
- Tutorial mode
- Practice mode
- Confidence building

**Technical:**
- No model file needed
- Instant inference (<1ms)
- Zero memory footprint
- Always available

**Player Experience:** "Practice against a beginner AI"

---

### Level 2: Easy AI (k-NN Agent)
**Win Rate:** 33% vs Heuristic | **Model:** `models/knn_agent_1000games.json`

**Strength:** Learns from examples but makes mistakes
**Play Style:** Pattern matching from 1K game memory
**Best For:**
- Casual players
- Second difficulty tier
- Players who beat Random consistently

**Technical:**
- Model file: ~500KB (1K game states)
- Inference: ~5ms (k=5 nearest neighbor search)
- Memory: 32K examples × 53 features

**Player Experience:** "A learning AI that remembers patterns"

**Note:** Slower inference than others, consider for mobile performance

---

### Level 3: Medium AI (DQN Agent)
**Win Rate:** 31% vs Heuristic | **Model:** `results/dqn/dqn_final.pth`

**Strength:** Neural network, moderate difficulty
**Play Style:** Learned through deep reinforcement learning
**Best For:**
- Intermediate players
- "Smart AI" marketing
- Players comfortable with basics

**Technical:**
- Model file: ~500KB (PyTorch weights)
- Inference: ~3ms (3-layer neural network)
- Parameters: 34,518
- Requires PyTorch or ONNX export for mobile

**Player Experience:** "Challenge a neural network AI"

**Note:** Requires ML framework on mobile (PyTorch Mobile or ONNX Runtime)

---

### Level 4: Hard AI (TD-Linear NoInt) ⭐
**Win Rate:** 47.5% vs Heuristic | **Model:** 27 weights (no interaction features)

**Strength:** Strongest learning agent — matches imitation learning from MC search
**Play Style:** Tactical, values material and control, avoids telegraphing lines
**Best For:**
- Experienced players
- Main competitive AI
- **RECOMMENDED** primary opponent

**Technical:**
- Model file: <1KB (27 weights)
- Inference: <0.25ms (simple dot product)
- Parameters: 27 weights
- No dependencies (pure math)

**Player Experience:** "Face our smartest AI opponent"

**Why This Is Best:**
- ✅ Fastest inference (<0.25ms)
- ✅ Smallest model (<1KB)
- ✅ No ML framework needed
- ✅ Works on any platform
- ✅ Best performance of all learning agents (Phase 3.4 confirmed)
- ✅ Matches imitation learning from MC search — no teacher needed

---

### Level 5: Expert AI (Heuristic Agent) 🏆
**Win Rate:** Baseline (beats Random 70%, beats Linear Value 58%) | **Model:** `src/utala/agents/heuristic_agent.py`

**Strength:** Hand-crafted expert system (strongest)
**Play Style:** Rule-based tactical decisions
**Best For:**
- Advanced players
- "Expert" difficulty
- Final challenge

**Technical:**
- No model file (coded rules)
- Inference: <1ms
- Zero memory footprint
- Pure algorithmic logic

**Player Experience:** "Master level - expert tactical AI"

**Note:** This is actually the strongest agent overall

---

## Recommended Difficulty Progression

```
┌─────────────────────────────────────────────────────────┐
│ Easy        │ Medium          │ Hard          │ Expert    │
│ Random      │ TD-Linear+30%ε  │ TD-Linear     │ Heuristic │
│ ~20%        │ ~30%            │ ~48%          │ ~58%      │
└─────────────────────────────────────────────────────────┘
   Learn         Practice         Compete         Master
```

---

## Final 4-Tier System (Phase 3.4 Recommendation)

### Easy
**Agent:** Random
**Why:** Learn the rules without pressure

### Medium
**Agent:** TD-Linear NoInt + 30% epsilon
**Why:** Same core agent with exploration noise for ~30% strength

### Hard
**Agent:** TD-Linear NoInt (greedy)
**Why:** 47.5% vs Heuristic — strongest learning agent, proven by Phase 3.4 Pareto analysis

### Expert
**Agent:** Heuristic
**Why:** Expert-level tactics, explicit 3-in-a-row logic

**Skip k-NN, DQN, and imitation models** — TD-Linear NoInt matches or beats all of them with zero dependencies.

---

## Implementation Guide

### Production Setup (4 tiers, 2 core agents)

```python
from src.utala.agents.random_agent import RandomAgent
from src.utala.agents.linear_value_agent import LinearValueAgent
from src.utala.agents.heuristic_agent import HeuristicAgent

# Easy mode
easy_ai = RandomAgent("Easy AI")

# Medium mode (TD-Linear with exploration)
medium_ai = LinearValueAgent("Medium AI", epsilon=0.3,
                             deck_awareness=False)
# Train or load TD-Linear NoInt weights (27 features, no interaction)
# medium_ai.load("models/td_linear_noint.json")

# Hard mode (TD-Linear greedy)
hard_ai = LinearValueAgent("Hard AI", epsilon=0.0,
                           deck_awareness=False)
# Same weights, no exploration

# Expert mode
expert_ai = HeuristicAgent("Expert AI")
```

**Note:** TD-Linear NoInt uses 27 features (the 32 standard features minus 5 interaction features). See `train_imitation.py` for the interaction feature mask (indices 26-30).

---

## Performance Comparison

| Agent | Difficulty | Win Rate | Model Size | Inference | Mobile Ready |
|-------|-----------|----------|------------|-----------|--------------|
| **Random** | Easy | ~20% | 0 | <0.01ms | ✅ Yes |
| **TD-Linear + 30%ε** | Medium | ~30% | <1KB | <0.25ms | ✅ Yes |
| **TD-Linear (greedy)** | Hard | 47.5% | <1KB | <0.25ms | ✅ Yes |
| **Heuristic** | Expert | ~58% | 0 | <0.05ms | ✅ Yes |

**All agents are mobile-ready.** No ML framework needed. Skip k-NN (slow), DQN (needs PyTorch), and imitation models (same performance, more complexity).

---

## Model Export Formats

### Option A: JSON Weights (Recommended)

TD-Linear NoInt is 27 floats — export as a JSON array, load on any platform, compute a dot product. No framework, no runtime, no dependencies.

```
Model:     27 weights × 4 bytes = 108 bytes (float32)
On disk:   <1KB JSON
Inference: dot product — works in any language
```

### Option B: ONNX Neural Network (Available)

The Imitation-NN (32-unit hidden layer, 1,089 parameters) is saved at `models/imitation_nn_32h.pth` and can be exported to ONNX. Performance is nearly identical (48.0% vs 47.5%) but requires an ONNX runtime.

```
Model:     1,089 parameters (32 inputs → 32 hidden ReLU → 1 output)
On disk:   ~5-7KB ONNX
Inference: requires ONNX Runtime, CoreML, or PyTorch Mobile
```

**Size/speed comparison:**

| Format | Model Size | Inference | Dependencies | vs Heuristic |
|--------|-----------|-----------|--------------|-------------|
| **JSON (27 weights)** | **<1KB** | **<0.25ms** | **None** | **47.5%** |
| ONNX (1,089 params) | ~5-7KB | <0.25ms | ONNX Runtime | 48.0% |

The NN adds 50x more parameters and a runtime dependency for +0.5pp. Use JSON unless you specifically need the ONNX export path (e.g. demonstrating the pipeline, or as a stepping stone for future larger models).

---

## Mobile Platform Considerations

### Android
**Recommended:** Random, TD-Linear (JSON), Heuristic
- Pure Java/Kotlin implementation — load JSON, compute dot product
- No ML framework needed
- **If using ONNX:** ONNX Runtime for Android (~5MB library)

### iOS
**Recommended:** Random, TD-Linear (JSON), Heuristic
- Pure Swift implementation — load JSON, simple math
- No ML framework needed
- **If using ONNX:** CoreML can import ONNX models

### Cross-Platform (React Native, Flutter)
**Recommended:** Random, TD-Linear (JSON), Heuristic
- JSON parsing built-in
- TD-Linear works everywhere with basic math
- **If using ONNX:** onnxruntime packages available for both platforms

---

## Player-Facing Names

### Marketing-Friendly Labels

```
Easy:       "Practice Bot"     (Random)
Medium:     "Tactical AI"      (TD-Linear + noise)
Hard:       "Strategic AI"     (TD-Linear greedy)
Expert:     "Master AI"        (Heuristic)
```

Optional: "Neural Network AI" label for ONNX variant (Imitation-NN) — same strength, but sounds impressive.

---

## Tunable Difficulty (Advanced)

### TD-Linear with Epsilon Exploration

```python
# Make TD-Linear easier by adding randomness
def select_action_with_difficulty(agent, state, legal_actions, difficulty):
    if difficulty == "easy":
        epsilon = 0.4  # 40% random moves
    elif difficulty == "medium":
        epsilon = 0.3  # 30% random moves (~30% strength)
    elif difficulty == "hard":
        epsilon = 0.0  # Pure greedy (47.5% vs Heuristic)

    if random.random() < epsilon:
        return random.choice(legal_actions)
    else:
        return agent.select_action(state, legal_actions)
```

### Result: One Agent, Multiple Difficulties
- Easy: TD-Linear + 40% random = ~25% strength
- Medium: TD-Linear + 30% random = ~30% strength
- Hard: Pure TD-Linear = ~48% strength
- Expert: Heuristic = ~58% strength

---

## Recommended Setup for Mobile Game

### Final Recommendation: 4-Tier System ⭐

```
Easy:     Random Agent (~20%)
Medium:   TD-Linear NoInt + 30% epsilon (~30%)
Hard:     TD-Linear NoInt greedy (~48%)
Expert:   Heuristic Agent (~58%)
```

**Why This Is Best (confirmed by Phase 3.4 Pareto analysis):**
- ✅ Smooth difficulty curve with meaningful gaps between tiers
- ✅ One core learning agent (TD-Linear) for Medium and Hard
- ✅ All agents <0.25ms, all <1KB, all zero-dependency
- ✅ TD-Linear matches imitation learning from MC search — no teacher/dataset needed
- ✅ Phase 3.4 proved this is the performance ceiling for current features

---

## Final Recommendation (Phase 3.4)

**Use 4 difficulty tiers with 2 core agents:**

1. **Easy** - Random Agent (~20% strength)
2. **Medium** - TD-Linear NoInt + 30% epsilon (~30% strength)
3. **Hard** - TD-Linear NoInt greedy (~48% strength)
4. **Expert** - Heuristic Agent (~58% strength)

**Implementation:**
```python
class GameAI:
    def __init__(self):
        self.random = RandomAgent()
        self.td_linear = LinearValueAgent(deck_awareness=False)
        # Load trained TD-Linear NoInt weights (27 features)
        # self.td_linear.load("models/td_linear_noint.json")
        self.heuristic = HeuristicAgent()

    def get_agent(self, difficulty):
        if difficulty == "easy":
            return self.random
        elif difficulty == "medium":
            return (self.td_linear, 0.3)  # 30% exploration
        elif difficulty == "hard":
            return (self.td_linear, 0.0)  # Pure greedy
        elif difficulty == "expert":
            return self.heuristic
```

---

## Summary

**Final difficulty tiers (confirmed by Phase 3.4 Pareto analysis):**

1. **Random** (~20%) - Easy
2. **TD-Linear NoInt + 30%ε** (~30%) - Medium
3. **TD-Linear NoInt greedy** (~48%) - Hard ⭐
4. **Heuristic** (~58%) - Expert

**Key Phase 3 findings for mobile:**
- TD-Linear NoInt is the best learning agent (47.5% vs Heuristic, best seed 49.5%)
- Imitation learning from MC search adds no value — TD already captures everything
- All fast models converge to ~48% — this is the performance ceiling for current features
- MC search (485ms/decision) is both slower AND weaker than TD-Linear (<0.25ms)
- Skip k-NN, DQN, imitation models, and deck awareness — none improve on TD-Linear

**Files You Need:**
- `src/utala/agents/random_agent.py`
- `src/utala/agents/linear_value_agent.py`
- `src/utala/agents/heuristic_agent.py`
- `src/utala/learning/state_action_features.py` (feature extraction)
- TD-Linear NoInt weights (27 floats, <1KB)
