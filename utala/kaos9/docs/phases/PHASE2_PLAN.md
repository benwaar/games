# Phase 2 Implementation Plan

## Goals

**Primary:** Implement learning systems by hand that beat Phase 1 baselines while remaining fully interpretable.

**Secondary:** Produce portable, well-documented AI agents suitable for integration into game implementations.

## Design Principles

### Research Constraints (from Phase 1)
- Fixed state encoding (86-dimensional action space)
- Fixed action enumeration (no variable action sets)
- Illegal actions masked by engine, never removed
- Agents propose actions, never apply them
- All randomness lives in the engine
- Deterministic replays (seed + action list)

### Production Readiness
- **Serializable models:** All learned state (tables, weights) must serialize to JSON/simple formats
- **Documented interfaces:** Clear API boundaries for game integration
- **Behavior quality:** Agents should feel intelligent and varied to human players
- **Performance:** Decision time suitable for real-time gameplay (target: <100ms per action)

## Learning Systems to Implement

### 1. Associative Memory Agent (k-NN)

**Concept:** Store (state, action, outcome) tuples from self-play. At decision time, find k nearest historical states and vote on actions weighted by similarity and outcome.

**Implementation:**
- **State representation:** Extract key features from GameState
  - Grid occupancy vector (27 values: 9 positions × 3 states: empty/P1/P2)
  - Resource counts (rocketmen remaining, weapons remaining, kaos deck size)
  - Phase indicator
  - Simplified feature set (~50 dimensions)
- **Distance metric:** Euclidean or Manhattan distance on normalized features
- **Memory storage:** List of tuples: (feature_vector, action_index, game_outcome, turn_number)
- **Action selection:**
  1. Find k=10-50 nearest neighbors
  2. Weight by inverse distance × outcome value (win=1.0, draw=0.5, loss=0.0)
  3. Aggregate votes per legal action
  4. Select highest-voted action (with epsilon exploration)

**Training:**
- Self-play 1000-5000 games against Phase 1 baselines
- Store every decision point
- Prune memory periodically (keep only diverse/high-value states)

**Deliverables:**
- `associative_memory_agent.py`
- `memory_store.json` (serialized k-NN database)
- Evaluation: win rate vs Random, Heuristic, MC-Fast

**Export format:**
```json
{
  "agent_type": "associative_memory",
  "version": "1.0",
  "k": 20,
  "memory": [
    {
      "features": [0.2, 0.5, ...],
      "action": 42,
      "outcome": 1.0,
      "turn": 5
    }
  ]
}
```

### 2. Linear Action-Value Agent

**Concept:** Learn linear weights that score state-action pairs. Use TD learning (temporal difference) to update weights based on self-play experience.

**Implementation:**
- **State-action features:** Combine state features with action properties
  - State: grid control, material balance, phase progress
  - Action: rocketman power (if placement), weapon type (if dogfight)
  - Combined: ~100 dimensional feature vector per (state, action) pair
- **Value function:** `Q(s, a) = w^T × φ(s, a)` where φ is feature extractor
- **Learning rule:** TD(0) update after each game
  - `w ← w + α × (r + γ × max_a' Q(s', a') - Q(s, a)) × φ(s, a)`
  - Simplified for terminal rewards: after win/loss, backpropagate credit
- **Action selection:** Epsilon-greedy over legal actions

**Training:**
- Initialize weights randomly or zero
- Self-play 5000-10000 games with decaying learning rate
- Track weight evolution, convergence metrics
- Checkpoint best-performing weight vectors

**Deliverables:**
- `linear_value_agent.py`
- `feature_extractor.py` (documented feature engineering)
- `weights.json` (serialized weight vectors)
- Training logs: convergence plots, intermediate evaluations

**Export format:**
```json
{
  "agent_type": "linear_value",
  "version": "1.0",
  "learning_rate": 0.01,
  "discount": 0.95,
  "weights": {
    "grid_control": 0.75,
    "material_advantage": 0.45,
    "placement_power_high": 0.32,
    ...
  }
}
```

### 3. Policy Network (Simple MLP) - Optional Stretch

**Concept:** Small multi-layer perceptron (hand-coded, no frameworks) that learns policy: π(a|s).

**Implementation:**
- 2-layer network: input → hidden (64 units) → output (86 actions)
- Manual backprop implementation
- Softmax output over legal actions (masked)
- Train via policy gradient (REINFORCE) on self-play games

**Note:** Only implement if time permits and first two agents succeed. This pushes boundaries of "learning by hand" but demonstrates gradient-based learning without frameworks.

## Evaluation Framework

### Metrics (from README.md)

**Performance:**
- Win rate vs Phase 1 baselines (Random, Heuristic, MC-Fast)
- Cross-play matrix: all agents vs all agents
- Skill gap smoothness: strong → medium → weak should be monotonic

**Game Balance:**
- First-player advantage: target 52-56%
- Weapon spend profile: avg usage + attack/defense split
- Rocket hit value: should stay around 7 (53.85% hit rate)
- Tie rate & wipeout frequency
- Comeback rate: underdog recoveries

**Behavior Quality:**
- Move diversity: agent shouldn't play identically every game
- Interpretability: can humans understand why agent made a decision?
- Reaction to opponent: does agent adapt to different playstyles?

### Harness Extensions

Extend `evaluation/harness.py`:
- Add detailed metrics collection (weapon usage, comeback tracking)
- Export rich game logs for analysis
- Support serialized agent loading (from JSON)
- Performance profiling (decision time per action)

## Data Management

### Training Data
- Store training games in `data/training/` as compact replays
- Version datasets: `training_v1_random_1000games.jsonl`
- Track provenance: which agents generated which data

### Learned Models
- Store in `models/` with version tags
- Each model includes:
  - Agent config/hyperparameters
  - Learned state (weights, memory)
  - Training metadata (games trained, performance)
  - Timestamp, git commit hash

### Export Format Standard
All agents export to common schema:
```json
{
  "agent_name": "AssociativeMemory-v1",
  "agent_type": "associative_memory",
  "version": "1.0",
  "timestamp": "2026-03-23T10:30:00Z",
  "trained_games": 5000,
  "performance": {
    "vs_random": 0.89,
    "vs_heuristic": 0.67
  },
  "model_data": { ... },
  "hyperparameters": { ... }
}
```

## Documentation

### Structure
Create `docs/phase2/` with a document for each component as we implement it.

Each document should include:
- **What we did:** Implementation details, code structure, key decisions
- **What it means:** Conceptual explanation, why this approach matters
- **Further reading:** Links to papers, articles, or tutorials
- **Video resources:** Links to short (5-15 min) YouTube videos explaining core concepts

### Documents to Create

**Foundation:**
- `docs/phase2/01_state_features.md` - Feature extraction design
- `docs/phase2/02_serialization.md` - Model save/load framework
- `docs/phase2/03_metrics.md` - Extended evaluation metrics

**Associative Memory:**
- `docs/phase2/04_knn_basics.md` - k-NN learning fundamentals
  - Videos: instance-based learning, similarity metrics
- `docs/phase2/05_memory_training.md` - Training process and results
  - Videos: memory-based learning in games

**Linear Value Learning:**
- `docs/phase2/06_td_learning.md` - Temporal difference learning
  - Videos: TD learning, value functions, Q-learning basics
- `docs/phase2/07_feature_engineering.md` - State-action features
  - Videos: feature engineering for RL
- `docs/phase2/08_linear_training.md` - Training process and convergence

**Optional (if implemented):**
- `docs/phase2/09_policy_gradients.md` - REINFORCE and backprop
  - Videos: policy gradient methods, backpropagation from scratch

**Final:**
- `docs/phase2/10_results.md` - Tournament results and analysis
- `docs/phase2/11_integration_guide.md` - How to integrate agents into games

## Implementation Phases

### Phase 2.1: Foundation (Week 1-2)
- [ ] Design state feature extraction (shared across agents)
- [ ] Implement serialization framework (model save/load)
- [ ] Extend evaluation harness with detailed metrics
- [ ] Create training data generation pipeline
- [ ] **Docs:** Write `01_state_features.md`, `02_serialization.md`, `03_metrics.md`

### Phase 2.2: Associative Memory (Week 2-3)
- [ ] Implement k-NN agent with similarity search
- [ ] Generate training data (self-play vs baselines)
- [ ] Tune k, distance metric, pruning strategy
- [ ] Evaluate and compare to Phase 1 baselines
- [ ] **Docs:** Write `04_knn_basics.md` with reading links and video resources
- [ ] **Docs:** Write `05_memory_training.md` with training results

### Phase 2.3: Linear Value Agent (Week 3-5)
- [ ] Implement TD learning for linear function approximation
- [ ] Design state-action feature extractor
- [ ] Train with learning rate schedule
- [ ] Analyze weight evolution and convergence
- [ ] Evaluate and compare
- [ ] **Docs:** Write `06_td_learning.md` with RL fundamentals and videos
- [ ] **Docs:** Write `07_feature_engineering.md` with design rationale
- [ ] **Docs:** Write `08_linear_training.md` with convergence analysis

### Phase 2.4: Evaluation & Documentation (Week 5-6)
- [ ] Run comprehensive tournament (all agents)
- [ ] Collect simulation metrics (weapon usage, balance)
- [ ] Document learned behaviors and interpretability
- [ ] Export final models in portable format
- [ ] Write Phase 2 report
- [ ] **Docs:** Write `10_results.md` with tournament analysis
- [ ] **Docs:** Write `11_integration_guide.md` for game developers

## Success Criteria

**Minimum:**
- At least one learning agent consistently beats Heuristic baseline
- Models are fully serializable and documented
- Training process is reproducible (deterministic given seed)

**Target:**
- Learning agent achieves 60%+ win rate vs MC-Fast
- Agents show interpretable decision-making patterns
- Models export to clean, documented format suitable for game integration
- Training completes in reasonable time (<24 hours on single machine)

**Stretch:**
- Multiple learning agents with different playstyles
- Ensemble method combining agents
- Human evaluation: agents feel intelligent and fun to play against

## Risks & Mitigations

**Risk:** Learning agents overfit to baselines, fail vs novel strategies
**Mitigation:** Train on diverse opponent pool, evaluate cross-play

**Risk:** State features don't capture key strategic elements
**Mitigation:** Start simple, iterate based on interpretability analysis

**Risk:** Training time becomes prohibitive
**Mitigation:** Use fast baselines for bulk training, checkpoint frequently

**Risk:** Learned behaviors are opaque/uninteresting
**Mitigation:** Prioritize interpretability from start, visualize decision factors

## Next Steps

1. Review and approve this plan
2. Set up Phase 2 directory structure:
   - `data/training/` - Training game replays
   - `models/` - Serialized learned models
   - `docs/phase2/` - Implementation documentation
   - `src/utala/learning/` - Learning agent implementations
3. Begin Phase 2.1 foundation work
4. Document each component as we build it (with readings + videos)
5. Checkpoint after associative memory agent before proceeding to linear value
