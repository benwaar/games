# Phase 3 Implementation Plan: Deep Learning

## Goals

**Primary:** Implement deep learning systems that significantly improve on Phase 2's Linear Value agent (42% vs Heuristic baseline), targeting 60-70% win rate.

**Secondary:** Demonstrate when and why neural networks help, maintaining interpretability where possible.

## Building on Phase 2

**Phase 2 Foundation (42% vs Heuristic):**
- Linear Value agent with TD learning ✅
- 43-dimensional state-action features ✅
- Self-play training infrastructure ✅
- Portable model serialization ✅

**Phase 3 Advances Beyond Linear:**
- Non-linear function approximation (neural networks)
- Experience replay (break temporal correlation)
- Target networks (stabilize learning)
- Advanced exploration strategies
- Deeper tactical understanding

## Design Principles

### Research Constraints (inherited from Phase 1 & 2)
- Fixed state encoding (53-dim state, 86-dim action space)
- Fixed action enumeration (no variable action sets)
- Illegal actions masked by engine
- Agents propose actions, never apply them
- All randomness lives in the engine
- Deterministic training (given seeds)

### Deep Learning Philosophy
- **Minimal dependencies:** Use NumPy + small PyTorch/JAX for autograd only
- **Interpretable where possible:** Visualize learned representations
- **Compare to baselines:** Always show why DL improves on linear
- **Reproducible:** Fixed seeds, versioned code, documented hyperparameters

### Production Readiness
- **Model export:** ONNX or serialized weights for cross-platform use
- **Inference time:** <100ms per action (suitable for real-time play)
- **Behavior quality:** Strategic, varied, human-understandable moves
- **Fallback modes:** Graceful degradation if network fails

## Deep Learning Systems to Implement

### 3.1: Deep Q-Network (DQN)

**Concept:** Neural network approximates Q(s, a) instead of linear weights. Adds experience replay and target networks for stability.

**Implementation:**

**Network Architecture:**
```
Input: 53-dim state features
↓
Dense(128) + ReLU
↓
Dense(128) + ReLU
↓
Dense(86) [Q-value for each action]
↓
Mask illegal actions → Select argmax
```

**Key Components:**
- **Q-network:** Main network, updated every step
- **Target network:** Frozen copy, updated every N steps (stabilizes targets)
- **Replay buffer:** Store (s, a, r, s') tuples, sample minibatches
- **Loss:** `(r + γ × max_a' Q_target(s', a') - Q(s, a))²`

**Training Process:**
1. Play game using ε-greedy over Q-network
2. Store transitions in replay buffer (capacity: 10K-50K)
3. Sample minibatch (32-64 transitions)
4. Compute TD target using target network
5. Backprop to update Q-network
6. Every 1000 steps: copy Q-network → target network
7. Decay ε over training

**Hyperparameters:**
- Learning rate: 0.001 (Adam optimizer)
- Discount (γ): 0.95
- Replay buffer: 20,000 transitions
- Minibatch: 64
- Target update freq: 1000 steps
- ε decay: 0.1 → 0.01 over 50K games

**Deliverables:**
- `deep_q_agent.py` - DQN implementation
- `replay_buffer.py` - Experience replay
- `dqn_network.py` - Q-network architecture
- `models/DQN-v1.pth` - Trained weights
- Training logs: loss curves, Q-value evolution

**Export Format:**
```json
{
  "agent_type": "deep_q_network",
  "version": "1.0",
  "architecture": {
    "layers": [53, 128, 128, 86],
    "activations": ["relu", "relu", "linear"]
  },
  "training": {
    "games_trained": 100000,
    "learning_rate": 0.001,
    "replay_buffer_size": 20000
  },
  "performance": {
    "vs_heuristic": 0.68,
    "vs_linear_value": 0.75
  },
  "weights": "DQN-v1.pth"
}
```

### 3.2: Actor-Critic (A2C)

**Concept:** Learn both policy π(a|s) and value V(s). Actor proposes actions, critic evaluates them. Lower variance than REINFORCE.

**Implementation:**

**Network Architecture:**
```
Input: 53-dim state features
↓
Shared: Dense(128) + ReLU
↓
        ├─ Actor: Dense(86) + Softmax → π(a|s)
        └─ Critic: Dense(1) → V(s)
```

**Key Components:**
- **Actor:** Policy network π(a|s), outputs action probabilities
- **Critic:** Value network V(s), evaluates state quality
- **Advantage:** `A(s,a) = r + γ×V(s') - V(s)`
- **Actor loss:** `-log π(a|s) × A(s,a)` (policy gradient)
- **Critic loss:** `(r + γ×V(s') - V(s))²` (TD error)

**Training Process:**
1. Play game, collect trajectory: (s₀, a₀, r₁, s₁, ...)
2. Compute advantages using critic V(s)
3. Update actor to increase prob of good actions
4. Update critic to better predict values
5. Use entropy bonus to encourage exploration

**Hyperparameters:**
- Learning rate: 0.0005 (separate for actor/critic)
- Discount (γ): 0.95
- Entropy coefficient: 0.01
- Value loss coefficient: 0.5
- Trajectory length: full game episodes

**Deliverables:**
- `actor_critic_agent.py` - A2C implementation
- `ac_network.py` - Shared backbone + actor/critic heads
- `models/A2C-v1.pth` - Trained weights
- Analysis: policy entropy over training, value accuracy

**Export Format:**
```json
{
  "agent_type": "actor_critic",
  "version": "1.0",
  "architecture": {
    "shared": [53, 128],
    "actor": [128, 86],
    "critic": [128, 1]
  },
  "training": {
    "games_trained": 100000,
    "entropy_coef": 0.01
  },
  "performance": {
    "vs_heuristic": 0.72,
    "policy_entropy": 2.3
  },
  "weights": "A2C-v1.pth"
}
```

### 3.3: AlphaZero-Style Agent (Stretch Goal)

**Concept:** Combine Monte Carlo Tree Search with neural network guidance. Network learns both policy and value from self-play.

**Implementation:**

**Network Architecture:**
```
Input: 53-dim state features
↓
Dense(256) + ReLU + Dense(256) + ReLU
↓
        ├─ Policy head: Dense(86) + Softmax → p(a|s)
        └─ Value head: Dense(1) + Tanh → v(s) ∈ [-1,1]
```

**Key Components:**
- **Neural network:** (p, v) = f(s) predicts both policy and value
- **MCTS:** Use network to guide tree search
  - Selection: pick actions with high p(a|s) + exploration bonus
  - Expansion: add new node to tree
  - Evaluation: use v(s) from network (no rollouts!)
  - Backup: propagate values up tree
- **Self-play:** Generate training data by playing using MCTS
- **Training:** Supervised learning on (s, π, z) tuples
  - π = improved MCTS policy
  - z = actual game outcome

**Training Process:**
1. Generate self-play games using MCTS + network
2. Store (s, π_MCTS, z_outcome) tuples
3. Train network to predict MCTS policy and game outcome
4. Iterate: better network → better MCTS → better data

**Deliverables:**
- `alphazero_agent.py` - Self-play + training loop
- `mcts.py` - MCTS implementation with neural guidance
- `alphazero_network.py` - Policy-value network
- `models/AlphaZero-v1.pth` - Trained weights
- Analysis: ELO progression, MCTS vs pure network play

**Note:** This is a **stretch goal** - implement only if DQN and A2C succeed and time permits.

## Evaluation Framework

### Performance Metrics

**Win Rates (Primary):**
- vs Random: baseline check
- vs Heuristic: main Phase 3 target (60-70%)
- vs MC-Fast: stretch goal
- vs Linear Value (Phase 2): show improvement (target: 60%+)
- vs other Phase 3 agents: cross-comparison

**Learning Curves:**
- Win rate over training games
- Loss/TD error convergence
- Q-value magnitude and variance
- Policy entropy (for actor-critic)

**Efficiency:**
- Games to reach performance milestones
- Wall-clock training time
- Inference time per action (<100ms target)

### Interpretability Analysis

**Feature Importance:**
- Which input features have highest gradient magnitude?
- Saliency maps: which features matter most for decision?
- Ablation: remove features, measure performance drop

**Learned Representations:**
- Visualize hidden layer activations
- t-SNE projection of states in learned space
- Cluster analysis: do similar positions cluster?

**Decision Analysis:**
- Compare DQN Q-values to Linear Value estimates
- Show cases where DL makes better decisions
- Identify strategic patterns DL learned (that linear missed)

### Harness Extensions

Add to `evaluation/harness.py`:
- Model loading from PyTorch/ONNX
- Batch evaluation for efficiency
- Detailed logging: Q-values, action probabilities
- Ablation testing support

## Training Infrastructure

### Curriculum Learning

**Stage 1: Foundation (0-20K games)**
- Train vs 70% Heuristic, 30% Random
- High exploration (ε=0.3)
- Goal: Learn basic tactics

**Stage 2: Refinement (20K-50K games)**
- Train vs 50% Heuristic, 30% Linear Value, 20% self-play
- Medium exploration (ε=0.1)
- Goal: Improve vs learned agents

**Stage 3: Mastery (50K-100K games)**
- Train vs 60% self-play, 20% Heuristic, 20% Linear Value
- Low exploration (ε=0.01)
- Goal: Discover advanced strategies

### Checkpointing Strategy

**Automatic checkpoints:**
- Every 10K games
- On new performance records
- On training milestones (loss < threshold)

**Checkpoint includes:**
- Network weights
- Optimizer state
- Replay buffer snapshot (for DQN)
- Training statistics
- Evaluation metrics

### Distributed Training (Optional)

**Single-machine parallelism:**
- Multiple self-play workers (CPU)
- Single learner (GPU if available)
- Shared replay buffer

**Stretch: Multi-machine**
- Implement if training >100K games needed
- Use simple parameter server architecture

## Data Management

### Training Data

**Replay Buffer:**
- Store in `data/replay_buffers/` as checkpointed arrays
- Circular buffer: oldest transitions evicted
- Serialize periodically for analysis

**Self-play Games:**
- Store selected games in `data/selfplay/` as replays
- Keep milestone games (new records, interesting positions)
- Version: `selfplay_dqn_v1_50K.jsonl`

### Learned Models

**Storage:**
- `models/phase3/` with subdirs per agent type
- Format: PyTorch `.pth` or ONNX `.onnx`
- Each model includes metadata JSON

**Version Control:**
- Git LFS for model files >10MB
- Tag releases: `v3.1-dqn-baseline`, `v3.2-a2c-final`
- Track lineage: which checkpoint trained from which

### Export Standards

**PyTorch Model:**
```python
{
    'model_state_dict': network.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'metadata': {
        'agent_type': 'dqn',
        'games_trained': 100000,
        'performance': {...},
        'hyperparameters': {...}
    }
}
```

**ONNX Export:**
- Standard ONNX format for cross-platform inference
- Include preprocessing/postprocessing in graph
- Test import in non-Python environment

## Documentation

### Structure: `docs/phase3/`

**Foundation:**
- `01_why_deep_learning.md` - When and why to use neural networks
  - Compare linear vs non-linear function approximation
  - Phase 2 limitations that DL addresses
- `02_pytorch_minimal.md` - Using PyTorch for learning, not magic
  - Autograd basics
  - Building networks from scratch understanding
- `03_training_stability.md` - Making DL work reliably
  - Experience replay, target networks, batch norm
  - Common failure modes and fixes

**DQN:**
- `04_dqn_algorithm.md` - Deep Q-Learning fundamentals
  - Videos: DQN paper explained, experience replay
  - Reading: Original DQN paper (Mnih et al. 2015)
- `05_dqn_implementation.md` - Our DQN agent
  - Architecture choices, hyperparameter tuning
  - Training curves and analysis
- `06_dqn_results.md` - Performance and learned behavior
  - vs baselines, interpretability analysis

**Actor-Critic:**
- `07_policy_gradients_deep.md` - From REINFORCE to A2C
  - Videos: Policy gradient theorem, advantage functions
  - Reading: A3C paper, advantage actor-critic
- `08_a2c_implementation.md` - Our A2C agent
  - Shared representations, actor-critic tradeoffs
- `09_a2c_results.md` - Performance comparison to DQN

**Advanced (if implemented):**
- `10_alphazero_concept.md` - MCTS + neural networks
  - Videos: AlphaGo documentary, MCTS explained
  - Reading: AlphaZero paper
- `11_alphazero_implementation.md` - Self-play training loop
- `12_alphazero_results.md` - Comparison to pure RL

**Analysis:**
- `13_linear_vs_deep.md` - When does deep learning help?
  - Feature learning: what DL discovered vs hand-crafted
  - Performance gaps and scaling laws
- `14_interpretability.md` - Understanding learned networks
  - Saliency maps, activation analysis
  - Decision case studies
- `15_phase3_results.md` - Final tournament and conclusions

**Integration:**
- `16_deployment_guide.md` - Using Phase 3 agents in games
  - ONNX export, inference optimization
  - Fallback strategies, error handling

### Video Resources (each doc)

Curate 2-4 short videos (5-15 min) per topic:
- Conceptual explanations (3Blue1Brown style)
- Implementation walkthroughs
- Paper summaries (Two Minute Papers, Yannic Kilcher)
- Practical tutorials (Andrej Karpathy, StatQuest)

## Implementation Phases

### Phase 3.1: DQN Foundation (Week 1-3)

**Week 1: Setup**
- [ ] Set up PyTorch/JAX minimal environment
- [ ] Implement neural network Q-function
- [ ] Build experience replay buffer
- [ ] Create target network infrastructure
- [ ] **Docs:** Write `01_why_deep_learning.md`, `02_pytorch_minimal.md`

**Week 2: Training**
- [ ] Implement DQN training loop
- [ ] Add curriculum learning (staged opponents)
- [ ] Build checkpointing system
- [ ] Initial training run (20K games)
- [ ] **Docs:** Write `04_dqn_algorithm.md` with readings/videos

**Week 3: Evaluation**
- [ ] Full training (100K games)
- [ ] Comprehensive evaluation vs all baselines
- [ ] Analyze learned Q-values and decisions
- [ ] Hyperparameter tuning (learning rate, replay size)
- [ ] **Docs:** Write `05_dqn_implementation.md`, `06_dqn_results.md`

### Phase 3.2: Actor-Critic (Week 4-5)

**Week 4: Implementation**
- [ ] Design shared-backbone architecture
- [ ] Implement actor and critic heads
- [ ] Build trajectory collection system
- [ ] Add entropy regularization
- [ ] **Docs:** Write `07_policy_gradients_deep.md`

**Week 5: Training & Analysis**
- [ ] Train A2C (100K games)
- [ ] Compare to DQN: sample efficiency, final performance
- [ ] Analyze policy entropy and value accuracy
- [ ] Study actor-critic interaction
- [ ] **Docs:** Write `08_a2c_implementation.md`, `09_a2c_results.md`

### Phase 3.3: Analysis & Documentation (Week 6-7)

**Week 6: Interpretability**
- [ ] Implement saliency map visualization
- [ ] Analyze learned features (PCA, t-SNE)
- [ ] Compare DL decisions to Linear Value
- [ ] Identify discovered tactical patterns
- [ ] **Docs:** Write `13_linear_vs_deep.md`, `14_interpretability.md`

**Week 7: Finalization**
- [ ] Run final tournament (all agents)
- [ ] Export models to ONNX
- [ ] Write deployment guide
- [ ] Create Phase 3 summary report
- [ ] **Docs:** Write `15_phase3_results.md`, `16_deployment_guide.md`

### Phase 3.4: AlphaZero (Stretch - Week 8-10)

Only if DQN/A2C succeed and time permits:
- [ ] Implement MCTS with neural guidance
- [ ] Build self-play generation pipeline
- [ ] Train policy-value network
- [ ] Compare MCTS+NN to pure NN
- [ ] **Docs:** Write `10-12_alphazero_*.md`

## Success Criteria

### Minimum (Phase 3 passes if achieved)
- ✅ At least one deep RL agent beats Linear Value (target: 60% vs Linear)
- ✅ DL agent reaches 60%+ win rate vs Heuristic (Phase 2 was 42%)
- ✅ Training is reproducible and documented
- ✅ Models export to portable format (ONNX)

### Target (Strong Phase 3 completion)
- ✅ Both DQN and A2C implemented and trained
- ✅ DL agents reach 65-70% vs Heuristic
- ✅ Clear analysis of what DL learned vs Linear
- ✅ Interpretability tools working (saliency, feature vis)
- ✅ Training completes in <48 hours on single machine

### Stretch (Exceptional Phase 3)
- ✅ AlphaZero-style agent implemented
- ✅ DL agents reach 70%+ vs Heuristic
- ✅ Discover non-obvious strategies (validated by human experts)
- ✅ Agents demonstrate varied, interesting playstyles
- ✅ Publication-quality analysis and documentation

## Comparison to Phase 2

**Phase 2 (Linear Value):**
- 42% vs Heuristic baseline
- 43-dimensional hand-crafted features
- Linear function: Q(s,a) = w^T × φ(s,a)
- Fast training (5K games, <1 min)
- Highly interpretable weights
- **Limitation:** Cannot learn feature interactions

**Phase 3 (Deep RL) - Expected:**
- 60-70% vs Heuristic baseline (+18-28 pp)
- Learn features automatically from state
- Non-linear function approximation
- Slower training (100K games, hours-days)
- Less interpretable (but tools help)
- **Advantage:** Discovers subtle tactical patterns

## Risks & Mitigations

**Risk:** Training instability (loss divergence, Q-value explosion)
**Mitigation:**
- Gradient clipping
- Target networks (DQN)
- Hyperparameter sweep
- Start with small networks

**Risk:** Overfitting to training opponents
**Mitigation:**
- Diverse opponent pool in curriculum
- Self-play in later stages
- Evaluate on held-out baselines

**Risk:** Sample inefficiency (need 1M+ games)
**Mitigation:**
- Start with DQN (most sample-efficient)
- Use experience replay aggressively
- Curriculum learning from simpler to harder

**Risk:** Learned behavior is opaque/uninteresting
**Mitigation:**
- Interpretability analysis from day 1
- Compare every decision to Linear Value
- Visualize what network "sees"

**Risk:** Phase 3 doesn't improve on Phase 2
**Mitigation:**
- This is a valid research finding!
- Document why (features, game complexity, etc.)
- Linear Value remains a great result

## Why Deep Learning Might Help

**Hypothesis:** DL can discover tactical patterns Linear Value cannot

**Examples of non-linear patterns:**
- **Feature interactions:** "High power rocketman in center" worth more than sum of parts
- **Positional motifs:** Diagonal formations, pinning patterns
- **Tempo reads:** Recognizing when to rush vs consolidate (state sequence patterns)
- **Opponent modeling:** Adapting to opponent's weapon usage tendencies

**Validation:**
- Compare DQN Q-values to Linear Value estimates
- Find positions where they disagree → analyze why
- Check if DQN's choice wins more often

## Next Steps

1. **Review and approve this plan**
2. **Set up Phase 3 infrastructure:**
   - `src/utala/deep_learning/` - DQN, A2C implementations
   - `data/replay_buffers/` - Experience replay storage
   - `data/selfplay/` - Self-play game logs
   - `models/phase3/` - Trained networks
   - `docs/phase3/` - Documentation
3. **Install minimal dependencies:**
   - PyTorch (or JAX) for autograd
   - ONNX for export
   - Tensorboard for training visualization
4. **Begin Phase 3.1: DQN implementation**
5. **Document each component with readings + videos**
6. **Checkpoint after DQN before proceeding to A2C**

---

**Ready to begin Phase 3?** This plan builds systematically on Phase 2's success while addressing its limitations through deep learning.
