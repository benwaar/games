# Phase 2 Complete: Learning Agents Implementation

**Date:** 2026-03-24
**Status:** ✅ COMPLETE
**Final Deliverable:** Linear Value Agent (42% vs Heuristic)

---

## Summary

Phase 2 successfully implemented and compared three learning algorithms for utala: kaos 9:

| Agent | Algorithm | Win Rate vs Heuristic | Notes |
|-------|-----------|----------------------|-------|
| **Linear Value** | TD Learning | **42%** | ✅ Phase 2 winner |
| Associative Memory | k-NN | 33% | Memory-based |
| Policy Network | REINFORCE | 18% | High variance |
| Enhanced Features | TD Learning | 22.5% | ❌ Failed (Phase 2.5) |

**Winner:** Linear Value Agent achieves 42% win rate with simple, interpretable TD learning.

---

## Achievements

### ✅ Core Deliverables
- **3 learning algorithms** implemented and evaluated
- **Training pipelines** established (reproducible, documented)
- **Model serialization** working (JSON format, portable)
- **Feature engineering** foundation (state & state-action features)
- **Evaluation framework** comprehensive metrics collection

### 📊 Performance
- All agents demonstrate learning capability
- Linear Value outperforms k-NN by 9 pp (42% vs 33%)
- Linear Value beats Policy Network by 24 pp (42% vs 18%)
- Agents train quickly (< 1 minute on laptop)

### 📚 Documentation
- 9 detailed documentation files in `docs/phase2/`
- Training procedures documented and reproducible
- Feature engineering strategies explained
- Learning algorithm comparisons analyzed

---

## Key Learnings

### What Worked
1. **TD Learning > Policy Gradients** for this problem (42% vs 18%)
2. **Linear function approximation** sufficient (no need for neural networks)
3. **Simple features** effective (53-dim state, 43-dim state-action)
4. **Fast iteration** enabled by Python implementation
5. **Interpretable weights** show learned strategy (material advantage, control)

### What Didn't Work
1. **Enhanced features (Phase 2.5)** made agent worse (22.5% vs 42%)
   - Line detection, tactical patterns, blocking logic
   - Even with bugs fixed, fundamentally flawed design
   - Simpler features consistently outperform complex ones
2. **Policy gradients (REINFORCE)** struggled with high variance
3. **Neural networks** unnecessary for current feature complexity

### Critical Insights
- **Simpler is better** - 43 features beat 49 features
- **Feature quality > quantity** - More features don't help if poorly designed
- **Incremental testing essential** - Adding 6 features at once = debugging nightmare
- **Know when to stop** - Two failures with same pattern = fundamental problem

---

## Technical Specifications

### Linear Value Agent (Final Deliverable)

**Architecture:**
```
Input: 43 state-action features
Model: Linear Q(s,a) = w^T × φ(s,a)
Learning: TD(0) with epsilon-greedy exploration
```

**Hyperparameters:**
- Learning rate (α): 0.01
- Discount (γ): 0.95
- Exploration (ε): 0.1
- Training: 5,000 games
- Opponent mix: 70% Heuristic, 30% Random

**Training Stats:**
- Time: 0.8 minutes
- Weight updates: 72,947
- Model size: ~50KB (JSON)

**Top Learned Weights:**
```
+0.31: material_advantage (more rocketmen = better)
+0.26: control_advantage (more squares = better)
+0.21: my_squares_controlled
-0.21: opp_rocketmen_count (opponent rocketmen = bad)
```

---

## Files & Artifacts

### Code
- `src/utala/agents/linear_value_agent.py` - Final agent implementation
- `src/utala/learning/state_action_features.py` - Feature extraction (43-dim)
- `train_linear_agent.py` - Training script
- `eval_knn_agent.py` - Evaluation harness

### Models (gitignored)
- `models/LinearValue-v1_final.json` - Trained agent (50KB)
- `models/LinearValue-v1_5000.json` - Checkpoint

### Results
- `results/linear_value/` - Training history and evaluation
- `results/linear_v2_BUGGY/` - Phase 2.5 failed attempt (archived)

### Documentation
- `PHASE2_STATUS.md` - Complete phase status
- `PHASE2.5_ANALYSIS.md` - Failed enhancement post-mortem
- `docs/phase2/*.md` - 9 detailed documents

---

## Phase 2.5 Post-Mortem

**Attempted:** Enhanced feature extraction to reach 55%+ win rate
**Result:** FAILED - Agent regressed to 22.5% (20 pp worse)

**Why it failed:**
1. Feature bugs initially (line detection, blocking logic)
2. **Bugs fixed, still failed** → fundamental design problem
3. Features conflicted, redundant, or poorly scaled
4. 49 features too many (overfitting)
5. Tactical features didn't capture game strategy

**Lesson:** Don't assume more features = better performance. The simpler 43-feature agent is objectively better.

---

## Comparison to Goals

**Original Phase 2 Goals:**
- ✅ Implement multiple learning algorithms
- ✅ Demonstrate agents can learn from self-play
- ✅ Create portable, reproducible models
- ⚠️  Beat Heuristic baseline (target: 55%, achieved: 42%)

**Assessment:**
- **Phase 2 successful** despite not reaching 55% target
- Learning infrastructure proven and working
- Clear path forward for Phase 3
- Valuable lessons about feature engineering

---

## Next Steps (Phase 3)

**Foundation:** Build on Linear Value agent (42% baseline)

**Promising approaches:**
1. **Better training strategy**
   - Curriculum learning (vs progressively stronger opponents)
   - Self-play + replay buffer
   - More training games (50K+)

2. **Improved algorithms**
   - Deep Q-Networks (DQN) with experience replay
   - Actor-Critic methods
   - Monte Carlo Tree Search integration

3. **Smarter features**
   - Domain-specific heuristics
   - Opponent modeling
   - Position evaluation functions

**Target:** 55%+ win rate vs Heuristic

---

## Conclusion

Phase 2 delivered a working learning agent (42% vs Heuristic) with:
- Clean, interpretable implementation
- Fast training (< 1 minute)
- Portable models (JSON)
- Comprehensive documentation

While the 55% target wasn't reached, Phase 2 established critical infrastructure and validated the learning approach. The Linear Value agent provides a solid foundation for Phase 3 improvements.

**Phase 2: Complete ✅**
