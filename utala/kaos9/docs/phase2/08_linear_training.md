# Linear Value Agent Training Results

## Training Configuration

```python
Training Games:     5,000
Learning Rate (α):  0.01
Discount (γ):       0.95
Exploration (ε):    0.10
Training Time:      0.8 minutes
```

**Opponent Mix:**
- 70% vs Heuristic
- 30% vs Random

**Evaluation:**
- Every 500 games
- 100 games per opponent (balanced P1/P2)

## Results

### Final Performance

**vs Random:**
- Win rate: 50% (100-11-89 in 200 games)
- Analysis: Breaks even with random - should be higher

**vs Heuristic:**
- Win rate: 42% (84-4-112 in 200 games)
- Analysis: Better than k-NN (33%) but below target (55%)

### Learning Progression

| Games | vs Random | vs Heuristic | Avg Weight |
|-------|-----------|--------------|------------|
| 0     | 53%       | 49%          | 0.0096     |
| 500   | 50%       | 42%          | 0.0458     |
| 1000  | 55%       | 30%          | 0.0535     |
| 1500  | 50%       | 35%          | 0.0596     |
| 2000  | 44%       | **52%**      | 0.0678     |
| 2500  | 59%       | 36%          | 0.0725     |
| 3000  | 50%       | 43%          | 0.0643     |
| 3500  | 53%       | 35%          | 0.0698     |
| 4000  | 51%       | **53%**      | 0.0722     |
| 4500  | 49%       | 48%          | 0.0702     |
| 5000  | 51%       | 30%          | 0.0723     |

**Observations:**
- High variance (30-53% against Heuristic)
- Peaked at games 2000 and 4000 (52-53%)
- Unstable convergence
- Weight magnitude grows steadily

### Learned Weights

**Top 15 by magnitude:**

| Rank | Feature | Weight | Interpretation |
|------|---------|--------|----------------|
| 1 | material_advantage | +0.31 | Piece advantage is critical |
| 2 | control_advantage | +0.26 | Territory matters |
| 3 | opp_rocketmen_count | -0.21 | Fear opponent strength |
| 4 | my_squares_controlled | +0.21 | Value territory |
| 5 | bias | +0.17 | Slight optimism |
| 6 | turn_normalized | +0.13 | Late-game value |
| 7 | dogfight_uses_flare | +0.12 | Flares are good |
| 8 | phase_dogfight | +0.12 | Dogfights valuable |
| 9 | strong_move_when_winning | +0.09 | Maintain lead |
| 10 | defensive_move_when_losing | +0.09 | Recover when behind |
| 11 | my_rocketmen_count | +0.08 | Keep pieces |
| 12 | opp_squares_controlled | -0.08 | Prevent opponent control |
| 13 | contested_squares | +0.08 | Contesting is okay |
| 14 | phase_placement | +0.06 | Placement phase value |
| 15 | dogfight_kaos_cards_remaining | -0.06 | Using Kaos is fine |

**Strategic lessons learned:**
1. **Material and control dominate** - Top 4 features are about advantage
2. **Relative > Absolute** - Advantages matter more than raw counts
3. **Defensive play valued** - Agent learned balanced strategy
4. **Flare preference** - Learned flares are useful

## Comparison to k-NN

| Metric | k-NN | Linear Value | Winner |
|--------|------|--------------|--------|
| vs Random | 58% | 50% | k-NN |
| vs Heuristic | 33% | **42%** | **Linear** |
| vs MC-Fast | 24% | (not tested) | - |
| Training time | Instant | 0.8 min | k-NN |
| Model size | 31 MB | 128 bytes | **Linear** |
| Interpretability | Similar states | **Weights** | **Linear** |
| Generalization | Local | **Global** | **Linear** |

**Overall:** Linear Value wins on key metrics despite lower performance vs Random.

## Training Analysis

### Weight Evolution

**Early training (0-1000 games):**
- Weights grow from ~0.01 to ~0.05
- Learning basic correlations
- Performance drops slightly (exploration)

**Mid training (1000-3000 games):**
- Weights stabilize around 0.06-0.07
- Best performance at game 2000 (52%)
- Some oscillation

**Late training (3000-5000 games):**
- Weights continue growing to ~0.07
- Performance varies 30-53%
- No clear convergence

**Issue:** High variance suggests:
- Need more games
- Learning rate may be too high
- Feature quality limits performance

### Update Statistics

```
Total weight updates: 72,947
Avg per game: 14.6 updates
Games trained: 5,000
```

**Per-game updates:** ~15 decisions per game, all updated once

### Convergence Issues

**Problem:** Performance oscillates, doesn't stabilize

**Possible causes:**
1. **Online learning variance** - Each game updates weights differently
2. **Opponent distribution** - 70/30 mix creates conflicting gradients
3. **No experience replay** - Temporal correlation in updates
4. **Small eval sample** - 100 games has ~10% variance

**Solutions to try:**
- Experience replay buffer
- Target network for stability
- Larger evaluation sets
- Learning rate decay

## What Worked

✅ **Better than k-NN** - 42% vs 33% against Heuristic
✅ **Fast training** - 5000 games in < 1 minute
✅ **Learned meaningful strategy** - Material/control advantages
✅ **Interpretable** - Can see what it learned
✅ **Small model** - 32 weights vs 32K examples

## What Didn't Work

❌ **Doesn't beat baseline** - 42% < 55% target
❌ **Unstable convergence** - High variance in performance
❌ **Weak vs Random** - Only 50% (should be higher)
❌ **Limited by linear model** - Can't learn complex patterns

## Insights

**Why linear model struggles:**

1. **Non-linear game** - Optimal strategy isn't linear combination of features
2. **Feature limitations** - Missing tactical patterns (lines, forks)
3. **Action space complexity** - 86 actions, many subtle differences
4. **Opponent adaptation** - Heuristic has hand-coded strategy

**Why better than k-NN:**

1. **Generalization** - Learns global patterns, not local similarity
2. **Efficiency** - Fast inference, small model
3. **Interpretability** - Clear strategic priorities

## Next Steps

### Short-term Improvements

1. **More training** - 10K-50K games
2. **Better features** - Line detection, threats
3. **Hyperparameter tuning** - Grid search α, γ, ε
4. **Experience replay** - Decorrelate updates
5. **Target network** - Stabilize learning

### Alternative Approaches

1. **Non-linear model** - Neural network (Phase 2.4)
2. **Ensemble** - Combine Linear + k-NN
3. **Curriculum learning** - Train vs progressively stronger opponents
4. **Opponent modeling** - Adapt to opponent style

### Evaluation Improvements

1. **More games** - 500-1000 per evaluation
2. **Cross-validation** - Multiple random seeds
3. **Detailed metrics** - Weapon usage, comebacks, etc.

## Lessons for Phase 2.5

**If improving Linear Value:**
- Focus on feature quality over quantity
- Implement line detection
- Add tactical pattern recognition
- Try non-linear combinations

**If trying neural network:**
- Can learn features automatically
- But need more training data
- And more complex to interpret

## Further Reading

- [Deep Q-Networks (DQN)](https://www.nature.com/articles/nature14236) - Neural network extension
- [Experience Replay](https://arxiv.org/abs/1712.01275) - Stabilizing online learning
- [Target Networks](https://arxiv.org/abs/1509.06461) - Reducing variance

## Video Resources

- [Reinforcement Learning in Practice](https://www.youtube.com/watch?v=JgvyzIkgxF0) (25 min)
- [Debugging RL Algorithms](https://www.youtube.com/watch?v=eeJ1-bUnwRI) (30 min)
