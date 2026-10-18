# k-NN Agent Training Results

## Training Configuration

**Date:** 2026-03-23

**Training data generation:**
- 1,000 games (500 Heuristic vs Random, 500 Random vs Heuristic)
- Starting seed: 500000
- Total examples: 32,445 decision points

**Agent configuration:**
- k = 20 neighbors
- Distance metric: Euclidean
- Epsilon: 0.05 (5% exploration)
- Memory size: 32,445 entries

**Training script:** `train_knn_agent.py`

## Training Data Statistics

**Memory composition:**
- Win examples: 47.1% (15,282 examples)
- Draw examples: 5.2% (1,687 examples)
- Loss examples: 47.7% (15,476 examples)

**Observation:** Balanced distribution between wins and losses is good - agent sees both successful and unsuccessful patterns.

## Evaluation Results

### Performance vs Baselines (100 games each, balanced)

**vs Random:**
- Win rate: **58.0%** (58-37-5)
- Result: ✅ **SUCCESS** - Clearly beats random baseline
- Analysis: Agent learned meaningful patterns

**vs Heuristic:**
- Win rate: **33.0%** (33-61-6)
- Result: ⚠️ **BELOW BASELINE** - Loses to Phase 1 heuristic
- Analysis: Not yet at target performance (need 55%+ to beat baseline)

**vs MC-Fast:**
- Win rate: **24.0%** (24-71-5)
- Result: ❌ **EXPECTED** - MC-Fast is strongest Phase 1 agent
- Analysis: k-NN struggles against lookahead search

### Progress Over Games (vs Heuristic)

Early games showed higher win rate (~35%), but regressed to 33% by game 100. This suggests:
- No clear learning trend (would need longer evaluation)
- Agent performance is stable but below target
- Not overfitting (stable performance)

## What Worked

✅ **Infrastructure validation:**
- Feature extraction produces meaningful state representations
- Training data pipeline works end-to-end
- Model serialization successful (31MB JSON)
- Evaluation harness runs cleanly

✅ **Basic learning confirmed:**
- Agent beats Random (58% >> 50% baseline)
- Makes legal moves consistently
- Decision time reasonable (~50ms average)

✅ **Interpretability:**
- Can inspect which past states influenced decisions
- Memory stats reveal training data composition
- Easy to understand agent behavior

## What Didn't Work

❌ **Below baseline performance:**
- 33% vs Heuristic (target: 55%+)
- Training data quality may be insufficient
- k=20 might be suboptimal

❌ **Training data source:**
- Mixed Heuristic/Random games → mixed quality
- Agent learned both good and bad patterns
- No filtering of low-quality examples

❌ **No hyperparameter tuning:**
- Didn't test different k values
- Didn't try Manhattan distance
- Didn't prune memory for quality

## Lessons Learned

### 1. Training Data Quality Matters More Than Quantity

**Observation:** 32K examples from mixed-skill games didn't produce strong agent.

**Hypothesis:** Better to have fewer high-quality examples than many mixed examples.

**Recommendation for future:**
- Generate training data from Heuristic vs Heuristic (uniform skill)
- Or filter examples: keep only winning player's decisions
- Or weight examples by game outcome

### 2. k-NN May Not Suit This Game

**Challenge:** Game state space is large relative to memory size.
- 53 dimensional feature space
- Many states won't have close neighbors
- k=20 may aggregate too many dissimilar states

**Alternative interpretation:** Linear value functions might generalize better than instance-based learning for tactical games.

### 3. Feature Engineering Is Critical

**Current features:** Grid occupancy, resource counts, material balance, position quality
- These capture game state reasonably
- But might miss strategic nuances (e.g., threat sequences, weapon timing)

**Evidence:** Agent makes legal moves but doesn't play strategically at heuristic level.

**Future work:** Could add features for:
- Kaos deck tracking (visible discard information)
- Weapon efficiency (when to spend vs save)
- Board control patterns (not just counts)

### 4. Memory-Based Learning Has Limitations

**Pros:**
- Simple, interpretable
- Works with small datasets
- No training phase

**Cons:**
- Slow at scale (linear search)
- Doesn't generalize patterns
- Sensitive to feature space quality

**Conclusion:** k-NN was good for validating infrastructure, but may not be best approach for this game.

## Comparison to Baselines

### Win Rate Summary

| Matchup | k-NN | Random | Heuristic | MC-Fast |
|---------|------|--------|-----------|---------|
| vs Random | **58%** | 50% | 78% | 92% |
| vs Heuristic | 33% | 22% | 50% | 65% |
| vs MC-Fast | 24% | 8% | 35% | 50% |

**Analysis:**
- k-NN sits between Random and Heuristic in skill
- Large gap to MC-Fast (lookahead is powerful)
- Room for improvement to reach Phase 2 goals

### Decision Time Comparison

| Agent | Avg Decision Time |
|-------|-------------------|
| Random | <1ms |
| Heuristic | ~2-5ms |
| k-NN (k=20) | ~50ms |
| MC-Fast | ~500-2000ms |

**Analysis:**
- k-NN is fast enough for real-time play (<100ms target)
- Significantly faster than MC-Fast
- Could be optimized further with approximate search

## Future Improvements

If revisiting k-NN agent, try:

### 1. Hyperparameter Tuning
- Test k ∈ {5, 10, 20, 50, 100}
- Try Manhattan distance metric
- Experiment with different feature weights

### 2. Memory Pruning
- Keep only winning examples: outcome = 1.0
- Prune similar states: diversity sampling
- Limit memory size: 10K best examples

### 3. Better Training Data
- Generate from Heuristic vs Heuristic (5000 games)
- Or include MC-Fast games for higher quality
- Weight examples by final margin of victory

### 4. Feature Engineering
- Add Kaos deck tracking features
- Add threat detection features
- Remove redundant features

### 5. Ensemble Methods
- Combine k-NN with heuristic evaluation
- Use k-NN for placement, heuristic for dogfights
- Weighted voting between multiple k values

## Recommendations

**For this research project:**
→ **Move to Linear Value Agent**
- k-NN validated infrastructure
- Linear methods may generalize better
- Can compare approaches at end of Phase 2

**If deploying k-NN to production:**
→ **Significant tuning required**
- Current performance (33% vs Heuristic) not production-ready
- Would need extensive hyperparameter search
- Consider hybrid approach (k-NN + rules)

## Code Artifacts

**Saved model:**
- `models/AssociativeMemory-v1_20260323_173057.json` (31MB)
- Contains full 32,445 example memory
- Can be loaded and used in games

**Training data:**
- `data/training/knn_training_data.jsonl` (14MB)
- 32,445 examples from 1,000 games
- Reusable for other learning methods

**Scripts:**
- `train_knn_agent.py` - Generate data and train
- `eval_knn_agent.py` - Comprehensive evaluation

## Conclusion

The k-NN agent **successfully demonstrated learning** (beats Random) but **doesn't yet meet Phase 2 baseline goals** (beat Heuristic).

**Key takeaway:** Instance-based learning works but may not be optimal for this game. Moving forward with Linear Value agent to explore alternative learning approach.

**Research value:** This experiment validated our infrastructure and provided baseline performance for comparison. The 58% vs Random establishes a "learned baseline" to measure future improvements against.
