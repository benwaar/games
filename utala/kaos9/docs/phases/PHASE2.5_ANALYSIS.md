# Phase 2.5: Linear Value Agent V2 - Post-Mortem Analysis

**Date:** 2025-03-24
**Status:** FAILED - Agent regressed from 42.5% to 25.5% vs Heuristic

## Objective

Improve Linear Value agent from 42% to 55%+ win rate vs Heuristic baseline by:
- Enhanced features (49 features, up from 43)
- Proper action decoding
- Line detection and tactical patterns
- 20K training games (up from 5K)

## Results

| Metric | Target | Achieved | Gap |
|--------|--------|----------|-----|
| Win rate vs Heuristic | 55%+ | 25.5% | -29.5 pp |
| Initial win rate | N/A | 42.5% | N/A |
| Final vs Initial | Improve | -17.0 pp | REGRESSION |

**Conclusion:** Complete failure. Agent became significantly worse with enhanced features.

## Training Progression

Win rate vs Heuristic over 20K games:
- 0K: 42.5% (baseline)
- 1K: 22.0% (collapsed immediately)
- 2K-14K: 14%-30% (unstable, fluctuating wildly)
- 15K-20K: 16%-30% (no convergence)

## Root Cause Analysis

### 1. Training Instability
- Performance collapsed after just 1K games (42.5% → 22%)
- Continued to fluctuate wildly throughout training
- No signs of convergence even after 20K games
- Suggests fundamental problems with feature set or learning dynamics

### 2. Enhanced Features May Have Bugs
Created 49 new features including:
- **Line detection** (`_check_line_at_position`, `_check_opponent_threat_at_position`)
- **Action decoding** (`_decode_placement_action`)
- **Tactical patterns** (blocks, forks, threatens)

Potential issues:
- Line detection may not properly account for contested squares
- Action decoding formula may be incorrect for action space layout
- Features may be returning incorrect values (always 0, or wrong signs)

### 3. Feature Redundancy/Conflict
- 49 features vs 43 in v1 (only 6 more)
- Many interaction features that might be redundant
- Features might be poorly scaled or conflicting with each other

### 4. Hyperparameters Not Tuned
Used same hyperparameters as Phase 2.3:
- Learning rate: 0.01
- Discount: 0.95
- Epsilon: 0.1

With more features and potentially different scaling, these may no longer be appropriate.

## Comparison to Phase 2.3

| Agent | Features | Training Games | Final Win Rate |
|-------|----------|----------------|----------------|
| Phase 2.3 Linear Value | 43 (basic) | 5K | 42% |
| Phase 2.5 Linear Value V2 | 49 (enhanced) | 20K | 25.5% |

**Phase 2.3 is better by 16.5 percentage points.**

## Weight Analysis

Final weights after 20K games:
- Weight norm: 0.79
- Weight mean: 0.034
- Weight std: 0.108
- Total updates: 291,483

Weights show learning occurred (norm increased from 0.06 to 0.79), but in the wrong direction.

## Next Steps (If Retrying)

### Debugging Approach
1. **Verify feature correctness**
   - Add unit tests for line detection
   - Verify action decoding matches action space
   - Print sample features for known positions to validate values

2. **Feature ablation study**
   - Start with Phase 2.3 features (43)
   - Add enhanced features one category at a time
   - Identify which features cause regression

3. **Hyperparameter sweep**
   - Try lower learning rates: 0.001, 0.005
   - Try different epsilon values: 0.05, 0.15, 0.2
   - Try different discount factors: 0.9, 0.99

4. **Simpler enhancement**
   - Just fix action decoding in Phase 2.3
   - Just add line detection (2-3 features)
   - Don't add all 6 new features at once

### Alternative Approaches
1. **Revert to Phase 2.3 + more training**
   - Take Phase 2.3 agent
   - Train for 20K games instead of 5K
   - See if more training alone helps

2. **Different learning algorithm**
   - Try eligibility traces (TD(λ))
   - Try different value function architecture
   - Try Q-learning instead of TD learning

3. **Feature engineering workshop**
   - Analyze what Phase 2.3 features work well
   - Design fewer, higher-quality features
   - Use domain knowledge more carefully

## Lessons Learned

1. **More features ≠ better performance** - The 6 additional features made things worse
2. **Test incrementally** - Should have added features one at a time and validated
3. **Feature quality > quantity** - Better to have fewer, well-designed features
4. **Hyperparameters matter** - Need to retune when changing feature set
5. **Early stopping signals** - Performance collapse after 1K games was a red flag

## Bug Fix Attempt (2026-03-24)

After identifying critical bugs in line detection and blocking logic, fixes were applied:

### Bugs Found and Fixed
1. **Line detection bug**: `_check_line_at_position()` checked current board state instead of simulating placement
   - **Fix**: Modified to simulate placing piece at (row, col) before counting lines
   - **Test**: ✓ All tests pass - correctly detects 2-in-a-row and 3-in-a-row after placement

2. **Blocking detection bug**: `_check_opponent_threat_at_position()` had flawed logic for checking blocks
   - **Fix**: Properly simulates blocking opponent's lines
   - **Test**: ✓ Correctly detects blocking winning moves

### Results After Bug Fixes

Training with fixed features (20K games):
- **Initial**: 43% vs Heuristic
- **Final**: 22.5% vs Heuristic
- **Change**: -20.5 pp regression (STILL FAILING)

Performance progression:
- 0K: 43%
- 2K: 12.5% (catastrophic collapse)
- 6K-14K: 15-28% (wild fluctuations)
- 20K: 22.5% (no convergence)

### Conclusion: Bugs Were Not The Root Cause

Even with correct feature implementation, the enhanced 49-feature agent **consistently underperforms** the simpler 43-feature Phase 2.3 agent by 15-20 pp.

The fundamental problem is **feature design**, not bugs:
- Features may be conflicting or redundant
- Feature scaling may be inappropriate
- 49 features may be too many (overfitting)
- Hyperparameters not tuned for larger feature set
- Tactical features may not capture what matters for this game

## Final Recommendation

**ABANDON Phase 2.5 enhanced features.**

After two complete training runs (one buggy, one fixed) with identical failure patterns, the evidence is clear: these enhanced features make the agent worse, not better.

**Phase 2 Completion:**
- Use Phase 2.3 Linear Value agent (42% vs Heuristic) as final Phase 2 deliverable
- Document Phase 2.5 as a failed experiment with valuable lessons
- Move to Phase 3 with simpler, proven approaches

## Lessons Learned

1. **Simpler is often better** - 43-feature agent (42%) beats 49-feature agent (22.5%)
2. **Test features incrementally** - Adding 6 features at once made debugging impossible
3. **Feature quality > quantity** - More features don't help if they're not informative
4. **Bugs can hide deeper problems** - Fixed bugs didn't fix performance
5. **Know when to stop** - Two failures with same pattern = fundamental design flaw
