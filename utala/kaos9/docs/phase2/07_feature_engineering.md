# Feature Engineering for Linear Value Learning

## What We Did

Designed 32 state-action features that capture game situation and action properties for the Linear Value agent.

**File:** `src/utala/learning/state_action_features.py`

**Feature categories:**
1. **State features** (10): Game phase, material, board control
2. **Placement action features** (10): Power level, position, tactical value
3. **Dogfight action features** (6): Weapon choice, power dynamics
4. **Interaction features** (5): Context-dependent action quality
5. **Bias term** (1): Baseline value adjustment

Total: **32 dimensions**

## What It Means

### Why State-Action Features?

**Problem:** Q-learning needs Q(s,a) - value of action a in state s.

**Options:**

1. **State features only:** Learn V(s), then pick best action
   - ❌ Doesn't distinguish between actions
   - ❌ Can't learn action-specific patterns

2. **State-action features:** Learn Q(s,a) directly
   - ✅ Each action gets unique evaluation
   - ✅ Can learn "this action is good in this situation"
   - ✅ More expressive

**Our choice:** State-action features for richer learning.

### Feature Design Philosophy

**Goals:**
- Capture strategic game state
- Encode action properties
- Express interactions (action quality given state)
- Keep computation fast (< 1ms per feature extraction)

**Constraints:**
- Must work with linear function (w^T × φ)
- Features in [0, 1] range (normalized)
- No look-ahead (no searching future states)
- Must handle both placement and dogfight phases

## Feature Breakdown

### State Features (10 dimensions)

These describe the current game situation:

```python
1. phase_placement         # 1.0 if placement phase, 0.0 otherwise
2. phase_dogfight          # 1.0 if dogfight phase, 0.0 otherwise
3. turn_normalized         # Turn number / 50 (progress through game)
4. my_rocketmen_count      # My remaining rocketmen / 9
5. opp_rocketmen_count     # Opponent rocketmen / 9
6. material_advantage      # (mine - opponent) / 9, range [-1, 1]
7. my_squares_controlled   # Squares I control / 9
8. opp_squares_controlled  # Squares opponent controls / 9
9. control_advantage       # (mine - opponent) / 9, range [-1, 1]
10. contested_squares      # Contested squares / 9
```

**Design rationale:**
- **Phase indicators:** Different strategies for placement vs dogfights
- **Material count:** Core strategic value (pieces = potential)
- **Board control:** Territory is valuable (3-in-a-row wins)
- **Advantages:** Relative position matters more than absolute

**What learned weights tell us:**
- material_advantage: +0.31 → **Most important feature!**
- control_advantage: +0.26 → Board control matters
- my_squares_controlled: +0.21 → Territory valuable
- opp_rocketmen_count: -0.21 → Fear opponent strength

### Placement Action Features (10 dimensions)

These describe properties of placement moves:

```python
11. placement_power_low     # 1.0 if power 2-4
12. placement_power_mid     # 1.0 if power 5-7
13. placement_power_high    # 1.0 if power 8-10
14. placement_center        # 1.0 if position (1,1)
15. placement_edge          # 1.0 if edge (not center/corner)
16. placement_corner        # 1.0 if corner position
17. placement_contests_square  # 1.0 if opponent already here
18. placement_takes_control # 1.0 if empty square
19. placement_forms_line_2  # 1.0 if creates 2-in-a-row (TODO)
20. placement_forms_line_3  # 1.0 if creates 3-in-a-row (TODO)
```

**Design rationale:**
- **Power indicators:** Learn which powers are valuable when
- **Position indicators:** Center/edges/corners have different tactical value
- **Tactical indicators:** Contesting vs controlling vs line-forming

**Current limitations:**
- Power encoded roughly (action index modulo)
- Line detection not implemented (would need board analysis)
- No threat detection (could add "blocks opponent line")

### Dogfight Action Features (6 dimensions)

These describe dogfight weapon choices:

```python
21. dogfight_uses_rocket    # 1.0 if playing rocket
22. dogfight_uses_flare     # 1.0 if playing flare
23. dogfight_power_diff_positive  # My rocketman stronger
24. dogfight_power_diff_negative  # Opponent stronger
25. dogfight_kaos_cards_remaining  # My kaos deck size / 9
26. dogfight_strategic_square      # Center/edge/corner (TODO)
```

**Design rationale:**
- **Weapon choice:** Learn when to rocket vs flare
- **Power dynamics:** Matters for Kaos card strategy
- **Resource tracking:** Kaos cards are finite

**What learned:**
- dogfight_uses_flare: +0.12 → Flares are valuable
- dogfight_kaos_cards_remaining: -0.06 → Using Kaos is okay

**Current limitations:**
- Power differential not properly computed (needs dogfight context)
- Strategic square not encoded (center contests more valuable)

### Interaction Features (5 dimensions)

These combine state + action for context-aware evaluation:

```python
27. strong_move_when_winning    # High material + aggressive action
28. defensive_move_when_losing  # Low material + defensive action
29. contests_with_high_power    # Contesting with strong rocketman
30. early_game_aggression       # Turn < 10 + aggressive
31. late_game_caution           # Turn > 30 + defensive
```

**Design rationale:**
- Learn situational strategies
- "This move is good when winning" vs "when losing"
- Time-dependent tactics (early aggression, late caution)

**What learned:**
- strong_move_when_winning: +0.09 → Keep pressure when ahead
- defensive_move_when_losing: +0.09 → Both equally valued

**Current limitations:**
- "Aggressive" vs "defensive" not well defined yet
- Could add more nuanced interactions

### Bias Term (1 dimension)

```python
32. bias  # Always 1.0
```

**Purpose:** Baseline value (like intercept in linear regression)

**What learned:** bias: +0.17 → Slight optimistic bias

## Feature Engineering Process

**How we designed features:**

1. **Domain knowledge:**
   - What do humans care about? (material, position, tempo)
   - What are the key decisions? (power selection, positioning)

2. **Iterative refinement:**
   - Start with basic features (material, phase)
   - Add tactical features (positioning, contesting)
   - Add interaction features (context-dependent)

3. **Normalization:**
   - All features scaled to [0, 1] or [-1, 1]
   - Prevents feature magnitude from dominating
   - Ensures stable learning

4. **Validation:**
   - Check feature extraction is fast (< 1ms)
   - Verify features capture meaningful information
   - Train and observe which weights grow large

## What Worked Well

✅ **Material features:**
- material_advantage has highest weight (+0.31)
- Simple but highly predictive
- Easy to compute

✅ **Board control:**
- control_advantage learned as important (+0.26)
- Natural strategic concept
- Correlates with winning

✅ **Phase indicators:**
- Different strategies for placement vs dogfight
- Both phases learned positive weights

✅ **Normalization:**
- All features in [0, 1] range
- Stable training, no explosion

## What Didn't Work / Needs Improvement

❌ **Placement power encoding:**
- Used rough action index approximation
- Should decode action properly to get actual power
- Currently: power_high = 1.0 if action >= 54

❌ **Line detection:**
- placement_forms_line_2/3 not implemented
- Would require board analysis logic
- Could be very valuable strategically

❌ **Dogfight power differential:**
- Not computed (needs current dogfight context)
- Would help Kaos card decisions
- Placeholder features set to 0.0

❌ **Interaction features:**
- "Aggressive" vs "defensive" not well defined
- Hard to determine from action index alone
- Need semantic action interpretation

❌ **Opponent modeling:**
- No features tracking opponent behavior
- Could learn "opponent likes to X, so counter with Y"
- Would need game history

## Feature Importance Analysis

From trained weights (absolute magnitude):

| Rank | Feature | Weight | Importance |
|------|---------|--------|-----------|
| 1 | material_advantage | +0.31 | Critical |
| 2 | control_advantage | +0.26 | High |
| 3 | opp_rocketmen_count | -0.21 | High |
| 4 | my_squares_controlled | +0.21 | High |
| 5 | bias | +0.17 | Medium |
| 6 | turn_normalized | +0.13 | Medium |
| 7 | phase_dogfight | +0.12 | Medium |
| 8 | dogfight_uses_flare | +0.12 | Medium |
| ... | (others) | <0.10 | Low |

**Insights:**
- **Advantage features dominate** (material, control)
- **Absolute counts matter less** than relative position
- **Tactical features** (placement position) low weights
- **Maybe too simple?** Linear function prefers simple patterns

## Ideas for Better Features

### Tactical Patterns

```python
# Threat detection
- threatens_line_3          # Placement creates win threat
- blocks_opponent_line      # Defensive placement
- fork_opportunity          # Creates two threats

# Position value
- center_control_value      # Center worth more early game
- corner_control_value      # Corners important for lines
```

### Game Tree Features

```python
# Shallow lookahead (1 ply)
- my_best_next_action_value    # Minimax depth-1
- opponent_threat_level        # Can they win next turn?
```

### Resource Management

```python
# Kaos deck composition
- high_kaos_cards_remaining    # 7-9 cards left
- kaos_card_advantage          # My deck stronger than opponent
```

### Temporal Features

```python
# Game stage
- early_placement    # Turns 1-6
- mid_placement      # Turns 7-12
- early_dogfight     # First few contests
- endgame            # Final contests
```

### Learned Features

```python
# Auto-generate combinations
- material_advantage × control_advantage  # Joint value
- turn_normalized × my_rocketmen_count    # Time-material
```

But these would increase dimensionality - maybe not worth it for linear model.

## Feature Extraction Performance

```python
# Profiling results (1000 extractions):
Total time: 12.3ms
Per extraction: 0.0123ms
Extractions/sec: 81,300
```

**Bottleneck:** Board iteration (counting rocketmen, squares)

**Fast enough?** Yes - even 100 extractions per decision < 2ms

## Comparison to k-NN Features

| Aspect | k-NN (State) | Linear Value (State-Action) |
|--------|--------------|----------------------------|
| **Dimensions** | 53 | 32 |
| **Encodes action** | No | Yes |
| **Phase-specific** | Placement only | Both phases |
| **Computation** | O(53) dot product | O(32) dot product |
| **Expressiveness** | State similarity | Action quality |

**Winner:** State-action features
- More focused (only 32 dims)
- Action-specific evaluation
- Handles both game phases

## Further Reading

**Feature Engineering for RL:**
- [Feature Construction for RL](https://www.microsoft.com/en-us/research/publication/feature-construction-for-reinforcement-learning/) - Research paper
- [Tile Coding](http://incompleteideas.net/book/ebook/node88.html) - Generalization technique
- [Radial Basis Functions](https://gibberblot.github.io/rl-notes/single-agent/function-approximation.html#radial-basis-functions) - Smooth features

**Game-Specific:**
- [Chess Feature Engineering](https://www.chessprogramming.org/Evaluation) - Classic game AI features
- [Go Features](https://arxiv.org/abs/1412.6564) - AlphaGo pattern features

**Automatic Feature Learning:**
- [Deep RL](https://www.deeplearningbook.org/contents/reinforcement_learning.html) - Learn features automatically
- [Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html) - Which features matter?

## Video Resources

- [Feature Engineering Explained](https://www.youtube.com/watch?v=qO-HUo0LsO4) (20 min) - DeepMind
- [Domain Knowledge in RL](https://www.youtube.com/watch?v=PnHCvfgC_ZA&t=3600s) (15 min) - David Silver
- [Tile Coding Tutorial](https://www.youtube.com/watch?v=5e1hLdyxPx8) (12 min) - Visual explanation

## Code Example

```python
from src.utala.learning.state_action_features import StateActionFeatureExtractor

# Create extractor
extractor = StateActionFeatureExtractor()

# Extract features for a state-action pair
features = extractor.extract(state, action=42, player=Player.ONE)

print(f"Feature vector: {features.shape}")  # (32,)
print(f"Feature names: {extractor.get_feature_names()}")

# Explain top features
explanation = extractor.explain_features(features, top_k=10)
print(explanation)
```

Output:
```
Top features:
  material_advantage        :   0.333
  control_advantage         :   0.222
  my_squares_controlled     :   0.556
  phase_placement           :   1.000
  placement_power_high      :   1.000
  placement_center          :   1.000
  ...
```

## Next Steps

**To improve features:**

1. **Decode actions properly:**
   - Parse action index → (position, power) or (weapon_type)
   - Use actual values instead of approximations

2. **Add tactical detection:**
   - Implement line-checking logic
   - Threat and block detection
   - Fork opportunities

3. **Opponent modeling:**
   - Track opponent tendencies
   - Adjust strategy based on opponent style

4. **Automatic feature selection:**
   - Try all possible features
   - Keep only those with large learned weights
   - Reduce dimensionality

5. **Non-linear features:**
   - Polynomial combinations
   - Or move to neural network (see Phase 2.4)
