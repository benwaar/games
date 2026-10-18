# State Feature Extraction

## What We Did

Implemented a fixed-dimensional feature extraction system that converts game states into numeric vectors suitable for machine learning.

**File:** `src/utala/learning/features.py`

**Key design:**
- 53-dimensional feature vector
- All values normalized to [0, 1] range
- Features extracted from player's perspective
- Shared across all learning agents (consistency)

**Feature groups:**

1. **Grid occupancy (27 features):** One-hot encoding for each of 9 squares: [empty, P1, P2]
2. **Resource counts (6 features):** Rocketmen, weapons, kaos cards for both players
3. **Material balance (3 features):** Relative advantage in each resource type
4. **Grid control (6 features):** Controlled squares, contested, threats
5. **Phase indicator (2 features):** Placement vs dogfights
6. **Position quality (9 features):** Strategic value of center/edges/corners

## What It Means

**Why fixed features matter:**

Machine learning agents need consistent input representation. By fixing the feature space:
- All agents see the same information in the same way
- Models are portable and comparable
- We can interpret what agents have learned by inspecting which features they weight heavily

**The tradeoff:**

Hand-crafted features vs raw state:
- **Hand-crafted** (our approach): Fast, interpretable, requires domain knowledge
- **Raw state**: More general, requires neural networks, harder to interpret

For Phase 2, we prioritize interpretability and simplicity. Hand-crafted features let us understand *why* agents make decisions.

**Feature engineering principles:**

1. **Normalization:** [0, 1] range prevents features with large values from dominating
2. **Relative encoding:** Balance features (my advantage - opponent advantage) capture relationships
3. **Strategic relevance:** Center square control matters more than corners → separate features
4. **Symmetry:** Features work from either player's perspective

## Further Reading

**Feature engineering for games:**
- [Feature Engineering for RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#states-and-observations) - OpenAI Spinning Up
- [State Representation in Game AI](https://arxiv.org/abs/1708.05866) - DeepMind's AlphaZero (comparison: they use raw state + neural nets)

**Vector representation:**
- [One-hot encoding](https://en.wikipedia.org/wiki/One-hot) - Wikipedia
- [Feature scaling and normalization](https://scikit-learn.org/stable/modules/preprocessing.html) - scikit-learn docs

**Game-specific features:**
- [Evaluation Functions in Game AI](https://www.chessprogramming.org/Evaluation) - Chess programming wiki (similar concept)

## Video Resources

**Feature engineering basics:**
- [Feature Engineering for Machine Learning](https://www.youtube.com/watch?v=6WDFfaYtN6s) (12 min) - Clear explanation of feature design
- [One-Hot Encoding Explained](https://www.youtube.com/watch?v=v_4KWmkwmsU) (8 min) - Visual guide to categorical encoding

**RL state representation:**
- [State Spaces in Reinforcement Learning](https://www.youtube.com/watch?v=TCCjZe0y4Qc) (15 min) - DeepMind x UCL lecture excerpt
- [Feature Design for Game AI](https://www.youtube.com/watch?v=kopoLzvh5jY) (10 min) - Practical game AI features

## Code Example

```python
from utala.state import GameState, Player
from utala.learning.features import get_feature_extractor

# Initialize
extractor = get_feature_extractor()
state = GameState()

# Extract features from Player ONE's perspective
features = extractor.extract(state, Player.ONE)

print(f"Feature vector shape: {features.shape}")  # (53,)
print(f"Feature range: [{features.min()}, {features.max()}]")  # [0.0, 1.0]

# Get human-readable feature names
names = extractor.feature_names()
for i, (name, value) in enumerate(zip(names[:10], features[:10])):
    print(f"{i}: {name} = {value:.3f}")
```

## Design Decisions

**Why 53 dimensions?**

Balance between information and efficiency:
- Too few: miss strategic nuances
- Too many: slower learning, overfitting risk

53 captures key game elements without redundancy.

**Why normalize to [0, 1]?**

Prevents scale dominance. Without normalization:
- Turn number (0-18) would dominate
- Binary features (0 or 1) would be ignored

Normalization puts all features on equal footing.

**Why player-perspective encoding?**

Agents learn relative strategies ("when I'm ahead, do X") not absolute positions ("when P1 is ahead, do X"). This makes learned models more general.
