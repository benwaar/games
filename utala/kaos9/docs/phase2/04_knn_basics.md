# k-Nearest Neighbors Learning (Associative Memory)

## What We Did

Implemented a learning agent that stores game experiences and makes decisions by finding similar past situations.

**File:** `src/utala/agents/associative_memory_agent.py`

**Core concept:**
- Store (state_features, action, outcome) tuples from training games
- At decision time, find k most similar past states
- Weight neighbors by similarity × outcome quality
- Vote on actions, select highest-weighted legal action

**Key components:**
1. **Memory storage:** List of (features, action, outcome) entries
2. **Similarity search:** Euclidean or Manhattan distance
3. **Action voting:** Weighted by inverse distance × outcome
4. **Epsilon-greedy:** Small random exploration (5%)

## What It Means

**k-NN is "instance-based learning"** (also called "memory-based learning"):
- Doesn't build a model
- Just remembers examples
- Makes decisions by analogy: "This situation looks like one where action X worked before"

**Advantages:**
- Simple to understand (very interpretable)
- No training phase (just store examples)
- Can learn from sparse data
- Handles non-linear patterns naturally

**Disadvantages:**
- Slow at decision time (must search all memory)
- Memory grows with training data
- Sensitive to irrelevant features
- Needs good distance metric

**How voting works:**

```
Current state: features = [0.2, 0.5, 0.8, ...]

Find k=20 nearest neighbors:
  Neighbor 1: distance=0.1, action=42, outcome=1.0 (win)
  Neighbor 2: distance=0.2, action=42, outcome=0.5 (draw)
  Neighbor 3: distance=0.15, action=7, outcome=0.0 (loss)
  ...

Calculate vote weights:
  Action 42: (1/(0.1+ε))×1.0 + (1/(0.2+ε))×0.5 = ~10 + 2.5 = 12.5
  Action 7: (1/(0.15+ε))×0.0 = 0
  ...

Select action with highest vote (action 42)
```

## Why k Matters

**k = 1** (only 1 nearest neighbor):
- Very sensitive to noise
- Memorizes training data exactly
- Overfits easily

**k = 100** (many neighbors):
- Smooths out noise
- But may include irrelevant examples
- "Washes out" specific patterns

**k = 10-50** (typical):
- Balance between noise and relevance
- Our initial choice: k=20

## Further Reading

**k-NN fundamentals:**
- [k-Nearest Neighbors Algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) - Wikipedia overview
- [Instance-Based Learning](https://machinelearningmastery.com/k-nearest-neighbors-for-machine-learning/) - Practical guide

**k-NN in games:**
- [Case-Based Reasoning in Games](https://www.aaai.org/Papers/AIIDE/2007/AIIDE07-044.pdf) - Using memory for game AI
- [Memory-Based Learning for Strategy Games](https://arxiv.org/abs/1811.02569) - Academic survey

**Distance metrics:**
- [Euclidean vs Manhattan Distance](https://www.analyticsvidhya.com/blog/2020/02/4-types-of-distance-metrics-in-machine-learning/) - When to use which
- [Feature scaling for k-NN](https://scikit-learn.org/stable/modules/preprocessing.html) - Why normalization matters

**Hyperparameter tuning:**
- [Choosing k in k-NN](https://towardsdatascience.com/how-to-find-the-optimal-value-of-k-in-knn-35d936e554eb) - Methods for selecting k
- [Cross-validation for k-NN](https://scikit-learn.org/stable/modules/cross_validation.html) - Systematic tuning

## Video Resources

**k-NN basics:**
- [k-Nearest Neighbors Explained](https://www.youtube.com/watch?v=HVXime0nQeI) (12 min) - Visual intuitive explanation
- [k-NN from Scratch](https://www.youtube.com/watch?v=ngLyX54e1LU) (15 min) - Implementation walkthrough

**Instance-based learning:**
- [Memory-Based Learning](https://www.youtube.com/watch?v=gR8QvFmNuLE) (10 min) - When to use k-NN
- [Lazy Learning Algorithms](https://www.youtube.com/watch?v=rj8h_bvjU-I) (8 min) - k-NN vs other methods

**Distance metrics:**
- [Euclidean vs Manhattan Distance](https://www.youtube.com/watch?v=vipwj_hPgPQ) (6 min) - Visual comparison
- [Feature Scaling for ML](https://www.youtube.com/watch?v=mnKm3YP56PY) (10 min) - Why it matters for k-NN

## Code Example

```python
from utala.agents.associative_memory_agent import AssociativeMemoryAgent
from utala.learning.training_data import generate_training_dataset

# Create agent
agent = AssociativeMemoryAgent(
    name="MyMemoryAgent",
    k=20,                      # Number of neighbors
    distance_metric="euclidean",  # or "manhattan"
    epsilon=0.05,              # Exploration rate
    seed=42
)

# Generate training data
examples = generate_training_dataset(
    agent_one=heuristic_agent,
    agent_two=random_agent,
    num_games=1000,
    output_path="data/training/my_data.jsonl"
)

# Train (just store examples)
agent.train(examples)

print(f"Memory size: {len(agent.memory)}")
print(f"Stats: {agent.get_memory_stats()}")

# Save trained agent
agent.save(
    agent_name="MyMemoryAgent-v1",
    agent_type="associative_memory",
    version="1.0",
    hyperparameters={'k': 20, 'distance_metric': 'euclidean'},
    performance={'vs_random': 0.58}
)

# Use agent in games
# Agent automatically finds similar states and votes on actions
```

## Design Decisions

**Why Euclidean distance by default?**

Euclidean = straight-line distance in feature space:
- Natural for continuous normalized features
- Captures overall similarity well
- Standard choice for k-NN

Manhattan = sum of absolute differences:
- Sometimes better for high-dimensional spaces
- Less sensitive to outliers
- Worth testing as alternative

**Why weight by outcome?**

Not all examples are equally valuable:
- Winning examples → positive reinforcement
- Losing examples → avoid those actions
- Draw examples → neutral

Outcome weighting: vote = (1/distance) × outcome
- Wins (1.0) get full weight
- Draws (0.5) get half weight
- Losses (0.0) contribute nothing

**Why epsilon-greedy exploration?**

Small random exploration (5%):
- Prevents getting stuck in local patterns
- Discovers new strategies during training
- Can be disabled (epsilon=0) during evaluation

## Common Issues

**Issue: Agent plays too cautiously**
- Cause: Too many losing examples in memory
- Fix: Prune memory, keep only winning examples

**Issue: Agent makes inconsistent decisions**
- Cause: k too large, includes irrelevant neighbors
- Fix: Reduce k to 5-10 for tighter similarity

**Issue: Slow decision time (>100ms)**
- Cause: Linear search through large memory
- Fix: Use approximate nearest neighbors (not implemented yet)
- Fix: Prune memory to smaller size

**Issue: Agent doesn't improve with more data**
- Cause: Feature space not capturing strategic information
- Fix: Review feature engineering (see `01_state_features.md`)
- Fix: Try different distance metric

## Interpretability

k-NN is highly interpretable - you can inspect why decisions were made:

```python
# After agent makes decision, inspect neighbors:
current_features = extractor.extract(state, player)
neighbors = agent._find_k_nearest(current_features, k=5)

print("Top 5 similar situations:")
for i, neighbor in enumerate(neighbors):
    print(f"{i+1}. Distance: {neighbor['distance']:.3f}")
    print(f"   Action: {neighbor['action']}")
    print(f"   Outcome: {neighbor['outcome']} ({'win' if neighbor['outcome']==1.0 else 'loss'})")
    print(f"   Turn: {neighbor['turn']}")
```

This lets you see exactly which past experiences influenced the decision.
