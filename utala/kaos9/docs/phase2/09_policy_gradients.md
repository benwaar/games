# Policy Gradient Learning (REINFORCE)

## What We Did

Implemented a Policy Network Agent using REINFORCE - a policy gradient algorithm that learns a neural network to directly output action probabilities.

**File:** `src/utala/agents/policy_network_agent.py`

**Architecture:**
- Input layer: 53 state features
- Hidden layer: 64 units with ReLU activation
- Output layer: 86 action logits with softmax
- Total parameters: ~8,000 weights

**Core concept:**
- Learn policy π(a|s) directly (not value function)
- Sample actions from policy distribution
- Increase probability of actions that led to wins
- Decrease probability of actions that led to losses

## What It Means

### Policy Gradient Methods

Policy gradient is a fundamentally different approach to reinforcement learning:

**Value-based methods (Q-learning, TD):**
- Learn Q(s,a) = value of actions
- Derive policy: π(s) = argmax Q(s,a)
- Indirect: learn values → extract policy

**Policy-based methods (REINFORCE):**
- Learn π(a|s) directly
- No value function needed
- Direct: optimize policy parameters

**Key insight:** Directly optimize what you care about (the policy), not an intermediate representation (values).

### Neural Network from Scratch

We implemented a 2-layer MLP without any ML frameworks:

```python
# Forward pass
z1 = input @ W1 + b1          # Linear: input → hidden
h1 = ReLU(z1)                 # Activation: max(0, z1)
z2 = h1 @ W2 + b2             # Linear: hidden → output
probs = softmax(z2, mask)     # Probabilities (legal actions only)
```

**Why from scratch?**
- Understand every computation
- See gradient flow clearly
- No hidden framework magic
- Educational value

**Components:**
1. **ReLU activation:** f(x) = max(0, x)
   - Non-linearity enables complex patterns
   - Easy derivative: f'(x) = 1 if x > 0, else 0

2. **Softmax:** Converts logits to probabilities
   - probs[i] = exp(logit[i]) / sum(exp(logits))
   - Illegal actions masked (set to -∞)

3. **Xavier initialization:** Weights ~ Uniform(-√6/(n_in + n_out), +√6/(n_in + n_out))
   - Prevents vanishing/exploding gradients
   - Scales with layer size

### REINFORCE Algorithm

Monte Carlo Policy Gradient - learn from complete episodes:

```
1. Play episode with current policy
   - For each state s:
     - Sample action a ~ π(a|s)
     - Store (state, action, reward)

2. After episode ends:
   - Compute returns (discounted rewards)
   - Update policy to increase prob of good actions

3. Repeat
```

**Policy Gradient Theorem:**
```
∇J(θ) = E[∇ log π(a|s) × G_t]
```

Where:
- θ = policy parameters (neural network weights)
- π(a|s) = policy (probability of action a in state s)
- G_t = return (discounted sum of future rewards)
- ∇ log π = gradient of log probability

**Intuition:**
- G_t > 0 (good outcome) → increase prob of action
- G_t < 0 (bad outcome) → decrease prob of action
- Magnitude of G_t determines strength of update

### Backpropagation

Manual gradient computation through the network:

**Output layer gradient:**
```python
# Gradient of log π(a|s) w.r.t. output logits
dz2 = -probs.copy()
dz2[action] += 1.0
dz2 *= advantage  # Scale by return - baseline
```

**Hidden layer gradient:**
```python
# Backprop through output weights
dW2 = outer(h1, dz2)
db2 = dz2
dh1 = dz2 @ W2.T

# Backprop through ReLU
dz1 = dh1 * relu_derivative(z1)

# Backprop through hidden weights
dW1 = outer(input, dz1)
db1 = dz1
```

**Weight update (gradient ascent):**
```python
W1 += learning_rate * dW1
W2 += learning_rate * dW2
```

Note: Gradient *ascent* (not descent) because we maximize expected return.

### Baseline Subtraction

Raw REINFORCE has high variance. Baseline reduces it:

**Without baseline:**
```
∇J = ∇ log π(a|s) × G_t
```
Problem: G_t varies wildly (0 to 1 in our game)

**With baseline:**
```
∇J = ∇ log π(a|s) × (G_t - b)
```
- b = moving average of returns
- Advantage A = G_t - b

**Why it helps:**
- If G_t = 0.5, baseline = 0.5 → A = 0 (no update)
- If G_t = 1.0, baseline = 0.5 → A = +0.5 (increase prob)
- If G_t = 0.0, baseline = 0.5 → A = -0.5 (decrease prob)

Baseline centers returns around 0, making updates more stable.

## Training Results

**PolicyNetwork-v1 Performance:**
- Training: 5,000 episodes (0.5 minutes)
- vs Random: 14% win rate
- vs Heuristic: 18% win rate

**Training progression:**
```
Episode    vs Heuristic (eval)    Baseline
500        25%                    0.000
1000       20%                    0.000
1500       30%                    0.000
2000       15%                    0.000
2500       15%                    0.000
3000       10%                    0.000
3500       20%                    0.000
4000       10%                    0.000
4500       15%                    0.000
5000       20%                    0.000
```

**Analysis:**
- Baseline stuck at 0.000 (not learning)
- Performance oscillates wildly (10-30%)
- Worse than random (50%)
- Much worse than Linear Value (42%)

**Why it failed:**
1. **Insufficient training data:** 5K episodes too few for neural net
2. **High variance:** REINFORCE needs many samples per state
3. **Sparse rewards:** Only terminal reward (win/loss/draw)
4. **Credit assignment:** Hard to know which actions caused outcome
5. **Feature space:** 53 features not complex enough to need neural net

## Comparison: Policy Gradient vs TD Learning

| Aspect | TD Learning (Linear) | Policy Gradient (MLP) |
|--------|---------------------|----------------------|
| **What it learns** | Q(s,a) values | π(a\|s) probabilities |
| **Update frequency** | After each game | After each game |
| **Gradient** | Value error | Log probability × advantage |
| **Variance** | Low (bootstrapping) | High (Monte Carlo) |
| **Sample efficiency** | Good | Poor |
| **Model complexity** | 32 weights | ~8,000 weights |
| **Training time** | 0.8 min | 0.5 min |
| **Performance** | 42% vs Heuristic | 18% vs Heuristic |

**Winner:** TD Learning (Linear Value)

## Advantages of Policy Gradients

Despite poor performance, policy gradients have strengths:

1. **Stochastic policies:** Can learn mixed strategies
   - "Play rocket 70%, flare 30%"
   - Value methods force deterministic policies

2. **Continuous actions:** Works with continuous action spaces
   - Not relevant for this discrete game
   - Critical for robotics, control

3. **Convergence guarantees:** Proven to converge to local optimum
   - Value methods can diverge with function approximation

4. **Natural for exploration:** Outputs probability distribution
   - Easy to sample and explore
   - Can add entropy bonus

## Disadvantages

Why policy gradients failed here:

1. **Sample inefficiency:** Needs many episodes per update
   - TD learning uses every transition
   - REINFORCE only uses final outcome

2. **High variance:** Returns vary greatly
   - Even with baseline subtraction
   - Needs large batch sizes

3. **Credit assignment:** Which action caused win?
   - Placement decisions 20 moves before outcome
   - All get equal credit/blame

4. **Local optima:** Can get stuck in suboptimal policies
   - No exploration pressure
   - May never find good strategies

5. **Overkill for simple problems:** Neural network not needed
   - Linear function works better
   - Simpler model generalizes with less data

## Extensions

Ways to improve policy gradient learning:

**1. Actor-Critic:**
- Learn both policy π(a|s) and value V(s)
- Use value as better baseline
- Reduces variance significantly

**2. Advantage Actor-Critic (A2C):**
- Advantage A(s,a) = Q(s,a) - V(s)
- Better credit assignment
- More stable learning

**3. Proximal Policy Optimization (PPO):**
- Clip policy updates to prevent large changes
- More stable than vanilla REINFORCE
- State-of-the-art for many tasks

**4. Entropy Regularization:**
- Add entropy bonus to encourage exploration
- Loss = policy_loss - β × entropy
- Prevents premature convergence

**5. Experience Replay:**
- Store episodes in buffer
- Sample mini-batches for updates
- Breaks correlation, improves stability

## Further Reading

**Policy Gradient Fundamentals:**
- [Sutton & Barto Chapter 13: Policy Gradient Methods](http://incompleteideas.net/book/the-book-2nd.html)
- [Policy Gradients in a Nutshell](https://towardsdatascience.com/policy-gradients-in-a-nutshell-8b72f9743c5d)
- [REINFORCE Algorithm Explained](https://medium.com/@thechrisyoon/deriving-policy-gradients-and-implementing-reinforce-f887949bd63)

**Neural Networks from Scratch:**
- [Neural Networks from Scratch](https://nnfs.io/) - Book
- [Backpropagation Calculus](https://www.3blue1brown.com/topics/neural-networks) - Visual explanation
- [Implementing Backprop](https://cs231n.github.io/optimization-2/) - Stanford CS231n

**Advanced Policy Gradients:**
- [Actor-Critic Methods](https://arxiv.org/abs/1602.01783)
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)

**When to Use Policy Gradients:**
- [Policy Gradient vs Value Methods](https://stats.stackexchange.com/questions/326788/when-to-choose-policy-gradient-vs-value-function-approximation)
- [Choosing RL Algorithms](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)

## Video Resources

**REINFORCE Basics:**
- [REINFORCE Algorithm Explained](https://www.youtube.com/watch?v=XaYhArxxMIY) (12 min)
- [Policy Gradients](https://www.youtube.com/watch?v=y3oqOjHilio) (15 min) - Stanford CS234

**Backpropagation:**
- [Backpropagation Explained](https://www.youtube.com/watch?v=Ilg3gGewQ5U) (15 min) - 3Blue1Brown
- [Neural Networks from Scratch](https://www.youtube.com/watch?v=Wo5dMEP_BbI) (20 min)

**Advanced Topics:**
- [Actor-Critic Methods](https://www.youtube.com/watch?v=EKqxumCuAAY) (10 min)
- [PPO Explained](https://www.youtube.com/watch?v=5P7I-xPq8u8) (12 min)
- [David Silver RL Lecture 7: Policy Gradients](https://www.youtube.com/watch?v=KHZVXao4qXs) (90 min)

## Code Example

```python
from utala.agents.policy_network_agent import PolicyNetworkAgent

# Create agent
agent = PolicyNetworkAgent(
    name="PolicyNet",
    hidden_size=64,
    learning_rate=0.001,
    discount=0.99,
    seed=42
)

# Enable training
agent.set_training_mode(True)

# Play game (agent samples from policy)
for episode in range(num_episodes):
    # Run game, agent stores trajectory
    result = play_game(agent, opponent)

    # Update policy with outcome
    outcome = 1.0 if result.winner == agent.name else 0.0
    agent.observe_outcome(outcome)

# Evaluate
agent.set_training_mode(False)
win_rate = evaluate(agent, opponent, 100)
```

## Implementation Details

**Softmax with Masking:**
```python
def softmax(logits, legal_mask):
    # Set illegal actions to -∞
    logits = np.where(legal_mask, logits, -1e9)

    # Numerical stability
    logits = logits - np.max(logits)

    # Softmax
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)
```

**Why masking works:**
- exp(-1e9) ≈ 0
- Illegal actions get ~0 probability
- Legal actions share remaining probability

**Gradient Clipping:**
Not implemented but recommended:
```python
# Clip gradients to prevent exploding
max_grad = 1.0
if np.linalg.norm(dW1) > max_grad:
    dW1 = dW1 / np.linalg.norm(dW1) * max_grad
```

## Why Linear Value Beat MLP

**Data efficiency:**
- Linear: 5K games → 42%
- MLP: 5K games → 18%

**Sample efficiency:**
- Linear: Updates from every transition
- MLP: Updates only from episode outcome

**Bias-variance tradeoff:**
- Linear: High bias (simple model), low variance
- MLP: Low bias (flexible model), high variance

**With limited data → simple models win.**

If we had 100K+ training games, MLP might catch up. But for this project, linear function approximation is the clear winner.

## Conclusion

Policy gradient learning with neural networks:
- ✅ Theoretically sound
- ✅ Works for complex problems (Atari, Go, robotics)
- ✅ Educational (implemented backprop from scratch)
- ❌ Overkill for this problem
- ❌ Sample inefficient
- ❌ High variance without sophisticated techniques

**For Utala Kaos 9:**
- Linear Value Agent (TD learning) is the winner
- Simple beats complex with limited data
- 42% vs Heuristic is a good Phase 2 result

**Next:** Phase 2.5 will improve Linear Value agent with better features and more training to reach 55%+ target.
