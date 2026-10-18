# Extended Evaluation Metrics

## What We Did

Created a detailed metrics collection system that goes beyond win/loss to track game balance, strategy patterns, and agent performance.

**File:** `src/utala/learning/metrics.py`

**Metrics tracked:**

1. **Weapon usage:** How many weapons used, attack vs defense split
2. **Rocket effectiveness:** Hit rate, deflection rate
3. **Game dynamics:** Wipeout rate (both rocketmen removed)
4. **Comeback tracking:** Does the underdog recover?
5. **First-player advantage:** P1 win rate (should be 52-56% for balanced game)
6. **Decision time:** Agent thinking time (for performance optimization)

**Classes:**

- `GameMetrics`: Detailed stats for a single game
- `DetailedMatchResult`: Aggregated stats across multiple games
- `MetricsCollector`: Helper for collecting metrics during gameplay

## What It Means

**Why win rate isn't enough:**

Two agents with 50% win rate could play very differently:
- Agent A: Always spends weapons, aggressive
- Agent B: Never spends weapons, defensive

Win rate says "equally strong" but doesn't reveal *how* they play.

**Metrics reveal strategy:**

- **Weapon spend profile:** Always-spend → risky/aggressive. Always-hoard → passive/weak.
- **Rocket hit rate:** Low hit rate → telegraphed attacks, opponent predicts well.
- **Comeback rate:** High comebacks → skilled agents who recover. Low → early game matters too much.
- **Wipeout rate:** High wipeouts → mutual destruction, luck-driven endgames.

**Game balance indicators:**

Good tactical games show:
- **First-player advantage:** 52-56% (slight edge, not overwhelming)
- **Weapon spend:** ~2-3 weapons/game (meaningful choices, not always/never)
- **Rocket hit value:** ~7 (53% hit rate means rockets are balanced)
- **Comeback rate:** 10-20% (skill matters, but early game has weight)

If metrics fall outside these ranges, the game may have balance issues.

## Further Reading

**Game balance analysis:**
- [Measuring Game Balance](https://www.gdcvault.com/play/1015360/Metrics-in-Game-Balance) - GDC talk on balance metrics
- [First-player advantage in games](https://en.wikipedia.org/wiki/First-move_advantage_in_chess) - Chess example

**Simulation metrics:**
- [Evaluating AI in Games](https://arxiv.org/abs/1807.01281) - Academic survey of game AI evaluation
- [Beyond Win Rate](https://www.microsoft.com/en-us/research/publication/beyond-win-rate-measuring-skill-in-competitive-games/) - Microsoft Research on skill metrics

**Performance profiling:**
- [Profiling Python Code](https://docs.python.org/3/library/profile.html) - Python profiler docs
- [Real-time AI performance](https://www.gamedeveloper.com/programming/performance-optimization-for-real-time-ai) - Game AI performance guide

## Video Resources

**Game balance:**
- [Balancing Multiplayer Games](https://www.youtube.com/watch?v=WXQzdXPTb5Q) (18 min) - GDC talk on balance principles
- [Metrics for Game Design](https://www.youtube.com/watch?v=O8YG7-nDh6g) (12 min) - What to measure and why

**Performance measurement:**
- [Profiling Python Applications](https://www.youtube.com/watch?v=m_a0fN48Alw) (15 min) - How to measure performance
- [Real-Time AI Constraints](https://www.youtube.com/watch?v=6f6kGVs7P5g) (10 min) - Why decision time matters

## Code Example

```python
from utala.agents.random_agent import RandomAgent
from utala.agents.heuristic_agent import HeuristicAgent
from utala.learning.metrics import DetailedMatchResult, MetricsCollector

# Run a match with detailed metrics
agent1 = HeuristicAgent("Heuristic")
agent2 = RandomAgent("Random")

# Normally integrated into evaluation harness
# For now, can use MetricsCollector manually:

collector = MetricsCollector(seed=12345)

# During game:
# collector.record_weapon_use(Player.ONE, is_rocket=True)
# collector.record_rocket_outcome(hit=True)
# collector.update_square_advantage(state)

# After game:
# metrics = collector.finalize(winner, final_state)

# View detailed summary
print(result.summary())
# Output:
# === DETAILED MATCH RESULTS ===
# Heuristic vs Random
# Games: 100
#
# OUTCOMES:
#   P1 Wins: 65 (65.0%)
#   P2 Wins: 30 (30.0%)
#   Draws: 5 (5.0%)
#   First-player advantage: 65.0%
#
# WEAPON USAGE:
#   P1 avg: 2.8 weapons/game
#   P2 avg: 3.1 weapons/game
#   Rocket hit rate: 54.3%
#   Wipeout rate: 12.5%
#
# GAME DYNAMICS:
#   Comeback rate: 15.0%
#
# PERFORMANCE:
#   P1 avg decision time: 45.2ms
#   P2 avg decision time: 0.8ms
```

## Design Decisions

**Why track decision time?**

Real-time games have performance budgets:
- Mobile games: ~16ms per frame (60fps)
- Turn-based: <100ms feels instant to humans

If an agent takes 2 seconds to think, it won't work in production. Decision time metrics flag performance problems early.

**Why comeback tracking?**

Measures game depth:
- Low comebacks (< 5%): Early game determines everything → shallow
- High comebacks (> 30%): Late game swings wildly → too random
- Moderate (10-20%): Skill matters throughout, but early lead helps → good balance

**Why weapon spend profile?**

Reveals strategic diversity:
- All agents spend 0 weapons: weapons are useless
- All agents spend 4 weapons: weapons are mandatory
- Agents vary (2-3 avg): meaningful strategic choice → good design

**Why rocket hit value ~7?**

Math: 7/13 = 53.85% hit rate

If rockets hit < 40%: too weak, never worth using
If rockets hit > 70%: too strong, always dominant

~54% means rockets are slightly favored (as attacker), but flares matter.

## Interpreting Results

**Example: Agent shows 90% rocket hit rate**

Possible causes:
- Opponent never uses flares → too defensive
- Agent fires only when sure → too conservative
- Opponent's defense is predictable → poor strategy

Action: Check opponent behavior, adjust training.

**Example: Agent has 80% win rate but 0% comeback rate**

Interpretation:
- Agent wins by early domination
- Falls apart if opponent gets early lead
- Lack of tactical flexibility

Action: Train on diverse game states, not just winning positions.

**Example: Agent decision time > 500ms**

Problem: Too slow for real-time use

Action:
- Profile the code (where is time spent?)
- Optimize hot paths
- Consider approximations (e.g., fewer rollouts for Monte Carlo)
