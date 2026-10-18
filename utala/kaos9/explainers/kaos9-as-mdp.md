# KAOS 9 as an MDP

Before any learning algorithm can run, the game has to be translated into a form a computer can reason about. That translation is the Markov Decision Process (MDP) — the formal framework that defines what the agent observes, what it can do, and what it's trying to maximise.

---

## The two-phase structure

Most games have a single loop: observe, act, observe, act. KAOS 9 has two fundamentally different phases that require different kinds of reasoning:

**Phase 1 — Deployment:** Players alternate placing Rocketmen (cards 2–10) onto the 3×3 grid. Any square can hold both players' pieces. No combat resolves. The highest and lowest value cards (2, 3, 9, 10) are placed face-down — their power is hidden from the opponent.

**Phase 2 — Dogfights:** Contested squares are resolved one at a time. Each dogfight is a mini-game: the weaker unit acts first, players spend dual-purpose weapons (A, K, Q, J) as rockets (attack) or flares (defence), and combat outcomes are resolved via personal 13-card Kaos decks.

The agent uses a single flat feature vector throughout both phases — the phase indicator is just another feature (see [state-encoding.md](state-encoding.md)). But the *meaning* of actions is completely different between phases, and the agent has to learn this from game outcomes alone.

> **In practice:** Any two-phase system has this structure — plan, then execute. A supply chain that books capacity (phase 1) then fulfils orders (phase 2), or a financial system that pre-allocates (phase 1) then settles (phase 2). The agent faces the same challenge: decisions made during planning constrain what's possible during execution, so it has to look ahead.

---

## State space

At each step the agent observes a 80-dimensional feature vector (see [state-encoding.md](state-encoding.md) for the full breakdown). The key categories:

| Category | What it captures |
|----------|-----------------|
| Per-square features (54) | For each of 9 grid squares: my occupancy, opponent occupancy, my power, opponent *visible* power, face-down flags |
| Resource counts (6) | My/opponent rocketmen remaining, weapons remaining, Kaos cards remaining |
| Board control (6) | Controlled squares, contested squares, 2-in-a-row threats |
| Phase + turn (3) | Deployment/dogfight flags, turn number normalised to [0,1] |
| Variant A context (3) | Awaiting dogfight choice, remaining contested count, joker holder |
| Deck awareness (4) | High-card ratio (≥8) in my/opponent Kaos deck, expected Kaos value |

**What the agent does NOT observe:** the exact power of opponent face-down cards. It only knows they're face-down. This is the hidden-information problem — bluffing exists because of this gap.

---

## Action space

95 discrete actions (Variant A):

| Range | Count | Action type |
|-------|-------|-------------|
| 0–80 | 81 | PLACE_ROCKETMAN — 9 powers × 9 grid positions |
| 81–84 | 4 | PLAY_WEAPON — 4 dual-purpose weapon cards |
| 85 | 1 | PASS |
| 86–94 | 9 | CHOOSE_DOGFIGHT — pick which contested square to fight next (Variant A only) |

**Illegal action masking:** The engine computes which actions are legal at each step and passes a binary mask to the agent. Illegal Q-values are set to −∞ before argmax — the agent never learns to play illegal moves, it simply can't select them. This design choice (fixed action space + masking) is much cleaner than a variable action space, because the network's output size never changes.

---

## Reward signal

Sparse and terminal:

```
+1  win
-1  loss
 0  draw
```

No intermediate rewards. The agent receives no feedback mid-game about whether a placement was good or a dogfight was well-played — it only learns from the final outcome.

**Why this makes learning hard:** A game lasts ~15–20 decisions. The reward at step 20 has to propagate back to step 1 through the Bellman equation. With TD learning, this works because each step bootstraps from the value estimate of the next step. With full Monte Carlo policy gradient (REINFORCE), the reward only arrives at game end and must be used to credit every decision — producing extremely high-variance gradient estimates. This is why the Phase 2 policy network (REINFORCE) reached only 18% vs Heuristic, while TD-Linear reached 42%.

---

## The hidden-information problem

Two sources of incomplete information:

1. **Face-down placement (Phase 1):** Cards 2, 3, 9, 10 are placed face-down. The agent knows its own face-down values but not the opponent's. The feature extractor reads `opp_rm.face_down` to suppress opponent power to 0 when the face-down flag is set — a deliberate asymmetry.

2. **Kaos deck order (Phase 2):** Each player's 13-card Kaos deck is shuffled. The agent tracks how many high cards remain (≥8) via the `high_card_ratio` feature, but not the exact draw order.

**The bluffing opportunity:** Placing a 2 (weakest Rocketman) face-down in the center looks identical to placing a 9. If the opponent bids too many weapons to win that dogfight, they've wasted resources. The agent can learn this exploitation purely from game outcomes — no explicit bluffing rule is needed.

**The information integrity issue (Phase 3):** Monte Carlo agents were silently using perfect information — `deepcopy(state)` copies all private fields. MC vs Heuristic was reported as 49% before the fix; after enforcing information sets for MC rollouts, it dropped to 30%. Learning agents (TD-Linear, DQN) were always clean because their feature extractors never read raw opponent face-down values.

---

## Why this game is worth studying

The game sits in a productive zone for AI research:

- **Simple enough** to understand completely — rules fit on two pages, full game tree is finite, outcomes are verifiable
- **Complex enough** to require learning — Heuristic beats Random by 30pp, but DQN beats Heuristic only under Variant A after 50K training games
- **Two distinct phases** that test planning vs execution reasoning
- **Hidden information** that makes bluffing a genuine first-class mechanic
- **Finite probability decks** (Kaos) that turn randomness into probability management

See [when-deep-learning-helps.md](when-deep-learning-helps.md) for how the game complexity directly determined which algorithms worked.
