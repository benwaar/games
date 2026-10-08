# Utala KAOS 9 — Explainers

Project-specific explainers for Utala KAOS 9. Shared concepts (reinforcement learning, training loops, neural networks, distillation theory) live at the programme level:

→ **[../../explainers/](../../explainers/README.md)** — all shared explainers

The most relevant shared explainers for this project:

→ [Reinforcement Learning](../../explainers/reinforcement-learning.md) — MDP, TD learning, DQN, experience replay, target network, curriculum training, distillation
→ [Libraries](../../explainers/libraries.md) — PyTorch, NumPy, checkpoint patterns
→ [Python Concepts](../../explainers/python-concepts.md) — patterns used throughout

---

## Utala KAOS 9 — Project-specific explainers

### KAOS 9 as an MDP

How a novel two-phase card game maps onto a Markov Decision Process — state space, action space, reward signal, and the hidden-information problem.

→ [kaos9-as-mdp.md](kaos9-as-mdp.md)

---

### State Encoding — The 80-Dimensional Feature Vector

Every agent decision starts here. What the 80 features are, why the information asymmetry is deliberate, and how the encoding evolved from 53 dimensions (Phase 2) to 80 (Phase 4).

→ [state-encoding.md](state-encoding.md)

---

### When Deep Learning Helps — The Goldilocks Zone

DQN failed on the original rules (31% vs Heuristic) but succeeded on Variant A (53% peak). Why the game structure determines whether neural networks add value. The interaction-feature finding. The negative-weight line-formation discovery.

→ [when-deep-learning-helps.md](when-deep-learning-helps.md)

---

### Distillation — When the Student Beats the Teacher

Both major distillation runs in this project produced students that outperformed their teachers. Why that happens, what it means for deployment, and the concrete numbers.

→ [distillation.md](distillation.md)

---

### Agent Progression — The Full Story Arc

All agents across all phases with win rates, parameter counts, and what each phase proved. The complete arc from random baseline to 54% TinyNN in Flutter.

→ [agent-progression.md](agent-progression.md)
