# echo — Adaptive RL System

A lightweight reinforcement learning agent that sits on top of the Signal Hunt classifier
and adapts task difficulty based on user performance. The model stays fixed — what changes
is which sounds are presented, how difficult the task is, and in what order.

---

## The idea

A static classifier gives you one answer per input. It doesn't know whether the user is
improving, struggling, or bored. echo treats user interaction as a Markov Decision Process:
the state is the user's recent accuracy history, the action is which task to present next,
and the reward is whether the user gets it right.

Over time the agent learns: this user finds whistles easy and claps hard. Present more claps.
When they improve, increase difficulty. When they plateau, simplify and rebuild confidence.

This is the pattern behind adaptive learning systems, recommendation engines, and dynamic
difficulty in games — not a sequence of fixed tasks, but a system that responds.

---

## Prerequisites

- Signal Hunt Phase 2 complete — trained classifier working
- Acoustic Odyssey cast (preferred) — fast on-device inference for real-time response

---

## Milestones

### M1 — MDP design

- [ ] Define the state space: recent N accuracy scores, current difficulty level, task history
- [ ] Define the action space: which sound class to present, at which augmentation difficulty
- [ ] Define the reward function: +1 correct, -1 incorrect, bonus for streaks, penalty for repeated failures
- [ ] Document the design decisions — reward shaping choices and why

**Gate:** MDP fully specified on paper before any code. State/action/reward clearly defined.

**Docs:**
- [ ] Explainer: Markov Decision Process — state, action, reward, transition, why "Markov" matters
- [ ] Explainer: reward shaping — how the reward function drives behaviour

---

### M2 — Simulated environment

- [ ] Build a `UserSimulator` — a synthetic user with configurable skill level and learning curve
- [ ] The simulator takes a task (sound + difficulty) and returns: correct/incorrect based on its skill model
- [ ] Train and evaluate the RL agent against the simulator before touching real users

**Gate:** Simulator produces realistic accuracy curves. Agent can be trained fully offline.

**Docs:**
- [ ] Explainer: why simulation before real users — test without risk, iterate fast

---

### M3 — RL agent (Q-learning or policy gradient)

- [ ] Start with tabular Q-learning on a discretised state space
- [ ] Train against the simulator: agent should learn to present harder tasks as simulated user improves
- [ ] Evaluate: does accuracy of simulated user improve faster with the agent vs random task selection?

**Gate:** Agent produces measurably better learning curves than random/fixed ordering in simulation.

**Docs:**
- [ ] Explainer: Q-learning — value function, Bellman equation, exploration vs exploitation (ε-greedy)
- [ ] Compare to Utala's TD learning — same family, different application

---

### M4 — Exploration vs exploitation

- [ ] Implement ε-greedy: explore random tasks with probability ε, exploit best known action otherwise
- [ ] Decay ε over time — start exploratory, become more confident
- [ ] Compare: fixed ε vs decayed ε vs UCB (upper confidence bound)

**Gate:** Exploration strategy documented. Agent doesn't get stuck exploiting one action.

**Docs:**
- [ ] Explainer: exploration vs exploitation — the fundamental RL tradeoff

---

### M5 — Integration with Signal Hunt

- [ ] Connect the RL agent to the actual Signal Hunt inference pipeline
- [ ] Real interaction loop: agent presents a task → user records sound → classifier judges → reward fed back
- [ ] Session history saved: what was presented, what the user did, how accuracy changed

**Gate:** One real session logged. Agent's task selection visible in the history.

---

### M6 — Demo + evaluation

- [ ] Compare two sessions: random task order vs agent-directed order
- [ ] Plot accuracy over time for each — does the agent produce better learning curves?
- [ ] Document: what the agent learned, where it struggled

**Gate:** Measurable difference between random and agent-directed sessions.

**Docs:**
- [ ] README: how to run a session, how to read the history
- [ ] Explainer: evaluation of RL systems — why cumulative reward isn't the only metric

---

## Skills this covers

| Skill | Gap filled |
|-------|-----------|
| MDP formalisation | ⬜ → ✅ |
| Q-learning / policy gradient | ✅ extends Utala RL |
| Exploration vs exploitation | ⬜ → ✅ |
| RL in a real interaction loop | ⬜ → ✅ |
| Reward shaping | ✅ extends Utala |

---

## Feeds from

[cast](../cast/) — the optimised inference model is echo's engine.
Signal Hunt Phase 2 — the classifier that judges each user response.
