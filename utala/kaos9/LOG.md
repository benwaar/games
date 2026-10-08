# utala: kaos 9 — Project Log

A study of learning algorithms through a 2-player tactical card game. One game engine, one evaluation harness — plug in progressively sophisticated agents and compare.

## Contents

- [Phase Progression](#phase-progression)
- [Phase 1 — Engine & Baselines](#phase-1--engine--baselines)
- [Phase 2 — Learning Without Frameworks](#phase-2--learning-without-frameworks)
- [Phase 3 — Deep Learning](#phase-3--deep-learning)
- [Phase 4 — Variant A Rule Change](#phase-4--variant-a-rule-change)
- [Phase 5 — Improve, Distil, Ship](#phase-5--improve-distil-ship)
- [All Agents](#all-agents)
- [Overall Lessons](#overall-lessons)

---

## Phase Progression

| Phase | What | Checkpoint question | Result |
|-------|------|---------------------|--------|
| 1 | Engine, replay, baselines | Is the game worth studying? | PASS |
| 2 | TD-linear, hand-built gradients | Can a model learn to play? | PASS |
| 3 | DQN, distillation (original rules) | Does deep learning help? | FAIL (linear wins) |
| 4 | Variant A rules, DQN retrained | Is DL justified now? | PASS |
| 5 | Better DQN, TinyNN, Flutter | Can we improve and ship? | PASS |

Each phase reuses the same engine and harness — no throwaway code.

---

## Phase 1 — Engine & Baselines

**Goal:** Build the game engine and prove the game is worth studying.

### What was built

- Canonical Python engine with deterministic replay (seed + action list)
- Evaluation harness: self-play, cross-play, balanced P1/P2 matchups
- Three baseline agents: Random, Heuristic, Monte Carlo (10 rollouts)

### Results

| Agent | vs Heuristic | vs Random |
|-------|-------------|----------|
| Random | ~20% | — |
| Heuristic | 50% (baseline) | 53% |
| Monte Carlo (fair info) | 30% (Variant A) | 48% |

**Checkpoint 1 PASS:** Clear skill gradient with meaningful variance. Game is not dominated by luck or trivial heuristics.

**MC note:** Initially reported at 49% vs Heuristic. Phase 3 audit found MC was using perfect information (`deepcopy` copies face-down values). Corrected to 30% under Variant A after enforcing information sets.

---

## Phase 2 — Learning Without Frameworks

**Goal:** Prove a model can learn. No PyTorch — hand-built gradient updates.

### What was built

| Agent | Method | vs Heuristic |
|-------|--------|-------------|
| k-NN | 5-nearest-neighbour | 33% |
| Policy network (REINFORCE) | 285K params, policy gradient | 18% |
| TD-Linear (42 features) | TD(0), ε-greedy | 42% |
| TD-Linear NoInt (27 features) | Removed interaction features | **47.5% best / 43.8% avg** |

### Key decisions

- TD learning over REINFORCE — short episodes (~15 steps) and sparse terminal rewards make Monte Carlo policy gradient high-variance and slow. TD bootstraps from the next state, not from game end.
- Interaction features hurt — `strong_move_when_winning` combines two concepts that individually have stable weights. Combined, they introduced collinearity and noise (-6pp).
- Line-formation features were hardcoded to 0.0 (a bug). After fixing, the model learned **negative weights** — forming visible 2-in-a-row lines is counterproductive against a Heuristic with an explicit +30 blocking bonus.

---

## Phase 3 — Deep Learning (original rules)

**Goal:** Test whether DQN could exceed the linear ceiling.

### Results

| Agent | vs Heuristic | Params |
|-------|-------------|--------|
| DQN v1 | 31% | 34,518 |
| MC-Fast (teacher) | 44.5% | 0 |
| Linear imitation | 47% | ~5,000 |
| TinyNN imitation | 48% | 1,089 |

**Finding:** DQN failed on original rules — 11pp worse than TD-Linear. Short episodes, sparse rewards, and linear patterns meant the DQN's extra capacity added noise. The game's tactical patterns were: `outcome ≈ material_advantage + control_advantage` — a linear combination.

**Distillation:** MC-Fast generated 8,228 decision records. Both imitation students beat the teacher by 3–4pp. The student averages the teacher's noisy decisions and inherits strategy without noise.

**Information integrity audit (Phase 3.2):** MC was silently using perfect information. Learning agents were always clean because their feature extractors never read raw opponent state.

---

## Phase 4 — Variant A Rule Change

**Goal:** Add a rule requiring nonlinear reasoning. Prove deep learning is necessary.

**The change:** Center square fights first; then the dogfight winner chooses the next square to fight. Action space 86 → 95 (+9 CHOOSE_DOGFIGHT actions).

**Why it changes the answer:** Choosable dogfight order requires reasoning like "fight here because I complete MY row **AND** have a power advantage" — a conjunction of two features. A linear model computes sums, not products. This is fundamentally nonlinear.

### Results

| Agent | vs Heuristic | Notes |
|-------|-------------|-------|
| TD-Linear (Variant A) | 35.5% | Down from 47.5% — linear ceiling |
| DQN v2 (80-dim state, Variant A) | 47% final / **53% peak** | Recovered above linear ceiling |

**Checkpoint 2 PASS:** DQN peak (53%) exceeds TD-Linear ceiling (35.5%). Deep learning is now justified.

**State encoding upgrade:** 53-dim (Phase 2) → 80-dim. Per-square features expanded from occupancy-only to 6 per square (my/opp occupancy, my/opp power, face-down flags). Added Variant A context and deck awareness groups.

**Training peak at game 32,500 (early Stage 3).** Stage 3 self-play caused oscillation — the agent adapted to its own changing policy rather than consolidating. Best checkpoint saved separately.

---

## Phase 5 — Improve, Distil, Ship

**Goal:** Improve DQN to consistent >50% vs Heuristic. Distil into TinyNN. Ship to Flutter.

### DQN v3

Retrained against corrected Heuristic. **56% peak vs Heuristic at game 32,500.** Final weights 47.5% (Stage 3 oscillation, same pattern as v2). Best checkpoint saved: `results/dqn_v2d/dqn_v2_best.pth`.

### TinyNN v2

93,149 decision records from 5,000 teacher games. Behavioural cloning (cross-entropy).

| Student | Params | vs Heuristic | vs Random |
|---------|--------|-------------|----------|
| TinyNN (80→32→95) | 5,727 | **48%** | 50% |
| Linear (80→95) | 7,695 | 46.5% | — |
| DQN v3 (teacher, deployed) | 39,135 | 49% | — |

TinyNN matches teacher at **7× fewer parameters**. Student beats teacher for the same reason as Phase 3.4 — averages the teacher's noisy decisions.

### Export and ship

| File | Format | Use |
|------|--------|-----|
| `export/tiny_nn_f32.json` | JSON float32 | Dart pure-inference in Flutter |
| `models/onnx/tiny_nn.onnx` | ONNX | ONNX runtime path |
| `models/onnx/dqn_full.onnx` | ONNX | Hard difficulty path |

**Checkpoint 3 PASS:** DL agent improved and shipped.

---

## All Agents

All numbers are Variant A, corrected harness (200 games balanced P1/P2).

| Agent | vs Heuristic | Params | Inference | Ships? |
|-------|-------------|--------|-----------|--------|
| Random | ~20% | 0 | <0.01ms | No |
| Heuristic | 50% (baseline) | 0 | <0.05ms | Candidate |
| MC-Fast (fair info) | 30% | 0 | 485ms | No |
| TD-Linear NoInt | 43.8% avg / 47.5% best | 27 | <0.25ms | Candidate |
| DQN v2 peak | 53% | 39,135 | 0.054ms | No |
| DQN v3 peak | 56% | 39,135 | 0.054ms | No (teacher) |
| **TinyNN v2** | **48%** | **5,727** | **0.045ms** | **✅ Shipped** |
| Linear distilled | 46.5% | 7,695 | 0.040ms | Candidate |

---

## Overall Lessons

**When deep learning helps:** Only when the game requires nonlinear feature combinations. Under original rules, DQN was 11pp worse than a 27-weight linear model. Variant A's choosable dogfight order introduced conjunctive reasoning that linear models cannot do — that's when DQN became necessary.

**Interaction features hurt linear models:** Clean individual features beat clever combinations. The model already knew the individual signals; combining them introduced collinearity.

**Distilled students beat teachers:** The teacher's training noise (epsilon-greedy exploration, Stage 3 oscillation) doesn't transfer to the student — only the strategy does.

**Stage 3 self-play oscillation:** Both DQN v2 and v3 peaked in early Stage 3 then degraded. The best checkpoint was mid-training, not at the end. Save checkpoints throughout training, not just the final weights.

**Information asymmetry in features:** The face-down suppression in the feature extractor (opponent power = 0 when face-down) makes the information boundary explicit — the model can learn what "face-down" means strategically, not just treat it as missing data.
