# utala: kaos 9 — Docs

Full command reference and project structure.

## Contents

- [Setup](#setup)
- [Play](#play)
- [Train](#train)
- [Evaluate](#evaluate)
- [Export](#export)
- [Tests](#tests)
- [Project structure](#project-structure)

---

## Setup

```bash
./setup.sh                  # Python 3.11 venv + dependencies
source .venv/bin/activate
```

---

## Play

```bash
./run.sh                    # human vs AI (TinyNN by default)
python scripts/demo/play_demo.py --agent heuristic
python scripts/demo/play_demo.py --agent tiny_nn
```

---

## Train

```bash
# Phase 2 — TD-Linear
python scripts/train/train_linear_agent.py

# Phase 4/5 — DQN (Variant A)
python scripts/train/variant_a/train_dqn.py

# Phase 5 — Distil DQN → TinyNN
python scripts/train/variant_a/distill_dqn.py
```

---

## Evaluate

```bash
# Run full evaluation (all agents)
python scripts/eval/run_evaluation.py

# Compare two specific agents
python scripts/eval/eval_agents.py --a heuristic --b tiny_nn --games 200

# Information sets comparison (MC fair vs perfect)
python scripts/eval/eval_info_sets.py
```

---

## Export

```bash
# Export TinyNN + DQN to JSON/ONNX for Flutter
python scripts/export/export_onnx_phase5.py
```

Outputs to `export/tiny_nn_f32.json` and `models/onnx/`.

---

## Tests

```bash
python run_tests.py
```

---

## Project structure

```
src/utala/
  engine.py          — game engine core (placement, dogfights, win detection)
  state.py           — game state representation
  actions.py         — action space (95 actions), legal mask
  agents/            — all agent implementations
  learning/          — feature extraction (80-dim), TD training infra
  deep_learning/     — DQN architecture and training
  evaluation/        — harness for running and scoring games

scripts/
  train/             — training scripts per phase
    train_linear_agent.py
    variant_a/train_dqn.py
    variant_a/distill_dqn.py
  eval/              — evaluation scripts
  export/            — model export (JSON + ONNX)
  analysis/          — weight analysis, ablation, feature importance
  demo/              — play/demo scripts

tests/               — unit tests
results/             — training output, checkpoints
  dqn_v2d/           — DQN v3 best checkpoint
  distill_v1/        — TinyNN v2 weights + dataset

export/
  tiny_nn_f32.json   — TinyNN v2 weights (float32, for Flutter)

models/
  onnx/
    tiny_nn.onnx     — TinyNN v2 (ONNX)
    dqn_full.onnx    — DQN v3 (ONNX)

docs/
  phases/            — phase plans and results (2–5)
  mobile/            — Flutter integration guides
  EXPLAINERS.md      — pointer to shared + local explainers

explainers/          — project-specific knowledge docs
  kaos9-as-mdp.md
  state-encoding.md
  when-deep-learning-helps.md
  distillation.md
  agent-progression.md
```
