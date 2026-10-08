# CLAUDE.md — utala: kaos 9

## Architecture rules (hard constraints — do not violate)

- **All randomness lives in the engine, never in agents.** Agents observe sampled state; they do not own RNG.
- **Agents only propose actions.** They never apply actions, mutate state, or validate legality.
- **The action space is fixed and fully enumerated.** Illegal actions are masked by the engine, never removed. A learning method that requires a variable action set is rejected.
- **State encoding and action enumeration are fixed** across all agents and phases — do not change without updating all agents and the engine.
- **Determinism required:** explicit RNG seeds, versioned replay format (seed + action list).

## Current state and action space (Variant A)

- **State:** 80-dim feature vector — see `src/utala/learning/dqn_features.py`
- **Actions:** 95 fixed (86 base + 9 CHOOSE_DOGFIGHT for Variant A) — see `src/utala/actions.py`
- **Rules:** Variant A (v1.9) is canon — see `utala-kaos-9.md`

## Shipped models

- `export/tiny_nn_f32.json` — TinyNN v2, float32, Flutter production model
- `models/onnx/tiny_nn.onnx` — ONNX runtime path
- `models/onnx/dqn_full.onnx` — DQN v3 full weights

## Tech stack

Python 3.11. NumPy throughout. PyTorch (Phase 3+) for DQN and distillation. ONNX for export. No graphics — text-only research environment.
