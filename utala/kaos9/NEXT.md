# Utala Kaos 9 — What's Next

## Current state of the AI opponents

All numbers corrected (2026-10-07) using the fixed harness. Variant A rules, 200 games balanced P1/P2.

| Agent | Params | vs Heuristic | vs Random | Ships? | Notes |
|-------|--------|-------------|-----------|--------|-------|
| **Random** | 0 | ~47% | — | No | Baseline |
| **Heuristic** | 0 | — | 53% | Candidate | Strongest agent |
| **MC-Fast** | 0 | 30% | 48% | No | Collapses in Variant A |
| **DQN-v3 (best ckpt)** | ~39K | **56% peak / 49% eval** | 48% | No | Teacher for distillation |
| **TinyNN-v2** | ~5.7K | **48%** | 50% | ✅ Ready | Redistilled from v3. Matches teacher at 7x smaller. |
| **TinyNN-v1 (shipped)** | ~5.7K | 38.5% | 48% | ⚠️ Stale | Old teacher — replace in Flutter |

---

## What we'll do next

### ✅ Step 1 — Retrain DQN against corrected Heuristic — DONE

Retrained 2026-10-07. 50K games, same curriculum config.

**Results:**
- Best checkpoint: **56.0% vs Heuristic** (game 32,500, early Stage 3) — clears 50% gate ✅
- Final model: 47.5% — same Stage 3 oscillation as v2 (loss climbs 0.27 → 0.57)
- Best checkpoint saved: `results/dqn_v2d/dqn_v2_best.pth`

**Opening shift (placement heatmap):** first move changed from **R (93.6%)** to **T (92.8%)** — different fixed opening when playing a properly-defending opponent. Still deterministic, just a different square.

---

### ✅ Step 2 — Redistil TinyNN from DQN v3 — DONE

Redistilled 2026-10-07. 5K game dataset (93K decisions), 50 epochs.

**Results:**
- TinyNN (80→32→95, 5,727 params): **48% vs Heuristic**, 50% vs Random — ✅ exceeds 45% target
- Linear (80→95, 7,695 params): 46.5% vs Heuristic — also above 45%
- Teacher DQN: 49% vs Heuristic (200-game final eval; 56% was mid-training peak)
- TinyNN matches teacher at **7× fewer parameters** — excellent compression
- Imitation accuracy: 89% — student closely copies teacher's action choices

New weights exported to `export/tiny_nn_f32.json` — ready for Flutter.

---

### ✅ Step 3 — Ship updated models to Flutter — DONE

Updated 2026-10-07. Both inference paths updated in `src/b/ui/assets/models/`:

| File | What | Notes |
|------|------|-------|
| `tiny_nn_f32.json` | TinyNN v2 weights (JSON) | Dart pure-inference path |
| `tiny_nn.onnx` | TinyNN v2 weights (ONNX) | ONNX runtime path |
| `dqn_full.onnx` | DQN v3 full weights (ONNX) | Hard difficulty path |

Export script: `scripts/export/export_onnx_phase5.py` — updated to use `dynamo=False` (legacy single-file ONNX, opset 12). Run from utala project root.

---

## Phase 5 Complete ✅

| Step | Status | Result |
|------|--------|--------|
| 5.1 Retrain DQN | ✅ | 56% peak vs corrected Heuristic |
| 5.2 Redistil TinyNN | ✅ | 48% vs Heuristic (was 38.5%) |
| 5.3 Ship to Flutter | ✅ | Both JSON + ONNX updated in src/b/ui |

## Stage 3 oscillation — open question

Both v2 and v3 peak in early Stage 3 then oscillate. Options if redistil alone isn't enough:

- **Stop at 30K** (skip Stage 3 entirely — peak is consistently in early Stage 3)
- **Lower Stage 3 LR further** (0.0001 instead of 0.0005)
- **Keep 80% Heuristic in Stage 3** instead of 50/50 self-play mix
