# Utala Kaos 9 — What's Next

## Current state of the AI opponents

All numbers corrected (2026-10-07) using the fixed harness. Variant A rules, 200 games balanced P1/P2.

| Agent | Params | vs Heuristic | vs Random | Ships? | Notes |
|-------|--------|-------------|-----------|--------|-------|
| **Random** | 0 | ~47% | — | No | Baseline |
| **Heuristic** | 0 | — | 53% | Candidate | Strongest agent |
| **MC-Fast** | 0 | 30% | 48% | No | Collapses in Variant A |
| **DQN-v3 (best ckpt)** | ~39K | **56%** | ~48% | No | ✅ Clears 50% gate. Stage 3 oscillation — 47.5% final. Redistil needed. |
| **TinyNN (shipped)** | ~5.7K | 38.5% | 48% | ⚠️ Stale | Distilled from v2 (weak teacher). Needs redistil from v3. |

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

### Step 2 — Redistil from DQN v3 best checkpoint

Best checkpoint (56%) is the teacher. Redistil into TinyNN (80→32→95, ~5.7K params).

**Script:** `scripts/train/variant_a/distill_dqn.py`
**Target:** TinyNN ≥45% vs corrected Heuristic
**Steps:**
1. Generate 5K–10K games with DQN v3 best checkpoint vs Heuristic + Random
2. Train TinyNN student on teacher logits
3. Eval: target ≥45% vs corrected Heuristic (100 games balanced)
4. Export to `export/tiny_nn_f32.json` for Flutter

**After redistil:** run visualise_agents.py, copy charts to explainers/images/, update this file.

**Commit:** `feat(utala): redistil TinyNN from DQN v3 — stronger teacher`

---

### Step 3 — Ship updated ONNX to Flutter

Replace weights in Flutter app with new `tiny_nn_f32.json`.

---

## Stage 3 oscillation — open question

Both v2 and v3 peak in early Stage 3 then oscillate. Options if redistil alone isn't enough:

- **Stop at 30K** (skip Stage 3 entirely — peak is consistently in early Stage 3)
- **Lower Stage 3 LR further** (0.0001 instead of 0.0005)
- **Keep 80% Heuristic in Stage 3** instead of 50/50 self-play mix
