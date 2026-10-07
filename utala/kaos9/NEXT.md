# Utala Kaos 9 — What's Next

## Current state of the AI opponents

All numbers below are from the corrected evaluation (2026-10-07) using the fixed harness
(`rocket_in_play` now visible to agents each dogfight turn, MC rollouts use correct config).
Variant A rules: `GameConfig(fixed_dogfight_order=False)`, 200 games balanced P1/P2.

| Agent | Params | vs Heuristic | vs Random | Ships? | Notes |
|-------|--------|-------------|-----------|--------|-------|
| **Random** | 0 | ~47% | — | No | Baseline. Picks legal moves at random. |
| **Heuristic** | 0 | — | 53% | Candidate | Rule-based. Prioritises 3-in-a-row, defends when rocket in play. Strongest agent so far. |
| **MC-Fast** | 0 | 30% | 48% | No | Monte Carlo rollouts. Collapses in Variant A — random rollouts can't evaluate fight-order choice. |
| **DQN-v2** | ~39K | 41.5% | 59% | No | Best trained agent. Below Heuristic. Phase 5.1 checkpoint (>50%) does not hold against corrected Heuristic. |
| **TinyNN (shipped)** | ~5.7K | 38.5% | 48% | ⚠️ Shipped | Distilled from DQN-v2. Loses 6 in 10 vs Heuristic. Barely above Random. Teacher was too weak. |

**Bottom line:** the Heuristic is the only opponent worth shipping right now. The TinyNN that went to Flutter loses more than it wins and is only marginally better than random.

---

## What we'll do next

### Step 1 — Retrain DQN against corrected Heuristic

The previous training used the buggy harness — the Heuristic opponent never defended. The DQN learned to exploit a passive opponent. Retraining with the fixes in place means it will face a properly-defending Heuristic from the start.

**Script:** `scripts/train/variant_a/train_dqn_v2.py`
**Target:** consistent >50% vs corrected Heuristic over 200-game eval
**Config:** same as `dqn_v2d` (50K games, Stage 1/2/3 curriculum, LR halve at Stage 3)

Key things to watch:
- Does the DQN learn to counter defensive dogfight play?
- Does it learn to pick favourable fight order (the dimension MC failed at)?
- Peak vs final — save the best checkpoint, not the last

**Commit:** `feat(utala): DQN v3 — retrain against corrected Heuristic`

---

### Step 2 — Evaluate and set new checkpoint

Run the 200-game corrected eval. Gate: consistent >50% vs Heuristic.

If it doesn't clear 50%: try extending Stage 2 (the Heuristic curriculum phase) before adding self-play. The original 50K run peaked at 53% at 30K games — Stage 3 self-play may have caused oscillation. Try ending at 30K or using a lower LR throughout Stage 3.

---

### Step 3 — Redistil once DQN clears the gate

Same approach as Phase 5.2:
- Generate 5K–10K games with new DQN vs Heuristic and Random
- Train TinyNN (80→32→95, ~5.7K params)
- Eval: target ≥45% vs corrected Heuristic
- Export to `export/tiny_nn_f32.json` for Flutter

**If TinyNN can't reach 45%:** try a slightly larger student (80→64→95, ~10K params). Size vs performance is the tradeoff for mobile.

---

### Step 4 — Ship updated ONNX to Flutter

Replace the weights in the Flutter app. Update `export/tiny_nn_f32.json`.

The Heuristic rule logic can also be ported as a deterministic fallback — if the neural net is uncertain (max logit < threshold), fall back to heuristic rules. Belt and braces.

---

## Longer term (if the above succeeds)

- **Harder Heuristic:** the current heuristic doesn't model opponent Kaos deck state or use deck-aware strategy. Adding that would raise the bar further and produce a more interesting opponent.
- **Curriculum difficulty:** ship multiple agents (Easy = Random, Medium = TinyNN, Hard = DQN) so players can choose.
- **Self-play leagues:** once DQN beats Heuristic, a self-play league can push further — but requires stable training first.
