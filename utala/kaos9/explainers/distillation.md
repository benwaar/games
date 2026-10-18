# Distillation — When the Student Beats the Teacher

This project ran two major distillation experiments, and in both cases the student model outperformed its teacher. This isn't a fluke — it's a predictable consequence of how distillation works.

---

## What distillation is

A large, expensive model (the teacher) plays many games and its decisions are recorded. A small, fast model (the student) is trained to mimic those decisions via supervised learning. The student learns from the teacher's *policy* — what to do — rather than from game outcomes directly.

The goal is to compress expertise into a model small enough to ship: run in a Flutter app, serve with low latency, or deploy to an edge device.

---

## Phase 3.4 — MC-Fast teaches imitation models

**Teacher:** Monte Carlo with 10 rollouts per decision. Plays ~500 games vs Heuristic/Random, generating 8,228 decision-point records. Win rate vs Heuristic: **44.5%**

**Students trained on MC-Fast decisions:**

| Student | Architecture | Params | vs Heuristic |
|---------|-------------|--------|-------------|
| Linear imitation | 53 → 95 | ~5,000 | 47% |
| TinyNN imitation | 53 → 32 → 95 | 1,089 | 48% |
| MC-Fast (teacher) | — (search) | 0 | 44.5% |

Both students beat the teacher by 3–4pp.

**Why:** MC-Fast with 10 rollouts has high per-decision variance. Each decision is based on a small sample of possible futures — some rollouts get lucky, some don't. The training dataset of 8,228 decisions averages over all of MC-Fast's choices, including its noisy ones. The student learns the central tendency of MC's policy — the strategic intent — without the noise of individual rollout variance.

---

## Phase 5.2 — DQN v3 teaches TinyNN

**Teacher:** DQN v3, best checkpoint (game 32,500, Stage 3 early). Win rate vs Heuristic: **56% at peak, ~49.5% average checkpoint**.

**Dataset:** 93,149 decision points from 5,000 teacher games vs Heuristic + Random.

**Students trained with behavioural cloning (cross-entropy loss):**

| Student | Architecture | Params | vs Heuristic |
|---------|-------------|--------|-------------|
| Linear distilled | 80 → 95 | 7,695 | **50%** |
| TinyNN distilled | 80 → 32 → 95 | 5,727 | **54%** |
| DQN v3 (teacher, deployed checkpoint) | 80 → 128 → 128 → 95 | 39,135 | 49.5% |

TinyNN with 5,727 parameters outperforms its 39,135-parameter teacher.

---

## Why the student beats the teacher

**The teacher's problem:** DQN trains with epsilon-greedy exploration. During late-stage training (the data collection phase), epsilon is low but non-zero — the teacher occasionally takes random exploratory actions. It also oscillates during Stage 3 self-play (see [when-deep-learning-helps.md](when-deep-learning-helps.md) and the training curve chart). The "best checkpoint" is a snapshot of a model mid-oscillation.

**What the student sees:** 93,149 decisions, each representing the teacher's best-effort action at that state. Some of those decisions are from strong teacher checkpoints, some from weaker moments. The student trains to predict the teacher's action — effectively averaging the teacher's policy across its entire training trajectory.

> The student inherits the teacher's strategic knowledge without inheriting its noise.

This is the same phenomenon as ensemble methods: the average of many noisy predictions is more accurate than any single prediction.

---

## What changes in deployment

| Property | DQN (teacher) | TinyNN (student) |
|----------|--------------|-----------------|
| Parameters | 39,135 | 5,727 |
| Architecture | 80→128→128→95 | 80→32→95 |
| Framework | PyTorch | None (Dart forward pass) |
| Model file | `dqn_v2_best.pth` (157KB) | `tiny_nn.json` (118KB) → `tiny_nn_f32.json` (~25KB) |
| Inference time | 0.054ms | 0.045ms |
| Deployment target | Python research | Flutter mobile app |

The TinyNN forward pass is implemented directly in Dart — two matrix multiplications and a ReLU, reading weights from a JSON asset. No ML framework required on the device.

---

## The export chain

```
DQN v3 training (PyTorch)
    ↓
5,000 teacher games → 93,149 decision records
    ↓
TinyNN trained (PyTorch, behavioural cloning)
    ↓
Export: results/distill_v1/tiny_nn.json (float64, 118KB)
    ↓
Convert: export/tiny_nn_f32.json (float32, ~25KB)
    ↓  also:
    ↓  models/onnx/tiny_nn.onnx (ONNX format)
    ↓
Flutter app: load JSON asset, Dart forward pass
```

---

## The general principle

Distillation is how production ML systems are often built:

1. Train the best model you can (large, slow, expensive)
2. Use it to label a dataset
3. Train a small model to mimic the large one
4. Ship the small model

The student frequently beats the teacher in evaluation because the teacher's training noise doesn't transfer — only its knowledge does. This is why GPT-class models can distil into models 10–100× smaller while retaining most of their capability on specific tasks.

> **In practice:** If you have a slow but accurate model (complex rules engine, expensive API, large ML model), you can often distil it into something deployable by generating a labelled dataset from it and training a smaller supervised model. The distilled model won't match the teacher perfectly, but it often matches or exceeds the teacher's practical performance while running orders of magnitude faster.

See: [scripts/train/variant_a/distill_dqn.py](../scripts/train/variant_a/distill_dqn.py) — distillation pipeline  
See: [results/distill_v1/tiny_nn.json](../results/distill_v1/tiny_nn.json) — exported TinyNN weights
