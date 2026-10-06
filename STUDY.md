# Study Map — What Was Built and Where It Applies

A record of what each project teaches, mapped to the types of problems those skills solve.
Not tied to specific products — the point is recognising the pattern when you see it.

---

## Projects

### Utala: KAOS 9 (Complete)

A card game AI researching how agents learn to play well — from random baselines through
TD learning to deep reinforcement learning and model distillation.

**Skills built:**

| Skill | What it is |
|-------|-----------|
| Temporal Difference learning | Agent updates its value estimate after each action, not just at the end |
| Deep RL (policy/value networks) | Neural network replaces hand-crafted value table — generalises to unseen states |
| Model distillation | Compress a large trained model into a smaller, faster one without retraining from scratch |
| Game tree evaluation | Lookahead search, minimax, risk/reward tradeoffs under uncertainty |
| Reward shaping | Designing feedback signals that produce the behaviour you want |

**Applies to any problem where:**
- An agent makes sequential decisions and learns from outcomes (not labelled examples)
- You need to personalise behaviour per user over time (the "agent" is responding to that user's history)
- A system needs to adapt its strategy based on feedback — A/B-style, but dynamic
- You have a large capable model and need a lightweight version for production
- You want to simulate human-like play, opponent modelling, or risk-aware decision making

*Examples: recommendation engines, adaptive difficulty, fraud response strategies, trading algorithms, coaching systems, game NPCs, user journey optimisation*

---

### Signal Hunt — Phase 1 & 2 (Complete)

An audio classifier — from raw `.wav` files to a trained CNN that identifies sound types
with 100% test accuracy, plus a full inference pipeline.

**Skills built:**

| Skill | What it is |
|-------|-----------|
| Data pipeline engineering | Ingest, normalise, augment, and version raw unstructured input at scale |
| Feature engineering | Transform raw signal into a representation a model can learn from (Mel-spectrogram) |
| CNN classification | Spatial pattern recognition from 2D input — learns what shapes mean |
| Training loop | Loss, backprop, optimiser, LR scheduling, early stopping, checkpointing |
| Evaluation | Precision, recall, F1, confusion matrix — understanding *where* a model fails |
| Inference pipeline | End-to-end: raw input → prediction + confidence, no manual steps |
| Train/val/test discipline | Held-out evaluation that isn't contaminated by development decisions |

**Applies to any problem where:**
- Raw unstructured data (audio, images, sensor readings, documents) needs to be turned into features a model can use
- You need to classify inputs into discrete categories from examples — not rules
- You want to detect patterns in 2D data (spectrograms, images, heatmaps, time-frequency plots)
- Small datasets need augmentation to train meaningfully
- You need a real-time inference endpoint: input arrives → prediction returned immediately
- You want to understand model confidence, not just yes/no outputs

*Examples: sound classification, image recognition, document triage, anomaly detection, sensor monitoring, medical imaging, defect detection, content moderation*

---

### Signal Hunt — Phase 3: Pitch Classification (Planned)

Extend the Phase 2 CNN to classify which musical note is being hummed or whistled —
12 classes instead of 3.

**Skills will include:**

| Skill | What it is |
|-------|-----------|
| Transfer learning | Reuse a model trained on a related task as a starting point — freeze early layers, retrain the head |
| Fine-grained classification | Distinguishing between similar-looking classes (adjacent musical notes) |
| Larger label spaces | Scaling from 3 classes to 12+ without rebuilding from scratch |
| Class imbalance | Some classes are harder to produce — handling uneven data distributions |
| Curriculum learning | Train on easy examples first, introduce harder ones gradually |

**Applies to any problem where:**
- You have a working model on a coarse task and need finer resolution
- Labels are expensive to collect and you want to start broad, refine later
- Classes are similar in some dimensions but distinguishable in others
- Data is naturally imbalanced (rare events, hard-to-produce categories)
- You're extending an existing system rather than rebuilding from scratch

*Examples: product categorisation (coarse → fine), intent classification (broad → specific), species identification, medical condition grading, customer segment refinement*

---

### Signal Hunt — Phase 4: Sequence Recognition (Planned)

Detect ordered patterns of sounds using a CNN-RNN hybrid — "hum then clap" vs "clap then hum".

**Skills will include:**

| Skill | What it is |
|-------|-----------|
| Recurrent networks (GRU/LSTM) | Models that carry memory across time steps — learns what happened before matters |
| Sequence classification | A variable-length input maps to a single label |
| Temporal reasoning | The order of events matters, not just which events occurred |
| CNN-RNN hybrid | CNN extracts per-frame features; RNN models how they evolve over time |
| Attention | Learns which time steps to focus on for a given prediction |

**Applies to any problem where:**
- The order of events matters, not just their presence
- Input is a sequence and the output depends on context across that sequence
- You need to model user journeys, event logs, or time-series data
- Pattern detection in streams: "X then Y within T seconds" style logic

*Examples: clickstream analysis, transaction sequence fraud, log anomaly detection, activity recognition, speech recognition, time-series classification, customer journey modelling*

---

### Acoustic Odyssey — Phase 5: Edge Optimisation (Planned)

Export a trained model to production-grade lightweight formats and benchmark real-world performance.

**Skills will include:**

| Skill | What it is |
|-------|-----------|
| ONNX / TFLite export | Convert PyTorch model to a portable, framework-agnostic format |
| INT8 quantisation | Reduce model weights from 32-bit floats to 8-bit integers — 4× smaller, faster |
| Benchmarking | Measure latency, memory, and accuracy tradeoffs under real constraints |
| Edge deployment | Run inference locally, no network round-trip, no cloud bill |

**Applies to any problem where:**
- Inference must happen on-device (mobile, IoT, embedded systems)
- Latency matters — response must be under a threshold (sub-100ms, real-time)
- Cloud inference cost is unsustainable at the usage volume you expect
- Privacy requires data to stay on-device

*Examples: mobile ML features, real-time audio/video processing, IoT sensor inference, offline-capable apps, privacy-preserving personalisation*

---

### Acoustic Odyssey — Phase 6: Adaptive Systems (Planned)

A lightweight RL agent that personalises behaviour based on user performance over time.

**Skills will include:**

| Skill | What it is |
|-------|-----------|
| Markov Decision Process | Formalise the problem: states, actions, rewards, transitions |
| RL in production | Apply RL where the environment is a real user, not a simulator |
| Personalisation loops | Update the model per-user based on ongoing feedback |
| Exploration vs exploitation | Balance trying new strategies vs using what works |

**Applies to any problem where:**
- You want the system to adapt to individual users over time
- Feedback is behavioural (clicks, completions, errors) rather than explicit labels
- The "right answer" differs per user or changes over time
- You want to move beyond static rules into learned, dynamic behaviour

*Examples: adaptive difficulty, content recommendation, coaching systems, dynamic pricing, onboarding flows, accessibility adjustments*

---

## Skill Dependency Map

```
Utala: KAOS 9
  └── RL fundamentals → Phase 6 (adaptive systems)
  └── Model distillation → Phase 5 (compression techniques)

Signal Hunt Ph 1–2
  └── Data pipelines → any ML project with unstructured input
  └── CNN classification → image/audio/sensor classification
  └── Training loop → universal ML foundation
  └── Inference pipeline → Phase 5 (optimise this pipeline)

Signal Hunt Ph 3
  └── Transfer learning → any domain where fine-tuning beats training from scratch
  └── Fine-grained classification → product/intent/condition grading

Signal Hunt Ph 4
  └── Sequence modelling → clickstreams, logs, journeys, time-series
  └── CNN-RNN hybrid → foundation for speech and activity recognition

Phase 5 (edge)
  └── Quantisation + export → production deployment of everything above

Phase 6 (adaptive)
  └── RL personalisation → sits on top of any classifier (Phase 2–4) as a decision layer
```

---

## Where these skills apply beyond the current projects

| Domain | Phases that apply |
|--------|-----------------|
| Real-time media apps (audio/video classification) | Ph 1–2, Ph 4, Ph 5 |
| Fraud detection & anomaly detection | Ph 1–2 (pattern recognition), Ph 4 (sequence), Ph 6 (adaptive thresholds) |
| Healthcare (imaging, monitoring, diagnostics) | Ph 1–2, Ph 3, Ph 5 (on-device privacy) |
| Accessibility tooling | Ph 2 (classification), Ph 4 (sequence/timing), Ph 6 (personalisation) |
| Recommendation & personalisation | Utala RL, Ph 6 |
| IoT / edge computing | Ph 5 |
| Developer tooling (code classification, log analysis) | Ph 1–2, Ph 3, Ph 4 |
| Customer journey analytics | Ph 4, Ph 6 |
| Content moderation | Ph 1–2, Ph 3 |
| Financial services (transaction patterns, risk) | Ph 4 (sequences), Utala (risk/reward), Ph 6 (adaptive) |

---

## ML/DL Skills Map

Assuming all six phases complete. ✅ = covered, ⬜ = gap.

---

### Foundations

| Skill | Status | Where |
|-------|--------|-------|
| NumPy array operations | ✅ | Signal Hunt Ph 1 |
| PyTorch tensors | ✅ | Signal Hunt Ph 1–2 |
| Data normalisation (z-score) | ✅ | Signal Hunt Ph 1 |
| Train/val/test split | ✅ | Signal Hunt Ph 2 |
| Stratified sampling | ✅ | Signal Hunt Ph 2 |
| Data augmentation | ✅ | Signal Hunt Ph 1 |
| Class imbalance handling | ✅ | Signal Hunt Ph 3 |
| Cross-validation | ⬜ | — |
| Hyperparameter search (grid/random/Bayesian) | ⬜ | — |
| Feature stores / data versioning | ⬜ | — |

---

### Classic ML (pre-deep learning)

| Skill | Status | Where |
|-------|--------|-------|
| Linear / logistic regression | ⬜ | — |
| Decision trees / random forests | ⬜ | — |
| Gradient boosting (XGBoost, LightGBM) | ⬜ | — |
| SVM | ⬜ | — |
| k-means clustering | ⬜ | — |
| PCA / dimensionality reduction | ⬜ | — |
| Naive Bayes | ⬜ | — |

> **Note:** Classic ML is the missing foundation. Gradient boosting in particular is
> the dominant approach for tabular data in production — most financial, operational,
> and CRM datasets are tabular, not images or audio.

---

### Neural Network Fundamentals

| Skill | Status | Where |
|-------|--------|-------|
| Forward pass / loss / backprop | ✅ | Signal Hunt Ph 2 |
| Activation functions (ReLU, softmax) | ✅ | Signal Hunt Ph 2 |
| Adam optimiser | ✅ | Signal Hunt Ph 2 |
| SGD / RMSprop | ⬜ | — |
| Learning rate scheduling | ✅ | Signal Hunt Ph 2 |
| Dropout | ✅ | Signal Hunt Ph 2 |
| Batch normalisation | ✅ | Signal Hunt Ph 2 |
| Weight initialisation | ⬜ | — |
| Early stopping + checkpointing | ✅ | Signal Hunt Ph 2 |
| Gradient clipping | ✅ (planned) | Signal Hunt Ph 4 |

---

### Architectures

| Skill | Status | Where |
|-------|--------|-------|
| Feedforward / MLP | ✅ (linear head) | Signal Hunt Ph 2 |
| CNN (Conv2d, pooling, GAP) | ✅ | Signal Hunt Ph 2 |
| RNN / GRU / LSTM | ✅ (planned) | Signal Hunt Ph 4 |
| Bidirectional RNN | ✅ (planned) | Signal Hunt Ph 4 |
| Attention mechanism | ✅ (planned) | Signal Hunt Ph 4 |
| Transformer | ⬜ | — |
| Autoencoder / VAE | ⬜ | — |
| GAN | ⬜ | — |
| Diffusion models | ⬜ | — |
| Graph neural networks | ⬜ | — |

---

### Transfer Learning & Advanced Training

| Skill | Status | Where |
|-------|--------|-------|
| Transfer learning (freeze + retrain head) | ✅ (planned) | Signal Hunt Ph 3 |
| Fine-tuning (full network) | ✅ (planned) | Signal Hunt Ph 3 |
| Curriculum learning | ✅ (planned) | Signal Hunt Ph 3 |
| Self-supervised learning | ⬜ | — |
| Few-shot / zero-shot learning | ⬜ | — |
| Multi-task learning | ⬜ | — |

---

### Evaluation

| Skill | Status | Where |
|-------|--------|-------|
| Accuracy, precision, recall, F1 | ✅ | Signal Hunt Ph 2 |
| Confusion matrix | ✅ | Signal Hunt Ph 2 |
| Loss curves / overfitting analysis | ✅ | Signal Hunt Ph 2 |
| ROC / AUC | ⬜ | — |
| Regression metrics (MSE, RMSE, MAE) | ⬜ | — |
| Model calibration | ⬜ | — |
| Interpretability (SHAP, LIME, attention viz) | ⬜ (attention partial) | Signal Hunt Ph 4 |

---

### Domains

| Domain | Status | Where |
|--------|--------|-------|
| Audio classification | ✅ | Signal Hunt Ph 1–2 |
| Time-frequency features (Mel, STFT) | ✅ | Signal Hunt Ph 1 |
| Sequence / temporal classification | ✅ (planned) | Signal Hunt Ph 4 |
| Tabular data | ⬜ | — |
| Computer vision (images beyond spectrograms) | ⬜ | — |
| Object detection / segmentation | ⬜ | — |
| NLP / text classification | ⬜ | — |
| Embeddings / word vectors | ⬜ | — |
| Large language models / transformers | ⬜ | — |
| Speech recognition (ASR) | ⬜ | — |
| Reinforcement learning | ✅ | Utala |
| Multi-modal (combine modalities) | ⬜ | — |

---

### Production / MLOps

| Skill | Status | Where |
|-------|--------|-------|
| Inference pipeline | ✅ | Signal Hunt Ph 2 |
| Model export (ONNX / TFLite) | ✅ (planned) | Acoustic Odyssey Ph 5 |
| Quantisation (INT8) | ✅ (planned) | Acoustic Odyssey Ph 5 |
| Model distillation | ✅ | Utala |
| Edge deployment | ✅ (planned) | Acoustic Odyssey Ph 5 |
| Model serving / REST API | ⬜ | — |
| Model monitoring / drift detection | ⬜ | — |
| Feature stores | ⬜ | — |
| A/B testing models | ⬜ | — |
| Federated learning | ⬜ | — |

---

### Reinforcement Learning

| Skill | Status | Where |
|-------|--------|-------|
| Temporal difference learning | ✅ | Utala |
| Deep RL (policy / value networks) | ✅ | Utala |
| MDP (Markov Decision Process) | ✅ (planned) | Acoustic Odyssey Ph 6 |
| Exploration vs exploitation | ✅ (planned) | Acoustic Odyssey Ph 6 |
| Reward shaping | ✅ | Utala |
| Multi-agent RL | ⬜ | — |
| Model-based RL | ⬜ | — |
| Offline RL | ⬜ | — |

---

### Summary

After all six phases:
- **Strong:** Audio/signal processing, CNNs, training discipline, inference pipelines, RL fundamentals, model distillation, edge deployment
- **Partial:** Sequence modelling, attention, transfer learning, quantisation
- **Gaps:** Classic ML (tabular, gradient boosting), NLP/transformers, computer vision beyond spectrograms, model monitoring, MLOps infrastructure, advanced RL variants
- **Biggest practical gap:** Classic ML on tabular data — most real-world business datasets are tables, not audio. Gradient boosting (XGBoost/LightGBM) solves the majority of them and is rarely taught in deep learning courses.
