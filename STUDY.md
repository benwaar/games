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

## ML/AI Skills Map

**Legend:** ✅ this learning track (games repo) · 🔷 separate project · 🔶 previous role · ⬜ gap

---

```
AI / ML
│
├── Classic ML ─────────────────────────────────── ⬜ gap (all)
│   ├── Linear / logistic regression
│   ├── Decision trees, random forests
│   ├── Gradient boosting (XGBoost, LightGBM) ← biggest practical gap
│   ├── SVM
│   ├── k-means clustering
│   └── PCA / dimensionality reduction
│
├── Deep Learning
│   │
│   ├── Foundations
│   │   ├── NumPy, PyTorch tensors ──────────────── ✅ Signal Hunt Ph 1
│   │   ├── Forward pass / loss / backprop ─────── ✅ Signal Hunt Ph 2
│   │   ├── Optimisers (Adam) ───────────────────── ✅ Signal Hunt Ph 2
│   │   ├── LR scheduling, early stopping ────────── ✅ Signal Hunt Ph 2
│   │   ├── Dropout, BatchNorm ──────────────────── ✅ Signal Hunt Ph 2
│   │   └── Weight initialisation ───────────────── ⬜
│   │
│   ├── Architectures
│   │   ├── CNN (Conv2d, pooling, GAP) ──────────── ✅ Signal Hunt Ph 2
│   │   ├── RNN / GRU / LSTM ────────────────────── ✅ Signal Hunt Ph 4
│   │   ├── Attention ───────────────────────────── ✅ Signal Hunt Ph 4
│   │   ├── Transformer ─────────────────────────── 🔷 TPP (via LoRA/RAG)
│   │   ├── Autoencoder / VAE ───────────────────── ⬜
│   │   ├── GAN ─────────────────────────────────── ⬜
│   │   ├── Diffusion ───────────────────────────── ⬜
│   │   └── Graph neural networks ───────────────── ⬜
│   │
│   ├── Training techniques
│   │   ├── Data augmentation ───────────────────── ✅ Signal Hunt Ph 1
│   │   ├── Class imbalance handling ─────────────── ✅ Signal Hunt Ph 3
│   │   ├── Transfer learning (freeze + head) ────── ✅ Signal Hunt Ph 3
│   │   ├── Full fine-tuning ────────────────────── ✅ Signal Hunt Ph 3
│   │   ├── LoRA (parameter-efficient fine-tuning) ─ 🔷 Art project
│   │   ├── Curriculum learning ─────────────────── ✅ Signal Hunt Ph 3
│   │   ├── Model distillation ──────────────────── ✅ Utala
│   │   ├── Self-supervised learning ─────────────── ⬜
│   │   └── Few-shot / zero-shot ─────────────────── ⬜
│   │
│   └── Evaluation
│       ├── Precision, recall, F1, confusion matrix ✅ Signal Hunt Ph 2
│       ├── Loss curves / overfitting analysis ────── ✅ Signal Hunt Ph 2
│       ├── ROC / AUC ───────────────────────────── ⬜
│       ├── Regression metrics (MSE, RMSE) ────────── ⬜
│       └── Interpretability (SHAP, LIME) ─────────── ⬜
│
├── Domains
│   ├── Audio / signal processing ───────────────── ✅ Signal Hunt Ph 1–4
│   ├── Sequence / temporal ─────────────────────── ✅ Signal Hunt Ph 4
│   ├── Reinforcement learning ──────────────────── ✅ Utala, AO Ph 6
│   ├── Tabular data ────────────────────────────── ⬜
│   ├── Computer vision (images) ────────────────── 🔶 NPR (first CNN)
│   ├── NLP / text ──────────────────────────────── 🔷 TPP
│   ├── Embeddings / vector search ──────────────── 🔷 TPP (RAG)
│   ├── LLMs (inference, fine-tuning) ───────────── 🔷 TPP, art project
│   ├── Speech recognition (ASR) ────────────────── ⬜
│   └── Multi-modal ─────────────────────────────── ⬜
│
├── LLM Engineering (separate track — TPP)
│   ├── Prompt engineering & skills ─────────────── 🔷 TPP
│   ├── Retrieval-augmented generation (RAG) ─────── 🔷 TPP
│   ├── Tool use / MCP ──────────────────────────── 🔷 TPP
│   ├── Deterministic evals ─────────────────────── 🔷 TPP
│   ├── Agent orchestration (iterate-until-green) ─── 🔷 TPP
│   └── Two-stage critic (correctness → quality) ──── 🔷 TPP
│
└── Production / MLOps
    ├── Inference pipeline ──────────────────────── ✅ Signal Hunt Ph 2
    ├── Model export (ONNX / TFLite) ───────────── ✅ AO Ph 5
    ├── Quantisation (INT8) ─────────────────────── ✅ AO Ph 5
    ├── Edge deployment ─────────────────────────── ✅ AO Ph 5
    ├── Model serving / REST API ────────────────── ⬜
    ├── Model monitoring / drift detection ──────── ⬜
    ├── Cross-validation, hyperparameter search ─── ⬜
    └── A/B testing models ──────────────────────── ⬜
```

---

### Summary

After all projects complete:

| Area | Status |
|------|--------|
| Audio / signal ML (end to end) | ✅ Strong |
| CNN training discipline | ✅ Strong |
| RL fundamentals + distillation | ✅ Strong |
| RNNs, attention, sequence modelling | ✅ Covered (Ph 4) |
| Transfer learning + LoRA | ✅ Covered (Ph 3 + TPP) |
| LLM engineering (RAG, MCP, evals, agents) | 🔷 Covered (TPP) |
| Edge deployment + quantisation | ✅ Covered (AO Ph 5) |
| Classic ML (regression, boosting, trees) | ⬜ Shallow gap |
| Computer vision beyond spectrograms | 🔶 NPR (first CNN) |
| Tabular data ML | ⬜ Shallow gap |
| NLP / text (beyond LLM use) | ⬜ Gap |
| MLOps (serving, monitoring, A/B) | ⬜ Gap |

**Note on classic ML:** Shallower than it looks. The algorithms (decision trees, gradient boosting) are simpler than backprop. The sklearn API is fit() / predict() — easier than a PyTorch training loop. The hard part of tabular ML is feature engineering and domain understanding, both of which come from working with data — already covered. Closing this gap is a 2-day project, not a study track.

---

## Project Ideas — Filling Gaps Through Interests

Three themes from your work: **tape/signal encoding**, **loss and reconstruction**, **compact protocol design**. These map directly onto several gaps.

The common thread: *how do you store, transmit, and recover information reliably under constraints?*

---

### Artefact: dig + bloom — binary art scanner and pixel upscaler

See [artefact/](artefact/) — dig finds hidden sprites in binary memory, bloom scales them to HD.
Together: scan → extract → restore. A full digital archaeology pipeline.

---

### Tape Rescue — degraded signal reconstruction

**The idea:** ZX Spectrum / C64 games were stored on cassette tape as audio tones. Real tapes degrade — dropouts, noise, bit errors. Can a model reconstruct corrupted tape audio and recover the original data?

**Gaps it fills:**
- **Autoencoder / denoising autoencoder** ← fills the ⬜ Autoencoder gap — encode degraded signal to latent space, decode clean signal
- **Signal reconstruction / missing data** — same family as QR recovery, JPEG artefact removal, medical image inpainting
- **Builds directly on Signal Hunt** — same audio pipeline, same spectrogram features, different task

**Why it fits your style:** The ZX Spectrum tape format (TZX/TAP) is fully documented — like TPP, it has a spec. Generate ground-truth clean audio, deliberately degrade it, deterministic eval: does the decoded data match the original bytes? Same pattern as TPP evals.

---

### QR Rescue — masked image recovery

**The idea:** QR code with chunks deliberately obscured (torn corner, sticker, dirt). Train a model to predict the missing modules and reconstruct a scannable code.

**Gaps it fills:**
- **Masked autoencoder (MAE)** ← fills ⬜ Self-supervised learning — predict missing patches from visible ones. Same technique as BERT (text masking) and MAE (vision, He et al. 2022)
- **Computer vision on real images** — grid detection, perspective correction, spatial reasoning

**Why it fits your style:** QR codes already have Reed-Solomon redundancy. The project asks: what if the damage exceeds what error correction handles? Evaluation is deterministic — the code scans or it doesn't.

---

### Pixel Art Upscaler — retro sprites to HD

**The idea:** Take 8×8 or 16×16 pixel art sprites (ZX Spectrum, C64, NES) and train a model to upscale them to HD while preserving the crisp pixel aesthetic — not blurring, not interpolating, but learning the style.

**Gaps it fills:**
- **Computer vision on real images** ← fills ⬜ CV gap — real 2D image input, spatial feature learning
- **Super-resolution / generative models** — a CNN encoder-decoder (U-Net style) or GAN. The discriminator learns "does this look like pixel art at scale?" — touches ⬜ GAN territory
- **Perceptual loss** — standard MSE blurs; perceptual loss (feature-space distance) preserves sharpness. New evaluation technique.

**Why it fits your style:** Pixel art is already a compression problem — 16 colours, fixed palette, every pixel deliberate. The upscaler is learning to decompress with style. Directly usable for Aerythen assets. Has a clear visual pass/fail — you can see immediately if the output looks right, and you can compare against NEAREST/BILINEAR/LANCZOS as baselines.

**Bonus:** existing datasets are free — CGA/EGA game rips, sprite sheets from public domain ROMs, Libresprite assets. No recording needed.

---

### The pattern

```
Tape Rescue    → corrupted audio waveform    → reconstruct bytes       (autoencoder)
QR Rescue      → occluded image grid         → reconstruct modules     (masked AE)
Pixel Upscaler → low-res pixel art           → high-res stylised image (super-resolution / GAN)
Tabular Rescue → missing sensor rows         → reconstruct values      (gradient boosting)
```

Same reconstruction idea. Different substrate, different architecture, different gap filled.
