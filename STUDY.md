# Study Map

Skills across all projects. ✅ covered · 🔷 separate project · 🔶 previous role · ⬜ gap

---

## Projects

| Project | Folder | Status |
|---------|--------|--------|
| Utala: KAOS 9 | [utala/kaos9/](utala/kaos9/) | ✅ Complete |
| Signal Hunt Ph 1–2 | [signal-hunt/](signal-hunt/) | ✅ Complete |
| Signal Hunt Ph 3–4 | [signal-hunt/](signal-hunt/) | Planned |
| Acoustic Odyssey: cast | [acoustic-odyssey/cast/](acoustic-odyssey/cast/) | Planned |
| Acoustic Odyssey: echo | [acoustic-odyssey/echo/](acoustic-odyssey/echo/) | Planned |
| Artefact: dig | [artefact/dig/](artefact/dig/) | Planned |
| Artefact: bloom | [artefact/bloom/](artefact/bloom/) | Planned |
| TPP | separate repo | 🔷 Complete |

---

## Skills Tree

```
AI / ML
│
├── Classic ML ──────────────────────────────────── ⬜ shallow gap (2-day project)
│   ├── Linear / logistic regression
│   ├── Decision trees, random forests
│   ├── Gradient boosting (XGBoost, LightGBM)
│   ├── SVM, k-means, PCA
│   └── ← tabular data baseline in artefact/dig M2 covers this partially
│
├── Deep Learning
│   │
│   ├── Foundations
│   │   ├── NumPy, PyTorch tensors ──────────────── ✅ signal-hunt Ph 1
│   │   ├── Forward pass / loss / backprop ─────── ✅ signal-hunt Ph 2
│   │   ├── Adam, LR scheduling, early stopping ── ✅ signal-hunt Ph 2
│   │   ├── Dropout, BatchNorm ──────────────────── ✅ signal-hunt Ph 2
│   │   └── Weight initialisation ───────────────── ⬜
│   │
│   ├── Architectures
│   │   ├── CNN (Conv2d, pooling, GAP) ──────────── ✅ signal-hunt Ph 2
│   │   ├── RNN / GRU / LSTM ────────────────────── signal-hunt Ph 4
│   │   ├── Attention ───────────────────────────── signal-hunt Ph 4
│   │   ├── U-Net / encoder-decoder ─────────────── artefact/bloom
│   │   ├── Transformer ─────────────────────────── 🔷 TPP
│   │   ├── Autoencoder ─────────────────────────── artefact/dig M5
│   │   ├── GAN ─────────────────────────────────── artefact/bloom M5 (stretch)
│   │   ├── Diffusion ───────────────────────────── ⬜
│   │   └── Graph neural networks ───────────────── ⬜
│   │
│   ├── Training techniques
│   │   ├── Data augmentation ───────────────────── ✅ signal-hunt Ph 1
│   │   ├── Class imbalance ─────────────────────── signal-hunt Ph 3
│   │   ├── Transfer learning ───────────────────── signal-hunt Ph 3
│   │   ├── LoRA ────────────────────────────────── 🔷 art project
│   │   ├── Curriculum learning ─────────────────── signal-hunt Ph 3
│   │   ├── Model distillation ──────────────────── ✅ utala
│   │   └── Self-supervised ─────────────────────── artefact/dig M5
│   │
│   └── Evaluation
│       ├── Precision, recall, F1, confusion matrix ✅ signal-hunt Ph 2
│       ├── Loss curves / overfitting ──────────────── ✅ signal-hunt Ph 2
│       ├── ROC / AUC ───────────────────────────── artefact/dig M2
│       ├── PSNR / SSIM / perceptual loss ────────── artefact/bloom M2–M4
│       ├── Regression metrics (MSE, RMSE) ────────── artefact/bloom M3
│       └── Interpretability (SHAP, LIME) ─────────── ⬜
│
├── Domains
│   ├── Audio / signal processing ───────────────── ✅ signal-hunt Ph 1–4
│   ├── Sequence / temporal ─────────────────────── signal-hunt Ph 4
│   ├── Computer vision (images) ────────────────── 🔶 NPR + artefact/dig+bloom
│   ├── Binary data / ROM scanning ──────────────── artefact/dig
│   ├── Image super-resolution ──────────────────── artefact/bloom
│   ├── Tabular data ────────────────────────────── ⬜ (artefact/dig baseline partial)
│   ├── NLP / text ──────────────────────────────── 🔷 TPP
│   ├── Embeddings / vector search ──────────────── 🔷 TPP
│   ├── LLM engineering ─────────────────────────── 🔷 TPP
│   └── Speech recognition (ASR) ────────────────── ⬜
│
├── Reinforcement Learning
│   ├── TD learning, deep RL ────────────────────── ✅ utala
│   ├── Reward shaping ──────────────────────────── ✅ utala
│   ├── MDP + Q-learning ────────────────────────── acoustic-odyssey/echo
│   ├── Exploration vs exploitation ─────────────── acoustic-odyssey/echo
│   └── Multi-agent / model-based / offline ─────── ⬜
│
├── LLM Engineering (TPP track) ─────────────────── 🔷 all covered
│   ├── Prompting, skills, RAG
│   ├── Tool use / MCP
│   ├── Deterministic evals
│   └── Agent orchestration
│
└── Production / MLOps
    ├── Inference pipeline ──────────────────────── ✅ signal-hunt Ph 2
    ├── Model export (ONNX / TFLite) ───────────── acoustic-odyssey/cast
    ├── Quantisation (INT8) ─────────────────────── acoustic-odyssey/cast
    ├── Edge deployment ─────────────────────────── acoustic-odyssey/cast
    ├── Latency benchmarking ────────────────────── acoustic-odyssey/cast
    ├── Model serving / REST API ────────────────── ⬜
    ├── Model monitoring / drift ────────────────── ⬜
    └── A/B testing models ──────────────────────── ⬜
```

---

## Gap Summary

| Area | Status |
|------|--------|
| Classic ML / tabular | ⬜ Shallow — 2-day project when needed |
| Diffusion models | ⬜ Not planned |
| Graph neural networks | ⬜ Not planned |
| NLP / text (from scratch) | ⬜ Covered at LLM level via TPP |
| MLOps (serving, monitoring) | ⬜ Not planned |
| ASR | ⬜ Not planned |

---

## Business Alignment

What these skills enable — described by problem type, not product name.

| Problem type | Projects that enable it |
|-------------|------------------------|
| Classify real-world signals (audio, sensor, image) in real time | signal-hunt + acoustic-odyssey/cast |
| Deploy a model to run on-device with no cloud dependency | acoustic-odyssey/cast |
| Adapt an experience to individual user behaviour over time | utala + acoustic-odyssey/echo |
| Find structure or patterns in raw/unlabelled data | artefact/dig |
| Restore or enhance degraded/low-quality media | artefact/bloom |
| Build LLM-powered tools with deterministic quality gates | TPP |
| Extend a working model to new classes without retraining from scratch | signal-hunt Ph 3 |
| Detect sequences and order-dependent patterns in streams | signal-hunt Ph 4 |
| Compress a research model into a production-grade artefact | acoustic-odyssey/cast + utala |
