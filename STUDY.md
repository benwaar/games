# AI/ML Capability Development

A personal R&D programme building practical AI and machine learning capabilities — from data pipelines and model training through to production deployment and adaptive systems. Each project is a working system, not a tutorial exercise.

The focus is applied: every skill maps to a class of real problem. The business alignment section below shows where each capability lands.

---

## Business Alignment

| Capability | What it enables | Examples |
|------------|----------------|---------|
| Classify real-world signals | Identify patterns in audio, documents, transaction data, or images automatically — no rules, learned from examples | Call centre audio → intent or sentiment; scanned documents → type and validity; transaction history → fraud or churn signal; cheque signature validation; accessibility testing — detect when a user is struggling |
| On-device inference | Run a trained model locally with no cloud round-trip — real time, zero latency, no ongoing API cost | Mobile banking feature; ATM-side fraud detection; branch kiosk; any product where cloud latency is too slow or too expensive |
| Adaptive systems | A system that learns from individual user behaviour and adjusts — not static rules, but a model that responds | Dynamic risk questionnaire; personalised onboarding; adaptive learning platform; accessibility adjustments that tune to how a specific user actually interacts |
| Structure detection in unlabelled data | Find patterns or anomalies in raw data without needing pre-labelled examples | Legacy system archaeology; anomaly detection in logs; discover clusters in transaction data before knowing what you're looking for |
| Media restoration and enhancement | Improve degraded, low-resolution, or damaged source material automatically | Scanned legacy documents cleaned up; old CCTV or archive footage enhanced; brand assets restored for remaster |
| LLM tools with quality gates | Build AI-powered internal tools that are verifiably correct — not just fluent | Internal knowledge assistant grounded in company documents; compliance checking tool; spec-driven code generation with automated test gates |
| Model extension without retraining | Add new categories to a working classifier without rebuilding it | New fraud type added to existing fraud model; new document category; new language added to existing intent model |
| Sequence and journey pattern detection | Detect order-dependent patterns — what happened, and in what order | Fraud sequence detection; user journey analysis (which paths precede churn); process mining to find the step order that leads to failure |
| Production deployment | Take a research model and ship it — compressed, fast, benchmarked, reliable | Any ML prototype that needs to become a product |

---

## Projects

| Project | What it is | Status |
|---------|-----------|--------|
| [Utala: KAOS 9](utala/kaos9/) | Card game AI — RL from scratch, TD learning, deep RL, model distillation | 🔷 Ph 1–5 done, redistil TinyNN from v3 pending |
| [Signal Hunt](signal-hunt/) | Audio classifier — data pipelines, CNN, training loop, inference, 100% test accuracy | ✅ Ph 1–3 complete |
| [Signal Hunt Ph 4](signal-hunt/) | Chord detection (multi-label) + chord progressions (CNN-RNN) | In progress — M18 next |
| [Acoustic Odyssey: cast](acoustic-odyssey/cast/) | Edge deployment — ONNX, TFLite, INT8 quantisation, sub-100ms on-device inference | Planned |
| [Acoustic Odyssey: echo](acoustic-odyssey/echo/) | Adaptive RL system — MDP, Q-learning, personalises task difficulty to the user | Planned |
| [Artefact: dig](artefact/dig/) | Binary art scanner — find hidden sprites in ROM/memory using CV and autoencoders | Planned |
| [Artefact: bloom](artefact/bloom/) | Pixel art upscaler — super-resolution CNN and GAN for retro game assets | Planned |
| [Artefact: dream](artefact/dream/) | LoRA fine-tune a diffusion model on extracted sprites → generate new pixel art in the same ROM style | Planned |
| [Void Duel](void-duel/) | 2P space shooter on ZX Spectrum + WASM emulator — classic ML, multi-agent RL, extreme-edge distillation | Planned (Ph 0: PoC) |
| TPP | LLM engineering — RAG, MCP, deterministic evals, agent orchestration | 🔷 Complete (separate repo) |

---

## Skills Coverage

Technical reference. ✅ complete · planned (project named) · 🔷 separate project · 🔶 previous role · ⬜ gap

```
AI / ML
│
├── Classic ML ──────────────────────────────────── 🔶 work project (not in this repo)
│   ├── Linear / logistic regression
│   ├── Decision trees, random forests
│   ├── Gradient boosting (XGBoost, LightGBM)
│   └── SVM, k-means, PCA
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
│   │   ├── Autoencoder ─────────────────────────── artefact/dig
│   │   ├── GAN ─────────────────────────────────── artefact/bloom (stretch)
│   │   ├── Diffusion ───────────────────────────── artefact/dream
│   │   └── Graph neural networks ───────────────── ⬜
│   │
│   ├── Training techniques
│   │   ├── Data augmentation ───────────────────── ✅ signal-hunt Ph 1
│   │   ├── Transfer learning + LoRA ────────────── ✅ signal-hunt Ph 3 + 🔷 TPP
│   │   ├── Curriculum learning ─────────────────── ✅ signal-hunt Ph 3
│   │   ├── Model distillation ──────────────────── ✅ utala
│   │   └── Self-supervised ─────────────────────── artefact/dig
│   │
│   └── Evaluation
│       ├── Precision, recall, F1, confusion matrix ✅ signal-hunt Ph 2
│       ├── Loss curves / overfitting analysis ────── ✅ signal-hunt Ph 2
│       ├── ROC / AUC ───────────────────────────── artefact/dig
│       ├── PSNR / SSIM / perceptual loss ────────── artefact/bloom
│       └── Interpretability (SHAP, LIME) ─────────── void-duel Ph 2
│
├── Domains
│   ├── Audio / signal processing ───────────────── ✅ signal-hunt
│   ├── Sequence / temporal ─────────────────────── signal-hunt Ph 4
│   ├── Computer vision ─────────────────────────── 🔶 NPR + artefact
│   ├── Binary / ROM data ───────────────────────── artefact/dig
│   ├── Image restoration ───────────────────────── artefact/bloom
│   ├── NLP / LLM engineering ───────────────────── 🔷 TPP
│   ├── Tabular data ────────────────────────────── 🔶 work project
│   └── ASR / speech recognition ────────────────── ⬜
│
├── Reinforcement Learning
│   ├── TD learning, deep RL, distillation ─────── ✅ utala
│   ├── MDP + Q-learning ────────────────────────── acoustic-odyssey/echo
│   ├── Exploration vs exploitation ─────────────── acoustic-odyssey/echo
│   └── Multi-agent / offline RL ────────────────── void-duel Ph 3
│
├── LLM Engineering ─────────────────────────────── 🔷 all covered (TPP)
│   └── RAG, MCP, skills, evals, agent orchestration
│
└── Production / MLOps
    ├── Inference pipeline ──────────────────────── ✅ signal-hunt Ph 2
    ├── ONNX / TFLite / INT8 / edge ────────────── acoustic-odyssey/cast
    ├── Latency benchmarking ────────────────────── acoustic-odyssey/cast
    ├── Model serving / REST API ────────────────── ⬜
    ├── Model monitoring / drift ────────────────── ⬜
    └── A/B testing models ──────────────────────── ⬜
```
