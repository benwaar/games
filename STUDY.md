# AI/ML Capability Development

A personal R&D programme building practical AI and machine learning capabilities — from data pipelines and model training through to production deployment and adaptive systems. Each project is a working system, not a tutorial exercise.

The focus is applied: every skill maps to a class of real problem. The business alignment section below shows where each capability lands.

---

## Business Alignment

| Capability | What it enables | Examples |
|------------|----------------|---------|
| Classify real-world signals | Identify patterns in audio, documents, transaction data, or images automatically — no rules, learned from examples | Call centre audio → intent or sentiment; scanned documents → type and validity; transaction history → fraud or churn signal; cheque signature validation; accessibility testing — detect when a user is struggling |
| On-device inference | Run a trained model locally with no cloud round-trip — real time, zero latency, no ongoing API cost. Native mobile via Rust + ONNX with hardware accelerators (Neural Engine, GPU/DSP) and UniFFI bindings to Swift/Kotlin | Mobile banking feature; ATM-side fraud detection; branch kiosk; real-time audio classification on phone; any product where cloud latency is too slow or too expensive |
| Adaptive systems | A system that learns from individual user behaviour and adjusts — not static rules, but a model that responds | Dynamic risk questionnaire; personalised onboarding; adaptive learning platform; accessibility adjustments that tune to how a specific user actually interacts |
| Structure detection in unlabelled data | Find patterns or anomalies in raw data without needing pre-labelled examples | Legacy system archaeology; anomaly detection in logs; discover clusters in transaction data before knowing what you're looking for |
| Media restoration and enhancement | Improve degraded, low-resolution, or damaged source material automatically | Legacy audio cleaned up; old recordings restored; 8-bit game soundtracks remastered; brand audio assets enhanced |
| LLM tools with quality gates | Build AI-powered internal tools that are verifiably correct — not just fluent | Internal knowledge assistant grounded in company documents; compliance checking tool; spec-driven code generation with automated test gates |
| Model extension without retraining | Add new categories to a working classifier without rebuilding it | New fraud type added to existing fraud model; new document category; new language added to existing intent model |
| Sequence and journey pattern detection | Detect order-dependent patterns — what happened, and in what order | Fraud sequence detection; user journey analysis (which paths precede churn); process mining to find the step order that leads to failure |
| Production deployment | Take a research model and ship it — compressed, fast, benchmarked, reliable | Any ML prototype that needs to become a product |
| Spec-driven code generation | LLM pipeline: spec → RAG → generate → assemble → test. Verifiably correct binary output | Automated code generation with quality gates; compliance-checkable output; any pipeline where "compiles and passes tests" replaces "looks right" |
| Wasm / binary security | Package and verify binary artifacts (SRI hashes, supply chain, size optimisation). Understand the attack surface — CDP exposes WASM linear memory the same way a Spectrum debugger exposes RAM | Shipping ML models or Wasm modules with integrity guarantees; security review of client-side WASM (licence checks, auth tokens, feature gates); defensive architecture that assumes the client is compromised |
| Synthetic data generation (diffusion) | Generate realistic synthetic data to augment imbalanced datasets, stress-test models, or satisfy privacy constraints — no real PII needed | Synthetic transaction data for fraud model training; rare-event augmentation (money laundering patterns too sparse to train on); privacy-safe test datasets for dev/QA; stress-test scenarios that haven't happened yet |

---

## Projects

| Project | What it is | Status |
|---------|-----------|--------|
| [Utala: KAOS 9](utala/kaos9/) | Card game AI — RL from scratch, TD learning, deep RL, model distillation | ✅ Complete |
| [Signal Hunt](signal-hunt/) | Audio classifier — data pipelines, CNN, training loop, inference, 100% test accuracy | ✅ Ph 1–3 complete |
| [Signal Hunt Ph 4](signal-hunt/) | Chord detection (multi-label) + chord progressions (CNN-RNN) | In progress — M18 next |
| [Signal Hunt Ph 5](signal-hunt/) | Weight initialisation experiments — Xavier vs He vs default, convergence analysis | Planned |
| [Acoustic Odyssey: cast](acoustic-odyssey/cast/) | Edge deployment — ONNX, TFLite, INT8 quantisation, sub-100ms on-device inference; Rust mobile runtime (native audio, rustfft, ort, UniFFI); model serving (REST API) | Planned |
| [Acoustic Odyssey: echo](acoustic-odyssey/echo/) | Adaptive RL system — MDP, Q-learning, personalises task difficulty to the user | Planned |
| [Artefact: dig](artefact/dig/) | ROM audio scanner — find hidden music/SFX in ROM dumps using autoencoders and anomaly detection | Planned |
| [Artefact: bloom](artefact/bloom/) | 8-bit audio restoration — super-resolution, denoising, lo-fi → hi-fi using U-Net and STFT loss | Planned |
| [Artefact: dream](artefact/dream/) | Generative chip-tunes — fine-tune sequence model on VGM/SID data, constraint-aware generation | Planned |
| [Artefact: drift](artefact/drift/) | Diffusion chip-tune generator — DDPM/DDIM on spectrograms, head-to-head with dream | Planned (after dream) |
| [Void Duel](void-duel/) | 2P space shooter on ZX Spectrum + WASM emulator — classic ML, multi-agent RL, extreme-edge distillation | Planned (Ph 0: PoC) |
| [zxw](zxw/) | ZX Spectrum 48K emulator — Rust → WASM, CDP debug integration, shared emulator for Void Duel + binary security research | ✅ M1–M7 complete |
| [Incant](incant/) | Spec-driven code gen — local LLM + RAG → WAT/WASM + Z80 assembly, ONNX → Z80 inference, binary protection PoC | ✅ M1–M10 complete |
| TPP | LLM engineering — RAG, MCP, deterministic evals, agent orchestration | 🔷 Complete (separate repo) |

---

## Skills Coverage

Technical reference. ✅ complete · planned (project named) · 🔷 separate project · 🔶 previous role · ⬜ gap

```
AI / ML
│
├── Classic ML ──────────────────────────────────── 🔶 prior role (~2004) + void-duel Ph 2
│   ├── Linear / logistic regression ────────────── void-duel Ph 2b
│   ├── Decision trees, random forests ──────────── void-duel Ph 2b
│   ├── Gradient boosting (XGBoost, LightGBM) ──── void-duel Ph 2b
│   ├── SVM ─────────────────────────────────────── void-duel Ph 2b
│   ├── k-means clustering ──────────────────────── void-duel Ph 2a
│   └── PCA ─────────────────────────────────────── void-duel Ph 2a
│
├── Deep Learning
│   │
│   ├── Foundations
│   │   ├── NumPy, PyTorch tensors ──────────────── ✅ signal-hunt Ph 1
│   │   ├── Forward pass / loss / backprop ─────── ✅ signal-hunt Ph 2
│   │   ├── Adam, LR scheduling, early stopping ── ✅ signal-hunt Ph 2
│   │   ├── Dropout, BatchNorm ──────────────────── ✅ signal-hunt Ph 2
│   │   └── Weight initialisation ───────────────── signal-hunt Ph 5
│   │
│   ├── Architectures
│   │   ├── CNN (Conv2d, pooling, GAP) ──────────── ✅ signal-hunt Ph 2
│   │   ├── RNN / GRU / LSTM ────────────────────── signal-hunt Ph 4
│   │   ├── Attention ───────────────────────────── signal-hunt Ph 4
│   │   ├── U-Net / encoder-decoder ─────────────── artefact/bloom
│   │   ├── Transformer ─────────────────────────── 🔷 TPP + artefact/dream
│   │   ├── Autoencoder ─────────────────────────── artefact/dig
│   │   ├── GAN ─────────────────────────────────── artefact/bloom (stretch)
│   │   ├── Diffusion ───────────────────────────── artefact/drift
│   │   └── Graph neural networks ───────────────── ⬜
│   │
│   ├── Training techniques
│   │   ├── Data augmentation ───────────────────── ✅ signal-hunt Ph 1
│   │   ├── Transfer learning + LoRA ────────────── ✅ signal-hunt Ph 3 + 🔷 TPP
│   │   ├── Curriculum learning ─────────────────── ✅ signal-hunt Ph 3
│   │   ├── Model distillation ──────────────────── ✅ utala
│   │   └── Self-supervised ─────────────────────── artefact/dig
│   │
│   ├── Generation
│   │   ├── Sequence generation (transformer/SSM) ── artefact/dream
│   │   ├── Constrained decoding ────────────────── artefact/dream
│   │   └── Style fine-tuning ───────────────────── artefact/dream
│   │
│   └── Evaluation
│       ├── Precision, recall, F1, confusion matrix ✅ signal-hunt Ph 2
│       ├── Loss curves / overfitting analysis ────── ✅ signal-hunt Ph 2
│       ├── ROC / AUC ───────────────────────────── artefact/dig
│       ├── STFT loss / perceptual audio metrics ──── artefact/bloom
│       └── Interpretability (SHAP, LIME) ─────────── void-duel Ph 2c
│
├── Domains
│   ├── Audio / signal processing ───────────────── ✅ signal-hunt
│   ├── Sequence / temporal ─────────────────────── signal-hunt Ph 4
│   ├── Computer vision ─────────────────────────── 🔶 NPR
│   ├── Binary / ROM data ───────────────────────── artefact/dig
│   ├── Audio restoration ───────────────────────── artefact/bloom
│   ├── Sequence generation (chip-tunes) ────────── artefact/dream
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
├── Code Generation
│   ├── Spec → assembly (WAT, Z80) ────────────── ✅ incant M1–M3
│   ├── RAG-grounded code gen ──────────────────── ✅ incant M1–M3
│   ├── Gate-driven iteration ──────────────────── ✅ incant M2–M3
│   ├── Multi-sigil orchestration ──────────────── ✅ incant M4
│   ├── Intent → spec → code pipeline ─────────── ✅ incant M6
│   ├── Unified test harness (WASM + Z80) ─────── ✅ incant M7
│   ├── Binary format read/write (TAP) ────────── ✅ incant M8
│   ├── Binary self-verification (Z80 + WASM) ─── ✅ incant M9
│   └── ONNX → INT8 → Z80 inference engine ────── ✅ incant M10
│
├── Systems / Emulation
│   ├── Z80 CPU emulation ──────────────────────── ✅ zxw M2, M6
│   ├── Rust → WASM compilation ────────────────── ✅ zxw M1–M7
│   ├── Browser/WASM interop (wasm-bindgen) ────── ✅ zxw M1, M3–M5
│   ├── CDP / browser debug protocol ──────────── ✅ zxw M7
│   └── Emulator architecture ──────────────────── ✅ zxw M2–M6
│
└── Production / MLOps
    ├── Inference pipeline ──────────────────────── ✅ signal-hunt Ph 2
    ├── ONNX / TFLite / INT8 / edge ────────────── 🔶 Aerythen demo (ONNX → Flutter) + acoustic-odyssey/cast
    ├── Latency benchmarking ────────────────────── acoustic-odyssey/cast
    ├── Real-time audio pipeline (CoreAudio/Oboe) ─ acoustic-odyssey/cast
    ├── Native mobile inference (Rust + ort) ────── acoustic-odyssey/cast
    ├── Cross-platform FFI (UniFFI → Swift/Kotlin)─ acoustic-odyssey/cast
    ├── Model serving / REST API ────────────────── acoustic-odyssey/cast
    ├── Model monitoring / drift ────────────────── ⬜
    └── A/B testing models ──────────────────────── ⬜
```

---

## External Projects

Work done outside this repo that contributes to the skills coverage above.

| Project | What it covers | Link |
|---------|---------------|------|
| TPP | LLM engineering — RAG, MCP, deterministic evals, agent orchestration, Transformer architecture | Separate repo |
| Aerythen Demo | ONNX model distilled into Flutter mobile app — on-device inference, edge deployment | [aerythen.com/demo](https://aerythen.com/demo) |
