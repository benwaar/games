# AI/ML Capability Development

A personal R&D programme building practical AI and machine learning capabilities — from data pipelines and model training through to production deployment and adaptive systems. Each project is a working system, not a tutorial exercise.

The focus is applied: every skill maps to a class of real problem. The business alignment section below shows where each capability lands.

---

## Business Alignment

**ML / AI** — Pattern recognition in audio, documents, images, and transaction data. On-device inference via ONNX/Rust (zero-latency, no cloud cost). Adaptive systems that learn from individual users. Sequence detection for fraud and journey analysis. Synthetic data generation for privacy-safe training. LLM tooling with quality gates and multi-agent orchestration. The skills tree below maps each technique to the project that teaches it.

**WASM, security & accessibility** — Binary packaging with integrity verification (SRI, supply chain). WASM linear memory inspection via CDP — the same attack surface as client-side licence checks, auth tokens, and feature gates. Spec-driven code generation with automated test gates. Web accessibility for canvas-based content: mirror DOM, preference bridging, palette override, spatial audio — transferable to any dashboard or data visualisation that needs WCAG compliance.

---

## Projects

| Project | What it is | Status |
|---------|-----------|--------|
| [KAOS 9](utala/kaos9/) | Card game AI — RL from scratch, TD learning, deep RL, model distillation | ✅ Complete |
| [Signal Hunt](signal-hunt/) | Audio classifier — data pipelines, CNN, transfer learning, multi-label chords, CNN-RNN progressions, browser inference | ✅ Complete (Ph 1–5) |
| [Acoustic Odyssey: cast](acoustic-odyssey/cast/) | Edge deployment — ONNX, TFLite, INT8 quantisation, sub-100ms on-device inference; Rust mobile runtime (native audio, rustfft, ort, UniFFI); model serving (REST API) | ▶ Ready |
| [Acoustic Odyssey: echo](acoustic-odyssey/echo/) | Adaptive RL system — MDP, Q-learning, personalises task difficulty to the user | ▶ Ready |
| [Artefact: dig](artefact/dig/) | ROM audio scanner — find hidden music/SFX in ROM dumps using autoencoders and anomaly detection | ▶ Ready |
| [Artefact: bloom](artefact/bloom/) | 8-bit audio restoration — super-resolution, denoising, lo-fi → hi-fi using U-Net and STFT loss | ▶ Ready |
| [Artefact: dream](artefact/dream/) | Generative chip-tunes — fine-tune sequence model on VGM/SID data, constraint-aware generation | ▶ Ready |
| [Artefact: drift](artefact/drift/) | Diffusion chip-tune generator — DDPM/DDIM on spectrograms, head-to-head with dream | Planned (after dream) |
| [zxw](zxw/) | ZX Spectrum 48K emulator — Rust → WASM, CDP debug integration, shared emulator for Void Duel + binary security research | ✅ M1–M7 complete |
| [Incant](incant/) | Spec-driven code gen — local LLM + RAG → WAT/WASM + Z80 assembly, ONNX → Z80 inference, binary protection PoC | ✅ M1–M10 complete |
| TPP | LLM engineering — RAG, MCP, deterministic evals, agent orchestration | 🔷 Complete (separate repo) |
| [wex](wex/) | WASM executor — static HTML player for pipeline output, responsive canvas, a11y (mirror DOM, settings panel, palette presets) | ▶ Active |
| [Glyph](glyph/) | LLM-first presentation layer — cross-platform HAL, image-to-SCR converter, prompt-driven art pipeline, a11y (mirror DOM, preference imports, settings panel) | ▶ POC + a11y Phase 1–2 complete |
| [Void Duel](void-duel/) | 2P space shooter on ZX Spectrum + WASM emulator — classic ML, multi-agent RL, extreme-edge distillation | Planned (Ph 0: PoC) |

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
│   ├── k-means clustering ──────────────────────── ✅ glyph POC + void-duel Ph 2a
│   └── PCA ─────────────────────────────────────── void-duel Ph 2a
│
├── Deep Learning
│   │
│   ├── Foundations
│   │   ├── NumPy, PyTorch tensors ──────────────── ✅ signal-hunt Ph 1
│   │   ├── Forward pass / loss / backprop ─────── ✅ signal-hunt Ph 2
│   │   ├── Adam, LR scheduling, early stopping ── ✅ signal-hunt Ph 2
│   │   ├── Dropout, BatchNorm ──────────────────── ✅ signal-hunt Ph 2
│   │   └── Weight initialisation ───────────────── ✅ signal-hunt Ph 5
│   │
│   ├── Architectures
│   │   ├── CNN (Conv2d, pooling, GAP) ──────────── ✅ signal-hunt Ph 2
│   │   ├── RNN / GRU / LSTM ────────────────────── ✅ signal-hunt Ph 4b
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
│   ├── Sequence / temporal ─────────────────────── ✅ signal-hunt Ph 4b
│   ├── Computer vision ─────────────────────────── 🔶 NPR + ✅ glyph POC (colour quantisation)
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
├── LLM Engineering
│   ├── Embeddings (sentence-transformers, ollama) ── ✅ knowledge corpus + incant
│   ├── Vector search (FAISS) ─────────────────────── ✅ knowledge corpus
│   ├── RAG pipeline ──────────────────────────────── ✅ incant + 🔷 TPP
│   ├── MCP server (custom, Rust) ─────────────────── ✅ knowledge-mcp
│   ├── Skills / CLI tools ────────────────────────── ✅ /asm skill
│   ├── Deterministic evals ───────────────────────── 🔷 TPP
│   └── Agent orchestration ────────────────────────── 🔷 TPP + ✅ /forge skill (this repo)
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
├── Web Accessibility (a11y)
│   ├── Mirror DOM (canvas → screen reader) ────── ✅ glyph/wasm a11y Ph 1
│   ├── ARIA roles, live regions, sr-only ──────── ✅ glyph/wasm a11y Ph 1
│   ├── Browser preference imports to WASM ─────── ✅ glyph/wasm a11y Ph 1
│   ├── Settings panel + palette override ──────── ✅ glyph/wasm a11y Ph 2
│   ├── Self-voicing (Web Speech API) ──────────── glyph/wasm a11y Ph 3
│   └── Spatial audio (PannerNode) ─────────────── glyph/wasm a11y Ph 3
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
