# Games AI Lab

Hands-on machine learning and game AI projects — reinforcement learning, deep learning, audio classification, neural networks, and model distillation. Built from scratch in Python with PyTorch.

## Development Ethos

These projects are about:
- understanding algorithms by building them
- seeing how different approaches *think*
- learning by stripping problems down to their **nuts and bolts**

Everything is inspectable and hackable.

---

## Projects

- [Utala: KAOS 9](utala/kaos9/) — card game AI: TD learning, deep RL, distillation → Flutter, WASM, Z80 ✅
- [Signal Hunt](signal-hunt/) — audio classifier: data pipelines, CNN, transfer learning, chords, CNN-RNN ✅
- [Acoustic Odyssey: cast](acoustic-odyssey/cast/) — edge deployment: ONNX, quantisation, sub-100ms on-device inference
- [Acoustic Odyssey: echo](acoustic-odyssey/echo/) — adaptive RL: personalises task difficulty to the user
- [Artefact: dig](artefact/dig/) — ROM audio scanner: find hidden music/SFX in ROM dumps using autoencoders
- [Artefact: bloom](artefact/bloom/) — 8-bit audio restoration: lo-fi → hi-fi using U-Net and STFT loss
- [Artefact: dream](artefact/dream/) — generative chip-tunes: sequence model fine-tuned on VGM/SID data
- [Artefact: drift](artefact/drift/) — diffusion chip-tune generator: DDPM/DDIM on spectrograms, head-to-head with dream
- [zxw](zxw/) — ZX Spectrum 48K emulator: Rust → WASM + CDP debug integration ✅
- [Incant](incant/) — spec-driven code gen: local LLM + RAG → WAT/WASM + Z80 assembly ✅
- [wex](wex/) — WASM executor: responsive player, a11y (mirror DOM, settings panel, palette presets) ▶
- [Glyph](glyph/) — LLM-first presentation layer: same spec → WASM (full colour) + Z80 (Spectrum constraints) ▶
- [Void Duel](void-duel/) — ZX Spectrum shooter: classic ML, multi-agent RL, extreme-edge distillation


---

## Utala: KAOS 9 

A competitive 2-player tactical duel playable with any standard 52-card deck.

AI research studying skill expression, risk management, and learning algorithms — from random baselines through hand-built TD learning to deep reinforcement learning and model distillation. **TinyNN v2 wins 54% against the hand-crafted heuristic** — the first learned agent to consistently beat the baseline — and deploys to three targets: Flutter (mobile), WASM (browser, 6.5 µs/inference), and Z80 assembly (ZX Spectrum, INT8 quantised, 100% argmax agreement with float32).

[One Brain, Three Runtimes](utala/kaos9/docs/DISTILLATION_TARGETS.md) · [Utala KAOS 9 Rules (PDF)](utala-kaos-9-rules.pdf) · [Project folder](./utala/kaos9/)

### Explore the Aerythen Universe

Utala: KAOS 9 contains artificial intelligence code for the card game in the [Aerythen](https://aerythen.com) universe — a Command Line Punk world created by [David Benoy](https://aerythen.com).

The overarching project bridges art, music, AI and a developing series of companion novels and games.

* 🎮 **Play the demo:** [Aerythen Interactive Demo](https://aerythen.com/demo)
* 🎵 **Listen to the Soundtrack:** [Aerythen Soundtrack](https://song.link/jtmzgwknhjt8g)
* 📖 **Read the novel:** [Utala — An Aerythen Novel](https://mybook.to/utala)

---

## Signal Hunt

A deep learning project — from raw audio to a live browser piano classifier.

`CNN` · `transfer learning` · `Mel spectrogram` · `multi-label` · `CNN-RNN` · `ONNX` · `WASM inference`

Turn raw audio into Mel-spectrogram tensors and build up from sound-type classification through piano note detection, chord recognition, and chord progressions. The trained model ships to the browser — real-time inference via `onnxruntime-web`, ~12 ms end-to-end, no server required.

**Highlights:** 89.5% piano note accuracy · 96.2% chord name accuracy · live browser demo (`demo/index.html`) · real-time sliding-window inference

[Project folder](./signal-hunt/)

---

## Incant

Write a plain-English spec, get a working binary — for both WASM and Z80.

An LLM pipeline that reads BDD-format specs, decomposes them into function graphs, generates assembly (WAT or Z80), assembles it, and verifies it against tests. RAG-grounded with instruction set knowledge. Also includes ONNX → INT8 → Z80 inference engine, Spectrum TAP file tools, and binary protection PoC (self-verifying code on both architectures).

```bash
# BDD spec → WASI binary (WAT)
bash incant/run_wasm_greet.sh        # "Type a name:" → hello Ben

# Same spec → Z80 machine code (42 bytes)
bash incant/run_z80_greet.sh         # "Type a name:" → hello Ben
```

[Project folder](./incant/)

---

## zxw

A ZX Spectrum 48K emulator written in Rust, compiled to WebAssembly. Built from scratch — small enough to understand completely.

A platform for exploring binary protection (patching live memory via the Chrome DevTools Protocol debug API), deploying trained AI models onto retro hardware, and understanding what computation actually costs when you strip away modern abstractions. The KAOS 9 card game AI (5,727 parameters, INT8 quantised) runs inference on it — 6M T-states for what WASM does in microseconds.

**Highlights:** live [Z80 tamper demo](zxw/www/demo-protection.html) — 26-byte XOR verifier prefix catches a single flipped bit, crashes to HALT · live [WASM tamper demo](zxw/www/demo-wasm-protect.html) — `unreachable` trap fires on checksum mismatch · full [writeup](zxw/docs/binary-protection-demos.md) with bypass analysis

[Project folder](./zxw/)

---

## Tooling

Development tools that support the projects — shared knowledge infrastructure, browser harnesses, and editor integration.

| Tool | What it does |
|------|-------------|
| [knowledge-mcp](knowledge-mcp/) | Rust MCP server — gives Claude searchable access to the Z80/WASM knowledge corpus |
| [wex](wex/) | WASM player — static HTML harness for testing `.wasm` output in the browser |
| [zxw harness](zxw/www/) | Spectrum TAP harness — loads `.tap` files into the emulator, zero-click debug mode |
| [/asm skill](.claude/skills/asm/) | CLI skill — browse and search the knowledge corpus from the terminal |
| VS Code extensions | Syntax highlighting and snippets for Z80 assembly and WAT |

The knowledge corpus has three access layers — all reading from the same `knowledge/chunks/*.md` files:
- **knowledge-mcp** → Claude (MCP, keyword index)
- **/asm skill** → user (CLI, browsable)
- **FAISS embeddings** → local LLM (incant RAG, semantic search)

---

## License

Source code is licensed under the MIT License.

Aerythen, the associated artwork, the game names, rulebook text, and branding are © 2026 David Benoy. All rights reserved.
