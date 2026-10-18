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

Ordered by dependency — each project builds on capabilities from earlier ones.

| # | Project | What it builds | Depends on | Status |
|---|---------|---------------|------------|--------|
| 1 | [Utala: KAOS 9](utala/kaos9/) | Card game AI: TD learning → deep RL → distillation (Flutter, WASM, Z80) | — | ✅ |
| 2 | [Signal Hunt](signal-hunt/) | Audio classifier: data pipelines → CNN → transfer learning → chords | — | ✅ |


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

## License

Source code is licensed under the MIT License.

Aerythen, the associated artwork, the game names, rulebook text, and branding are © 2026 David Benoy. All rights reserved.
