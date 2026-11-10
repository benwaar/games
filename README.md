# Games AI Lab

Hands-on machine learning and game AI projects.

## Development Ethos

These projects are about:
- understanding algorithms by building them
- seeing how different approaches *think*
- learning by stripping problems down to their **nuts and bolts**

Everything is inspectable and hackable.

---

## Projects

| # | Project | What it builds | Depends on | Status |
|---|---------|---------------|------------|--------|
| 1 | [Utala: KAOS 9](utala/kaos9/) | Card game AI: TD learning → deep RL → distillation (Flutter, WASM, Z80) | — | ✅ |


---

## Utala: KAOS 9 

A competitive 2-player tactical duel playable with any standard 52-card deck.

`TD learning` · `deep RL` · `model distillation` · `TinyNN` · `WASM` · `INT8 quantisation`

AI research studying skill expression, risk management, and learning algorithms — from random baselines through hand-built TD learning to deep reinforcement learning and model distillation. **TinyNN v2 wins 54% against the hand-crafted heuristic** — the first learned agent to consistently beat the baseline — and deploys to three targets: Flutter (mobile), WASM (browser, 6.5 µs/inference), and Z80 assembly (ZX Spectrum, INT8 quantised, 100% argmax agreement with float32).

[One Brain, Three Runtimes](utala/kaos9/docs/DISTILLATION_TARGETS.md) · [Utala KAOS 9 Rules (PDF)](utala-kaos-9-rules.pdf) · [Project folder](./utala/kaos9/)

### Explore the Aerythen Universe

Utala: KAOS 9 contains artificial intelligence code for the card game in the [Aerythen](https://aerythen.com) universe — a Command Line Punk world created by [David Benoy](https://aerythen.com).

The overarching project bridges art, music, AI and a developing series of companion novels and games.

* 🎮 **Play the demo:** [Aerythen Interactive Demo](https://aerythen.com/demo)
* 🎵 **Listen to the Soundtrack:** [Aerythen Soundtrack](https://song.link/jtmzgwknhjt8g)
* 📖 **Read the novel:** [Utala — An Aerythen Novel](https://mybook.to/utala)

---


## License

Source code is licensed under the MIT License.

Aerythen, the associated artwork, the game names, rulebook text, and branding are © 2026 David Benoy. All rights reserved.
