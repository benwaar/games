# Games AI Lab

Hands-on machine learning and game AI projects — reinforcement learning, deep learning, audio classification, neural networks, and model distillation. Built from scratch in Python with PyTorch.

## Development Ethos

These projects are about:
- understanding algorithms by building them
- seeing how different approaches *think*
- learning by stripping problems down to their **nuts and bolts**

Everything is text-based, inspectable, and hackable.

---

## Projects

- [Utala: KAOS 9](utala/kaos9/) — card game AI: TD learning, deep RL, distillation ✅
- [Signal Hunt](signal-hunt/) — audio classifier: data pipelines, CNN, transfer learning, chords, CNN-RNN ✅
- [Acoustic Odyssey: cast](acoustic-odyssey/cast/) — edge deployment: ONNX, quantisation, sub-100ms on-device inference
- [Acoustic Odyssey: echo](acoustic-odyssey/echo/) — adaptive RL: personalises task difficulty to the user
- [Artefact: dig](artefact/dig/) — binary art scanner: find hidden sprites in ROM using CV and autoencoders
- [Artefact: bloom](artefact/bloom/) — pixel art upscaler: super-resolution CNN and GAN
- [Artefact: dream](artefact/dream/) — LoRA fine-tune a diffusion model on extracted sprites
- [Void Duel](void-duel/) — ZX Spectrum shooter: classic ML, multi-agent RL, extreme-edge distillation
- [Incant](incant/) — spec-driven code gen: local LLM + RAG → WAT/WASM + Z80 assembly 🔨

---

## Utala: KAOS 9 

A competitive 2-player tactical duel playable with any standard 52-card deck.

AI research studying skill expression, risk management, and learning algorithms — from random baselines through hand-built TD learning to deep reinforcement learning and model distillation.

[Utala KAOS 9 Rules (PDF)](utala-kaos-9-rules.pdf) · [Project folder](./utala/kaos9/)

### Explore the Aerythen Universe

Utala: KAOS 9 contains artificial intelligence code for the card game in the [Aerythen](https://aerythen.com) universe — a Command Line Punk world created by [David Benoy](https://aerythen.com).

The overarching project bridges art, music, AI and a developing series of companion novels and games.

* 🎮 **Play the demo:** [Aerythen Interactive Demo](https://aerythen.com/demo)
* 🎵 **Listen to the Soundtrack:** [Aerythen Soundtrack](https://song.link/jtmzgwknhjt8g)
* 📖 **Read the novel:** [Utala — An Aerythen Novel](https://mybook.to/utala)

---

## Signal Hunt

A deep learning project — from raw audio to a trained classifier.

Turn raw audio (hums, whistles, claps) into clean Mel-spectrogram tensors, then train a hybrid CNN-RNN to classify them.

[Project folder](./signal-hunt/)

---

## License

Source code is licensed under the MIT License.

Aerythen, the associated artwork, the game names, rulebook text, and branding are © 2026 David Benoy. All rights reserved.
