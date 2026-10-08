# Games AI Lab

Hands-on ML and game AI — built from scratch, phase by phase. Each project is a working system, not a tutorial exercise.

→ [STUDY.md](STUDY.md) — capability map and business alignment

---

## Contents

- [Up next](#up-next)
  - [Acoustic Odyssey](#acoustic-odyssey)
  - [Artefact](#artefact)
  - [Void Duel](#void-duel)
- [Complete](#complete)
  - [Signal Hunt](#signal-hunt)
  - [Utala: KAOS 9](#utala-kaos-9)

---

## Up next

### Acoustic Odyssey

Deploy Signal Hunt's trained models into the real world — on-device inference and an adaptive difficulty system.

| Project | What | Status |
|---------|------|--------|
| [cast](acoustic-odyssey/cast/) | Export to ONNX/TFLite, INT8 quantisation, sub-100ms on-device inference | Planned |
| [echo](acoustic-odyssey/echo/) | RL agent that adapts task difficulty to the user in real time | Planned |

Depends on Signal Hunt Phase 4 ✅

---

### Artefact

Extract hidden sprites from ROM/memory using computer vision, upscale them, and generate new pixel art in the same style using a fine-tuned diffusion model.

| Project | What | Status |
|---------|------|--------|
| [dig](artefact/dig/) | Binary art scanner — find hidden sprites using CV and autoencoders | Planned |
| [bloom](artefact/bloom/) | Pixel art upscaler — super-resolution CNN and GAN | Planned |
| [dream](artefact/dream/) | LoRA fine-tune a diffusion model on extracted sprites | Planned |

---

### Void Duel

2-player space shooter on a ZX Spectrum emulator (WASM). Classic ML baselines → multi-agent RL → extreme-edge distillation into a model that runs in 48KB.

[void-duel/](void-duel/) · Planned (Phase 0: PoC)

---

## Complete

### Signal Hunt

Audio classifier — from raw audio to a piano teacher prototype, phase by phase.

| Phase | What | Result |
|-------|------|--------|
| 1 | Data pipeline (ingest, augment, Mel-spectrograms) | `(1,128,65)` tensors |
| 2 | 3-class CNN (hum / whistle / clap) | 100% test accuracy |
| 3 | 12-class piano note classifier, transfer learning | 89.5% test accuracy |
| 4a | Chord detection — two-model pipeline | 96.2% name / 73.1% exact-match |
| 4b | Chord progressions, CNN-RNN | 66.7% on 4 classes |

```bash
bash demo_chord.sh chord.wav --expected Cmaj
# Chord:   Amin  ✗  (expected Cmaj)
# Notes:   A4 ✓  C4 ✓  E4 ✓
# Missing: G4
```

[signal-hunt/](signal-hunt/) · [Log](signal-hunt/LOG.md) · [Explainers](signal-hunt/explainers/README.md)

---

### Utala: KAOS 9

Card game AI — one engine, one harness, progressively sophisticated agents.

| Phase | What | Result |
|-------|------|--------|
| 1 | Engine, replay, baselines | Heuristic 65% vs Random — game worth studying |
| 2 | TD-linear, hand-built gradients | 47.5% vs Heuristic (27 weights) |
| 3 | DQN, distillation (original rules) | DQN failed; TinyNN imitation 48% |
| 4 | Variant A rules, DQN retrained | DQN 53% peak — deep learning justified |
| 5 | Better DQN → TinyNN v2, Flutter | 48% shipped at 5,727 params |

[utala/kaos9/](utala/kaos9/) · [Log](utala/kaos9/LOG.md) · [Explainers](utala/kaos9/explainers/README.md) · [Rules (PDF)](utala-kaos-9-rules.pdf)

Part of the [Aerythen](https://aerythen.com) universe — [play the demo](https://aerythen.com/demo) · [read the novel](https://mybook.to/utala)

---

## License

Source code: MIT License.  
Aerythen, artwork, game names, rulebook text, and branding: © 2026 David Benoy. All rights reserved.
