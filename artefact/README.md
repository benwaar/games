# Artefact

Digital archaeology for 8-bit audio.

**dig** finds hidden music and sound data buried in ROM dumps and memory banks.
**bloom** takes degraded chip audio and restores it to clean hi-fi.
**dream** generates new chip-tunes in the style of a specific composer or sound chip.
**drift** generates chip-tunes using diffusion — a head-to-head comparison with dream.

Together they form a pipeline: scan an old game file for audio data, restore it, then generate new material in the same style.

---

## The problem

Old game ROMs pack music into register streams (AY-3-8910, SID, YM2149), tracker patterns, or raw PCM samples — no headers, no labels. Finding the music data means knowing the chip architecture or scanning by hand. Extracting it cleanly means dealing with noise floors, bit-crushed samples, and hardware-specific envelope shapes.

**dig** automates the scanning with unsupervised feature extraction.
**bloom** restores and enhances what dig finds.
**dream** learns a composer's style and generates new tracks.

---

## Projects

| Project | What it does | Key techniques |
|---------|-------------|----------------|
| [dig](dig/) | Scan ROMs/memory dumps for music data, sound effects, unindexed PCM samples | Autoencoder, anomaly detection, binary data feature extraction |
| [bloom](bloom/) | Audio super-resolution — clean noise floors, upsample lo-fi chip audio, restore degraded samples | U-Net encoder-decoder, STFT-based perceptual loss, GAN |
| [dream](dream/) | Generate new chip-tunes in the style of a specific sound chip or composer | Sequence model (transformer/SSM), fine-tuning, constraint-aware generation |
| [drift](drift/) | Generate chip-tunes via diffusion — denoise spectrograms from pure noise | DDPM, DDIM, classifier-free guidance, generative model comparison |

---

## Pipeline

```
game.tap / game.z80 / .sid / .vgm
     ↓
  dig scan
     ↓
ranked audio regions → sample_001.wav, music_data_002.bin ...
     ↓
  bloom restore
     ↓
sample_001_clean.wav ...
     ↓
  dream generate --style=<composer/chip>
     ↓
new_track_001.vgm ...
```

---

## Why audio, not pixels?

The original Artefact plan targeted pixel art (sprites, tile sets). The pivot to 8-bit audio:

- **Builds on Signal Hunt** — reuses FFT, spectrogram, and audio pipeline expertise instead of context-switching to 2D spatial operations
- **Hardware constraints are a training signal** — PSG chips have strict polyphony limits, hard envelope shapes, and register constraints. Models must respect these, adding a constraint-validation layer
- **Same ML techniques, richer domain** — autoencoders, U-Nets, GANs, and sequence models all apply, but in a domain where the user already has deep feature extraction tooling

---

## Status

- dig: planned
- bloom: planned
- dream: planned
- drift: planned (after dream — reuses its data pipeline)

See each project's PLAN.md for milestones and approach.
