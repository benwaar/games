# drift — Diffusion Chip-Tune Generator

Generate new 8-bit chip-tunes using a denoising diffusion model operating on spectrograms.
Start from pure noise, iteratively denoise into authentic chip music.

A direct comparison to dream: same domain, same training data, fundamentally different
generative architecture. dream generates register sequences autoregressively; drift
generates spectrograms by reversing a noise process.

---

## The idea

Diffusion models learn to reverse noise. During training, you gradually corrupt real
spectrograms with Gaussian noise over many steps, then train a U-Net to predict the
noise at each step. At generation time, you start from pure static and denoise step by
step until a coherent spectrogram emerges.

The advantage over sequence models: diffusion can capture global structure (overall
timbre, frequency balance, harmonic relationships) that autoregressive models struggle
with because they only see left-context. The disadvantage: no discrete register-level
control, and generation is slow (many denoising steps).

Comparing drift vs dream on the same chip-tune corpus is the point — when does each
architecture win, and why?

---

## Prerequisites

- artefact/dream M1 complete — tokenised chip-tune corpus and spectrogram rendering
- Shared explainers: training loops, loss functions, U-Net architecture (from bloom)

---

## Milestones

### M1 — Spectrogram dataset

- [ ] Render chip-tune corpus (VGM/SID files from dream's dataset) as Mel-spectrograms
- [ ] Fixed-length segments: 4-bar or 8-bar chunks, padded/truncated to uniform shape
- [ ] Normalise to [-1, 1] range (standard for diffusion)
- [ ] Train/val/test split, stratified by chip type or composer

**Gate:** DataLoader yields `(spectrogram,)` tensors at uniform shape. Coverage across chip types.

**Docs:**
- [ ] Explainer: diffusion models — the forward process (add noise), the reverse process (denoise), why it works
- [ ] Explainer: noise schedules — linear vs cosine, how the schedule affects generation quality

---

### M2 — DDPM baseline

- [ ] Implement Denoising Diffusion Probabilistic Model (DDPM) with a small U-Net backbone
- [ ] Train on chip-tune spectrograms with linear noise schedule
- [ ] Generate samples: denoise from pure Gaussian noise over 1000 steps
- [ ] Convert generated spectrograms to audio via Griffin-Lim
- [ ] Listen test: does the output sound like structured music, or just filtered noise?

**Gate:** Generated audio has recognisable pitch structure and rhythm. Not just noise.

**Docs:**
- [ ] Explainer: DDPM — the maths in plain terms (forward process, reverse process, loss function)
- [ ] Note: why 1000 steps — the quality/speed tradeoff

---

### M3 — Improved sampling (DDIM)

- [ ] Implement DDIM (Denoising Diffusion Implicit Models) for faster sampling
- [ ] Compare: 1000-step DDPM vs 50-step DDIM — quality vs speed
- [ ] Benchmark: generation time per sample for each method

**Gate:** DDIM produces comparable quality in 10–20× fewer steps. Documented with spectrograms and audio.

**Docs:**
- [ ] Explainer: DDIM — deterministic sampling, why fewer steps work, the connection to ODEs

---

### M4 — Conditional generation (chip type)

- [ ] Add class conditioning: generate "in the style of AY" vs "in the style of SID"
- [ ] Classifier-free guidance: train with random label dropout, scale guidance at inference
- [ ] Compare: unconditional vs chip-conditioned outputs — does conditioning improve authenticity?

**Gate:** Conditioned outputs are distinguishably different per chip type. A/B comparison documented.

**Docs:**
- [ ] Explainer: classifier-free guidance — how dropping labels during training enables guided generation
- [ ] Note: conditioning signals for audio — chip type, tempo, key, composer

---

### M5 — Head-to-head: drift vs dream

- [ ] Generate 20 samples each from drift (diffusion) and dream (sequence model)
- [ ] Same chip type, same corpus, same evaluation criteria
- [ ] Compare: musical coherence, chip authenticity, generation speed, controllability
- [ ] Document: when each architecture wins and why

**Gate:** Documented comparison with audio samples, spectrograms, and analysis. Clear conclusion on trade-offs.

**Docs:**
- [ ] Explainer: diffusion vs autoregressive generation — architectural trade-offs for audio
- [ ] Summary: which approach for which use case

---

### M6 — CLI + demo

- [ ] `python -m drift generate --chip ay --bars 8 --steps 50`
- [ ] `python -m drift generate --chip sid --bars 16` → outputs .wav + spectrogram
- [ ] `python -m drift compare --with dream` → generates matched samples from both models
- [ ] Demo: generate a track, play it alongside a dream-generated track

**Gate:** CLI generates playable chip-tunes from a cold clone. Comparison demo works.

**Docs:**
- [ ] README: usage, chip types, step count vs quality, how to compare with dream

---

## Skills this covers

| Skill | Gap filled |
|-------|-----------|
| Diffusion models (DDPM) | ⬜ → ✅ |
| Noise schedules | ⬜ → ✅ |
| DDIM / fast sampling | ⬜ → ✅ |
| Classifier-free guidance | ⬜ → ✅ |
| Generative model comparison | ⬜ → ✅ |

---

## Feeds from

[dream](../dream/) — shares the chip-tune corpus and spectrogram rendering pipeline.
[bloom](../bloom/) — U-Net architecture and audio quality metrics carry across.
