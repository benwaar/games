# bloom — 8-Bit Audio Restoration

Take degraded, lo-fi chip audio — noisy samples, bit-crushed waveforms, low-bitrate
tracker modules — and restore them to clean hi-fi. Audio super-resolution and denoising
for retro game sound.

---

## The idea

8-bit audio has character, but it's also constrained: low sample rates, quantisation noise,
hardware noise floors, and lossy compression in tracker formats. Standard resampling
(NEAREST, LINEAR) either keeps the crunch or introduces artifacts.

A trained encoder-decoder learns what clean audio looks like in spectrogram space and
reconstructs it from the degraded input — similar to how speech enhancement and audio
source separation models work.

bloom takes whatever dig finds and restores it to usable quality.

---

## Milestones

### M1 — Dataset

- [ ] Collect paired audio: clean reference recordings of chip music (emulated at high quality)
  alongside degraded versions (low sample rate, added noise, hardware-accurate artifacts)
- [ ] Sources: VGM archives rendered at multiple quality levels, HVSC SID collection,
  tracker modules (.mod, .xm) at original vs upsampled rates
- [ ] Generate training pairs: clean → degraded (downsample, add noise, quantise)
- [ ] Represent as Mel-spectrograms or STFT magnitude/phase

**Gate:** DataLoader yields `(degraded_spectrogram, clean_spectrogram)` pairs. Shapes correct.

**Docs:**
- [ ] Explainer: audio super-resolution as a reconstruction task — why paired data matters

---

### M2 — Baselines

- [ ] Implement standard resampling: nearest-neighbour, linear interpolation, sinc resampling
- [ ] Score each with STFT-based metrics: spectral convergence, log-spectral distance
- [ ] Listen test: compare side-by-side — these are the bar the model must beat

**Gate:** Baseline scores documented. Audio comparisons saved.

**Docs:**
- [ ] Explainer: audio quality metrics — STFT loss, spectral convergence, why SNR alone misleads

---

### M3 — U-Net encoder-decoder (MSE loss)

- [ ] Build a U-Net style encoder-decoder operating on spectrograms
- [ ] Train with MSE loss on spectrogram reconstruction
- [ ] Compare metrics against baselines
- [ ] Griffin-Lim or learned vocoder for spectrogram → waveform

**Gate:** Model beats linear resampling on spectral metrics. Output sounds cleaner.

**Docs:**
- [ ] Explainer: U-Net / encoder-decoder for audio — skip connections preserve detail
- [ ] Note: why MSE on spectrograms produces "average" reconstructions

---

### M4 — Perceptual / STFT loss

- [ ] Replace MSE with multi-resolution STFT loss: compare spectral content at multiple
  time-frequency resolutions simultaneously
- [ ] Perceptual loss preserves transients, harmonics, and texture that MSE smooths away
- [ ] Compare outputs: MSE vs STFT loss — audible difference in clarity

**Gate:** STFT-loss output audibly sharper than MSE. Documented with before/after spectrograms.

**Docs:**
- [ ] Explainer: perceptual audio loss — why frequency-domain distance beats time-domain MSE
  for audio quality

---

### M5 — GAN (stretch)

- [ ] Add a discriminator: "does this restored audio sound like real clean chip music?"
- [ ] Generator (the restorer) vs discriminator — adversarial training
- [ ] Compare GAN output against STFT-loss output

**Gate:** GAN output passes a listening test — sounds like clean chip audio, not filtered noise.

**Docs:**
- [ ] Explainer: audio GANs — generator vs discriminator for audio enhancement

---

### M6 — CLI tool + pipeline

- [ ] `python -m bloom restore sample.wav --quality high`
- [ ] `python -m bloom restore candidates/ --quality high` → processes folder from dig
- [ ] Demo: run bloom on audio found by dig from a real game file

**Gate:** Full pipeline works: `dig scan` → `bloom restore`. Output audio is audibly cleaner.

**Docs:**
- [ ] README: usage, quality settings, how to feed from dig

---

## Skills this covers

| Skill | Gap filled |
|-------|-----------|
| Audio restoration / enhancement | ⬜ → ✅ |
| STFT-based perceptual loss | ⬜ → ✅ |
| GAN (stretch) | ⬜ → ✅ |
| Encoder-decoder / U-Net architecture | ⬜ → ✅ |

---

## Feeds from

[dig](../dig/) — pass candidate audio from dig scan directly into bloom restore.

## Feeds into

[dream](../dream/) — cleaned audio serves as higher-quality training data for style learning.
