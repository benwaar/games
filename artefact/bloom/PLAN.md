# bloom — Pixel Art Upscaler

Take a low-resolution pixel art sprite — 8×8, 16×16, 32×32 — and scale it to HD
without blurring, without interpolating, preserving the crisp pixel aesthetic.

---

## The idea

Standard upscaling (NEAREST, BILINEAR, LANCZOS) either stays blocky or blurs edges.
Neither looks right. A CNN trained on pixel art learns the style — when to keep a hard
edge, when to anti-alias, what a diagonal pixel boundary should look like at 4×.

bloom takes the sprites dig finds and makes them usable — for game remasters,
Aerythen assets, or anything else built from retro source material.

---

## Milestones

### M1 — Dataset

- [ ] Collect public-domain pixel art sprites: ZX Spectrum, C64, NES, Game Boy
  (many available via OpenGameArt, LibreSprite, public-domain ROM rips)
- [ ] Generate pairs: original sprite at 1× (ground truth), downscaled version (model input)
  — train the model to reconstruct the original from the downscaled version
- [ ] Augment: random crops, flips, palette swaps

**Gate:** DataLoader yields `(low_res, high_res)` pairs. Both normalised. Shapes correct.

**Docs:**
- [ ] Explainer: super-resolution as a reconstruction task — why pairs beat single images

---

### M2 — Baselines

- [ ] Implement NEAREST, BILINEAR, LANCZOS upscaling
- [ ] Score each with PSNR (peak signal-to-noise ratio) and SSIM (structural similarity)
- [ ] Visualise side-by-side — these are the bar the model must beat

**Gate:** Baseline scores documented. Visual comparison saved to `images/`.

**Docs:**
- [ ] Explainer: PSNR and SSIM — why pixel-level metrics aren't the whole story

---

### M3 — CNN upscaler (MSE loss)

- [ ] Build a U-Net style encoder-decoder: compress → reconstruct at higher resolution
- [ ] Train with MSE loss (pixel-level difference)
- [ ] Compare PSNR/SSIM against baselines

**Gate:** CNN beats BILINEAR on PSNR. Outputs don't blur hard edges.

**Docs:**
- [ ] Explainer: U-Net / encoder-decoder for spatial tasks
- [ ] Note: why MSE produces blurry outputs (averages over uncertainty)

---

### M4 — Perceptual loss

- [ ] Replace MSE with perceptual loss: compare feature maps from a pretrained VGG,
  not raw pixel values
- [ ] Perceptual loss preserves sharpness and texture; MSE averages it away
- [ ] Compare outputs: MSE vs perceptual — visible difference in edge quality

**Gate:** Perceptual loss output visually sharper than MSE. Documented with side-by-side images.

**Docs:**
- [ ] Explainer: perceptual loss — why feature-space distance beats pixel-space distance
  for image quality

---

### M5 — GAN (stretch)

- [ ] Add a discriminator: "does this upscaled sprite look like real pixel art at scale?"
- [ ] Generator (the upscaler) vs discriminator — adversarial training
- [ ] Compare GAN output against perceptual loss output

**Gate:** GAN output passes a visual test — looks like hand-drawn HD pixel art, not a
scaled photo.

**Docs:**
- [ ] Explainer: GANs — generator vs discriminator, why adversarial training produces
  sharper outputs than reconstruction loss alone

---

### M6 — CLI tool + pipeline

- [ ] `python -m bloom scale sprite.png --scale 4` → `sprite_4x.png`
- [ ] `python -m bloom scale candidates/ --scale 4` → processes a folder (output from dig)
- [ ] Demo: run bloom on sprites found by dig from a real game file

**Gate:** Full pipeline works: `python -m dig scan game.tap | python -m bloom scale`
(or equivalent). Output sprites are visually clean at 4×.

**Docs:**
- [ ] README: usage, scale factors, how to feed from dig

---

## Skills this covers

| Skill | Gap filled |
|-------|-----------|
| Computer vision on real images | ⬜ → ✅ (image-to-image, not just classification) |
| Perceptual loss / image quality metrics | ⬜ → ✅ |
| GAN (stretch) | ⬜ → ✅ |
| Encoder-decoder / U-Net architecture | ⬜ → ✅ |

---

## Feeds from

[dig](../dig/) — pass candidate PNGs from dig scan directly into bloom scale.
