# dig — Binary Art Scanner

Scan a game binary, render every byte window as pixels, and score each one:
*is this intentional art, or noise?*

Automates what BinViewer does by hand.

---

## The idea

A ZX Spectrum `.tap` or `.z80` memory dump is just bytes. Sprites, fonts, and tile sets
live somewhere in there with no headers or addresses — the developer knew where they were,
and nobody wrote it down. The only way to find them is to render every region and look.

dig trains a classifier on known sprite data, then scans unknown binaries scoring every
window. High score = probably art. Output: ranked list of candidate windows, ready for bloom.

---

## Milestones

### M1 — Data pipeline

- [ ] Parse `.tap` and `.z80` files → extract raw bytes
- [ ] Render byte windows as 1bpp images (8×8, 16×16, 32×8 grids)
- [ ] Build labelled dataset: known sprites from public-domain ZX games (art=1) vs random
  code/data regions (noise=0)
- [ ] Use BinViewer as ground truth source for known-art addresses

**Gate:** DataLoader yields `(image, label)` pairs. Images are `(1, H, W)` tensors. Labels balanced.

**Docs:**
- [ ] Explainer: ZX Spectrum memory layout and display file format
- [ ] Explainer: 1bpp pixel rendering — how bytes become pixels

---

### M2 — Classical ML baseline

- [ ] Extract features per window: entropy, byte value distribution, run-length stats,
  checkerboard score (alternating bytes = probable pixel pattern)
- [ ] Train a gradient boosting classifier (XGBoost / sklearn) on those features
- [ ] Evaluate with ROC/AUC — sets the bar the CNN must beat

**Gate:** ROC/AUC above 0.7 on held-out windows. Baseline documented.

**Docs:**
- [ ] Explainer: feature engineering for binary data — what makes a byte window look structured
- [ ] Explainer: ROC/AUC — what it measures, why accuracy alone misleads on imbalanced data

---

### M3 — CNN classifier

- [ ] Train a small CNN on rendered window images (binary classification: art / noise)
- [ ] Compare ROC/AUC against M2 baseline — does visual learning beat hand-crafted features?
- [ ] Evaluate: precision, recall, F1, confusion matrix

**Gate:** CNN ROC/AUC beats baseline. Precision/recall tradeoff understood.

**Docs:**
- [ ] Update evaluation explainer with ROC/AUC from real results

---

### M4 — Sliding window scanner

- [ ] Scan a full binary with overlapping windows at multiple sizes
- [ ] Score each window with the trained CNN
- [ ] Output: ranked list of windows sorted by art probability
- [ ] Visualise: render top-N candidates as a contact sheet

**Gate:** Running on a known game surfaces actual sprites in the top-20 results.

**Docs:**
- [ ] Explainer: sliding window detection — the spatial scanning pattern

---

### M5 — Autoencoder approach (unsupervised)

- [ ] Train a convolutional autoencoder on known sprites only (no noise needed)
- [ ] At scan time: high reconstruction error = probably not art
- [ ] Compare: supervised CNN (M3) vs unsupervised autoencoder (M5)
  — which is more useful when you have no labels for the target game?

**Gate:** Autoencoder reconstruction error correlates with art probability on held-out games.

**Docs:**
- [ ] Explainer: autoencoders — encode to latent space, decode, use reconstruction error as anomaly score
- [ ] Explainer: self-supervised learning — why the sprites themselves are the only label needed

---

### M6 — CLI tool + demo

- [ ] `python -m dig scan game.tap` → outputs `candidates/` folder of ranked PNGs
- [ ] `python -m dig scan game.z80 --top 20 --size 16x16`
- [ ] Demo: run against a known public-domain ZX game, show sprites found

**Gate:** CLI works on a fresh clone. Demo output contains recognisable sprites.

**Docs:**
- [ ] README: usage, how to interpret output, how to pass results to bloom

---

## Skills this covers

| Skill | Gap filled |
|-------|-----------|
| Computer vision on real images | ⬜ → ✅ |
| ROC / AUC | ⬜ → ✅ |
| Sliding window / detection pattern | ⬜ → ✅ |
| Classic ML baseline (gradient boosting) | ⬜ → ✅ |
| Autoencoder | ⬜ → ✅ |
| Self-supervised learning | ⬜ → ✅ |

---

## Feeds into

[bloom](../bloom/) — pass candidate PNGs from dig directly into bloom for HD scaling.
