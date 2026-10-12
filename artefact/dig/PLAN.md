# dig — ROM Audio Scanner

Scan a game ROM or memory dump for hidden music data, sound effects, and unindexed
PCM samples. Score every region: *is this structured audio data, or noise?*

Automates what manual ROM reverse-engineering does by hand.

---

## The idea

A ZX Spectrum `.tap`, `.z80`, a C64 `.sid`, or a raw memory dump contains music packed as
chip register streams, tracker patterns, or PCM samples — with no headers, no index, no
documentation. The developer knew the memory layout; nobody wrote it down.

dig trains an unsupervised model on known chip audio data, then scans unknown binaries
scoring every region. High reconstruction confidence = probably music or SFX data.
Output: ranked list of candidate regions, ready for bloom.

---

## Milestones

### M1 — Data pipeline

- [ ] Parse ROM formats: `.tap`, `.z80`, `.sid`, `.vgm`, raw memory dumps
- [ ] Extract byte windows at multiple sizes (matching common sound data block sizes)
- [ ] Build reference dataset: known music data from public-domain game files
  (VGM archives, HVSC SID collection, AY chip dumps)
- [ ] Convert reference audio data to spectrogram / feature representation

**Gate:** DataLoader yields `(feature_tensor, label)` pairs. Reference data covers multiple chip types.

**Docs:**
- [ ] Explainer: 8-bit sound chip architectures — AY-3-8910, SID, YM2149, register layout
- [ ] Explainer: VGM / SID file formats — how music is stored as register streams

---

### M2 — Feature extraction

- [ ] Extract per-window features: byte entropy, register-pattern frequency,
  spectral characteristics of rendered audio, periodicity detection
- [ ] Classical ML baseline: gradient boosting classifier on hand-crafted features
- [ ] Evaluate with ROC/AUC — sets the bar the autoencoder must beat

**Gate:** ROC/AUC above 0.7 on held-out data. Baseline documented.

**Docs:**
- [ ] Explainer: feature engineering for binary audio data — what makes a byte window look like music
- [ ] Explainer: ROC/AUC — what it measures, why accuracy alone misleads on imbalanced data

---

### M3 — Autoencoder (unsupervised)

- [ ] Train a convolutional autoencoder on known chip audio feature representations only
- [ ] At scan time: low reconstruction error = structured audio data; high = noise/code
- [ ] Compare: supervised baseline (M2) vs unsupervised autoencoder (M3)
  — which works better when you have no labels for the target ROM?

**Gate:** Autoencoder reconstruction error correlates with audio data presence on held-out ROMs.

**Docs:**
- [ ] Explainer: autoencoders — encode to latent space, decode, use reconstruction error as anomaly score
- [ ] Explainer: self-supervised learning — why the audio data itself is the only label needed

---

### M4 — Sliding window scanner

- [ ] Scan a full binary with overlapping windows at multiple block sizes
- [ ] Score each window with the trained autoencoder
- [ ] Output: ranked list of regions sorted by audio-data probability
- [ ] Render top-N candidates as spectrograms + playable audio previews

**Gate:** Running on a known game surfaces actual music data in the top-20 results.

**Docs:**
- [ ] Explainer: sliding window detection applied to binary audio scanning

---

### M5 — CLI tool + demo

- [ ] `python -m dig scan game.tap` → outputs `candidates/` folder of ranked audio regions
- [ ] `python -m dig scan game.z80 --top 20 --chip ay`
- [ ] Demo: run against a known public-domain game, show music data found

**Gate:** CLI works on a fresh clone. Demo output contains recognisable music/SFX data.

**Docs:**
- [ ] README: usage, how to interpret output, how to pass results to bloom

---

## Skills this covers

| Skill | Gap filled |
|-------|-----------|
| Unsupervised structure detection | ⬜ → ✅ |
| ROC / AUC | ⬜ → ✅ |
| Sliding window / detection pattern | ⬜ → ✅ |
| Classic ML baseline (gradient boosting) | ⬜ → ✅ |
| Autoencoder | ⬜ → ✅ |
| Self-supervised learning | ⬜ → ✅ |
| Binary / ROM data domain | ⬜ → ✅ |

---

## Feeds into

[bloom](../bloom/) — pass candidate audio regions from dig directly into bloom for restoration.
