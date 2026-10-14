# Next — Glyph POC: Image → Both Targets

**Goal:** Take an image, render it on both targets from one spec. Proves the HAL principle — same intent, native implementation per platform.

```
source image (any size, full colour)
  ├── WASM: resize to canvas → render full colour (native)
  └── Z80:  resize to 256×192 → k-means quantise per cell → SCR → load into ULA
```

---

## Why this POC

1. **Proves the HAL principle.** Same spec, different implementations. WASM gets the good version, Z80 gets the Spectrum version. Not a clone — each target plays to its hardware.
2. **K-means clustering warm-up.** The image-to-SCR quantisation is a real clustering problem — 768 tiny k=2 optimisations, one per 8×8 cell. Currently a gap in the study guide (void-duel Ph 2a).
3. **Computer vision touch.** Colour space, downscaling, palette mapping — small but hands-on CV work (currently only prior role experience).
4. **SCR is the simplest glyph artefact.** No game loop, no input, no animation. Static image → both targets. Minimal surface to prove the pipeline.

---

## Milestones

### P1 — Explainer: k-means clustering

Write the shared explainer at `explainers/k-means-clustering.md`.

- [ ] Core concept: partition N points into k clusters by minimising within-cluster distance
- [ ] Algorithm: initialise centroids → assign points → update centroids → repeat
- [ ] Visual intuition: 2D colour-space example (plot pixels, show cluster assignment)
- [ ] Coming from C/JS/TS: k-means is the grouping step in any "find natural categories" problem
- [ ] In practice: customer segmentation, anomaly detection, colour quantisation, feature grouping
- [ ] Connection to this POC: each 8×8 cell is a k=2 clustering in RGB space, centroids mapped to Spectrum palette

**Gate:** Explainer reviewed. Someone could read it cold and understand what k-means does and why we're using it here.

---

### P2 — Image-to-SCR converter (Python)

The meaty bit. Quantise an image to the Spectrum's constraints.

- [ ] Load image, resize to 256×192
- [ ] For each 8×8 cell (768 cells total):
  - Extract 64 pixels
  - K-means with k=2 in RGB space → two cluster centroids
  - Map each centroid to nearest Spectrum palette colour (15 colours: 7 normal + 7 bright + black)
  - Assign each pixel to ink (1) or paper (0) based on cluster membership
  - Pick bright flag from whether either mapped colour is in the bright set
- [ ] Pack into SCR format: 6144 bitmap bytes (interleaved row order) + 768 attribute bytes
- [ ] Write `.scr` file
- [ ] Visual comparison: source image vs SCR rendered back to PNG (using ULA decode logic)
- [ ] Test with a few images — photo, pixel art, high contrast, low contrast

**Gate:** `python -m glyph.scr convert test.png --output test.scr` produces a valid 6912-byte SCR. Visual comparison shows recognisable output with expected colour clash.

---

### P3 — Z80 SCR loader

Trivial asm — but proves the Z80 target pipeline.

- [ ] Z80 asm: `LD HL, scr_data` / `LD DE, 0x4000` / `LD BC, 6912` / `LDIR` / `HALT`
- [ ] SCR data appended after code as raw bytes
- [ ] Runs in zxw emulator — image visible on screen
- [ ] Build script: assemble + append SCR → single binary

**Gate:** Binary loads in zxw, screen shows the converted image.

---

### P4 — WASM image renderer

The WASM target doesn't use SCR — it renders the source image natively.

- [x] Rust crate: `glyph-wasm/` with macroquad dependency
- [x] Loads source image (PNG/JPEG) at build time (WASM) or runtime arg (desktop)
- [x] Renders to canvas — full colour, scaled to window, centred
- [x] No ULA decode, no colour clash, no Spectrum constraints
- [x] Same spec as P3, different output

**Gate:** `cargo build --target wasm32-unknown-unknown` → loads in browser, shows the source image at full colour.

**Status:** DONE — both desktop (`cargo run`) and WASM (`./build.sh`) build clean. 1.5 MB .wasm.

---

### P5 — One spec, both targets

Wire it together. One spec file, two outputs.

- [ ] Define spec format (YAML): `type: display_image`, `source: path/to/image.png`
- [ ] CLI: `python -m glyph cast spec.yaml --target z80` → SCR binary
- [ ] CLI: `python -m glyph cast spec.yaml --target wasm` → WASM build
- [ ] Both outputs from same spec, same source image
- [ ] Side-by-side comparison: Z80 (colour clash, 256×192) vs WASM (full colour)
- [ ] Document the HAL principle with this concrete example

**Gate:** One command per target, same spec, visually correct output on both. The difference in quality demonstrates why the HAL doesn't emulate — each target is native.

---

## Skills covered

| Skill | Study guide ref | Status before |
|-------|----------------|---------------|
| K-means clustering | Classic ML → k-means | Gap (void-duel Ph 2a) |
| Colour quantisation | Computer vision | Gap (prior role only) |
| Cross-platform HAL | New (glyph) | New |
| SCR / ULA format | Systems / Emulation | Extends zxw |
| WASM sprite rendering | New (glyph) | New |

---

## Multi-agent build approach

P2–P4 are independent and can run in parallel with separate Sonnet agents:

- **Agent A** — P2: image-to-SCR converter (Python, k-means, palette mapping)
- **Agent B** — P3: Z80 SCR loader (asm, zxw integration)
- **Agent C** — P4: WASM renderer (Rust, macroquad)

P1 (explainer) is done first. P5 (integration) is done after all three converge.
