# Glyph — LOG

## POC: Image → Both Targets ✅

**Goal:** Take an image, render it on both targets from one spec. Proves the HAL principle.

### P1 — Explainer: k-means clustering ✅

Shared explainer at `explainers/k-means-clustering.md`. Core concept, algorithm, visual intuition, business parallels, connection to Spectrum colour quantisation.

### P2 — Image-to-SCR converter ✅

`glyph/scr/` — Python pipeline: load image → letterbox to 256×192 → k-means (k=2) per 8×8 cell → map centroids to Spectrum palette → pack into SCR format. 39 tests.

- `palette.py` — 15-colour Spectrum palette, `nearest_colour()` via Euclidean RGB distance
- `quantise.py` — k-means per cell, handles uniform cells, bright flag logic
- `pack.py` — SCR packer with interleaved ULA pixel address formula, roundtrip pack/unpack
- `render.py` — SCR → PIL Image decoder
- `__main__.py` — CLI: `convert`, `render`, `compare` subcommands

### P3 — Z80 SCR loader ✅

`glyph/z80/` — 12-byte Z80 loader (`LD HL/DE/BC` + `LDIR` + `HALT`). Build script assembles asm via `z80` pip package, concatenates with SCR data. 6 tests.

### P4 — WASM image renderer ✅

`glyph/glyph-wasm/` — Rust + macroquad. Loads image (embedded for WASM, CLI arg for desktop), renders centred/scaled to window. No Spectrum constraints — full colour native rendering.

### P5 — Design iteration + prompt skeleton ✅

Three design iterations (v1→v3) proved that prompt-engineering art for the worst target produces better results on both platforms. Added `prompt-skeleton.md` — template for generating LLM art that works on both targets.

**Key learnings:**
- K-means quality depends on input art design, not algorithm tuning
- QR codes impractical at Spectrum resolution (modules smaller than 8×8 cells)
- Image optimisation: 1.4MB PNG → 247KB JPEG (WASM), 6.8KB SCR (Z80)

### Tools ✅

- `tools/zx-viewer/` — VS Code extension: renders .scr files, TAP hex inspector with decoded headers + colour-coded SCR bytes
- `tools/wasm-viewer/` — VS Code extension: WASM section breakdown, colour-coded hex dump, decoded exports/imports
- `tools/EXTENSIONS.md` — catalogue of useful VS Code extensions for the repo
