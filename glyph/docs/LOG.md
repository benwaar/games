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

## POC: Audio → Both Targets ✅

**Goal:** Play audio on both targets from one spec. Same intent, native implementation per platform.

### A0 — zxw beeper support ✅

T-state accurate audio sampling in the Rust emulator + Web Audio playback. Foundation for all Z80 audio output.

### A1 — Explainer: beeper audio ✅

`glyph/explainers/beeper-audio.md` — 1-bit audio, cycle-counted Z80 loops, port 0xFE bit 4, frequency calculation, mapping to Web Audio oscillators.

### A2 — Z80 beeper engine ✅

`glyph/glyphz80/beeper.py` — compiles glyph-audio JSON to Z80 machine code. Cycle-counted half-period loops for accurate pitch. 34 tests. TAP output via `incant.tap`.

### A3 — Pure-WAT audio WASM ✅

Replaced wasm-pack/Rust approach with pure WAT generation. Freq/duration arrays embedded as f64 data segments. Compiled via `wat2wasm`. Audio WASM is 201 bytes vs 19KB from wasm-pack.

### A4 — WASM player ✅

Rewrote `wasm/index.html` as a generic player. Handles image, audio, keyboard via `type()` bitmask export. X crosshair when no image. Web Audio oscillators (square wave) with click-to-start overlay for autoplay policy. Pause (P) button.

### A5 — zxw debug mode ✅

`zxw/www/debug.html` — auto-loads ROM + TAP, boots emulator, types `LOAD ""` after 3s. Zero clicks. Pause (Ctrl+P), Reload, full keyboard + Kempston. Debug API at `window.zxw`.

### A6 — Combined image+audio WASM ✅

`combined_to_wat()` in `pipeline.py` — pixels at offset 0, audio f64 arrays after pixel data. `type()→3` (image+audio bitmask). CLI: `--audio` flag. 34 pipeline tests covering WAT generation, type bitmask, f64 encoding, memory pages.

### A7 — Cleanup ✅

Removed `glyph/glyphwasm/` Rust crate entirely. Updated gitignore, CLAUDE.md, pipeline docs. 113 tests total (34 beeper + 6 scr_loader + 39 scr + 34 pipeline).

**Key learnings:**
- Pure WAT is vastly smaller and simpler than wasm-pack for data-embedding use cases
- Browser autoplay policy needs click-to-start fallback — AudioContext starts suspended
- f64 encoding in WAT: `struct.pack('<d', value)` → hex escape sequences in data segments
- The `type()` bitmask pattern makes one static HTML handle any content combination

---

### Tools ✅

- `tools/zx-viewer/` — VS Code extension: renders .scr files, TAP hex inspector with decoded headers + colour-coded SCR bytes
- `tools/wasm-viewer/` — VS Code extension: WASM section breakdown, colour-coded hex dump, decoded exports/imports
- `tools/EXTENSIONS.md` — catalogue of useful VS Code extensions for the repo
