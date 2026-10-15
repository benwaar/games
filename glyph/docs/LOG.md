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

`glyph/glyph-wex/` — Rust + macroquad. Loads image (embedded for WASM, CLI arg for desktop), renders centred/scaled to window. No Spectrum constraints — full colour native rendering.

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

Rewrote `wex/index.html` as a generic player. Handles image, audio, keyboard via `type()` bitmask export. X crosshair when no image. Web Audio oscillators (square wave) with click-to-start overlay for autoplay policy. Pause (P) button.

### A5 — zxw debug mode ✅

`zxw/www/debug.html` — auto-loads ROM + TAP, boots emulator, types `LOAD ""` after 3s. Zero clicks. Pause (Ctrl+P), Reload, full keyboard + Kempston. Debug API at `window.zxw`.

### A6 — Combined image+audio WASM ✅

`combined_to_wat()` in `pipeline.py` — pixels at offset 0, audio f64 arrays after pixel data. `type()→3` (image+audio bitmask). CLI: `--audio` flag. 34 pipeline tests covering WAT generation, type bitmask, f64 encoding, memory pages.

### A7 — Cleanup ✅

Removed `glyph/glyphwex/` Rust crate entirely. Updated gitignore, CLAUDE.md, pipeline docs. 113 tests total (34 beeper + 6 scr_loader + 39 scr + 34 pipeline).

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

---

## a11y Phase 1: Mirror DOM ✅ (2026-10-15)

**Goal:** Make canvas-based WASM content accessible to screen readers.

- Pipeline encodes a11y scene JSON as UTF-8 data segment (`type()` bit 3). New exports: `a11y_offset()`, `a11y_size()`.
- Player reads a11y data, builds hidden mirror DOM overlay with `role`, `aria-label`, percentage-based positioning.
- `aria-live="polite"` region for announcements, sr-only clip-rect pattern.
- Optional `env` imports: `prefers_dark`, `prefers_reduced_motion`, `prefers_high_contrast` — browser preferences bridged to WASM.
- Fallback: no bit 3 → `aria-label="Game content"` + `role="region"` on canvas.
- Explainer: `wex/explainers/mirror-dom.md`.

## a11y Phase 2: Settings panel + palette override ✅ (2026-10-15)

**Goal:** Let users override browser preferences and switch palettes.

- Settings gear icon (top-right of canvas), toggle overrides: dark mode, reduced motion, high contrast (Auto/On/Off).
- 3 palette presets: default, high-contrast, deuteranopia-friendly. CSS custom properties on `:root`.
- `localStorage` persistence under `wasm-player-settings`.
- `reloadWasm()` re-instantiates WASM when overrides change (imports return new values).
- Multi-agent code review found and fixed: keyboard listener leak (AbortController), redundant matchMedia, dead code.

## a11y Phase 3: Sound customisation ✅ (2026-10-15)

**Goal:** Volume control, self-voicing, and spatial audio for blind/low-vision players.

- Volume slider (range 0–100) with master `GainNode` — persisted to localStorage.
- Self-voicing toggle: Web Speech API reads scene descriptions on load and mirror DOM node focus.
- Spatial audio: `PannerNode` with HRTF model positions audio per-note using a11y node x coordinates.
- All settings saved/loaded alongside existing preferences under `wasm-player-settings` key.
- Review fix: replaced deprecated `setPosition()` with `positionX/Y/Z` AudioParams.

**Goal:** Let users override browser preferences and switch palettes.

- Settings gear icon (top-right of canvas), toggle overrides: dark mode, reduced motion, high contrast (Auto/On/Off).
- 3 palette presets: default, high-contrast, deuteranopia-friendly. CSS custom properties on `:root`.
- `localStorage` persistence under `wasm-player-settings`.
- `reloadWasm()` re-instantiates WASM when overrides change (imports return new values).
- Multi-agent code review found and fixed: keyboard listener leak (AbortController), redundant matchMedia, dead code.
