# Knowledge Corpus — Deferred Work

## Content gaps

Sources already fetched: SkoolKit ROM disassembly, Z80 instruction set (z80-heaven), WASM spec, MDN WAT reference, WASI preview1.

Still missing:
- [ ] Spectrum hardware content — skoolkid/WoS pages had different structure than expected, only stubs fetched
- [ ] MDN WAT section parsing — MDN redirects fragment URLs, content extraction needs rework
- [ ] Undocumented Z80 opcodes — Sean Young doc not yet integrated
- [ ] 1-bit music routines (utz82) — beeper engine patterns: pulse interleaving, PWM, sample playback
- [ ] Z80 library routines (breakintoprogram / Dean Belfield) — game routines: sound, sprites, vectors

---

# Next

https://skoolkid.github.io/rom/asm/03B5.html

## zxw — fix emulator audio pipeline

The beeper audio in zxw drops/loses notes. Investigation (2026-10-15) found:

- **`ScriptProcessorNode` is deprecated** — Chrome warns on every load. Replace with `AudioWorklet`.
- **Queue overflow wipes audio** — `audio.js` line 41: if `sampleQueue > 44100` it clears the entire buffer. ROM BEEP runs with `DI` (interrupts disabled), so thousands of samples accumulate between frame drains and get nuked.
- **`drain_audio()` only called per `requestAnimationFrame`** (~16ms) — not fast enough for continuous beeper output. Samples pile up in Rust between drains.
- **Test case:** `CALL 0x03B5` (ROM BEEP) with C4/D4/E4/F4 notes, 500ms warmup, 150ms gaps. Should hear 4 distinct rising tones. Currently hears 0–2 notes depending on timing.
- Reference: ROM BEEP disassembly at https://skoolkid.github.io/rom/asm/03B5.html

Steps:
- [ ] Replace `ScriptProcessorNode` with `AudioWorkletNode` in `zxw/www/audio.js`
- [ ] Fix queue overflow — don't wipe, ring buffer or backpressure instead
- [ ] Drain audio more frequently (per-instruction or per-scanline accumulation, not per-frame)
- [ ] Test with ROM BEEP test TAP (C D E F notes) — all 4 notes audible with clear gaps
- [ ] Test with a known game (Jetpac) — sound effects should be clean

## glyph — extract Z80 assembly from beeper.py

`beeper.py` has ~200 lines of hand-encoded Z80 machine code as raw hex bytes with a mini linker (patching CALL addresses after layout). The SCR loader already does it right — `scr_loader.asm` assembled by the z80 lib. Beeper should follow the same pattern.

- [ ] Extract `play_note`, `delay_hl`, `rest_loop` routines into `.asm` files under `glyphz80/`
- [ ] Use the z80 assembler (already a dep) with labels — let it resolve addresses instead of hand-patching
- [ ] Python parameterises (note frequencies, durations) and stitches the assembled binary
- [ ] Fix audio playback — TAP beeper output currently doesn't play well, needs investigation (timing? cycle counts? emulator accuracy?)
- [ ] Tests: existing beeper tests must still pass, add assembly-level sanity checks

## wex — indexed colour pipeline (256-colour LUT)

Design decision: content art uses a 256-colour indexed palette. Why:

- **Palette remapping becomes a table swap.** Each palette variant (high-contrast, deuteranopia) is a curated 256-entry LUT. The WASM `render()` walks the pixel buffer, looks up each colour, writes the remapped value. 1KB table, trivial loop.
- **k-means quantisation already in our pipeline** — same technique we use for Spectrum (k=2 per cell), just with k=256 globally. No new tooling.
- **256 is the sweet spot for illustrated/pixel art.** Bold line art with flat fills (like Aerythen) rarely exceeds 256 unique colours. Perceptually complete, matches indexed colour (PNG-8), well-understood format.
- **Designer controls every remap.** No algorithmic surprises — each colour maps to an exact target per palette. Semantic meaning survives (danger stays visually distinct, not just "was red").
- **Fallback for photographic content:** 3×3 matrix transform per pixel (Brettel for colour blindness, gamma for dark/light). Works on any art, but results are approximate — LUT is preferred when colour count allows.

Steps:
- [ ] Quantise source art to 256 colours in pipeline (k-means, store palette + indices)
- [ ] New memory layout: 256×4 bytes (palette RGBA) + W×H bytes (indices), replacing raw RGBA
- [ ] Embed palette + index buffer in WAT data segments
- [ ] Author palette variants: default, high-contrast, deuteranopia (256 entries each)
- [ ] WAT `render()` reads preference imports, selects LUT, remaps indices → RGBA into a separate output region for canvas
- [ ] Update player JS to call `render()` on palette change instead of full `reloadWasm()` — cheaper, just re-renders with new LUT
- [ ] Fallback: matrix transform path for content exceeding 256 colours

## Glyph — deferred from audio POC

- HAL audio surface (`tone_play`, `sfx_play`) — needed for game sigils
- SFX: sweep, noise burst — melody proved the pipeline
- ADSR envelopes on WASM side
- WASM player as wrapper (bundle player + content for deploy)
- Keyboard input mechanism (store key state in WASM memory)

## Explainer: indexed colour and palette tables

- Indexed colour: palette (N colours) + index buffer (1 byte per pixel) vs raw RGBA (4 bytes per pixel)
- Where it appears: PNG-8, GIF, VGA mode 13h, Spectrum SCR (2 colours per 8×8 cell)
- k-means quantisation: reduce arbitrary art to N representative colours. Same algorithm, different k.
- LUT remapping: swap the palette, pixels follow. Why this is fast and predictable.
- Bridge: Spectrum SCR is indexed colour with extra constraints (attribute cells). Our 256-colour LUT is the same idea without the spatial restriction.
- Where: `wex/explainers/indexed-colour.md`

## Explainer: mirror DOM + a11y pipeline

- How a11y data flows: scene JSON → pipeline → WASM data segment → JS → mirror DOM → screen reader
- The `type()` bitmask, `a11y_offset()` / `a11y_size()` contract
- `buildMirrorDom()` pattern: percentage-based positioning, sr-only clip-rect, aria-live
- Self-voicing and spatial audio: Web Speech API + PannerNode HRTF
- Where: `wex/explainers/a11y-pipeline.md`

## Explainer: browser preferences → WASM imports

- The full bridge: OS/browser preference → CSS media query → JS host function → settings override check → WASM import → integer return
- How `(import "env" "prefers_dark" (func (result i32)))` works: WASM declares what it needs, JS provides the implementation at instantiation
- Settings panel override layer: Auto (browser default) vs On/Off (localStorage)
- CSS custom properties (`--bg` etc.) are player-side chrome only — WASM never sees CSS, only integers through imports
- Not M3 design tokens — flat, six-token set. Note how M3/design systems solve the same problem at larger scale.
- Where: `wex/explainers/preference-bridge.md`

## Explainer: WASM code reuse — from WAT snippets to Component Model

- Our approach: text composition in Python — pipeline stitches WAT fragments (a11y segment, imports, shared render). Pipeline is the linker.
- Single module composition: multiple `(func)` blocks, everything in one `(module)`. Simple, no tooling overhead.
- WASM module linking: separate `.wasm` files importing/exporting to each other. Browser support exists (`WebAssembly.Module` imports). Better separation, heavier setup.
- wasm-merge (wabt): mechanical merging of two `.wasm` into one. Linking without the component model.
- Component Model + WIT: the official "WASM library" story. Interface types, composable modules, multi-language. Production-grade, still maturing.
- Why text composition works for us (learning, small modules, one pipeline) vs why you'd graduate to linking/components in production (third-party modules, separate compilation, multi-language teams)
- Where: `wex/explainers/wasm-code-reuse.md`

## Cross-project

- incant to follow same debug pipeline as glyph, with wrappers for testing
- Better build pipeline later that combines stuff
