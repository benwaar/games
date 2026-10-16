# Next

## Knowledge corpus — Z80 binary analysis

Add a `z80-binary-analysis` category to the corpus for reverse engineering game binaries. Needed for artefact (ROM scanner, sound extraction). The j83 sound analysis was done by hand — this would give the MCP the knowledge to assist.

Topics to cover:
- [ ] Opcode scanning patterns (finding OUT (FE), CALL targets, data vs code discrimination)
- [ ] Recursive descent disassembly vs linear sweep — tradeoffs for 8-bit binaries
- [ ] Call graph tracing — identifying subroutine boundaries from CALL/RET/JP
- [ ] Data table detection — distinguishing address tables, lookup tables, sprite data from code
- [ ] Sound routine signatures — common beeper patterns (DJNZ delay loops, OUT (FE) toggle, sweep/decay structures)
- [ ] TAP/SNA/Z80 format parsing for extraction
- [ ] RAM workspace analysis — tracing engine state through IX-indexed structures

Resources:
- [z80-smart-disassembler](https://github.com/cormacj/z80-smart-disassembler) — labels strings/data areas automatically
- [z80dasm](https://github.com/sharkcz/z80dasm) — Spectrum-focused disassembler with control file support
- [Spectrum Analyser](https://colourclash.co.uk/spectrum-analyser/) — emulator + debugger + interactive disassembler
- [Wise Owl Z80 disassembler](https://www.wiseowl.com/tools/z80-disassembler/) — recursive-descent walker, data classification, sjasmplus output
- [zxrecomp](https://github.com/sp00nznet/zxrecomp) — static recompilation toolkit, Z80 → C (good analysis patterns)
- [ZX Spectrum 1-Bit Routines](https://github.com/utz82/ZX-Spectrum-1-Bit-Routines) — reference collection of beeper sound engines (already in corpus as z80-beeper)
- [Project AY](https://worldofspectrum.org/projectay/tech.htm) — player routine + note table structures
- [Reverse engineering ZX Spectrum games](https://mrcook.uk/reverse-engineering-zx-spectrum-games/) — walkthrough of disassembly and annotation process
- [z80-rom-reverse-engineering skill](https://skillsmp.com/creators/fstark/micral-p2/copilot-skills-z80-rom-reverse-engineering) — Claude Code agent skill for Z80 ROM analysis (reference for our approach)


## glyph — extract Z80 assembly from beeper.py

`beeper.py` has ~200 lines of hand-encoded Z80 machine code as raw hex bytes with a mini linker (patching CALL addresses after layout). The SCR loader already does it right — `scr_loader.asm` assembled by the z80 lib. Beeper should follow the same pattern.

- [ ] Extract `play_note`, `delay_hl`, `rest_loop` routines into `.asm` files under `glyphz80/`
- [ ] Use the z80 assembler (already a dep) with labels — let it resolve addresses instead of hand-patching
- [ ] Python parameterises (note frequencies, durations) and stitches the assembled binary
- [ ] Fix audio playback — TAP beeper output currently doesn't play well, needs investigation (timing? cycle counts? emulator accuracy?)
- [ ] Tests: existing beeper tests must still pass, add assembly-level sanity checks

---

## Rosetta — POC analysis (j83 sound extraction)

Round-trip pipeline: Z80 binary → extract → glyph IR → Z80 + WASM. See [rosetta/README.md](rosetta/README.md).

### What we learned from j83

The j83 binary has **two layers** of sound:

**4 standalone beeper routines** (0x6800–0x6833 + 0x684E, 117 bytes):
- Fixed-pitch tone (D=period, C=repeat count) — pickup and hit sounds
- Rising sweep (C increments from 0x08 to 0x38) — laser/thrust
- Descending sweep with decay (IX-indexed workspace) — explosion
- All self-contained, no dependencies, relocatable (relative jumps only)

**Integrated animation+sound engine** (0x6875–0x6920):
- Not standalone sound — calls `729B`, `71CF`, `7327`, `72A5` etc (all graphics routines)
- Title screen "sound" is really a combined animation system
- Channel-based driver with RAM workspace at `0x5D80`, frame tick at `0x68AB`
- Cannot be extracted without pulling in the entire graphics engine

### POC result

Built `zxw/tools/build_j83_snd_test.py` — extracts the 4 standalone routines, wraps in a test harness with varied parameters (5 sections: original tones, pitch range, sweep, decay variations, rapid-fire combos). Loads as single CODE block, plays in zxw emulator.

**Sound quality is poor** — the beeper routines use tight DJNZ loops (toggling port FE every few dozen T-states) and the emulator's frame-based audio sampler averages out the sharp edges. This is an emulator accuracy issue, not an extraction problem.

**The extraction pipeline works.** Binary → analyse → identify routines → extract → wrap → play. This is the Rosetta M1 approach proven at POC level.

### Implications for Rosetta M1

- Standalone beeper routines are clean extraction targets — pattern: OUT (FE) + DJNZ delay loop
- Engine-driven sound is deeply coupled to game state/graphics — needs a different approach (full binary recompilation or selective stubbing)
- The glyph beeper work (above) should happen first — gives us clean Z80 assembly to compare extracted code against
- Emulator audio accuracy needs work before we can do meaningful waveform comparison (Rosetta M3)

Full disassembly and call map: `_data/j83-analysis-sound.md`


---

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
