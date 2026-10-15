# CLAUDE.md — glyph

## What this is

An LLM-first presentation layer. Takes image or audio specs and outputs to two targets: WASM (full fidelity) and Z80 TAP (Spectrum-constrained). The pipeline generates pure WAT/WASM — no Rust, no wasm-pack, no JS glue.

## Pipeline

```
python glyph/pipeline.py glyph/assets/i1.png    # image → both targets
python glyph/pipeline.py glyph/assets/a1.json   # audio → both targets
```

Output:
- `wex/debug.wasm` — loaded by WASM player on `:8999`
- `zxw/www/tap/debug.tap` — loaded by zxw debug mode on `:8998/debug.html`

## WASM export contract

Content WASM (`debug.wasm`) exports:

```
type()         → i32    # bitmask: bit 0=image, bit 1=audio, bit 2=keyboard
width()        → i32    # canvas width (image only)
height()       → i32    # canvas height (image only)
render()                # populate framebuffer (image only)
audio_count()  → i32    # number of notes (audio only)
audio_offset() → i32    # byte offset of note data (audio only)
memory                  # linear memory
```

## WASM import contract (optional)

Content WASM may import preference queries from the host:

```
(import "env" "prefers_dark" (func (result i32)))
(import "env" "prefers_reduced_motion" (func (result i32)))
(import "env" "prefers_high_contrast" (func (result i32)))
```

Returns 1 (true) or 0 (false). Browser default unless overridden by player settings.
Games opt in by declaring the imports they need — unused imports are ignored.

Audio data layout at `audio_offset()`:
- `f64[audio_count]` frequencies in Hz
- `f64[audio_count]` durations in ms

## Architecture

```
input (png/jpg/json)
  ├── WASM: full-colour/full-fidelity → WAT → wat2wasm → wex/debug.wasm
  └── TAP:  Spectrum-constrained → Z80 binary → incant.tap → zxw/www/tap/debug.tap
```

Image WASM: RGBA pixels embedded as WAT data segments at native resolution.
Audio WASM: freq/duration f64 arrays embedded as WAT data segments.
Image TAP: k-means quantised to 256×192 Spectrum palette → SCR → Z80 LDIR loader.
Audio TAP: Z80 beeper binary — cycle-counted loops toggling port 0xFE bit 4.

## Key files

```
glyph/
  pipeline.py            — main pipeline: input → debug.wasm + debug.tap
  assets/                — test inputs (i1.png, a1.json)
  scr/                   — shared image processing: quantise, pack, palette, render
  glyphz80/              — Z80 target: beeper compiler, SCR loader, assembler
  tests/                 — 34 pipeline tests (WAT generation, type bitmask, f64 encoding)
  glyphz80/tests/        — 34 beeper + 6 scr_loader tests
  explainers/            — beeper-audio.md, hal-principle.md
```

## Test harnesses

- **WASM player** (`wex/index.html`, `:8999`): generic player — handles image, audio, keyboard via `type()` bitmask. Shows X crosshair when no image. Includes a settings panel (gear icon, top-right of canvas) with palette presets (default, high-contrast, deuteranopia-friendly), preference override toggles (dark mode, reduced motion, high contrast), and `localStorage` persistence under key `wasm-player-settings`. Player chrome uses CSS custom properties on `:root` (`--bg`, `--bg-panel`, `--border`, `--text`, `--text-muted`, `--accent`) — pipeline-generated WASM can inherit these for consistent theming.
- **zxw debug** (`zxw/www/debug.html`, `:8998`): auto-loads ROM + TAP, boots, types `LOAD ""`. Zero clicks.
- **zxw full** (`zxw/www/index.html`, `:8998`): full emulator with sidebar, file pickers, POK support.

## Conventions

- Pure WAT → `wat2wasm` for all WASM output. No Rust/wasm-pack in the pipeline.
- TAP files via `incant.tap.write_spectrum_tap()` — canonical TAP writer.
- Tests: `python -m pytest glyph/ -v` (113 tests).
