# Next: M3 — ULA screen rendering

Decode the ZX Spectrum's interleaved screen layout and attribute memory into an RGBA pixel buffer, then blit it to the canvas via wasm-bindgen.

## What to do

1. `src/screen.rs` — `render_frame(mem: &Memory) -> Vec<u8>`:
   - Screen pixels at 0x4000–0x57FF — 192 rows of 256 pixels (1 bit each), stored in thirds (lines 0–63, 64–127, 128–191) with a non-linear row order within each third.
   - Attributes at 0x5800–0x5AFF — one byte per 8×8 cell: `FBPPPIII` (FLASH / BRIGHT / PAPER×3 / INK×3).
   - For each pixel: look up INK (foreground) or PAPER (background) colour from attribute cell, apply BRIGHT, emit as RGBA.
   - Spectrum palette: 8 base colours × 2 brightness levels = 16 entries.

2. Address calculation — row `y` (0–191) maps to pixel address:
   - Third: `t = y / 64`
   - Line within third: `l = y % 64` → `row_in_char = l / 8`, `char_line = l % 8`
   - Pixel addr: `0x4000 | (t << 11) | (char_line << 8) | (row_in_char << 5)`
   - Each byte covers 8 horizontal pixels; x byte = `x / 8`, bit = `7 - (x % 8)`.

3. `src/lib.rs` — expose `render_frame` via wasm-bindgen; return `Vec<u8>` as `Uint8Array` to JS.

4. `www/index.html` — update JS to call `render_frame()` on each `requestAnimationFrame`.

5. `tests/screen_tests.rs` — unit tests:
   - Solid INK colour: fill attribute mem with `0x07` (white ink, black paper), fill pixel mem with `0xFF` → all pixels white.
   - Solid PAPER colour: attribute `0x00`, pixel `0x00` → all pixels black.
   - BRIGHT flag doubles intensity (e.g. red `0x02` vs bright red `0x42`).
   - Attribute cell boundary check: pixel at (0,0) uses attribute (0,0), pixel at (8,8) uses attribute (1,1).

## Gate

Load the Jetpac loading screen (6912 bytes: 6144 pixel + 768 attribute) directly into memory at 0x4000 → `render_frame()` produces correct RGBA output verifiable by inspecting a known pixel coordinate's colour.

## Reference

- Screen memory map: Sean Young's Z80 Undocumented, Section 5; Spectrum ROM Disassembly.
- Attribute byte: bit 7 = FLASH, bit 6 = BRIGHT, bits 5–3 = PAPER, bits 2–0 = INK.
- Spectrum colours: Black=0, Blue=1, Red=2, Magenta=3, Green=4, Cyan=5, Yellow=6, White=7. Normal brightness ≈ 0xD7 (215); bright = 0xFF.
