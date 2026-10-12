# Next: M4 — ROM + TAP loading

Wire up the Spectrum ROM (16K binary) and a TAP file loader so real programs can boot and run.

## What to do

1. `src/lib.rs` — expose `load_rom(data: &[u8])` and `load_tap(data: &[u8])` via wasm-bindgen on `Emulator`.

2. ROM loader — `Memory::load_rom` already exists; just plumb it through `Emulator::load_rom`.

3. TAP parser in Rust (`src/tap.rs`):
   - Block structure: 2-byte length LE + `length` bytes (flag + data + checksum).
   - Parse into `Vec<TapBlock>` where each block has `flag: u8` and `data: Vec<u8>`.
   - Validate XOR checksum on each block.

4. Instant-load trap (`src/spectrum.rs`):
   - Before each `cpu.step`, check if PC == 0x0556 (LD_BYTES entry in the standard ROM).
   - If so, find the matching TAP block (type from IX register, length from DE), copy data to (IX), set carry flag, set PC to the return address on the stack, and return.
   - This skips the ROM's tape-reading loop entirely — programs load in zero time.

5. `src/lib.rs` — `Emulator::load_tap(data: &[u8])` stores the parsed TAP, ready for the trap.

6. `www/index.html` — add file inputs for `.rom` and `.tap` files. On load: call `load_rom` / `load_tap`, then `step_frame` in a RAF loop.

7. Tests (`tests/tap_tests.rs`):
   - Parse a hand-crafted 2-block TAP (header + data) and verify block count, flags, payload bytes.
   - Checksum mismatch returns an error.
   - Instant-load trap: build a minimal TAP + ROM stub, step CPU, verify data landed in memory.

## Gate

Load `48.rom` + `jetpac_fixed.tap` → CPU runs → title screen appears on canvas without manual screen injection.

## Reference

- TAP format: WoS FAQ; each block is `u16 length` + `length` bytes.
- LD_BYTES ROM address: 0x0556 (48K ROM).
- At LD_BYTES entry: IX = destination, DE = byte count, A′ = flag (0 = header, 0xFF = data).
