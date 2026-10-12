# zxw — Plan

Minimal ZX Spectrum 48K emulator: Rust → WASM, with CDP debug integration.

---

## M1 — Scaffold + WASM canvas hello world

- [x] Create `Cargo.toml` (wasm-bindgen, web-sys with Canvas features, js-sys)
- [x] Write `setup.sh`: check/install rustup, add `wasm32-unknown-unknown`, install `wasm-pack`
- [x] Implement `init(canvas_id)` → fill a 256×192 canvas with a checkerboard pattern
- [x] Add `www/index.html` with canvas + module import
- [x] Add `Makefile` with `build` (wasm-pack) and `serve` (python3 http.server) targets
- [x] README with prereqs, build instructions, project overview

**Gate:** `make build && make serve` → open localhost:8080 → checkerboard renders, no console errors.

---

## M2 — Z80 CPU core (foundational opcodes)

- [ ] CPU state struct: AF/BC/DE/HL/IX/IY/SP/PC + alternates + I/R/IFF/IM/halted
- [ ] Memory trait + SpectrumMemory: 16K ROM (read-only) + 48K RAM
- [ ] Fetch-decode-execute: `step(&mut self, mem) -> u32` returns T-states
- [ ] Unprefixed opcodes: LD r,r / LD r,n / LD r,(HL) / arithmetic / logic / JP/JR/CALL/RET / DJNZ / PUSH/POP / EX / LDIR / HALT / NOP / EI/DI / IN/OUT stubs
- [ ] CB prefix: BIT/SET/RES / rotates and shifts
- [ ] ED prefix subset: LD A,I / LD A,R / IM modes / LDIR/LDDR/CPIR / NEG / RETI
- [ ] Unit tests: ADD, DJNZ loop, LDIR block copy, CALL+RET, bit test

**Gate:** `cargo test` — all CPU unit tests pass. A test program (load regs, arithmetic, halt) produces expected register state.

---

## M3 — ULA screen rendering

- [ ] Spectrum screen address calculation (interleaved thirds layout)
- [ ] Attribute decoding: INK/PAPER/BRIGHT/FLASH per 8×8 cell
- [ ] `render_frame(memory) -> Vec<u8>` → RGBA pixel buffer (256×192×4)
- [ ] Spectrum palette (16 colours: 8 normal + 8 bright)
- [ ] Expose to JS via wasm-bindgen, blit with `ImageData` + `putImageData`
- [ ] `requestAnimationFrame` loop: step CPU ~69888 T-states per frame, then render

**Gate:** Load the Jetpac loading screen (6912 bytes extracted from TAP) directly into memory at 0x4000 → loading screen renders correctly with proper colours.

---

## M4 — ROM + TAP loading

- [ ] ROM loader: read 16K binary into 0x0000–0x3FFF, enforce read-only
- [ ] TAP parser in Rust: block lengths, flags, headers, data payloads, XOR checksums
- [ ] Instant-load trap: intercept ROM's LD_BYTES (0x0556), inject TAP data directly, set carry flag
- [ ] Expose `load_tap(data)` + `load_rom(data)` via wasm-bindgen
- [ ] File input in HTML: drag-and-drop / `<input type="file">` for .tap and .rom

**Gate:** Load Spectrum ROM + a test TAP (simple program that writes to screen memory and halts) → character appears on canvas.

---

## M5 — Keyboard input

- [ ] Keyboard port matrix: 8 half-rows, 5 bits each via IN port 0xFE
- [ ] Map browser keydown/keyup → Spectrum key matrix via JS events
- [ ] Wire into IN instruction handler (port 0xFE, address line selection)
- [ ] Kempston joystick (port 0x1F) as alternative
- [ ] Test program: poll keys in loop, change border colour per key

**Gate:** Load keyboard test program → pressing keys changes border colour in real time.

---

## M6 — Jetpac bring-up (incremental opcode completion)

- [ ] Load `jetpac_fixed.tap` (rebuilt 3-block version) → screen at 0x4000, code at 0x6000, entry 0x61E5
- [ ] Unimplemented-opcode trap: log PC, opcode, context, halt — feedback loop for incremental implementation
- [ ] Implement IX/IY prefix opcodes (DD/FD): indexed addressing `(IX+d)` / `(IY+d)`
- [ ] Fill remaining ED-prefix gaps as Jetpac hits them
- [ ] IM 1 interrupt handling: RST 0x38 every ~69888 T-states
- [ ] Verify: title screen → key press → gameplay → movement + fire works

**Gate:** Jetpac loads, title screen renders, game starts on keypress, player moves and collects rockets. ~50fps, no visible slowdown.

---

## M7 — CDP debug integration

- [ ] WASM debug API: `read_memory`, `write_memory`, `read_registers`, `set_breakpoint`, `step`, `pause`, `resume`
- [ ] Register as `window.zxw.*` functions for CDP access
- [ ] `tools/cdp_debug.py`: connect via WebSocket, commands: `mem`, `regs`, `poke`, `bp`, `screen`, `watch`
- [ ] Memory-write watchpoint mechanism (optional callback on address range write)
- [ ] Test: while Jetpac runs, read score address, poke lives, dump screen

**Gate:** Jetpac running → `python3 tools/cdp_debug.py` connects → `regs` prints Z80 state → `mem 0x4000 32` dumps screen bytes → `poke <lives_addr> 9` changes displayed lives.

---

## Explicitly deferred

- **Sound** — beeper + AY not needed for Jetpac gameplay
- **TZX** — TAP is sufficient
- **Cycle-exact timing** — contended memory, floating bus
- **Snapshot formats** — .z80 / .sna can come later
- **Full opcode set** — only what Jetpac executes (grows organically in M6)

---

## Skills this covers

| Skill | Where |
|-------|-------|
| Rust systems programming | M1–M7 |
| WASM compilation + deployment | M1, M3–M7 |
| Z80 CPU architecture | M2, M6 |
| Browser/WASM interop (wasm-bindgen) | M1, M3–M5 |
| CDP / browser debugging protocol | M7 |
| Binary security (live demo) | M7 |
| Emulator architecture | M2–M6 |

## Feeds into

- **void-duel** — Layer 3 emulator host, P2P input injection via debug API
- **incant** — binary-protection.md live demo (CDP memory patching)
- **acoustic-odyssey** — Rust + WASM build pipeline patterns
