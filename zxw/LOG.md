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

- [x] CPU state struct: AF/BC/DE/HL/IX/IY/SP/PC + alternates + I/R/IFF/IM/halted
- [x] Memory trait + SpectrumMemory: 16K ROM (read-only) + 48K RAM
- [x] Fetch-decode-execute: `step(&mut self, mem) -> u32` returns T-states
- [x] Unprefixed opcodes: LD r,r / LD r,n / LD r,(HL) / arithmetic / logic / JP/JR/CALL/RET / DJNZ / PUSH/POP / EX / LDIR / HALT / NOP / EI/DI / IN/OUT stubs
- [x] CB prefix: BIT/SET/RES / rotates and shifts
- [x] ED prefix subset: LD A,I / LD A,R / IM modes / LDIR/LDDR/CPIR / NEG / RETI
- [x] Unit tests: ADD, DJNZ loop, LDIR block copy, CALL+RET, bit test

**Gate:** `cargo test` — all CPU unit tests pass. A test program (load regs, arithmetic, halt) produces expected register state.

---

## M3 — ULA screen rendering

- [x] Spectrum screen address calculation (interleaved thirds layout)
- [x] Attribute decoding: INK/PAPER/BRIGHT/FLASH per 8×8 cell
- [x] `render_frame(memory) -> Vec<u8>` → RGBA pixel buffer (256×192×4)
- [x] Spectrum palette (16 colours: 8 normal + 8 bright)
- [x] Expose to JS via wasm-bindgen, blit with `ImageData` + `putImageData`
- [x] `requestAnimationFrame` loop: step CPU ~69888 T-states per frame, then render

**Gate:** Load a 1983 Spectrum game's loading screen (6912 bytes extracted from TAP) directly into memory at 0x4000 → loading screen renders correctly with proper colours.

---

## M4 — ROM + TAP loading

- [x] ROM loader: read 16K binary into 0x0000–0x3FFF, enforce read-only
- [x] TAP parser in Rust: block lengths, flags, headers, data payloads, XOR checksums
- [x] Instant-load trap: intercept ROM's LD_BYTES (0x0556), inject TAP data directly, set carry flag
- [x] Expose `load_tap(data)` + `load_rom(data)` via wasm-bindgen
- [x] File input in HTML: drag-and-drop / `<input type="file">` for .tap and .rom

**Gate:** Load Spectrum ROM + a test TAP (simple program that writes to screen memory and halts) → character appears on canvas.

---

## M5 — Keyboard input

- [x] Keyboard port matrix: 8 half-rows, 5 bits each via IN port 0xFE
- [x] Map browser keydown/keyup → Spectrum key matrix via JS events
- [x] Wire into IN instruction handler (port 0xFE, address line selection)
- [x] Kempston joystick (port 0x1F) as alternative
- [x] Test program: poll keys in loop, change border colour per key

**Gate:** Load keyboard test program → pressing keys changes border colour in real time.

---

## M6 — 1983 game bring-up (incremental opcode completion)

- [x] Load `example_fixed.tap` (rebuilt 3-block version) → screen at 0x4000, code at 0x6000, entry 0x61E5
- [x] Unimplemented-opcode trap: log PC, opcode, context, halt — feedback loop for incremental implementation
- [x] Implement IX/IY prefix opcodes (DD/FD): indexed addressing `(IX+d)` / `(IY+d)`
- [x] Fill remaining ED-prefix gaps as the game hits them
- [x] IM 1 interrupt handling: RST 0x38 every ~69888 T-states
- [ ] Verify: title screen → key press → gameplay → movement + fire works

**Gate:** Game loads, title screen renders, game starts on keypress, player moves. ~50fps, no visible slowdown.

---

## M7 — CDP debug integration

- [x] WASM debug API: `read_memory`, `write_memory`, `read_registers`, `set_breakpoint`, `step`, `pause`, `resume`
- [x] Register as `window.zxw.*` functions for CDP access
- [x] `tools/cdp_debug.py`: connect via WebSocket, commands: `mem`, `regs`, `poke`, `bp`, `screen`, `watch`
- [x] Memory-write watchpoint mechanism (optional callback on address range write)
- [x] Test: while game runs, read score address, poke lives, dump screen

**Gate:** Game running → `python3 tools/cdp_debug.py` connects → `regs` prints Z80 state → `mem 0x4000 32` dumps screen bytes → `poke <lives_addr> 9` changes displayed lives.

---

## Demo 1 — Z80 self-verifying binary

Shows incant's `wrap_z80_checksum` in action on the live emulator.

- [x] `incant/scripts/make_protected_tap.py` — 26-byte XOR verifier prefix + payload (writes 0xFF stripe to screen), output to `zxw/www/tap/protected.tap`
- [x] `zxw/src/lib.rs` — added `set_pc` / `get_pc` to WASM API
- [x] `zxw/www/demo-protection.html` — 4-step browser demo

**Gate:** Load binary → run intact (verifier passes, screen memory written at 0x4000) → tamper byte at 0x801A → run again (verifier JP NZ → HALT at 0x8049). ✅

---

## Demo 3 — WASM self-verifying module

Shows incant's `generate_self_verify_wat` in a pure browser context.

- [x] `incant/scripts/make_protected_wasm.py` — generates `protect.wasm` (168 bytes, XOR checksum 0x1F)
- [x] `zxw/www/demo-wasm-protect.html` — 3-step browser demo

**Gate:** intact → `run()` returns 42; tamper payload byte → `WebAssembly.RuntimeError: unreachable`. ✅

---

## Explicitly deferred

- **Sound** — beeper + AY not needed for basic gameplay
- **TZX** — TAP is sufficient
- **Cycle-exact timing** — contended memory, floating bus
- **Snapshot formats** — .z80 / .sna can come later
- **Full opcode set** — only what the test game executes (grows organically in M6)

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
