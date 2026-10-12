# Next: zxw M7 — CDP debug integration

zxw M6 (Jetpac bring-up) complete — DD/FD indexed opcodes, DDCB/FDCB double prefix, IM 1 interrupts, unimplemented-opcode trap. 61 tests passing. Gameplay verification pending (needs actual Jetpac TAP + browser test).

---

## zxw M7 — CDP debug integration

- [ ] WASM debug API: `read_memory`, `write_memory`, `read_registers`, `set_breakpoint`, `step`, `pause`, `resume`
- [ ] Register as `window.zxw.*` functions for CDP access
- [ ] `tools/cdp_debug.py`: connect via WebSocket, commands: `mem`, `regs`, `poke`, `bp`, `screen`, `watch`
- [ ] Memory-write watchpoint mechanism (optional callback on address range write)
- [ ] Test: while Jetpac runs, read score address, poke lives, dump screen

**Gate:** Jetpac running → `python3 tools/cdp_debug.py` connects → `regs` prints Z80 state → `mem 0x4000 32` dumps screen bytes → `poke <lives_addr> 9` changes displayed lives.

**Dir:** `zxw/`

---

## Also ready (independent)

- [ ] Incant M10 (ONNX → Z80) — depends only on incant M8 ✅

---

## Completed

- [x] zxw M1 — scaffold + WASM canvas hello world
- [x] zxw M2 — Z80 CPU core (11 tests, all unprefixed + CB + ED opcodes)
- [x] zxw M3 — ULA screen rendering (17 tests, Spectrum palette + interleaved layout)
- [x] zxw M4 — ROM + TAP loading (TAP parser, instant-load trap, 11 integration tests)
- [x] zxw M5 — Keyboard input (port 0xFE matrix, Kempston joystick, browser mapping, 11 tests)
- [x] zxw M6 — Jetpac bring-up (DD/FD indexed, DDCB/FDCB, IM 1 interrupts, 22 CPU tests)
- [x] Incant M7 — unified harness library (Z80Harness + WasmHarness)
- [x] Incant M8 — TAP read/write + disassembly (25 tests, --tap flag, disasm CLI)

---

## What's next after M7

| Milestone | Depends on | Parallel? |
|-----------|-----------|-----------|
| Incant M9 (binary protection PoC) | zxw M7 + incant M8 | Both projects converge here |
| Incant M10 (ONNX → Z80) | incant M8 | Independent — could start now |
