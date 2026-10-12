# Next: zxw M2 + Incant M8 (parallel tracks)

Two independent workstreams. Both predecessors complete (zxw M1 ✅, incant M7 ✅). Different directories, no shared files — safe to run simultaneously via `/research`.

---

## Track 1: zxw M2 — Z80 CPU core (foundational opcodes)

Build the Z80 CPU struct and decode-execute loop. Pure Rust, tested with `cargo test`.

- [ ] `src/cpu.rs` — Cpu struct: AF/BC/DE/HL/IX/IY/SP/PC + alternates + I/R/IFF/IM/halted
- [ ] `src/memory.rs` — 64K flat array, ROM read-only guard, read/write helpers
- [ ] Fetch-decode-execute: `step(&mut self, mem) -> u32` returns T-states
- [ ] Unprefixed opcodes: LD, arithmetic, logic, JP/JR/CALL/RET, DJNZ, PUSH/POP, EX, LDIR, HALT, EI/DI, IN/OUT stubs
- [ ] CB prefix: BIT/SET/RES, rotates and shifts
- [ ] ED prefix subset: LD A,I/R, IM modes, LDIR/LDDR/CPIR, NEG, RETI
- [ ] Unit tests: ADD, DJNZ loop, LDIR block copy, CALL+RET, bit test

**Gate:** `cargo test` — all CPU unit tests pass.

**Dir:** `zxw/`

---

## Track 2: Incant M8 — TAP read/write + disassembly

TAP file pipeline: write `.bin` → `.tap`, read `.tap` → disassemble, feed into RAG.

- [ ] TAP writer: wrap `.bin` → `.tap` (header + data blocks, XOR checksums)
- [ ] Add `--tap` flag to Z80 gate output
- [ ] TAP reader: parse blocks, extract code bytes
- [ ] Disassembler: code bytes → Z80 asm (`z80dis`)
- [ ] `disasm` CLI subcommand: `.tap` → `.asm`
- [ ] Knowledge pipeline: disassembled code → chunked → embedded into RAG
- [ ] Round-trip test: sigil → asm → bin → tap → extract → disasm → compare

**Gate:** `python -m incant cast sigils/examples/z80_add.sigil.yaml --tap` produces a loadable `.tap`. `python -m incant disasm game.tap` extracts annotated Z80 assembly.

**Dir:** `incant/`

---

## Completed

- [x] zxw M1 — scaffold + WASM canvas hello world
- [x] Incant M7 — unified harness library (Z80Harness + WasmHarness)

---

## What's next after these

| Milestone | Depends on | Parallel? |
|-----------|-----------|-----------|
| zxw M3 (ULA screen) | zxw M2 | Sequential after Track 1 |
| zxw M4 (ROM + TAP loading) | zxw M3 | Sequential |
| zxw M5 (keyboard) | zxw M3 | Can parallel with M4 |
| zxw M6 (Jetpac bring-up) | M4 + M5 | Sequential |
| zxw M7 (CDP debug) | M6 | Sequential |
| Incant M9 (binary protection PoC) | zxw M7 + incant M8 | Both tracks converge here |
| Incant M10 (ONNX → Z80) | incant M8 | Independent |
