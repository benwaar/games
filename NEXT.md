# Next: Incant M9 — Binary Protection PoC

All other milestones complete. One remaining: self-verifying code in both Z80 and WASM.

---

## Incant M9 — Binary Protection PoC: Self-Verifying Code

- [ ] Write Z80 sigil: `self_verify.sigil.yaml` — generates code with XOR self-check
- [ ] Post-assembly checksum patching in the Z80 gate
- [ ] Test: modify one byte of the binary, verify the check catches it
- [ ] Write WAT sigil: `self_verify_wasm.sigil.yaml` — data-segment mirror + hash check
- [ ] Test: modify the data segment, verify the check catches it
- [ ] Explainer update: add concrete examples from the PoC to binary-protection.md
- [ ] Document the attacker's bypass for each (patch analysis)

**Gate:** Both sigils produce binaries that pass tests normally but detect and reject single-byte tampering. The explainer includes worked examples of both the protection and the bypass.

**Dir:** `incant/`

**Depends on:** zxw M7 ✅ (CDP debug for live demo) + incant M8 ✅

---

## Completed

### zxw (all 7 milestones — project complete)
- [x] M1 — scaffold + WASM canvas hello world
- [x] M2 — Z80 CPU core (11 tests)
- [x] M3 — ULA screen rendering (17 tests)
- [x] M4 — ROM + TAP loading (11 tests)
- [x] M5 — Keyboard input (11 tests)
- [x] M6 — DD/FD indexed opcodes + IM 1 interrupts (22 CPU tests)
- [x] M7 — CDP debug integration (11 tests, 72 total)

### incant (M1–M8 + M10 complete, M9 remaining)
- [x] M1–M6 — sigil pipeline, Z80/WAT backends, multi-sigil, BDD smelt
- [x] M7 — unified harness library
- [x] M8 — TAP read/write + disassembly
- [x] M10 — ONNX → Z80 inference engine (11 tests, 100% accuracy)
