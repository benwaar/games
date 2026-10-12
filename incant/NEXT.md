# Next: All milestones complete

M9 (Binary Protection PoC) is complete. All incant milestones (M1–M10) are done.

**What M9 built:**

- `incant/protect.py` — self-verifying binary wrappers for Z80 and WASM
  - `wrap_z80_checksum()` — 26-byte XOR verifier prefix with post-assembly patching
  - `generate_self_verify_wat()` — data-segment mirror + hash check WAT module
  - Bypass helpers: `bypass_nop_branch()`, `bypass_fix_checksum()`, `tamper_wasm_data_segment()`
- 16 new tests in `tests/test_protect.py` — intact runs, tamper detection, every-byte sweep, attacker bypasses
- Explainer update: `explainers/binary-protection.md` now has worked PoC sections for both targets

149 tests pass.

**Gate:** Both Z80 and WASM binaries detect single-byte tampering. Attacker bypasses demonstrated and documented.

**What's next:**

The incant pipeline is feature-complete. Possible extensions:
- **WASM inference engine** — same INT8 quantised model, but generating WAT instead of Z80
- **More protection techniques** — anti-debug timing checks, multi-stage loading, obfuscation
- **Integration with zxw** — run protected Z80 binaries in the emulator, demonstrate CDP bypass live
