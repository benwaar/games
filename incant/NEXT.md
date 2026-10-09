# Next: M7 — TAP: Spectrum tape files + disassembly

M6 is complete — smelt pipeline works end-to-end. BDD spec → LLM decomposition → WASI binary with stdin/stdout I/O. All 69 tests pass.

**What's next:**

M7 adds TAP file support: wrap Z80 `.bin` output into loadable `.tap` files, extract code from real Spectrum games, disassemble it, and feed it back into the pipeline as RAG knowledge.

```
sigil → asm → bin → TAP (write)
TAP → extract → disasm → asm (read)
disasm → chunk → embed → RAG (knowledge)
```

**Steps:**
1. TAP writer: wrap `.bin` → `.tap` (header + data blocks, checksums)
2. Add `--tap` flag to Z80 gate output
3. TAP reader: parse blocks, extract code bytes
4. Disassembler: code bytes → Z80 asm (`z80dis`)
5. `disasm` CLI subcommand: `.tap` → `.asm`
6. Knowledge pipeline: disassembled code → chunked → embedded into RAG
7. Round-trip test: sigil → asm → bin → tap → extract → disasm → compare

**Gate:** `python -m incant cast sigils/examples/z80_add.sigil.yaml --tap` produces a `.tap` loadable in FUSE. `python -m incant disasm game.tap` extracts annotated Z80 assembly.
