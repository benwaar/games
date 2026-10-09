# incant — Plan

Spec-driven code gen: sigil → LLM → WAT/Z80 asm → assemble → test → binary.

---

## M1 — Project scaffold + knowledge base

- [x] Create project structure, README, PLAN, NEXT, CLAUDE.md
- [x] Curate Z80 knowledge: core opcodes, registers, flags, memory map
- [x] Curate WAT knowledge: S-expression syntax, types, instructions, memory
- [x] Embed knowledge into vector store via nomic-embed-text
- [x] Write sigil format parser
- [x] Write 3 example sigils per target (6 total)
- [x] Test: RAG query returns relevant docs

**Gate:** `python -m incant rag query "add two numbers" --collection z80` returns relevant chunks.

---

## M2 — Z80 backend (end-to-end)

- [x] Implement Z80 prompt template
- [x] Implement Z80 gate: assemble with `z80.Asm()`, run on `z80.Z80Machine()`, check registers
- [x] Retry loop: feed assembler errors back to LLM (max 3 retries)
- [x] Run on 3 example sigils: add, loop counter, memory store/load

**Gate:** All 3 sigils produce assembled code that passes all test cases.

---

## M3 — WAT backend (end-to-end)

- [x] Implement WAT prompt template
- [x] Implement WAT gate: `wat2wasm` to assemble, `wasm-interp -r name -a type:val` to test
- [x] Parse wasm-interp output to check against sigil expected values
- [x] Handle signed/unsigned i32 conversion (wasm-interp rejects negative args)
- [x] Run on 3 example sigils: add, factorial, memory swap

**Gate:** All 3 sigils produce .wasm that passes all test cases.

---

## M4 — Multi-sigil orchestration

- [x] Manifest format: YAML listing sigils + target
- [x] WAT stitcher: merge modules, dedup functions/memory/exports
- [x] Z80 stitcher: concatenate asm, strip intermediate halts
- [x] `multi` CLI subcommand
- [x] Topological sort + dependency-aware context injection
- [x] Dependent sigil example: `sum_of_factorials` calls `factorial` + `add`
- [x] Live run: `wat_math` (2 funcs), `z80_basics` (3 funcs), `wat_composed` (3 funcs with deps)

**Gate:** Multi-sigil builds produce working binaries. WAT composed build has cross-function calls.

---

## M5 — Docs + integration

- [x] Explainers: WAT format, Z80 instruction set, RAG for code gen, sigil design
- [x] Update STUDY.md skills coverage
- [x] Update root CLAUDE.md project table
- [x] Demo script: `bash demo.sh`

**Gate:** `bash demo.sh` works from cold clone after `bash setup.sh`.

---

## M6 — Smelt: BDD spec → sigils → binary

Human writes intent as a BDD-format markdown file. The pipeline interprets it, checks available libraries, generates sigils for anything missing, and runs the existing pipeline to produce WASM and Z80 binaries.

### Input format (BDD markdown)

```markdown
# Greet user

## Scenario: basic greeting
Given the program starts
When the user enters "Ben"
Then output "hello Ben"

## Scenario: empty name
Given the program starts
When the user enters ""
Then output "hello "
```

### Pipeline

```
human writes .spec.md (BDD)
  → smelt reads spec
  → LLM checks library catalogue (what already exists?)
  → LLM generates sigils for missing functions only
  → LLM writes manifest (new sigils + library sigils)
  → existing cast/multi pipeline runs
  → .wasm + .asm output
```

### Library catalogue

Reusable sigils the LLM can reference without regenerating:

```
libs/
  wat/
    io.sigil.yaml       — read_line, print_string (WASI)
    string.sigil.yaml   — concat, length
    math.sigil.yaml     — add, multiply, factorial
  z80/
    io.sigil.yaml       — print_char, read_key (Spectrum RST calls)
    math.sigil.yaml     — add, multiply, divide
```

The LLM gets the catalogue as context and picks what it needs. Only generates new sigils for logic not in the library.

### Steps

- [x] Define BDD `.spec.md` input format
- [x] Build library catalogue (reusable sigils with pre-generated code)
- [x] Add WASI target support (fd_read, fd_write for stdin/stdout)
- [x] `smelt` CLI subcommand: spec.md → sigils + manifest
- [x] LLM decomposes BDD scenarios into function graph with deps
- [x] Generate test cases from BDD When/Then pairs
- [x] Run: write BDD spec, get working binary with no YAML by hand

**Gate:** ✅ `python -m incant smelt specs/greet.spec.md -v` produces sigils + manifest + working WASI binary that passes all 3 BDD scenarios.

---

## M7 — Test Harness: unified I/O for both targets

A reusable test harness library with the same API for WASM and Z80. Both targets get: load input → run program → read output → compare. The harness is a Python module that the gate, run scripts, and future experiments all import. No more copy-pasting Z80 memory logic into shell scripts.

### What the harness provides

```python
from incant.harness import WasmHarness, Z80Harness

# Same interface, different backends
h = Z80Harness("output/greet.bin")
result = h.run("Ben")          # → "hello Ben"

h = WasmHarness("output/greet_user.wasm")
result = h.run("Ben")          # → "hello Ben"
```

Both harnesses:
- Accept a compiled binary (`.wasm` or `.bin`)
- Load a string input (stdin for WASM, memory buffer at 0x8000 for Z80)
- Execute the program
- Return the string output (stdout for WASM, memory buffer at 0x9000 for Z80)
- Timeout protection (no infinite loops)

### Why this matters

Right now the test logic is scattered: `run_wasi_test()` in `wat.py`, `run_harness_io_test()` in `z80.py`, inline Python in shell scripts. A unified harness means:
- Run scripts are one-liners
- Future experiments (M8 binary protection, M9+ TAP) import the harness
- Same test pattern for both architectures
- Easy to add new targets later

### Steps

- [ ] Create `incant/harness.py` — `Z80Harness` and `WasmHarness` classes with shared interface
- [ ] Refactor `z80.py` `run_harness_io_test()` to use `Z80Harness`
- [ ] Refactor `wat.py` `run_wasi_test()` to use `WasmHarness`
- [ ] Simplify run scripts to use the harness module
- [ ] CLI: `python -m incant run output/greet.bin "Ben"` — run any compiled output
- [ ] Tests for both harnesses (deterministic, no LLM)
- [ ] Docs: explainer, update README + CLAUDE.md

**Gate:** `python -m incant run output/greet.bin "Ben"` and `python -m incant run output/greet_user.wasm "Ben"` both print `hello Ben`. Same command, either target.

---

## M8 — TAP: Spectrum tape files + disassembly

Read and write ZX Spectrum TAP files. Extract machine code from real Spectrum games, disassemble it, and feed it back into the pipeline as knowledge. Closes the loop: sigil → asm → binary → TAP → extract → asm.

### TAP output (write)

Wrap incant's Z80 `.bin` output into loadable `.tap` files. A TAP is two blocks — a 19-byte header (type=3 Code, filename, start address, length) and a data block — each with a length prefix and XOR checksum. Pure Python, no dependencies.

### Disassembly (read)

Extract code blocks from `.tap` files and disassemble to Z80 assembly. Use `z80dis` (pure Python) for programmatic disassembly, with `skoolkit` as an optional path for annotated output from full game ROMs.

### Knowledge extraction

Feed disassembled real-world Z80 patterns into RAG — actual Spectrum game code as grounding context for the LLM.

### Steps

- [ ] TAP writer: wrap `.bin` → `.tap` (header + data blocks, checksums)
- [ ] Add `--tap` flag to Z80 gate output
- [ ] TAP reader: parse blocks, extract code bytes
- [ ] Disassembler: code bytes → Z80 asm (`z80dis`)
- [ ] `disasm` CLI subcommand: `.tap` → `.asm`
- [ ] Knowledge pipeline: disassembled code → chunked → embedded into RAG
- [ ] Round-trip test: sigil → asm → bin → tap → extract → disasm → compare

**Gate:** `python -m incant cast sigils/examples/z80_add.sigil.yaml --tap` produces a `.tap` loadable in FUSE. `python -m incant disasm game.tap` extracts annotated Z80 assembly.

---

## M9 — Binary Protection PoC: Self-Verifying Code

Explore binary protection techniques from the [explainer](explainers/binary-protection.md) by building a **checksum self-verification** PoC — in both Z80 and WASM. The program computes a hash of its own code at runtime and refuses to run if it's been tampered with.

This is the simplest protection technique that works identically on both targets, and it directly uses incant's existing pipeline. It's also the foundation for more advanced techniques (anti-debug, obfuscation, staged loading) if we go further.

### Z80: self-verifying binary

A sigil that generates Z80 code which:
1. Computes XOR checksum of its own code region in memory
2. Compares against an expected value
3. Runs normally if valid, halts/crashes if tampered

The gate assembles the code, patches in the correct checksum post-assembly, and verifies that modifying any byte causes the check to fail.

### WASM: self-verifying module

A sigil that generates WAT code which:
1. Copies its own function body into linear memory (via a data segment or init routine)
2. Computes a hash over that region
3. Branches to real logic or trap based on the result

WASM can't read its own code section directly (Harvard architecture — code and data are separate). So the verification has to work over a data segment that mirrors the code, or over a known region of linear memory. This is a real constraint that makes WASM protection harder than Z80.

### What we learn

- How self-verification works at the byte level on two very different architectures
- Why WASM's code/data split changes the game (and what workarounds exist)
- How to write sigils that generate protection code, not just application code
- The attacker's perspective: how to patch a self-verifying binary (remove the check, fix the checksum, or NOP the branch)

### Steps

- [ ] Write Z80 sigil: `self_verify.sigil.yaml` — generates code with XOR self-check
- [ ] Post-assembly checksum patching in the Z80 gate
- [ ] Test: modify one byte of the binary, verify the check catches it
- [ ] Write WAT sigil: `self_verify_wasm.sigil.yaml` — data-segment mirror + hash check
- [ ] Test: modify the data segment, verify the check catches it
- [ ] Explainer update: add concrete examples from the PoC to [binary-protection.md](explainers/binary-protection.md)
- [ ] Document the attacker's bypass for each (patch analysis)

**Gate:** Both sigils produce binaries that pass tests normally but detect and reject single-byte tampering. The explainer includes worked examples of both the protection and the bypass.

---

## Skills this covers

| Skill | Gap filled |
|-------|-----------|
| Spec → assembly (WAT, Z80) | ⬜ → ✅ |
| RAG-grounded code generation | ⬜ → ✅ |
| Gate-driven LLM iteration | ⬜ → ✅ |
| WASM binary pipeline | ⬜ → ✅ |
| Z80 assembly | ⬜ → ✅ |
| Multi-step LLM decomposition | ⬜ → ✅ (M6) |
| Intent → spec → code pipeline | ⬜ → ✅ (M6) |
| Unified test harness (WASM + Z80) | ⬜ → ✅ (M7) |
| Binary format read/write (TAP) | ⬜ → ✅ (M8) |
| Disassembly + reverse engineering | ⬜ → ✅ (M8) |
| Real-world code → RAG knowledge | ⬜ → ✅ (M8) |
| Binary self-verification (Z80 + WASM) | ⬜ → ✅ (M9) |
| Protection vs analysis (attacker model) | ⬜ → ✅ (M9) |

---

## Feeds into

- [Void Duel](../void-duel/) — Z80 game code generation
- [Acoustic Odyssey: cast](../acoustic-odyssey/cast/) — WASM deployment pipeline patterns
