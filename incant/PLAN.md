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

- [ ] Define BDD `.spec.md` input format
- [ ] Build library catalogue (reusable sigils with pre-generated code)
- [ ] Add WASI target support (fd_read, fd_write for stdin/stdout)
- [ ] `smelt` CLI subcommand: spec.md → sigils + manifest
- [ ] LLM decomposes BDD scenarios into function graph with deps
- [ ] Generate test cases from BDD When/Then pairs
- [ ] Run: write BDD spec, get working binary with no YAML by hand

**Gate:** Write a BDD spec, run `python -m incant smelt greet.spec.md`, get sigils + manifest + working .wasm that passes the scenarios.

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

---

## Feeds into

- [Void Duel](../void-duel/) — Z80 game code generation
- [Acoustic Odyssey: cast](../acoustic-odyssey/cast/) — WASM deployment pipeline patterns
