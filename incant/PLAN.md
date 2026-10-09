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

- [ ] Explainers: WAT format, Z80 instruction set, RAG for code gen, sigil design
- [ ] Update STUDY.md skills coverage
- [ ] Update root CLAUDE.md project table
- [ ] Demo script: `bash demo.sh`

**Gate:** `bash demo.sh` works from cold clone after `bash setup.sh`.

---

## Skills this covers

| Skill | Gap filled |
|-------|-----------|
| Spec → assembly (WAT, Z80) | ⬜ → ✅ |
| RAG-grounded code generation | ⬜ → ✅ |
| Gate-driven LLM iteration | ⬜ → ✅ |
| WASM binary pipeline | ⬜ → ✅ |
| Z80 assembly | ⬜ → ✅ |

---

## Feeds into

- [Void Duel](../void-duel/) — Z80 game code generation
- [Acoustic Odyssey: cast](../acoustic-odyssey/cast/) — WASM deployment pipeline patterns
