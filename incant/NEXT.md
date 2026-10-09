# Next: M2 — Z80 backend (end-to-end)

M1 is complete — project scaffolded, knowledge embedded (33 Z80 + 38 WAT chunks), RAG retrieval verified.

**What's next:**
1. Z80 prompt template — system prompt + sigil-to-prompt formatting
2. Z80 gate — assemble with `z80.Asm()`, run on `z80.Z80Machine()`, check registers
3. Retry loop — feed assembler errors back to LLM (max 3 retries)
4. Run on 3 example sigils: add, loop counter, memory store/load

**Gate:** All 3 Z80 sigils produce assembled code that passes all test cases.
