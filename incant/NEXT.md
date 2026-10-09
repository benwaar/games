# Next: M4 — Multi-sigil orchestration

M3 is complete — WAT backend generates, assembles, and tests all 3 sigils. Fixed wasm-interp CLI syntax (`-r name -a i32:N` instead of fabricated `-- args`), added signed/unsigned i32 conversion for negative values. memory_swap needed 3 attempts — clarifying the sigil description fixed the LLM's output.

**What's next:**
1. Read a sequence of sigils, generate in dependency order
2. Pass prior outputs as context (later sigils reference earlier functions)
3. Build a small Z80 program from 5+ sigils
4. Build a small WASM module from 5+ sigils

**Gate:** Multi-sigil builds produce working binaries.
