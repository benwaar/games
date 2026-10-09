# Next: M3 — WAT backend (end-to-end)

M2 is complete — Z80 backend generates, assembles, and tests all 3 sigils on the first attempt. Two bugs fixed: assembler requires lowercase mnemonics, and `code.resolve()` must be called before `code.encode()` for label resolution.

**What's next:**
1. WAT prompt template — system prompt + sigil-to-prompt formatting
2. WAT gate — `wat2wasm` to assemble, `wasm-interp --run-all-exports` to test
3. Parse wasm-interp output to check against sigil expected values
4. Run on 3 example sigils: add, factorial, memory read/write

**Gate:** All 3 WAT sigils produce .wasm that passes all test cases.
