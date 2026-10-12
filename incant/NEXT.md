# Next: M8 — TAP: Spectrum tape files + disassembly

M7 complete — unified harness library in place. Both targets use the same `.run(input)` interface. 33 tests pass.

**What was built:**

`incant/harness.py` exports `Z80Harness` and `WasmHarness`:

```python
from incant.harness import WasmHarness, Z80Harness

h = Z80Harness("output/greet.bin")
result = h.run("Ben")          # → "hello Ben"

h = WasmHarness("output/greet_user.wasm")
result = h.run("Ben")          # → "hello Ben"
```

Both harnesses accept a compiled binary, load string input, execute, return string output, and raise `TimeoutError` on runaway programs.

The `run` CLI subcommand uses the harness:

```bash
python -m incant run output/greet.bin "Ben"
python -m incant run output/greet_user.wasm "Ben"
```

**What's next:**

M8 reads and writes ZX Spectrum TAP files. Two directions:

1. **TAP output (write):** wrap incant's Z80 `.bin` output into loadable `.tap` files (header + data block, XOR checksum).
2. **Disassembly (read):** parse `.tap` files, extract code bytes, disassemble with `z80dis`.
3. **Knowledge extraction:** feed disassembled real-world patterns into RAG.

**Steps:**
1. TAP writer: `.bin` → `.tap` (header + data blocks, checksums)
2. Add `--tap` flag to Z80 gate output
3. TAP reader: parse blocks, extract code bytes
4. Disassembler: code bytes → Z80 asm (`z80dis`)
5. `disasm` CLI subcommand: `.tap` → `.asm`
6. Knowledge pipeline: disassembled code → embedded into RAG
7. Round-trip test: sigil → asm → bin → tap → extract → disasm → compare
