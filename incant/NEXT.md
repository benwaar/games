# Next: M7 — Test Harness: unified I/O for both targets

M6 is complete — smelt pipeline works end-to-end. Z80 harness_io added for memory-mapped string I/O. 72 tests pass.

**What's next:**

M7 extracts the test/run logic into a reusable harness library with the same API for both WASM and Z80. One command runs either target.

```python
from incant.harness import WasmHarness, Z80Harness

h = Z80Harness("output/greet.bin")
result = h.run("Ben")          # → "hello Ben"

h = WasmHarness("output/greet_user.wasm")
result = h.run("Ben")          # → "hello Ben"
```

**Steps:**
1. Create `incant/harness.py` — `Z80Harness` and `WasmHarness` with shared interface
2. Refactor gate test runners to use the harness
3. Simplify run scripts to one-liners
4. CLI: `python -m incant run output/greet.bin "Ben"`
5. Tests for both harnesses
6. Docs

**Gate:** `python -m incant run output/greet.bin "Ben"` and `python -m incant run output/greet_user.wasm "Ben"` both print `hello Ben`.
