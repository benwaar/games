# WAT Backend — Assembly, Execution, and Signed Integer Handling

How incant generates, assembles, and tests WebAssembly modules using a local LLM.

## The pipeline

```
sigil → RAG context → prompt → LLM → extract code → wat2wasm → wasm-interp → .wasm
                                        ↑                          |
                                        └── retry with error ──────┘  (max 3)
```

Same shape as the Z80 backend, different tools at each step.

## Prompt construction

The WAT target builds a prompt from three parts:

1. **RAG context** — top-5 relevant chunks from the WAT knowledge base (types, instructions, memory patterns)
2. **System prompt** — rules: output only a `(module ...)`, use flat stack style, export with the exact name, declare memory if needed
3. **Sigil spec** — function name, signature, memory pages, one example test case

```python
# From incant/targets/wat.py
def build_prompt(sigil, rag_context):
    params = ", ".join(f"{p.name}: {p.type}" for p in sigil.signature.inputs)
    memory_note = ""
    if sigil.memory:
        memory_note = f"\nMemory: {sigil.memory} page(s) — include (memory {sigil.memory}) and export it."
```

The prompt prefers flat (stack) style over folded S-expressions. This is a readability choice — flat style maps more directly to the WASM stack machine, making generated code easier to reason about.

## Assembly

`wat2wasm` from the WABT toolkit compiles WAT text to `.wasm` binary:

```python
def assemble(source):
    result = subprocess.run(
        ["wat2wasm", str(wat_path), "-o", str(wasm_path)],
        capture_output=True, text=True,
    )
```

> **Coming from C:** This is the equivalent of `gcc -c` — text to object code. The difference is that WAT → WASM is a 1:1 translation (no optimisation pass), so what you write is exactly what runs.

> **Coming from JS/TS:** WAT is to WASM what TypeScript is to JavaScript — a human-readable text format that compiles to a binary format the runtime actually executes.

## Test execution with wasm-interp

`wasm-interp` (also from WABT) runs a `.wasm` file. Key flags:

- `-r name` — call the exported function `name`
- `-a type:value` — pass a typed argument (e.g. `-a i32:5`)

```bash
wasm-interp output.wasm -r add -a i32:2 -a i32:3
# => add(i32:2, i32:3) => i32:5
```

### The signed integer problem

WASM uses two's complement internally, but `wasm-interp` only accepts unsigned values on the command line. `-a i32:-1` fails. You must convert:

```python
def _to_unsigned(value: int, bits: int = 32) -> int:
    if value < 0:
        return value + (1 << bits)  # -1 → 4294967295
    return value
```

Output comparison has the inverse problem — `wasm-interp` returns unsigned, but the sigil's expected value may be signed:

```python
def _to_signed(value: int, bits: int = 32) -> int:
    if value >= (1 << (bits - 1)):
        return value - (1 << bits)  # 4294967295 → -1
    return value
```

The `run_test` function converts inputs to unsigned for the CLI and converts outputs back to signed for comparison when the expected value is negative.

> **Coming from C:** This is the same `(int32_t)` / `(uint32_t)` cast you'd do, just explicit because Python doesn't have fixed-width integer types.

> **Coming from JS/TS:** JavaScript's `Int32Array` does this implicitly — `new Int32Array([4294967295])[0]` gives `-1`. Python makes you do the bit arithmetic yourself.

### Output parsing

`wasm-interp` prints results in the format `func_name(type:val, ...) => type:val`. We parse the result after `=>`, strip the type prefix, and compare:

```python
result_str = output.split("=>")[-1].strip()
if ":" in result_str:
    result_str = result_str.split(":")[-1]
actual = int(result_str)
```

## Code extraction

Same pattern as Z80 — the LLM sometimes wraps output in markdown fences despite being told not to. `extract_code()` handles three cases:

1. **Fenced code** — strips ` ```wat ` / ` ``` ` markers
2. **Module detection** — finds `(module` and matches parentheses to extract the complete module
3. **Fallback** — returns raw text and lets `wat2wasm` report the error

The parenthesis-matching approach (case 2) is more robust than the Z80 version because WAT's S-expression syntax guarantees balanced parens.

## Retry loop

Same gate-driven retry as Z80 — assembler errors and test failures are fed back to the LLM for correction. In practice:

- **add** — passes on attempt 1 (trivial)
- **factorial** — passes on attempt 1 (standard loop pattern)
- **memory_swap** — needed 3 attempts (the LLM initially returned the wrong value after the swap; error feedback showing "Expected 10, got 20" helped it self-correct)

The memory_swap case is instructive: the sigil description had to be made very explicit ("Return the i32 at offset 4 after the swap, which is the original value of a") to guide the LLM. Ambiguous specs produce ambiguous code — the gate catches it, but a clearer spec avoids wasted retries.

> **In practice:** This is a core production RAG lesson — when the LLM repeatedly fails on the same test, the issue is often the spec, not the model. Tightening the description (or adding a second example) is cheaper than adding retries. The gate measures correctness; the spec controls intent.

## Example: what the LLM generates

### add (a + b → i32)
```wat
(module
  (func $add (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.add
  )
  (export "add" (func $add))
)
```
Minimal stack machine: push two locals, add, return.

### factorial (n → i32, loop)
```wat
(module
  (func $factorial (param $n i32) (result i32)
    (local $result i32)
    i32.const 1
    local.set $result
    (block $break
      (loop $loop
        local.get $n
        i32.const 1
        i32.le_s
        br_if $break
        local.get $result
        local.get $n
        i32.mul
        local.set $result
        local.get $n
        i32.const 1
        i32.sub
        local.set $n
        br $loop))
    local.get $result)
  (export "factorial" (func $factorial))
)
```
Uses `block`/`loop`/`br_if` — WASM's structured control flow (no goto, no arbitrary jumps).

### memory_swap (store, swap, return)
```wat
(module
  (memory 1)
  (func $memory_swap (param $a i32) (param $b i32) (result i32)
    i32.const 0  local.get $a  i32.store    ;; mem[0] = a
    i32.const 4  local.get $b  i32.store    ;; mem[4] = b
    ;; swap via locals
    i32.const 4  i32.load  local.set 0      ;; temp0 = mem[4]
    i32.const 0  i32.load  local.set 1      ;; temp1 = mem[0]
    i32.const 0  local.get 0  i32.store     ;; mem[0] = temp0
    i32.const 4  local.get 1  i32.store     ;; mem[4] = temp1
    i32.const 4  i32.load)                  ;; return mem[4] (original a)
  (export "memory_swap" (func $memory_swap))
  (export "memory" (memory 0))
)
```
Linear memory: `i32.store` writes 4 bytes at an offset, `i32.load` reads them back. Memory is exported so a host could inspect it.

## Z80 vs WAT comparison

| Aspect | Z80 | WAT |
|--------|-----|-----|
| Assembly tool | `z80.Asm()` (Python) | `wat2wasm` (CLI) |
| Test runner | `z80.Z80Machine()` (Python) | `wasm-interp` (CLI) |
| I/O model | Registers (A, B, C...) | Stack machine (params → stack → result) |
| Memory | Flat 64K, direct addressing | Linear memory, page-based (64K pages) |
| Control flow | JP/JR/DJNZ (goto) | block/loop/br (structured) |
| Signed handling | CPU flags (sign bit) | Separate instructions (`i32.add` vs `i32.lt_s` / `i32.lt_u`) |
| Case quirk | Assembler requires lowercase | No case sensitivity issues |
