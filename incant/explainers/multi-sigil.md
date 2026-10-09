# Multi-Sigil Orchestration — Manifests, Stitching, and Dependencies

How incant composes multiple functions into a single program.

## The problem

M1–M3 generate one function per sigil: one `.wasm` or one `.bin`. Real programs have multiple functions that call each other. M4 adds:

1. **Manifests** — list multiple sigils to build together
2. **Stitching** — merge generated outputs into one file
3. **Dependencies** — later functions can call earlier ones

## Manifests

A manifest is a YAML file that names the target and lists sigil paths:

```yaml
# sigils/programs/wat_composed.manifest.yaml
name: math_composed
target: wat
sigils:
  - ../examples/wat_add.sigil.yaml
  - ../examples/wat_factorial.sigil.yaml
  - ../examples/wat_sum_of_factorials.sigil.yaml
```

All sigils must share the same target. Paths are relative to the manifest file.

```python
# From incant/manifest.py
@dataclass
class Manifest:
    name: str
    target: str
    sigils: list[Sigil]
    sigil_paths: list[Path]
```

> **Coming from JS/TS:** Think of it as a `tsconfig.json` that lists the files to compile — except here, the "compile" step also generates the source code.

## Stitching

Each sigil is generated independently (each passes its own tests), then the outputs are merged.

### WAT stitching

WAT modules are S-expressions. The stitcher:

1. Parses each `(module ...)` into top-level declarations
2. Deduplicates functions by name (first definition wins)
3. Deduplicates exports by name
4. Merges memory declarations (keeps the largest page count)
5. Wraps everything in a single `(module ...)`

```python
# From incant/stitch.py
def stitch_wat(modules: list[str]) -> str:
    # split each module into declarations
    # skip if func name or export name already seen
    # combine into one module
```

The dedup is essential when a dependent sigil re-defines its dependencies. The LLM often includes `$add` and `$factorial` inside `sum_of_factorials` to make it self-contained — the stitcher strips the copies and keeps the originals.

> **Coming from C:** This is the linker's "multiple definition" resolution. In C, multiple definitions are an error; here, we silently keep the first (since all definitions came from the same pipeline).

### Z80 stitching

Z80 programs are flat — concatenate them, strip intermediate `halt` instructions so execution falls through, add section comments:

```asm
; --- add ---
add a, b

; --- counter ---
ld a, 0
loop:
    inc a
    djnz loop

; --- store_load ---
ld (0x8000), a
ld a, (0x8000)
halt
```

Only the last program keeps its `halt`.

**Z80 limitation:** Stitched Z80 programs share registers and memory. You can't re-test individual functions against the combined binary because earlier programs modify the machine state. Each function is tested in isolation during generation; the combined binary only verifies that it assembles.

> **Coming from JS/TS:** Z80 stitching is like concatenating scripts — they share global scope. WAT stitching is like bundling ES modules — each function is isolated and callable by name.

## Dependencies

Sigils can declare dependencies on other sigils by name:

```yaml
# wat_sum_of_factorials.sigil.yaml
dependencies:
  - factorial
  - add
```

### Topological sort

The manifest sorts sigils so dependencies are generated before dependents. Uses a standard DFS-based topological sort with cycle detection.

```python
def topological_sort(sigils: list[Sigil]) -> list[Sigil]:
    # DFS with cycle detection
    # raises ValueError on circular dependencies
```

> **Coming from C:** Same algorithm as `make` uses for build dependencies. `a → b → c` means generate `a` first, then `b`, then `c`.

### Context injection

When generating a sigil with dependencies, the already-generated code of its dependencies is injected into the LLM prompt:

```python
dep_context = ""
if sigil.dependencies:
    dep_parts = []
    for dep_name in sigil.dependencies:
        if dep_name in generated_code:
            dep_parts.append(f"```\n{generated_code[dep_name]}\n```")
    dep_context = "\n\n".join(dep_parts)
```

This means the LLM sees the real function signatures it can call — not documentation about them, but the actual generated code. This is why `sum_of_factorials` correctly emits `call $factorial` and `call $add` on the first attempt.

> **In practice:** This is the same pattern used in multi-step code generation systems. Earlier outputs become context for later steps. The risk is that errors in early outputs propagate — but since each step is gated (assemble + test), errors are caught before they can compound.

## Combined verification

After stitching, the combined binary is assembled and verified:

- **WAT:** Each function is independently callable, so all test cases from all sigils are re-run against the combined `.wasm`
- **Z80:** Programs share state, so only assembly success is verified (individual tests already passed)

## Example: what gets built

### `math_composed` (WAT, 3 functions)

```
wat_add.sigil.yaml        → $add(a, b) → i32
wat_factorial.sigil.yaml  → $factorial(n) → i32
wat_sum_of_factorials     → $sum_of_factorials(a, b) → i32
                              calls $factorial(a) + $factorial(b) via $add
```

The generated `sum_of_factorials` function:
```wat
(func $sum_of_factorials (param $a i32) (param $b i32) (result i32)
    local.get $a
    call $factorial
    local.get $b
    call $factorial
    call $add)
```

Three lines of generated code that compose two other generated functions. The LLM wrote correct cross-function calls because it saw the dependency code in its prompt.
