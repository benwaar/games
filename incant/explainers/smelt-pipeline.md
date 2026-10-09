# Smelt Pipeline — BDD Spec → Sigils → Binary

How incant goes from a plain-English spec to a working binary in one command.

## The problem

M1–M4 require humans to write sigil YAML by hand — the function spec, signature, test cases, everything. For someone who knows WAT or Z80, that's fine. But the spec format itself is a barrier: you need to know how to decompose a program into function-shaped pieces before you can use the tool.

M6 adds a layer: write a BDD spec in markdown, and the LLM figures out what functions to build.

## Pipeline

```
greet.spec.md (BDD markdown)
  → parse_spec() — extract title, target, scenarios
  → load_catalogue() — what library functions exist?
  → decompose_spec() — LLM reads spec + catalogue, outputs sigil YAML
  → write_sigils_and_manifest() — sigils + manifest to disk
  → cast_multi() — existing pipeline generates + stitches code
  → run_wasi_test() — BDD scenario tests against the binary
```

> **Coming from JS/TS:** Think of this like `npm init` generating a `package.json` from prompts, except the "prompts" are BDD scenarios and the generator writes actual function implementations, not just metadata.

## BDD spec format

Specs use Given/When/Then syntax with YAML frontmatter for the target:

```markdown
---
target: wat
---
# Greet user

## Scenario: basic greeting
Given the program starts
When the user enters "Ben"
Then output "hello Ben\n"

## Scenario: empty name
Given the program starts
When the user enters ""
Then output "hello \n"
```

The parser extracts structured `Scenario` objects:

```python
@dataclass
class Scenario:
    name: str
    given: list[str]
    when: list[str]
    then: list[str]

@dataclass
class Spec:
    title: str
    target: str
    scenarios: list[Scenario]
```

"And" clauses continue the previous clause type — `And output "done"` after a `Then` line becomes another `then` entry.

> **Coming from C:** BDD is a testing pattern from Ruby/JS land. The structure is just a way to write test cases in natural language: "Given" is setup, "When" is the action, "Then" is the assertion.

## Library catalogue

Reusable sigils live in `sigils/libs/{target}/`. The catalogue is loaded and formatted as text context for the LLM:

```
sigils/libs/
  wat/
    add.sigil.yaml
    factorial.sigil.yaml
  z80/
    add.sigil.yaml
```

The LLM sees a summary like:

```
## Available library functions
- **add**(a: i32, b: i32) → i32
  Add two 32-bit integers
- **factorial**(n: i32) → i32
  Calculate factorial of n
```

If the LLM decides the spec needs `add`, it references the existing library sigil instead of generating a new one. This avoids regenerating code that already works.

> **Coming from JS/TS:** Like `node_modules` — pre-built functions the build system knows about. The LLM acts as a package resolver: "do I need to write this, or can I import it?"

## LLM decomposition

The core new capability. `decompose_spec()` sends the parsed BDD spec + catalogue to the LLM, which outputs a YAML document with:

- `sigils`: list of sigil definitions (name, description, signature, tests, WASI flag)
- `manifest`: which sigils to build and how they connect

The system prompt (`DECOMPOSE_SYSTEM_PROMPT`) explains the sigil format and WASI conventions. The LLM:

1. Reads the BDD scenarios
2. Decides what functions are needed
3. Checks the catalogue for reusable ones
4. Generates sigil specs for everything else
5. Derives test cases from the When/Then pairs

Retries up to 3 times on YAML parse failure or missing required fields.

> **In practice:** This is a structured-output pattern. The LLM generates YAML (not free text), and we parse + validate it before proceeding. If the output is malformed, we retry with the error message. Same pattern as any API that returns structured data from an LLM — you always validate and retry.

## WASI support

WASI (WebAssembly System Interface) lets WASM programs read stdin and write stdout. The greet example uses:

- `fd_read` — read bytes from stdin (fd 0)
- `fd_write` — write bytes to stdout (fd 1)

WASI sigils have `wasi: true` and use `_start` as the entry point instead of a named export. Testing uses `wasm-interp --wasi` with stdin piped in:

```python
result = subprocess.run(
    ["wasm-interp", "--wasi", str(wasm_path)],
    input=stdin_input,  # piped to fd 0
    capture_output=True, text=True,
)
actual = result.stdout  # captured from fd 1
```

The WASI system prompt includes a complete working example (read stdin, prefix with "hello ", write stdout) with explicit WAT syntax rules. Key gotchas the LLM needs to avoid:

- `i32.store` takes TWO stack args (address, value) — not `offset=`
- `i32.load` takes ONE stack arg (address) — result goes on the stack, not into a local
- `memory.copy` for string concatenation in linear memory

## WAT stitching for WASI

The stitcher (`stitch_wat`) was extended to handle WASI modules:

- **Imports** are preserved and deduplicated (same import string → keep one)
- **Memory with export** — `(memory (export "memory") 1)` is parsed as a memory declaration, and the page count is tracked
- **Data segments** — `(data ...)` declarations are collected separately and emitted after the memory declaration
- **Ordering**: imports → memory → data → functions

A subtle bug caught during development: the `_is_memory_export` check used `'(export' in line and '(memory' in line`, which matched functions containing both `(export "_start")` and `memory.copy` instructions. Fixed by requiring the line to **start** with `(export`.

> **Coming from C:** The stitcher is doing what a linker does — resolving symbols across compilation units. But because WAT modules are text, it's string manipulation rather than binary relocation. The import dedup is like `-Wl,--as-needed` — only keep unique imports.

## Running it

```bash
# Full pipeline: BDD spec → WASI binary
python -m incant smelt specs/greet.spec.md -v

# What it does:
# 1. Parses greet.spec.md → Spec with 3 scenarios
# 2. Loads library catalogue (3 WAT functions available)
# 3. LLM decomposes → 1 new sigil (greet_user, wasi: true)
# 4. Writes sigil YAML + manifest to output/smelt/
# 5. cast_multi generates + stitches WAT
# 6. Runs all 3 BDD scenarios as WASI tests
```

The output is a working WASM binary that reads a name from stdin and writes "hello {name}\n" to stdout.
