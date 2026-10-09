# incant

Spec-driven code generation — local LLM + RAG produces WAT (WebAssembly Text) and Z80 assembly from human-written specs called "sigils".

## What it builds

A pipeline that reads a sigil (spec file), retrieves relevant instruction set docs via RAG, prompts a local LLM, and gates the output by assembling and running it.

```
sigil.yaml → RAG context → LLM → code → assemble → test → binary
manifest.yaml → toposort → generate each → stitch → assemble → verify → combined binary
```

| Target | LLM outputs | Assembler | Test runner |
|--------|-------------|-----------|-------------|
| WASM | WAT (S-expressions) | `wat2wasm` | `wasm-interp -r <func> -a type:val` |
| Z80 | Raw Z80 asm | `z80.Asm()` | `z80.Z80Machine()` |

No intermediate languages. The LLM writes the target format directly.

## Quick start

```bash
cd incant
bash setup.sh
source .venv/bin/activate

# Generate WASM module from a sigil
python -m incant cast sigils/examples/wat_add.sigil.yaml

# Generate Z80 binary from a sigil
python -m incant cast sigils/examples/z80_add.sigil.yaml

# Multi-sigil build (independent functions)
python -m incant multi sigils/programs/wat_math.manifest.yaml

# Multi-sigil build with dependencies (sum_of_factorials calls factorial + add)
python -m incant multi sigils/programs/wat_composed.manifest.yaml

# Run all examples end-to-end
bash demo.sh
```

## Sigils

A sigil is a YAML spec — one function, with signature, description, and test cases:

```yaml
name: add
description: Add two 8-bit numbers
target: z80

signature:
  inputs:
    - name: a
      type: u8
    - name: b
      type: u8
  output:
    type: u8

tests:
  - inputs: { a: 2, b: 3 }
    expect: { a: 5 }
```

## Manifests (multi-sigil)

A manifest lists sigils to compose into one program. All sigils must share the same target.

```yaml
name: math
target: wat
sigils:
  - sigils/examples/wat_add.sigil.yaml
  - sigils/examples/wat_factorial.sigil.yaml
```

Sigils can declare dependencies — the pipeline sorts them topologically, injects generated code from dependencies into the LLM prompt, and stitches the outputs into a single binary.

```yaml
name: sum_of_factorials
target: wat
dependencies:
  - factorial
  - add
tests:
  - inputs: { a: 3, b: 4 }
    expect: 30
```

## Stack

- **LLM:** `qwen3-coder:latest` via Ollama (30b MoE)
- **Embeddings:** `nomic-embed-text` via Ollama
- **Z80 toolchain:** `z80` pip package (assembler + CPU emulator)
- **WASM toolchain:** `wabt` brew package (wat2wasm, wasm-interp, wasm-objdump)

## Learn

- Project plan and milestones: [PLAN.md](PLAN.md)
- Current milestone: [NEXT.md](NEXT.md)
- Explainers: [explainers/](explainers/)
- Shared concept explainers: [../explainers/](../explainers/README.md)
