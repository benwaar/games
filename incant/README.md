# incant

Spec-driven code generation — local LLM + RAG produces WAT (WebAssembly Text) and Z80 assembly from human-written specs called "sigils".

## What it builds

A pipeline that reads a sigil (spec file), retrieves relevant instruction set docs via RAG, prompts a local LLM, and gates the output by assembling and running it.

```
sigil → RAG context → LLM → code → assemble → test → binary
```

| Target | LLM outputs | Assembler | Test runner |
|--------|-------------|-----------|-------------|
| WASM | WAT (S-expressions) | `wat2wasm` | `wasm-interp --run-all-exports` |
| Z80 | Raw Z80 asm | `z80.Asm()` | `z80.Z80Machine()` |

No intermediate languages. The LLM writes the target format directly.

## Quick start

```bash
cd incant
bash setup.sh
source .venv/bin/activate

# Generate Z80 binary from a sigil
python -m incant cast sigils/examples/z80_add.sigil.yaml

# Generate WASM module from a sigil
python -m incant cast sigils/examples/wat_add.sigil.yaml
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
