# CLAUDE.md — incant

## What this is

A spec-driven code generation pipeline. Sigils (YAML specs) go in, assembled binaries come out. One local LLM (`qwen3-coder`), RAG for instruction set knowledge, two output targets (WAT → WASM, Z80 assembly).

## Architecture

```
sigil.yaml → rag.py (retrieve docs) → gen.py (prompt LLM) → gate.py (assemble + test)
                                                                  ↓
                                                         pass → output/
                                                         fail → retry with error (max 3)
```

## Conventions

- **One sigil = one function.** No multi-function sigils.
- **Tests in every sigil.** Input/output pairs the gate checks automatically.
- **No intermediate languages.** LLM outputs WAT or Z80 asm directly — no C, no Rust.
- **Gate = assembler + test runner.** If it assembles and tests pass, it ships.

## Tech stack

- Python 3.12, venv
- Ollama: `qwen3-coder:latest` (generation), `nomic-embed-text` (embeddings)
- Z80: `z80` pip package (assembler + CPU emulator)
- WASM: `wabt` brew package (wat2wasm, wasm-interp, wasm-objdump)

## Running

```bash
bash setup.sh                                              # venv, deps, embed knowledge
python -m incant cast sigils/examples/z80_add.sigil.yaml   # generate + test
python -m incant rag query "add two numbers" --collection z80  # test RAG
python -m pytest tests/ -v                                 # run tests
```

## Key files

```
incant/
  __main__.py    — CLI entry point
  rag.py         — embed + query (Ollama nomic-embed-text, JSONL store)
  gen.py         — prompt construction + LLM call
  gate.py        — assemble + run + check
  targets/
    z80.py       — Z80-specific prompt template + gate
    wat.py       — WAT-specific prompt template + gate
```
