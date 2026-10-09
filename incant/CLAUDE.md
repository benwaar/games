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
python -m incant cast sigils/examples/z80_add.sigil.yaml   # generate + test one sigil
python -m incant multi sigils/programs/wat_composed.manifest.yaml  # multi-sigil build
python -m incant smelt specs/greet.spec.md -v              # BDD spec → WASI binary
python -m incant rag query "add two numbers" --collection z80  # test RAG
python -m pytest tests/ -v                                 # run tests
```

## Key files

```
incant/
  __main__.py    — CLI entry point (cast, multi, smelt, rag)
  rag.py         — embed + query (Ollama nomic-embed-text, JSONL store)
  gen.py         — prompt construction + LLM call + multi-sigil orchestration
  manifest.py    — manifest parser + topological sort
  stitch.py      — WAT module merger + Z80 concatenator (handles WASI imports/memory)
  sigil.py       — sigil YAML parser
  smelt.py       — BDD spec parser + library catalogue + LLM decomposition
  targets/
    z80.py       — Z80-specific prompt template + gate
    wat.py       — WAT-specific prompt template + gate + WASI support
specs/
  greet.spec.md  — BDD spec example (stdin → stdout greeting)
sigils/
  libs/          — reusable library sigils (catalogue for LLM)
```
