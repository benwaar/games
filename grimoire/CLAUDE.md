# CLAUDE.md — grimoire

## What this is

Debug maps for LLM-first toolchains. A sidecar JSON artefact emitted alongside every incant build, capturing the full composition graph from sigil specs down to binary byte offsets.

## Architecture

```
incant cast / multi / smelt
  ├── code generation (existing pipeline)
  ├── assembly + test gate (existing pipeline)
  └── grimoire emission (new)
        ├── sigil → function mapping
        ├── WAT line ranges → WASM byte offsets
        ├── composition graph (dependencies, stitch order, memory layout)
        └── test expectations + retry history
```

Two builds, one pipeline:
- **Debug**: full name sections, source maps, unoptimised. LLM debugs against this.
- **Release**: `wasm-opt`, name-stripped, obfuscated. Ships to users.

Grimoire exists at build time only. Never in the binary.

## Key files

```
grimoire/
  schema/           — JSON schema definitions (v1+)
  emit.py           — grimoire emission from incant pipeline metadata
  source_map.py     — WASM source map generator (sigil YAML → byte offsets)
  debug_prompt.py   — structured debug prompt builder for LLM diagnosis
  tests/            — schema validation, round-trip, composition graph tests
```

## Conventions

- Grimoire files: `{name}.grimoire.json` alongside build output
- Source maps: `{name}.wasm.map` (debug builds only)
- Schema is versioned — bump `version` field on breaking changes
- All byte offsets are hex strings (`"0x00A4"`)
- Tests: `python -m pytest grimoire/tests/ -v`

## Running

```bash
# Emit grimoire with a single sigil build
python -m incant cast sigils/examples/wat_add.sigil.yaml
# → output/wat_add.wasm + output/wat_add.grimoire.json

# Emit grimoire with a multi-sigil build
python -m incant multi sigils/programs/wat_composed.manifest.yaml
# → output/wat_composed.wasm + output/wat_composed.grimoire.json

# Debug build (default) vs release build
python -m incant cast sigils/examples/wat_add.sigil.yaml --release
# → output/wat_add.wasm (optimised, stripped — grimoire still emitted for pipeline use)
```
