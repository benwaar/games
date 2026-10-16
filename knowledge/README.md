# Knowledge Corpus — Z80 / Spectrum / WASM

One corpus, three access interfaces: Incant RAG, MCP server for Claude, skill for the user.

---

## What it is

A curated reference corpus covering Z80 assembly, ZX Spectrum hardware/ROM, and WebAssembly (spec + WAT + WASI). Content is fetched from public, openly-licensed sources, normalised into uniform markdown chunks with structured frontmatter, and stored locally.

Designed to be queried by:
- **Incant** — RAG context during Z80/WASM code generation
- **MCP server** (`knowledge/mcp/`) — Claude queries mid-session (planned, M4)
- **Skill** (`.claude/skills/`) — user queries from the terminal (planned, M3)

---

## Directory structure

```
knowledge/
  scripts/          — fetch + normalise scripts
  raw/              — fetched source content (gitignored, regenerated)
  chunks/           — normalised chunks ready for indexing (gitignored, regenerated)
  schema/           — chunk schema definition
    chunk.md        — frontmatter spec + field definitions
  index/            — vector index (gitignored, future — M5)
  mcp/              — MCP server (future — M4)
  requirements.txt  — Python dependencies
```

---

## How to rebuild

```bash
pip install -r knowledge/requirements.txt
./knowledge/scripts/fetch_all.sh
python knowledge/scripts/normalise.py
```

`fetch_all.sh` runs all three fetch scripts. Each writes raw markdown to `knowledge/raw/{source}/`. The normaliser reads `raw/` and writes uniform chunks to `knowledge/chunks/`.

---

## Chunk schema

See [`schema/chunk.md`](schema/chunk.md) for the full frontmatter spec.

Every chunk carries: `topic`, `category`, `source_url`, `licence`, `tags`.

**Categories:**

| Category | Content |
|----------|---------|
| `z80-instruction` | Opcode definitions, timing, flags |
| `z80-pattern` | Common Z80 idioms and patterns |
| `spectrum-rom` | ROM entry points and routines |
| `spectrum-hardware` | System variables, ULA, port I/O |
| `wasm-spec` | WASM binary format, instruction set, types |
| `wasm-pattern` | WAT idioms and common patterns |
| `wasi` | WASI preview1 host function signatures |

---

## Sources and licences

| Source | Content | Licence |
|--------|---------|---------|
| clrhome.org | Z80 opcode table | Community/open |
| z80-heaven.wikidot.com | Z80 instruction descriptions | Community/open |
| skoolkid.github.io/rom | Spectrum ROM disassembly | GPL-3.0 |
| worldofspectrum.org | System variables, ULA docs | Community archive |
| webassembly.github.io/spec | WASM spec (instructions, types) | W3C |
| developer.mozilla.org | WAT text format, API reference | CC-BY-SA-2.5 |
| github.com/WebAssembly/WASI | WASI preview1 signatures | Apache-2.0 |

---

## What is NOT included

- Copyrighted books (full text) — e.g. "The Complete Spectrum ROM Disassembly", Rodney Zaks Z80 CPU manual
- Paywalled or subscription content
- Forum posts and wiki pages without an explicit open licence
- Any content requiring registration or login to access
