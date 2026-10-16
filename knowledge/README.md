# Knowledge Corpus — Z80 / Spectrum / WASM

One corpus, three access interfaces: Incant RAG, MCP server for Claude, skill for the user.

---

## What it is

A curated reference corpus covering Z80 assembly, ZX Spectrum hardware/ROM, and WebAssembly (spec + WAT + WASI). Content is fetched from public, openly-licensed sources, normalised into uniform markdown chunks with structured frontmatter, and stored locally.

Designed to be queried by:
- **Incant** — RAG context during Z80/WASM code generation
- **MCP server** (`knowledge-mcp/`) — Rust server with `search`, `read`, `search_semantic` tools
- **Skill** (`.claude/skills/asm/`) — `/asm` slash command for browsing from the terminal
- **kb** (`_data/kb/`) — human-readable markdown reference (gitignored, built by `build_kb.py`)

---

## Directory structure

```
knowledge/
  scripts/          — fetch, normalise, embed, build_kb scripts
  raw/              — fetched source content (gitignored, regenerated)
  chunks/           — normalised chunks ready for indexing (gitignored, regenerated)
  curated/          — hand-written reference files (z80/, wat/)
  schema/           — chunk schema definition
    chunk.md        — frontmatter spec + field definitions
  index/            — vector index + ONNX model (gitignored, regenerated)
```

---

## How to rebuild

```bash
bash knowledge/setup.sh
```

This fetches raw data, normalises into chunks, builds the FAISS/ONNX index, and generates the human-readable `_data/kb/` reference (gitignored).

---

## Chunk schema

See [`schema/chunk.md`](schema/chunk.md) for the full frontmatter spec.

Every chunk carries: `topic`, `category`, `source_url`, `licence`, `tags`.

**Categories:**

| Category | Content |
|----------|---------|
| `z80-instruction` | Opcode definitions, timing, flags |
| `z80-pattern` | Common Z80 idioms and patterns |
| `z80-undocumented` | Undocumented opcodes (Sean Young) |
| `z80-beeper` | 1-bit beeper engine source (utz82) |
| `z80-library` | Library routines (Dean Belfield) |
| `spectrum-rom` | ROM entry points and routines |
| `spectrum-hardware` | System variables, ULA, port I/O |
| `wasm-spec` | WASM instruction set, WAT text format, MDN pages |
| `wasm-binary` | WASM binary format (values, types, modules) |
| `wasm-pattern` | WAT idioms and common patterns |
| `wasi` | WASI preview1 host function signatures |

---

## Sources and licences

| Source | Content | Licence |
|--------|---------|---------|
| clrhome.org | Z80 opcode table | Community/open |
| z80-heaven.wikidot.com | Z80 instruction descriptions | Community/open |
| skoolkid.github.io/rom | Spectrum ROM disassembly, system variables | GPL-3.0 |
| worldofspectrum.org | ULA docs, I/O ports | Community archive |
| Sean Young (z80.info) | Undocumented Z80 opcodes (57 chunks) | GFDL |
| github.com/utz82/ZX-Spectrum-1-Bit-Routines | 37 beeper engines (138 chunks) | MIT |
| github.com/breakintoprogram/spectrumutils | Z80 library routines — Dean Belfield (32 chunks) | MIT |
| webassembly.github.io/spec | WASM spec (instructions, types, binary format) | W3C |
| developer.mozilla.org | WAT text format, per-instruction pages (210 pages) | CC-BY-SA-2.5 |
| github.com/WebAssembly/WASI | WASI preview1 signatures | Apache-2.0 |

---

## What is NOT included

- Copyrighted books (full text) — e.g. "The Complete Spectrum ROM Disassembly", Rodney Zaks Z80 CPU manual
- Paywalled or subscription content
- Forum posts and wiki pages without an explicit open licence
- Any content requiring registration or login to access
