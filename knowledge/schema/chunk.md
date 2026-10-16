# Chunk Schema

Every chunk in `knowledge/chunks/` is a markdown file with YAML frontmatter.

## Required frontmatter

```yaml
---
topic: "LD A, n"              # short title for the concept
category: z80-instruction     # one of the categories below
source_url: "https://..."     # where this was fetched from
licence: "public-domain"      # licence or status of the source
tags: [load, 8-bit, immediate] # searchable tags
---
```

## Categories

- `z80-instruction` — individual Z80 opcodes and addressing modes
- `z80-pattern` — common Z80 code patterns (scroll, multiply, etc.)
- `spectrum-rom` — ROM entry points and routines
- `spectrum-hardware` — ULA, display file, system variables, I/O ports
- `wasm-spec` — WASM binary format, WAT text format, type system
- `wasm-pattern` — common WASM/WAT code patterns
- `wasi` — WASI preview signatures and conventions
