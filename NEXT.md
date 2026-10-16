# Knowledge Corpus — Z80 / Spectrum / WASM Reference

One corpus, three access methods: Incant RAG, MCP server for Claude, skill for the user.

**Branch:** `feature/no-ticket-knowledge-corpus`

---

## M1 — Scaffold + fetch scripts

Set up the directory structure and write fetch scripts for public sources.

- [x] Create `knowledge/` dir structure: `scripts/`, `raw/`, `chunks/`, `schema/`
- [x] Add `knowledge/raw/` and `knowledge/chunks/` to `.gitignore`
- [x] Define chunk schema (frontmatter: topic, source_url, licence, category)
- [x] `scripts/fetch_z80.py` — Z80 instruction set from z80.info + Zilog public docs
- [x] `scripts/fetch_spectrum.py` — ROM entry points, system vars, ULA timing from public sources
- [x] `scripts/fetch_wasm.py` — W3C WASM spec sections, WAT text format, WASI signatures
- [x] Each script outputs raw markdown to `knowledge/raw/{source}/`
- [x] Add `scripts/fetch_all.sh` — runs all fetch scripts
- [x] Test: scripts run clean, raw files populated

## M2 — Normaliser + chunking

Transform raw fetched content into uniform, searchable chunks.

- [ ] `scripts/normalise.py` — reads `raw/`, writes uniform markdown to `chunks/`
- [ ] Chunks split by topic (one concept per file, ~500-1000 tokens)
- [ ] Frontmatter on every chunk: `topic`, `category`, `source_url`, `licence`, `tags`
- [ ] Categories: `z80-instruction`, `z80-pattern`, `spectrum-rom`, `spectrum-hardware`, `wasm-spec`, `wasm-pattern`, `wasi`
- [ ] Test: normaliser produces valid chunks, frontmatter parses, no empty files
- [ ] Docs: `knowledge/README.md` — what's in the corpus, how to rebuild, licence audit

## M3 — Skill (user query interface)

Slash command so the user can query the corpus from the terminal.

- [ ] Skill file: `.claude/skills/z80.md` (or `specref`)
- [ ] Skill searches chunks by keyword/topic, returns formatted result
- [ ] Handles multi-word queries, category filters (`/z80 rom CHAN-OPEN`, `/z80 wasm memory.grow`)
- [ ] Test: manual — query returns relevant chunks

## M4 — MCP server (Claude query interface)

Expose the corpus as an MCP server so Claude can query it mid-session.

- [ ] MCP server in `knowledge/mcp/` — Python (FastAPI or stdio)
- [ ] Tools: `search(query, category?)` → ranked chunks, `read(chunk_id)` → full content
- [ ] Resources: corpus metadata, category list
- [ ] Register in `.claude/settings.json` as MCP server
- [ ] Test: Claude can query Z80 instructions, Spectrum ROM routines, WASM specs mid-session

## M5 — Incant integration (RAG pipeline)

Wire the corpus into Incant's local LLM code generation pipeline.

- [ ] Embedding script: `scripts/embed.py` — generates vector index from chunks
- [ ] Vector store: FAISS or ChromaDB index in `knowledge/index/` (gitignored)
- [ ] Incant queries the index during code gen for Z80/WASM context
- [ ] Test: Incant generates better Z80/WASM with corpus vs without
- [ ] `knowledge/index/` added to `.gitignore`

---

## Sources (public, freely available)

| Source | Content | Licence/Status |
|--------|---------|---------------|
| z80.info | Instruction set, timing tables | Community-maintained, open |
| Zilog Z80 CPU manual | Official instruction reference | Public domain |
| "Undocumented Z80" (Sean Young) | Undocumented opcodes, flags | Freely distributed |
| Spectrum ROM disassembly (Logan/O'Hara) | ROM entry points, routines | Freely published |
| World of Spectrum | System variables, ULA docs | Community archive |
| Skoolkit disassemblies | Annotated ROM | GPL |
| W3C WebAssembly spec | WASM binary + text format | W3C licence (open) |
| MDN WebAssembly docs | WAT, API reference | CC-BY-SA |
| WASI spec | WASI preview signatures | Apache 2.0 |

**Not included:** copyrighted books (full text), paywalled content, forum posts without explicit licence.
