# LOG — Knowledge Corpus

## 2026-10-16 — Knowledge Corpus: Z80 / Spectrum / WASM Reference

One corpus, three access methods. Built fetch → normalise → serve pipeline for Z80, Spectrum, and WASM reference material from public sources.

**What was built:**

- **Fetch scripts** (`knowledge/scripts/fetch_z80.py`, `fetch_spectrum.py`, `fetch_wasm.py`) — pull from clrhome.org, z80-heaven, skoolkid, WoS, W3C spec, MDN. 517 raw files across 4 categories.
- **Normaliser** (`knowledge/scripts/normalise.py`) — validates frontmatter, splits oversized files, writes uniform chunks to `knowledge/chunks/{category}/`.
- **Skill** (`.claude/skills/asm/`) — `/asm` slash command with two domains (`z80`, `wasm`), browse → list → detail navigation. Helper script pre-allowed in settings.json.
- **MCP server** (`knowledge/mcp/server.py`) — stdio MCP with `search` + `read` tools and `corpus://` resources. Registered in `settings.local.json`.
- **Embedding pipeline** (`knowledge/scripts/embed.py`, `query_index.py`) — all-MiniLM-L6-v2 embeddings → FAISS index for RAG retrieval.
- **Review fixes** — path traversal guard, atomic writes, error handling, licence accuracy.
- **Curated knowledge** — moved 11 hand-written Z80/WAT reference files from `incant/knowledge/` to `knowledge/curated/`. Updated all incant references.
- **Setup script** (`knowledge/setup.sh`) — one-command install: deps, fetch, normalise, embed.

**Deferred:**
- Spectrum hardware content (skoolkid/WoS pages had different structure than expected — stubs only)
- MDN WAT section parsing (MDN redirects fragment URLs, content extraction needs rework)

**Branch:** `feature/no-ticket-knowledge-corpus`

## 2026-10-16 — MCP server rewrite + Incant pipeline integration

Rewrote MCP server in Rust, wired knowledge corpus into incant's RAG store, added embeddings explainer.

**What was built:**

- **Rust MCP server** (`knowledge-mcp/`) — rewrote Python server in Rust using `rust-mcp-sdk` v2. Same tools (`search`, `read`) and resources (`corpus://categories`, `corpus://stats`). Registered in `.mcp.json`.
- **Incant RAG integration** — corpus chunks (529 files across 6 categories) embedded into incant's existing ollama-based RAG store via `--append` flag. Curated files (11) kept as base, corpus appended. 78 → 1,794 chunks.
- **rag.py improvements** — frontmatter stripping in `chunk_markdown()`, `replace` parameter on `embed_directory()` for append-mode embedding.
- **Embeddings explainer** (`explainers/embeddings-and-rag.md`) — covers embeddings, FAISS, RAG pattern, and why the project uses sentence-transformers for the shared corpus and ollama for incant.
- **STUDY.md** — expanded LLM Engineering from 2 lines to 7: embeddings, vector search, RAG, MCP, skills, evals, agent orchestration.
- **Setup** — `setup.sh` now embeds all corpus categories into incant (z80-instruction, z80-pattern, spectrum-rom → z80 collection; wasm-spec, wasm-pattern, wasi → wat collection).

**Gate:** 151 tests passing. 4 end-to-end code gen tests (z80_add, wat_factorial, z80_store_load, z80_greet) all generate and pass gates with enriched RAG context.

**Branch:** `feature/no-ticket-knowledge-corpus`
