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
- Incant pipeline integration (wiring query_index into code gen)
- Spectrum hardware content (skoolkid/WoS pages had different structure than expected — stubs only)
- MDN WAT section parsing (MDN redirects fragment URLs, content extraction needs rework)

**Branch:** `feature/no-ticket-knowledge-corpus`
