# LOG — Games AI Lab

## 2026-10-16 — zxw: AudioWorklet + SAB ring buffer

Replaced the deprecated `ScriptProcessorNode` audio pipeline with `AudioWorkletNode` + `SharedArrayBuffer` ring buffer.

**What changed:**

- **`audio-worklet.js`** (new) — `BeeperProcessor` AudioWorkletProcessor. Reads f32 samples from a 16384-sample SAB ring buffer using atomic read/write indices. Silence on underrun.
- **`audio.js`** (rewrite) — creates `AudioWorkletNode`, feeds samples into the SAB ring buffer. Overflow drops oldest samples (advances read pointer) instead of wiping the entire queue. Same export API (`initAudio`, `feedAudio`, `resumeAudio`, `toggleMute`, `isMuted`, `audioStats`).
- **`serve.sh`** — Python server now sends `Cross-Origin-Opener-Policy: same-origin` + `Cross-Origin-Embedder-Policy: require-corp` headers (required for `SharedArrayBuffer`).
- **`debug.html` / `index.html`** — `initAudio()` is now async, call sites updated to `await`.

**Why:** The old pipeline used a plain JS array as a FIFO (`push` + `splice(0, n)` = O(n) per callback), wiped the entire queue on overflow (>1s backlog), and ran on the main thread. ROM BEEP runs with `DI` (interrupts disabled) in a tight loop — samples accumulated correctly in Rust but got dropped by the JS overflow wipe. The AudioWorklet runs on a dedicated audio thread, and the ring buffer handles backpressure without data loss.

**Rust side unchanged** — `spectrum.rs` sample accumulation (fractional T-state accumulator, ~880 samples/frame at 50Hz) was already correct.

**Needs listening test** — pipeline verified working (zero console errors/warnings, worklet running, buffer flowing steady at ~365ms fill), but audio quality needs human verification.

**Branch:** `feature/no-ticket-zxw-audio-pipeline`

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

## 2026-10-16 — Content gaps + semantic search

Filled all deferred content gaps from the initial corpus build and added semantic search to the Rust MCP server.

**Content gaps filled (6 commits):**

- **Spectrum hardware** — fixed `fetch_spectrum.py` selectors, fetched 22 priority ROM routines + 274 stubs + system variables + I/O ports/ULA. Real content, not stubs.
- **Z80 extras** (`fetch_z80_extras.py`) — undocumented opcodes (Sean Young PDF, GFDL, 17 sections → 57 chunks), 1-bit beeper engines (utz82 GitHub, MIT, 37 engines → 138 chunks), library routines (Dean Belfield GitHub, 14 routines → 32 chunks).
- **WASM binary format** — W3C spec sections (values, types, instructions, modules) → 23 chunks.
- **MDN WAT pages** — 210 per-instruction pages fetched. Fixed fragment-URL redirect issue (MDN redirects `#section` anchors, so fetch once and extract by heading ID instead).

**Semantic search:**

- ONNX embedding pipeline: `embed.py` generates `vectors.npy` + `metadata.json`, `export_onnx.py` exports all-MiniLM-L6-v2 to ONNX + tokenizer.
- Rust MCP server gains `search_semantic` tool — loads ONNX model at startup, cosine similarity ranking. No Python dependency at query time.
- 7 acceptance tests verified (beeper engines, binary format, LEB128, category filter, keyword vs semantic comparison, tape loading, corpus stats).

**Corpus totals:** ~800 chunks across 10 categories (z80-instruction, z80-pattern, z80-undocumented, z80-beeper, z80-library, spectrum-rom, spectrum-hardware, wasm-spec, wasm-pattern, wasi).

**Branch:** `feature/no-ticket-knowledge-content-gaps`

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
