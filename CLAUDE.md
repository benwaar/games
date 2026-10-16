# CLAUDE.md — Games AI Lab

This repo is a personal ML/AI R&D programme. Each project builds real, working systems — not tutorial exercises. Every capability maps to a class of real-world problem.

---

## Repo structure

```
games/
  explainers/          — shared knowledge: CNNs, training loops, RL, Mel-spectrograms etc.
  knowledge/           — Z80/Spectrum/WASM corpus: chunks, curated docs, MCP server, FAISS index
  _data/               — copyrighted reference material (gitignored, never commit)
  signal-hunt/         — audio classifier: data pipelines → CNN → transfer learning → chords
  utala/kaos9/         — card game AI: TD learning → deep RL → distillation
  acoustic-odyssey/    — deployment layer for Signal Hunt: on-device inference + adaptive RL
  artefact/            — 8-bit audio: ROM scanner, restoration, generative chip-tunes + diffusion
  void-duel/           — ZX Spectrum shooter: classic ML → multi-agent RL → extreme-edge distil
  zxw/                 — ZX Spectrum 48K emulator: Rust → WASM + CDP debug + AudioWorklet audio
  incant/              — spec-driven code gen: local LLM + RAG → WAT/WASM + Z80 assembly
  glyph/               — LLM-first presentation layer: game sigils → WASM (macroquad) + Z80 (ULA)
  grimoire/            — debug maps for LLM-first toolchains: spec → binary composition graph
  rosetta/             — round-trip translation: Z80 binary → glyph IR → Z80 + WASM, verified
  wex/                 — WASM executor: static HTML host + swappable debug.wasm
```

---

## Test harnesses

Two static test containers for checking build output in the browser.

### WASM player (`wex/`)

- **Output path:** `wex/debug.wasm`
- **Serve:** `./serve.sh` → `http://localhost:8999`
- **HTML is static** — generic player that handles image, audio, and keyboard via `type()` bitmask.
- WASM exports: `type()` (bitmask), `width()`, `height()`, `render()`, `audio_count()`, `audio_offset()`, `memory`.
- No image → shows blue X crosshair. Audio → Play button. Both → image + Play button.
- Pipeline generates WAT → `wat2wasm` → `debug.wasm`. No Rust/wasm-pack.

### Z80 TAP harness (`zxw/`)

- **Output path:** `zxw/www/tap/debug.tap`
- **Serve:** `zxw/serve.sh` → `http://localhost:8998`
- **Debug mode:** `http://localhost:8998/debug.html` — auto-loads ROM + TAP, boots, types `LOAD ""`. Zero clicks.
- **Full mode:** `http://localhost:8998/` — sidebar with file pickers, POK support.

---

## Explainers

Shared knowledge lives at the top level. Project-specific knowledge lives in the project.

- **`explainers/`** — shared concepts reused across projects (CNNs, training loops, RL, spectrograms, transfer learning). Add here when a concept will be referenced by more than one project.
- **`{project}/explainers/`** — project-specific knowledge only (e.g. iowa piano data, chord synthesis, KAOS 9 game mechanics). Link out to `../../explainers/` for shared concepts rather than duplicating.
- **`{project}/docs/`** — progress, process, checkpoints, quality docs. Not knowledge — status.

---

## Project ethos

These projects are about understanding algorithms by building them — seeing how different approaches think, and learning by stripping problems down to their nuts and bolts.

- **Real systems, not tutorials.** Each project ships a working thing. Checkpoints train. Models infer. Games play.
- **Build incrementally.** Milestones are small, gated, and committed one at a time.
- **Document as you build.** Explainers are written alongside the code, not after. The teaching test: could someone read the explainer cold and understand what was built? Add to docs with each milestone, not in a batch at the end.
- **Every skill maps to a real problem.** The business alignment in STUDY.md shows where each capability lands. This isn't abstract — it's practice for specific classes of production ML problem.
- **Learning first, speed second.** When there's a choice between the fast way and the way that teaches more, pick the one that teaches. Cut corners on polish, not understanding.
- **Bridge from what you know.** When a Python or ML concept appears, map it to the equivalent in C, JS/TS, or business domains. The explainers use "Coming from C/JS/TS" callouts and "In practice" business parallels — add these where the connection is strong and non-obvious.
- **Connect to business.** Resampling is normalising data from different sources. Augmentation is synthetic data for imbalanced datasets. Normalisation is feature scaling. Flag these parallels in explainers so it's clear why these skills transfer.

---

## Conventions

**Branches:** `feature/no-ticket-{project}-{description}` (no Jira key for this repo)

**Commits:** conventional commits — `feat`, `fix`, `docs`, `refactor`, `test`

**Milestones:** each milestone has a gate (measurable pass/fail), a commit, and a tick in LOG.md.

**Tests:** every new module gets tests. Run `python -m pytest tests/ -v` from the project root.

**Data:** `data/raw/` and `data/processed/` are gitignored — regenerated by setup scripts. Never commit training data.

**`_data/`:** copyrighted reference material (ROMs, game TAPs, analysis docs). Gitignored — never commit. `incant/data/raw/tap/48.rom` must stay in place (needed by incant). Use short codes for game files (e.g. `j83`, `joust`), never copyrighted game titles.

**Knowledge corpus:** `knowledge/` contains the Z80/Spectrum/WASM reference corpus. Three access layers — all reading from the same `knowledge/chunks/*.md` files:
- **knowledge-mcp** → Claude (MCP server, keyword + semantic search)
- **/asm skill** → user (CLI, browsable)
- **FAISS embeddings** → local LLM (incant RAG, semantic search)
Rebuild with `bash knowledge/setup.sh`. Human-readable version at `_data/kb/`.

**AI attribution:** every commit includes the AI trailer lines per the global CLAUDE.md.

**Subagents:** default to sonnet model. Prompts must be directive ("read this file, make this edit"), not descriptive. If `NEXT.md` has checkboxes for the current work, include the path in agent prompts and tell agents to tick `[x]` as they complete steps. Follow the `/research` skill loop: implement → test → tick → next.

---

## Projects and status

See [STUDY.md](STUDY.md) for the full capability map and business alignment.

| Project | Status |
|---------|--------|
| [Utala: KAOS 9](utala/kaos9/) | ✅ Complete |
| [Signal Hunt](signal-hunt/) | ✅ Complete (Ph 1–5) |
| [zxw](zxw/) | ✅ M1–M7 complete — Rust → WASM Spectrum emulator + CDP debug + AudioWorklet |
| [Incant](incant/) | ✅ M1–M10 complete — spec → code gen, TAP tools, ONNX inference, binary protection |
| [wex](wex/) | ▶ Active — WASM executor, responsive canvas, a11y (mirror DOM, settings, palettes) |
| [Glyph](glyph/) | ▶ POC complete — image + audio → both targets, pure WAT pipeline, 113 tests |
| [Rosetta](rosetta/) | POC — j83 sound extraction proved the pipeline, glyph beeper next |
| [Grimoire](grimoire/) | Planned — debug maps: sigil → binary composition graph, LLM debug loop |
| [Acoustic Odyssey](acoustic-odyssey/) | Planned — edge deployment: ONNX quantisation + adaptive RL |
| [Artefact](artefact/) | Planned — 8-bit audio: ROM scanner, restoration, generative chip-tunes |
| [Void Duel](void-duel/) | Planned — ZX Spectrum shooter: classic ML → multi-agent RL |
