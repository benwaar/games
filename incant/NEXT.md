# Next: M6 — Smelt: BDD spec → sigils → binary

M5 is complete — demo script passes all 10 steps (6 single sigils + 3 multi-sigil builds + RAG query), docs updated, STUDY.md skills marked complete.

**What's next:**

M6 adds a "smelt" stage: human writes intent as a BDD-format markdown file, the pipeline interprets it, checks a library catalogue of reusable sigils, generates sigils for anything missing, writes a manifest, and runs the existing cast/multi pipeline.

```
human writes .spec.md (BDD)
  → smelt reads spec
  → LLM checks library catalogue (what already exists?)
  → LLM generates sigils for missing functions only
  → LLM writes manifest (new sigils + library sigils)
  → existing cast/multi pipeline runs
  → .wasm + .asm output
```

**Steps:**
1. Define BDD `.spec.md` input format
2. Build library catalogue (reusable sigils with pre-generated code)
3. Add WASI target support (fd_read, fd_write for stdin/stdout)
4. `smelt` CLI subcommand: spec.md → sigils + manifest
5. LLM decomposes BDD scenarios into function graph with deps
6. Generate test cases from BDD When/Then pairs
7. Run: write BDD spec, get working binary with no YAML by hand

**Gate:** Write a BDD spec, run `python -m incant smelt greet.spec.md`, get sigils + manifest + working .wasm that passes the scenarios.
