# Next: M5 — Docs + integration

M4 is complete — multi-sigil orchestration with manifests, stitching, and dependency-aware generation. WAT composed build produces one .wasm with 3 exports where `sum_of_factorials` calls `factorial` and `add`. Z80 basics concatenates 3 programs into one binary.

**What's next:**
1. Demo script: `bash demo.sh` that runs single + multi sigil examples
2. Update STUDY.md skills coverage
3. Update root CLAUDE.md project table
4. Final explainer review pass

**Gate:** `bash demo.sh` works from cold clone after `bash setup.sh`.
