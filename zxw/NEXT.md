
## Next: Binary protection demos

- Load self-verifying binary on zxw, tamper via debug API, show the crash
- POK file support — classic Spectrum cheat format, ties into the debug API
- Same demo on WASM — tamper data segment, show detection + bypass
- 

# Next: incant M9 — multi-step decomposition + dependency graph

zxw M7 (CDP debug integration) is complete. All milestones M1–M7 done.

The next parallel track is incant M9.

---

## incant M9 — multi-step decomposition + dependency graph

Extend the incant code-gen pipeline to handle specifications that decompose into
multiple interdependent WAT/WASM modules or Z80 assembly routines.

### What to do

1. **Dependency graph** — represent a multi-step spec as a DAG of sub-tasks
   (each node is a spec → output pair; edges are data dependencies).
2. **Topological execution** — resolve build order, execute leaf nodes first,
   pass outputs forward as context to dependent nodes.
3. **LLM decomposition** — given a high-level spec, ask the local LLM to break
   it into a dependency graph (JSON) then execute the graph.
4. **Test** — a two-module WASM program: module A exports a helper,
   module B imports it; the pipeline produces both binaries in dependency order.

### Gate

Given a spec like "write a Z80 routine that calls two sub-routines in sequence",
incant M9 decomposes it, generates each routine independently, links them, and
produces a working binary.

## Reference

- incant project: `incant/`
- Previous milestone: M8 — TAP read/write + disassembly pipeline
- Incant LOG: `incant/LOG.md`
