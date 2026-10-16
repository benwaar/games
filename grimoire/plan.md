# grimoire — Plan

Debug maps for LLM-first toolchains. Sidecar artefact linking specs → compiled output → composition graph.

---

## M1 — Grimoire schema + single-sigil emission

Define the grimoire format and emit it from a single `incant cast` run.

- [ ] Define JSON schema v1: sigil source, functions, WAT line ranges, WASM byte offsets, memory layout, tests
- [ ] Emit `.grimoire.json` alongside every `incant cast` output
- [ ] WAT line tracking: instrument `gen.py` to record which lines the LLM produced per sigil
- [ ] WASM offset mapping: parse `wat2wasm` output to map function names → byte offsets
- [ ] Include test expectations in the grimoire (input/output pairs from sigil)
- [ ] Z80 support: map label addresses from assembler output
- [ ] Test: cast a sigil, verify grimoire contains correct function names, line ranges, and offsets

**Gate:** `incant cast sigils/examples/wat_add.sigil.yaml` produces `output/wat_add.grimoire.json` with valid schema.

---

## M2 — Multi-sigil composition graph

Extend the grimoire to capture how sigils compose in manifests and smelt builds.

- [ ] Composition graph: stitch order, dependency edges, entry point
- [ ] Memory layout map: which sigil owns which address ranges (stack, heap, shared)
- [ ] Per-function call graph: which generated functions call which (from stitch metadata)
- [ ] Smelt integration: grimoire tracks BDD scenario → decomposed sigils → functions
- [ ] Manifest integration: grimoire captures topological sort order and stitch decisions
- [ ] Test: multi-sigil build produces grimoire with full dependency graph

**Gate:** `incant multi sigils/programs/wat_composed.manifest.yaml` produces grimoire showing `sum_of_factorials → factorial + add` dependency chain with memory layout.

---

## M3 — LLM debug loop integration

Use the grimoire to give the LLM compositional debugging context when tests fail.

- [ ] On test failure, load grimoire + failing sigil + dependency sigils
- [ ] Build structured debug prompt: failing test, relevant WAT, composition context, memory layout
- [ ] Feed to LLM for diagnosis — output is spec-level explanation, not code-level
- [ ] Retry loop: LLM fix → regenerate → re-test → update grimoire
- [ ] Track retry history in grimoire (what failed, what the LLM tried, what worked)
- [ ] Test: introduce a deliberate type mismatch across sigils, verify LLM diagnoses the composition error

**Gate:** Pipeline auto-diagnoses a cross-sigil type mismatch and fixes it within 3 retries, using grimoire context.

---

## M4 — Debug/release build split

Two builds from one pipeline — debug for the LLM, release for production.

- [ ] Debug build: `wat2wasm --debug-names`, full name section, unoptimised, comments preserved
- [ ] Release build: `wasm-opt -O3`, name stripping, optional obfuscation pass
- [ ] Grimoire is debug-only — never referenced by release build
- [ ] CLI flag: `incant cast --release` vs default debug
- [ ] Z80 equivalent: debug build keeps labels in symbol table, release strips
- [ ] Test: release binary is smaller than debug, has no name section, but grimoire still maps to it via byte offsets

**Gate:** Same sigil produces both builds. Release is measurably smaller. Grimoire maps function names to release byte offsets despite name stripping.

---

## M5 — Browser DevTools integration (source maps)

Generate WASM source maps that point DevTools back to sigil YAML, not WAT.

- [ ] Study WASM source map format (URL scheme, mappings section, Chrome DevTools support)
- [ ] Generate `.wasm.map` from grimoire: WASM byte offsets → sigil YAML line ranges
- [ ] Embed source map URL in debug WASM (`sourceMappingURL` custom section)
- [ ] DevTools shows sigil YAML when stepping through WASM in debug builds
- [ ] Test: set breakpoint in Chrome DevTools, verify it shows sigil context not WAT

**Gate:** Chrome DevTools displays sigil YAML source when debugging a grimoire-annotated WASM module.

---

## Skills this covers

| Skill | Gap filled |
|-------|-----------|
| Debug map design (schema, composition graphs) | new |
| WASM binary analysis (byte offsets, name sections) | new |
| WASM source maps (custom sections, DevTools integration) | new |
| LLM-as-debugger (structured diagnosis from artefacts) | new |
| Build variants (debug/release from one pipeline) | new |
| Obfuscation-compatible debugging | new |

---

## Feeds into

- [incant](../incant/) — grimoire becomes a standard output of every build
- [glyph](../glyph/) — game sigils get grimoire support (sprite composition, frame buffer mapping)
- [void-duel](../void-duel/) — ML rollouts can trace failures through grimoire back to game specs
