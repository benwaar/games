# grimoire

Debug maps for LLM-first toolchains — a first-class pipeline artefact that links specs to compiled output.

Part of the [incant](../incant/) pipeline. When incant generates code from sigils, grimoire generates a sidecar debug map that records the full composition graph: which sigil produced which function, what called what, where it lives in memory, and what the tests expected.

---

## The problem

Three constraints pull in different directions:

1. **LLM needs compositional debugging.** Not just "this function failed" but "this function failed because it was called with the wrong memory layout from *that* function, which was stitched from *that* sigil."
2. **Production WASM is obfuscated/optimised.** Name stripping, `wasm-opt`, `wasm-snip` — debug info doesn't survive to runtime.
3. **Spec is the source of truth.** Humans think in sigils. The LLM thinks in sigils. Generated code is a build artefact.

Traditional source maps assume a higher-level language exists and survives into the binary. In an LLM-first pipeline, neither is true — the LLM writes the target format directly, and production builds are stripped.

---

## The idea

**Debug maps as a first-class pipeline artefact, not a compiler bolt-on.**

Because the pipeline *is* the compiler, the debug map captures richer information than any traditional source map — intent, composition, memory layout, and test expectations. Not just line numbers.

The grimoire is emitted alongside every build. It never ships in the binary. The LLM reads it to debug; the human reads sigils and test results.

---

## Two builds, one pipeline

| Build | Contains | Who reads it |
|-------|----------|-------------|
| Debug | Full name sections, sigil-to-WAT source maps, unoptimised, comments | LLM debugger |
| Release | Obfuscated, optimised, stripped | End user |

The grimoire sidecar exists at build time only. Obfuscation can be aggressive — debuggability lives in the pipeline, not the binary.

---

## What the grimoire captures

```json
{
  "version": 1,
  "build_id": "a3f7c...",
  "target": "wat",
  "sigils": [
    {
      "source": "sigils/factorial.sigil.yaml",
      "functions": ["factorial"],
      "wat_lines": [12, 38],
      "wasm_offsets": ["0x0040", "0x00A2"],
      "memory": { "locals": "0x00-0x0F" },
      "tests": [
        { "inputs": { "n": 5 }, "expect": 120 }
      ]
    },
    {
      "source": "sigils/sum_of_factorials.sigil.yaml",
      "functions": ["sum_of_factorials", "_call_factorial", "_call_add"],
      "wat_lines": [40, 93],
      "wasm_offsets": ["0x00A4", "0x01FF"],
      "memory": { "locals": "0x10-0x1F", "shared_heap": "0x1000+" },
      "depends_on": ["factorial.sigil.yaml", "add.sigil.yaml"],
      "tests": [
        { "inputs": { "a": 3, "b": 4 }, "expect": 30 }
      ]
    }
  ],
  "composition": {
    "stitch_order": ["add", "factorial", "sum_of_factorials"],
    "entry_point": "sum_of_factorials",
    "memory_layout": {
      "stack": "0x0000-0x0FFF",
      "heap": "0x1000+"
    }
  }
}
```

---

## Debug flow

```
test fails
  → pipeline reads grimoire
  → identifies which sigil produced the failing function
  → traces dependencies through composition graph
  → feeds failing test + relevant sigils + memory layout to LLM
  → LLM diagnoses at spec level, not code level
  → human sees: "sum_of_factorials failed because factorial returns i64 but add expects i32"
```

The human never reads WAT. The LLM never reads obfuscated WASM. Both work at the spec layer, with the grimoire bridging to the binary.

---

## Layers

| Layer | Human sees | LLM sees |
|-------|-----------|----------|
| Spec | Sigil YAML, test results | Sigil YAML, test results |
| Grimoire | Nothing (optional) | Full composition graph, offsets, memory layout |
| Debug WASM | DevTools if curious | WAT with names + comments |
| Release WASM | Working app | Never touches this |

---

## Connections

| Project | Relationship |
|---------|-------------|
| [incant](../incant/) | Core codegen pipeline — grimoire is a sidecar to every incant build |
| [glyph](../glyph/) | Presentation layer — grimoire tracks image/audio/game sigil composition |
| [wex](../wex/) | WASM player — debug builds load here with full name sections |
| [zxw](../zxw/) | Z80 harness — grimoire covers Z80 builds too (offsets into TAP) |

---

## Project status

| Milestone | Status |
|-----------|--------|
| M1: Grimoire schema + single-sigil emission | Planned |
| M2: Multi-sigil composition graph | Planned |
| M3: LLM debug loop integration | Planned |
| M4: Debug/release build split | Planned |
| M5: Browser DevTools integration (source maps) | Planned |

See [plan.md](plan.md) for milestone details.
