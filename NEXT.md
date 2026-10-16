# Knowledge Corpus — Deferred Work

## Incant pipeline integration

Wire the knowledge corpus into Incant's local LLM code generation pipeline.

- [ ] Incant queries the FAISS index during code gen for Z80/WASM context
- [ ] Test: Incant generates better Z80/WASM with corpus vs without

## Content gaps

- [ ] Spectrum hardware content — skoolkid/WoS pages had different structure than expected, only stubs fetched
- [ ] MDN WAT section parsing — MDN redirects fragment URLs, content extraction needs rework
- [ ] Undocumented Z80 opcodes — Sean Young doc not yet integrated
