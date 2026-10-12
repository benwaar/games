# dream — Generative Chip-Tunes

Fine-tune a sequence model on a composer's VGM, MIDI, or tracker files to generate
authentic new chip-tunes — respecting the constraints of the target sound chip.

---

## The idea

Chip music isn't just "lo-fi" — it's composed within strict hardware constraints:
3–4 channels, specific waveform types (square, triangle, noise), hard envelope shapes,
and register-level timing. A PSG composer doesn't just write music, they program a chip.

dream learns the patterns of a specific composer or chip architecture from their existing
work, then generates new tracks that sound authentic — not generic MIDI with a chiptune
filter, but sequences that respect the actual hardware constraints.

---

## Milestones

### M1 — Dataset

- [ ] Collect chip-tune data: VGM files (register streams), SID files, tracker modules (.mod, .xm)
- [ ] Parse into tokenised sequences: register writes, note events, timing, channel assignments
- [ ] Build per-composer and per-chip datasets from public archives (VGM Rips, HVSC, Mod Archive)
- [ ] Represent as token sequences suitable for transformer/SSM input

**Gate:** DataLoader yields tokenised sequences. Vocabulary covers register ops and timing.

**Docs:**
- [ ] Explainer: chip-tune as a sequence problem — why register streams are token sequences
- [ ] Explainer: VGM register format — how music is encoded as chip instructions

---

### M2 — Baseline (Markov / n-gram)

- [ ] Build a Markov chain or n-gram model on register sequences
- [ ] Generate sequences, render to audio via chip emulator
- [ ] Listen test: does it sound like music, or random noise? Sets the bar.

**Gate:** Generated sequences render to recognisable (if primitive) chip audio. Baseline documented.

**Docs:**
- [ ] Explainer: Markov chains for sequence generation — simple but no long-range structure

---

### M3 — Transformer sequence model

- [ ] Train a small transformer (or state-space model) on tokenised chip-tune sequences
- [ ] Autoregressive generation: predict next register write given context
- [ ] Compare against Markov baseline: coherence, musical structure, constraint adherence

**Gate:** Generated sequences have recognisable musical structure (phrases, repetition, chord movement).

**Docs:**
- [ ] Explainer: sequence generation with transformers — attention over register streams
- [ ] Note: state-space models (Mamba/S4) as an alternative for long sequences

---

### M4 — Constraint-aware generation

- [ ] Encode hardware constraints as hard rules: max channels, valid waveforms,
  register timing limits, envelope shapes
- [ ] Reject or mask invalid tokens during generation (constrained decoding)
- [ ] Validate: every generated sequence must be playable on the target chip emulator

**Gate:** 100% of generated sequences pass chip emulator validation. No invalid register writes.

**Docs:**
- [ ] Explainer: constrained decoding — masking invalid tokens vs post-hoc rejection
- [ ] Explainer: hardware constraints as training signal — how limits shape the model

---

### M5 — Style fine-tuning (LoRA or full)

- [ ] Fine-tune the base model on a single composer's catalogue (e.g. Tim Follin, Rob Hubbard)
- [ ] Compare: base model vs fine-tuned — does the output capture the composer's signature patterns?
- [ ] A/B test: can a listener tell fine-tuned output from base model output?

**Gate:** Fine-tuned model produces output that is distinguishably "in the style of" the target composer.

**Docs:**
- [ ] Explainer: fine-tuning for style transfer — why a small dataset works when the base model is strong
- [ ] Note: LoRA vs full fine-tune trade-offs for small sequence models

---

### M6 — CLI tool + demo

- [ ] `python -m dream generate --chip ay --style follin --bars 32`
- [ ] `python -m dream generate --chip sid --bars 16` → renders .vgm + .wav
- [ ] Demo: generate a track, play it in a chip emulator

**Gate:** CLI generates playable chip-tunes from a cold clone. Demo output sounds like authentic chip music.

**Docs:**
- [ ] README: usage, chip types, style options, how to render output

---

## Skills this covers

| Skill | Gap filled |
|-------|-----------|
| Sequence generation (transformer/SSM) | ⬜ → ✅ |
| Constrained decoding | ⬜ → ✅ |
| Fine-tuning for style transfer | ⬜ → ✅ |
| Generative modelling | ⬜ → ✅ |

---

## Feeds from

[bloom](../bloom/) — cleaned audio from bloom serves as higher-quality training data.
[dig](../dig/) — discovered music data from dig provides raw material for the training corpus.
