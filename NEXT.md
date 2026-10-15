# Next

## Glyph — deferred from audio POC

- HAL audio surface (`tone_play`, `sfx_play`) — needed for game sigils
- SFX: sweep, noise burst — melody proved the pipeline
- ADSR envelopes on WASM side
- WASM player as wrapper (bundle player + content for deploy)
- Keyboard input mechanism (store key state in WASM memory)

## Cross-project

- incant to follow same debug pipeline as glyph, with wrappers for testing
- Better build pipeline later that combines stuff
