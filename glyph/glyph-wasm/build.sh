#!/usr/bin/env bash
# Build glyph-wasm for WASM and copy output alongside the HTML loader.
set -euo pipefail

CRATE_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$CRATE_DIR/web"

echo "Building WASM target..."
cargo build --manifest-path "$CRATE_DIR/Cargo.toml" \
    --target wasm32-unknown-unknown \
    --release

mkdir -p "$OUT_DIR"

WASM_SRC="$CRATE_DIR/target/wasm32-unknown-unknown/release/glyph-wasm.wasm"
cp "$WASM_SRC" "$OUT_DIR/glyph-wasm.wasm"

# macroquad ships a minimal JS loader — copy it next to the WASM.
# If it is bundled inside the binary, macroquad will generate it at runtime.
# The index.html below loads it by convention.
cp "$CRATE_DIR/index.html" "$OUT_DIR/index.html"

echo "Output in $OUT_DIR"
echo "Serve with: python3 -m http.server 8080 --directory $OUT_DIR"
