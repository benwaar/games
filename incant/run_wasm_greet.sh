#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

WAT="output/greet_user.wat"
WASM="output/greet_user.wasm"

if [ ! -f "$WAT" ]; then
    echo "No greet_user.wat found. Run: python -m incant smelt specs/greet.spec.md -v"
    exit 1
fi

wat2wasm "$WAT" -o "$WASM"

if [ -n "${1:-}" ]; then
    echo "$1" | wasm-interp --wasi "$WASM"
else
    echo "Type a name, press Enter:"
    wasm-interp --wasi "$WASM"
fi
