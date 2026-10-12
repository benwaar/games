#!/usr/bin/env bash
set -e

echo "==> Checking prerequisites for zxw"

# rustup
if ! command -v rustup &>/dev/null; then
  echo "  [MISSING] rustup — install from https://rustup.rs"
  exit 1
fi
echo "  [OK] rustup"

# wasm32 target
if ! rustup target list --installed | grep -q wasm32-unknown-unknown; then
  echo "  [INSTALLING] wasm32-unknown-unknown target"
  rustup target add wasm32-unknown-unknown
fi
echo "  [OK] wasm32-unknown-unknown"

# wasm-pack
if ! command -v wasm-pack &>/dev/null; then
  echo "  [INSTALLING] wasm-pack"
  curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
fi
echo "  [OK] wasm-pack"

# python3 (for dev server)
if ! command -v python3 &>/dev/null; then
  echo "  [MISSING] python3 — needed for 'make serve'"
  exit 1
fi
echo "  [OK] python3"

echo ""
echo "All prerequisites met. Run 'make build && make serve'."
