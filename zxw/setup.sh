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

# ---------------------------------------------------------------------------
# Download ROM files (copyrighted — Amstrad permits free distribution for
# emulation). Source: spectrumforeveryone/zx-roms on GitHub.
# ---------------------------------------------------------------------------

ROM_DIR="www/rom"
ROM_BASE="https://github.com/spectrumforeveryone/zx-roms/raw/main/spectrum16-48"

mkdir -p "$ROM_DIR"

download_rom() {
  local filename="$1"
  local url="$2"
  local expected_sha="$3"

  if [ -f "$ROM_DIR/$filename" ]; then
    echo "  [OK] $filename (already exists)"
    return
  fi

  echo "  [DOWNLOADING] $filename"
  curl -sL "$url" -o "$ROM_DIR/$filename"

  local actual_sha
  actual_sha=$(shasum -a 256 "$ROM_DIR/$filename" | cut -d' ' -f1)
  if [ "$actual_sha" != "$expected_sha" ]; then
    echo "  [FAIL] $filename checksum mismatch — expected $expected_sha, got $actual_sha"
    rm -f "$ROM_DIR/$filename"
    exit 1
  else
    echo "  [OK] $filename (verified)"
  fi
}

download_rom "48.rom" \
  "$ROM_BASE/spec48.rom" \
  "d55daa439b673b0e3f5897f99ac37ecb45f974d1862b4dadb85dec34af99cb42"

download_rom "prototype.rom" \
  "$ROM_BASE/spec48-prototype.rom" \
  "75097b46c59b03458bb3e18856d39fc21d577576392efa1a513c13f123d75132"

# ---------------------------------------------------------------------------
# Generate TAP files from incant tools (if available)
# ---------------------------------------------------------------------------

TAP_DIR="www/tap"
mkdir -p "$TAP_DIR"

INCANT_DIR="../incant"
if [ -d "$INCANT_DIR" ] && [ -f "$INCANT_DIR/tools/gen_hello_world.py" ]; then
  echo ""
  echo "==> Generating TAP files from incant"

  # hello_world.tap
  if [ ! -f "$TAP_DIR/hello_world.tap" ]; then
    echo "  [GENERATING] hello_world.tap"
    (cd "$INCANT_DIR" && python3 -I tools/gen_hello_world.py) 2>&1 | head -3
    cp "$INCANT_DIR/output/hello_world.tap" "$TAP_DIR/"
    echo "  [OK] hello_world.tap"
  else
    echo "  [OK] hello_world.tap (already exists)"
  fi
else
  echo ""
  echo "  [SKIP] incant tools not found — hello_world.tap not generated"
  echo "         Run from games/ root, or copy TAP files into $TAP_DIR manually"
fi

# SCR directory
mkdir -p "www/scr"

# ---------------------------------------------------------------------------
# Assets manifest
# ---------------------------------------------------------------------------

echo ""
echo "==> Generating assets.json"

gen_assets() {
  local dir="$1"
  local first=true
  if [ -d "www/$dir" ]; then
    for f in "www/$dir"/*; do
      [ -f "$f" ] || continue
      $first || printf ','
      printf '"%s"' "$(basename "$f")"
      first=false
    done
  fi
}

printf '{\n  "rom": [' > www/assets.json
gen_assets rom >> www/assets.json
printf '],\n  "tap": [' >> www/assets.json
gen_assets tap >> www/assets.json
printf '],\n  "scr": [' >> www/assets.json
gen_assets scr >> www/assets.json
printf ']\n}\n' >> www/assets.json

echo "  [OK] www/assets.json"

echo ""
echo "All prerequisites met. Run 'make build && make serve'."
