#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/fetch_z80.py"
python "$SCRIPT_DIR/fetch_z80_extras.py"
python "$SCRIPT_DIR/fetch_spectrum.py"
python "$SCRIPT_DIR/fetch_wasm.py"

echo "Done — raw files in knowledge/raw/"
