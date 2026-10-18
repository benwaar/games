#!/bin/bash
# Signal Hunt demo — identifies an unknown sound using the trained model.
#
# Usage:
#   bash demo.sh                      # uses data/raw/unknown.wav (committed demo file)
#   bash demo.sh path/to/my_sound.wav # use your own recording

set -e

source .venv/bin/activate

CHECKPOINT="output/best_model.pt"
DEFAULT_FILE="data/raw/unknown.wav"
TARGET="${1:-$DEFAULT_FILE}"

echo "=== Signal Hunt Demo ==="
echo ""

# -- check checkpoint exists --
if [ ! -f "$CHECKPOINT" ]; then
  echo "No trained model found at $CHECKPOINT"
  echo "Run this first:  python -m model.train --epochs 50"
  exit 1
fi

# -- check target file exists --
if [ ! -f "$TARGET" ]; then
  echo "File not found: $TARGET"
  exit 1
fi

echo "File:   $TARGET"
echo ""
python -m model.predict "$TARGET" --verbose
echo ""
echo "To identify your own recording:"
echo "  bash demo.sh path/to/my_sound.wav"
echo ""
echo "Or drop any .wav into data/raw/ and run:"
echo "  python -m model.predict --scan"
