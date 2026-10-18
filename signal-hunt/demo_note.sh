#!/bin/bash
# Signal Hunt — Phase 3 note demo.
# Runs an unknown piano note through the trained classifier and shows the result.
#
# Usage:
#   bash demo_note.sh                         # uses data/raw/unknown_note.wav
#   bash demo_note.sh path/to/my_note.wav     # use your own recording

set -e

source .venv/bin/activate

CHECKPOINT="output/transfer/finetune/best_model.pt"
DEFAULT_FILE="data/raw/unknown_note.wav"
TARGET="${1:-$DEFAULT_FILE}"

echo "=== Signal Hunt — Phase 3: Note Classifier ==="
echo ""

# -- check checkpoint exists --
if [ ! -f "$CHECKPOINT" ]; then
  echo "No Phase 3 model found at $CHECKPOINT"
  echo "Run this first:  python -m model.transfer_train --no-freeze"
  exit 1
fi

# -- check target file exists --
if [ ! -f "$TARGET" ]; then
  echo "File not found: $TARGET"
  exit 1
fi

echo "File:  $TARGET"
echo ""

python -m model.predict "$TARGET" --mode note --verbose

echo ""
echo "To identify your own recording:"
echo "  bash demo_note.sh path/to/my_note.wav"
