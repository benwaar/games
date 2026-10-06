#!/bin/bash
# Signal Hunt demo — shows the full pipeline end to end.
# Run after setup.sh and model.train.
#
# Usage:
#   bash demo.sh                      # uses a built-in example from data/raw/
#   bash demo.sh path/to/my_sound.wav # use your own recording

set -e

source .venv/bin/activate

CHECKPOINT="output/best_model.pt"

echo "=== Signal Hunt Demo ==="
echo ""

# -- check checkpoint exists --
if [ ! -f "$CHECKPOINT" ]; then
  echo "No trained model found at $CHECKPOINT"
  echo "Run this first:  python -m model.train --epochs 50"
  exit 1
fi

# -- single file mode --
if [ -n "$1" ]; then
  echo "Predicting: $1"
  echo ""
  python -m model.predict "$1" --verbose
  exit 0
fi

# -- built-in demo: use one file from each class --
echo "Predicting one file from each class (from data/raw/):"
echo ""

for CLASS in hum whistle clap; do
  FILE=$(ls data/raw/${CLASS}/*.wav 2>/dev/null | head -1)
  if [ -n "$FILE" ]; then
    RESULT=$(python -m model.predict "$FILE")
    printf "  %-40s → %s\n" "$(basename $FILE) ($CLASS)" "$RESULT"
  fi
done

echo ""
echo "--- Scan mode ---"
echo "Drop any .wav into data/raw/ (not in a subfolder) and run:"
echo ""
echo "  python -m model.predict --scan"
echo ""
echo "--- Train from scratch ---"
echo ""
echo "  python -m model.train --epochs 50"
echo "  python -m model.evaluate"
