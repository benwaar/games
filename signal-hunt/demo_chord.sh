#!/usr/bin/env bash
# Two-pass chord inference: verdict (chord name) + correction (missing notes)
# Usage: bash demo_chord.sh                   — run on data/raw/unknown_chord.wav
#        bash demo_chord.sh path/to/chord.wav  — any recording
#        bash demo_chord.sh chord.wav --expected Cmaj  — show what's missing
set -e
cd "$(dirname "$0")"
source .venv/bin/activate

WAV="${1:-data/raw/unknown_chord.wav}"
shift 2>/dev/null || true

if [ ! -f "$WAV" ]; then
  echo "No file at $WAV — drop a chord recording there and re-run."
  exit 1
fi

python -m model.predict_chord "$WAV" "$@"
