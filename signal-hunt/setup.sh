#!/bin/bash
# Signal Hunt setup — idempotent, safe to re-run
# Requires: pyenv, brew

set -e

echo "=== Signal Hunt Setup ==="

PYTHON_VERSION="3.12.10"

# -- pyenv --
if ! command -v pyenv &>/dev/null; then
  echo "MISSING: pyenv — install from https://github.com/pyenv/pyenv"
  exit 1
fi
echo "OK: pyenv"

# Install Python version if missing
if ! pyenv versions --bare | grep -q "^${PYTHON_VERSION}$"; then
  echo "Installing Python ${PYTHON_VERSION} via pyenv..."
  pyenv install "$PYTHON_VERSION"
fi
echo "OK: Python ${PYTHON_VERSION} available"

# Set local version
pyenv local "$PYTHON_VERSION"
eval "$(pyenv init -)"
echo "OK: .python-version set to ${PYTHON_VERSION}"

# -- venv --
if [ ! -d .venv ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi
source .venv/bin/activate
echo "OK: venv active ($(python3 --version))"

# -- portaudio (optional, for mic recording) --
if ! brew list portaudio &>/dev/null 2>&1; then
  echo "Installing portaudio via brew (needed for mic recording)..."
  brew install portaudio
fi
echo "OK: portaudio"

# -- Python deps --
echo "Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Verify imports
python3 -c "import librosa; import soundfile; import numpy; import torch; import matplotlib" 2>/dev/null
echo "OK: all Python imports verified"

# -- directories --
mkdir -p data/raw data/processed output tests pipeline model explainers

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  source .venv/bin/activate"
echo "  python hello_audio.py    # M1: verify everything works"
