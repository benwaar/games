#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "=== incant setup ==="

# Python venv
if [ ! -d .venv ]; then
    python3 -m venv .venv
    echo "Created venv"
fi
source .venv/bin/activate
pip install -q -r requirements.txt

# Check Ollama
if ! command -v ollama &>/dev/null; then
    echo "ERROR: ollama not found. Install from https://ollama.ai"
    exit 1
fi

# Check required models
for model in qwen3-coder:latest nomic-embed-text:latest; do
    if ! ollama list | grep -q "${model%%:*}"; then
        echo "Pulling $model..."
        ollama pull "$model"
    fi
done

# Check wabt (wat2wasm)
if ! command -v wat2wasm &>/dev/null; then
    echo "ERROR: wabt not found. Install with: brew install wabt"
    exit 1
fi

# Embed knowledge (when RAG module exists)
if [ -f incant/rag.py ]; then
    echo "Embedding knowledge..."
    python -m incant rag embed knowledge/z80 z80
    python -m incant rag embed knowledge/wat wat
fi

echo "=== setup complete ==="
echo "Activate with: source .venv/bin/activate"
