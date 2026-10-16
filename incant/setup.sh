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

# Embed knowledge (curated + corpus)
if [ -f incant/rag.py ]; then
    echo "Embedding curated knowledge..."
    python -m incant rag embed ../knowledge/curated/z80 z80
    python -m incant rag embed ../knowledge/curated/wat wat

    echo "Embedding knowledge corpus chunks..."
    for dir in ../knowledge/chunks/z80-instruction ../knowledge/chunks/z80-pattern ../knowledge/chunks/spectrum-rom ../knowledge/chunks/z80-undocumented ../knowledge/chunks/z80-beeper ../knowledge/chunks/z80-library; do
        [ -d "$dir" ] && python -m incant rag embed "$dir" z80 --append
    done
    for dir in ../knowledge/chunks/wasm-spec ../knowledge/chunks/wasm-pattern ../knowledge/chunks/wasi ../knowledge/chunks/wasm-binary; do
        [ -d "$dir" ] && python -m incant rag embed "$dir" wat --append
    done
fi

echo "=== setup complete ==="
echo "Activate with: source .venv/bin/activate"
