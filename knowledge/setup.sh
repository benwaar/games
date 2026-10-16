#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=== Knowledge Corpus Setup ==="

# 1. Install Python dependencies
echo "Installing dependencies..."
pip install -q -r knowledge/requirements.txt

# 2. Fetch raw data from public sources
echo "Fetching raw data..."
bash knowledge/scripts/fetch_all.sh

# 3. Normalise into chunks
echo "Normalising chunks..."
python knowledge/scripts/normalise.py

# 4. Build FAISS index (optional — needs sentence-transformers + faiss-cpu)
echo "Building FAISS index..."
python knowledge/scripts/embed.py

# 5. Build human-readable kb/
echo "Building kb/..."
python knowledge/scripts/build_kb.py

echo ""
echo "=== Done ==="
echo "Corpus ready: $(find knowledge/chunks -name '*.md' | wc -l | tr -d ' ') chunks"
echo ""
echo "MCP server registered in .claude/settings.local.json"
echo "Restart Claude Code to activate the knowledge-corpus MCP server."
