#!/bin/bash
# Copy pyodide directory to dist after build
mkdir -p dist/pyodide
cp -r pyodide/*.whl dist/pyodide/ 2>/dev/null || true
echo "✓ Copied wheel to dist/pyodide/"
ls -lh dist/pyodide/
