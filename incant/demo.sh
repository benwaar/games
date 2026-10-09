#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Activate venv
if [ ! -d .venv ]; then
    echo "ERROR: No .venv found. Run 'bash setup.sh' first."
    exit 1
fi
source .venv/bin/activate

# Check embeddings exist
if [ ! -f data/embeddings.jsonl ]; then
    echo "ERROR: No embeddings found. Run 'bash setup.sh' first."
    exit 1
fi

# Clean previous output
rm -rf output/
mkdir -p output

PASS=0
FAIL=0

run_step() {
    local label="$1"
    shift
    echo ""
    echo "── $label ──"
    if "$@"; then
        echo "  ✓ $label"
        PASS=$((PASS + 1))
    else
        echo "  ✗ $label FAILED"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== incant demo ==="

# --- Single sigils ---

echo ""
echo "━━━ Single sigils ━━━"

run_step "WAT add"          python -m incant cast sigils/examples/wat_add.sigil.yaml
run_step "WAT factorial"    python -m incant cast sigils/examples/wat_factorial.sigil.yaml
run_step "WAT memory swap"  python -m incant cast sigils/examples/wat_memory_swap.sigil.yaml
run_step "Z80 add"          python -m incant cast sigils/examples/z80_add.sigil.yaml
run_step "Z80 counter"      python -m incant cast sigils/examples/z80_counter.sigil.yaml
run_step "Z80 store/load"   python -m incant cast sigils/examples/z80_store_load.sigil.yaml

# --- Multi-sigil builds ---

echo ""
echo "━━━ Multi-sigil builds ━━━"

run_step "WAT math (add + factorial)"         python -m incant multi sigils/programs/wat_math.manifest.yaml
run_step "Z80 basics (add + counter + store)" python -m incant multi sigils/programs/z80_basics.manifest.yaml
run_step "WAT composed (deps: sum_of_factorials → factorial + add)" \
    python -m incant multi sigils/programs/wat_composed.manifest.yaml

# --- Smelt (BDD spec → binary) ---

echo ""
echo "━━━ Smelt (BDD spec → binary) ━━━"

run_step "Smelt greet spec → WASI binary" python -m incant smelt specs/greet.spec.md -v

# --- RAG query ---

echo ""
echo "━━━ RAG query ━━━"

run_step "RAG: Z80 add" python -m incant rag query "add two numbers" --collection z80

# --- Summary ---

echo ""
echo "━━━━━━━━━━━━━━━━━━━━"
echo "Results: $PASS passed, $FAIL failed"

if [ "$FAIL" -gt 0 ]; then
    echo "Some steps failed — check output above."
    exit 1
fi

echo "All demos passed."
echo "Output files in: output/"
ls -la output/
