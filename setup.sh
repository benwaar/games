#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

BOLD='\033[1m'
DIM='\033[2m'
GREEN='\033[32m'
RED='\033[31m'
RESET='\033[0m'

ok()   { echo -e "  ${GREEN}[OK]${RESET} $1"; }
fail() { echo -e "  ${RED}[MISSING]${RESET} $1"; }
step() { echo -e "\n${BOLD}==> $1${RESET}"; }

ERRORS=0

# ─── Prerequisites ──────────────────────────────────────────────

step "Checking prerequisites"

command -v rustc &>/dev/null   && ok "rustc $(rustc --version | awk '{print $2}')"  || { fail "rustc — install from https://rustup.rs"; ERRORS=$((ERRORS+1)); }
command -v cargo &>/dev/null   && ok "cargo"                                         || { fail "cargo"; ERRORS=$((ERRORS+1)); }
command -v python3 &>/dev/null && ok "python3 $(python3 --version | awk '{print $2}')" || { fail "python3"; ERRORS=$((ERRORS+1)); }
command -v pip &>/dev/null     && ok "pip"                                            || { fail "pip"; ERRORS=$((ERRORS+1)); }
command -v ollama &>/dev/null  && ok "ollama"                                         || { fail "ollama — install from https://ollama.ai"; ERRORS=$((ERRORS+1)); }
command -v wat2wasm &>/dev/null && ok "wabt (wat2wasm)"                               || { fail "wabt — brew install wabt"; ERRORS=$((ERRORS+1)); }

if [ "$ERRORS" -gt 0 ]; then
    echo -e "\n${RED}${ERRORS} missing prerequisite(s). Install them and re-run.${RESET}"
    exit 1
fi

# ─── knowledge-mcp (Rust MCP server) ────────────────────────────

step "Building knowledge-mcp"
(cd knowledge-mcp && cargo build --release)
ok "knowledge-mcp binary"

# ─── Knowledge corpus (fetch + normalise + embed) ───────────────

step "Setting up knowledge corpus"
bash knowledge/setup.sh
ok "knowledge corpus"

# ─── incant (spec → code gen) ───────────────────────────────────

step "Setting up incant"
bash incant/setup.sh
ok "incant"

# ─── zxw (Spectrum emulator) ────────────────────────────────────

step "Setting up zxw"
bash zxw/setup.sh
ok "zxw"

# ─── Summary ────────────────────────────────────────────────────

step "Done"
echo -e "  ${DIM}Restart Claude Code to activate the knowledge-corpus MCP server.${RESET}"
echo ""
