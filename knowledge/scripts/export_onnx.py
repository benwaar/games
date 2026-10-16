"""
Download pre-exported ONNX model + tokenizer for the embedding model.

Produces two files the Rust MCP server needs at query time:
  - knowledge/index/model.onnx       — the transformer model (from HuggingFace)
  - knowledge/index/tokenizer.json   — the HuggingFace tokenizer

Run from repo root:
    python knowledge/scripts/export_onnx.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = REPO_ROOT / "knowledge" / "index"

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
ONNX_PATH = INDEX_DIR / "model.onnx"
TOKENIZER_PATH = INDEX_DIR / "tokenizer.json"


def main() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading ONNX model from {MODEL_ID}...")
    onnx_src = Path(hf_hub_download(MODEL_ID, "onnx/model.onnx"))
    shutil.copy2(onnx_src, ONNX_PATH)

    print("Downloading tokenizer...")
    tok_src = Path(hf_hub_download(MODEL_ID, "tokenizer.json"))
    shutil.copy2(tok_src, TOKENIZER_PATH)

    onnx_size = ONNX_PATH.stat().st_size / 1024 / 1024
    print(f"\nModel saved     → {ONNX_PATH}  ({onnx_size:.1f} MB)")
    print(f"Tokenizer saved → {TOKENIZER_PATH}")


if __name__ == "__main__":
    main()
