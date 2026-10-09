"""incant CLI — cast sigils into binaries."""

import argparse
import sys
from pathlib import Path

from .gen import cast_sigil
from .rag import VectorStore
from .sigil import parse_sigil

PROJECT_ROOT = Path(__file__).parent.parent
STORE_PATH = PROJECT_ROOT / "data" / "embeddings.jsonl"


def cmd_cast(args):
    store = VectorStore(STORE_PATH)
    sigil = parse_sigil(Path(args.sigil))
    output_dir = Path(args.output) if args.output else PROJECT_ROOT / "output"

    print(f"Casting sigil: {sigil.name} (target: {sigil.target})")

    success, message = cast_sigil(
        sigil, store, output_dir, verbose=args.verbose
    )

    if success:
        print(f"  {message}")
        print(f"  Output: {output_dir / sigil.name}.*")
    else:
        print(f"  FAILED: {message}", file=sys.stderr)
        sys.exit(1)


def cmd_rag(args):
    store = VectorStore(STORE_PATH)

    if args.rag_action == "embed":
        dir_path = Path(args.path)
        collection = args.collection
        print(f"Embedding {dir_path} into collection '{collection}'...")
        store.embed_directory(dir_path, collection)
        count = sum(1 for c in store.chunks if c.collection == collection)
        print(f"  {count} chunks embedded")

    elif args.rag_action == "query":
        query = args.query_text
        collection = args.collection
        print(f"Query: '{query}' (collection: {collection or 'all'})")
        results = store.query(query, collection=collection, top_k=args.top_k)
        for chunk, score in results:
            print(f"\n  [{score:.3f}] {chunk.heading} ({chunk.source})")
            preview = chunk.text[:200].replace("\n", " ")
            print(f"  {preview}...")


def main():
    parser = argparse.ArgumentParser(
        prog="incant",
        description="Spec-driven code generation for WAT and Z80 assembly",
    )
    subparsers = parser.add_subparsers(dest="command")

    # cast
    cast_parser = subparsers.add_parser("cast", help="Generate code from a sigil")
    cast_parser.add_argument("sigil", help="Path to sigil YAML file")
    cast_parser.add_argument("-o", "--output", help="Output directory")
    cast_parser.add_argument("-v", "--verbose", action="store_true")
    cast_parser.set_defaults(func=cmd_cast)

    # rag
    rag_parser = subparsers.add_parser("rag", help="Manage RAG knowledge base")
    rag_sub = rag_parser.add_subparsers(dest="rag_action")

    embed_parser = rag_sub.add_parser("embed", help="Embed a directory")
    embed_parser.add_argument("path", help="Directory of markdown files")
    embed_parser.add_argument("collection", help="Collection name")

    query_parser = rag_sub.add_parser("query", help="Query the knowledge base")
    query_parser.add_argument("query_text", help="Query string")
    query_parser.add_argument("-c", "--collection", help="Filter by collection")
    query_parser.add_argument("-k", "--top-k", type=int, default=5)

    rag_parser.set_defaults(func=cmd_rag)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
