#!/usr/bin/env bash
# Knowledge corpus query helper for /z80 skill
set -euo pipefail

CHUNKS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/knowledge/chunks"

case "${1:-overview}" in
  overview)
    for d in "$CHUNKS_DIR"/*/; do
      name=$(basename "$d")
      count=$(find "$d" -name '*.md' -maxdepth 1 | wc -l | tr -d ' ')
      echo "$name $count"
    done
    ;;
  list)
    dir="$CHUNKS_DIR/${2:?missing category dir}"
    if [ ! -d "$dir" ]; then
      echo "ERROR: no such category directory: $2"
      exit 1
    fi
    for f in "$dir"/*.md; do
      [ -f "$f" ] || continue
      basename "$f" .md
    done | sort
    ;;
  search)
    dir="${2:-}"
    query="${3:?missing query}"
    if [ -n "$dir" ]; then
      target="$CHUNKS_DIR/$dir"
    else
      target="$CHUNKS_DIR"
    fi
    # filename match
    find "$target" -name '*.md' -maxdepth 2 | while read -r f; do
      stem=$(basename "$f" .md)
      if echo "$stem" | grep -qi "$query"; then
        echo "FILE:$f"
      fi
    done
    # content match
    grep -ril "$query" "$target" --include='*.md' 2>/dev/null | while read -r f; do
      echo "GREP:$f"
    done
    ;;
  *)
    echo "Usage: z80-query.sh [overview|list <dir>|search <dir|''> <query>]"
    exit 1
    ;;
esac
