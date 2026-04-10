#!/usr/bin/env bash
# Execution boundary: clears ERROR.txt, runs train.py, updates results.tsv / ERROR.txt.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

mkdir -p .cache
: > ERROR.txt

OUT="$(mktemp)"
ERR="$(mktemp)"
trap 'rm -f "$OUT" "$ERR"' EXIT

COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo "nogit")"
STATUS="KEEP"
METRIC=""
VRAM=""
DESC=""

if python train.py >"$OUT" 2>"$ERR"; then
  METRIC="$(grep -oE 'METRIC=[0-9.eE+-]+' "$OUT" | head -1 | cut -d= -f2 || true)"
  VRAM="$(grep -oE 'VRAM_MB=[0-9]+' "$OUT" | head -1 | cut -d= -f2 || true)"
  DESC="$(head -1 "$OUT" | tr '\t\r\n' '    ' | sed 's/  */ /g' || true)"
  [[ -z "$METRIC" ]] && METRIC="nan"
  [[ -z "$VRAM" ]] && VRAM="0"
  [[ -z "$DESC" ]] && DESC="ok"
else
  cat "$ERR" > ERROR.txt
  STATUS="CRASH"
  METRIC="nan"
  VRAM="0"
  DESC="$(head -c 200 "$ERR" | tr '\t\r\n' '    ' | sed 's/  */ /g' || true)"
  [[ -z "$DESC" ]] && DESC="train.py failed"
  printf '%s\t%s\t%s\t%s\t%s\n' "$COMMIT" "$METRIC" "$VRAM" "$STATUS" "$DESC" >> "$ROOT/results.tsv"
  exit 1
fi

printf '%s\t%s\t%s\t%s\t%s\n' "$COMMIT" "$METRIC" "$VRAM" "$STATUS" "$DESC" >> "$ROOT/results.tsv"
