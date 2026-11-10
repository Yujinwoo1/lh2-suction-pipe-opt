#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA=""
OUT=""
QUICK=()
while (($#)); do
  case "$1" in
    --quick) QUICK=(--quick); shift ;;
    --out) OUT="${2:?--out requires a path}"; shift 2 ;;
    -*) echo "Unknown option: $1" >&2; exit 2 ;;
    *) if [[ -n "$DATA" ]]; then echo "Provide exactly one data path." >&2; exit 2; fi; DATA="$1"; shift ;;
  esac
done
[[ -n "$DATA" ]] || { echo "Usage: $0 MERGED.csv [--quick] [--out DIRECTORY]" >&2; exit 2; }
DATA="$(realpath -- "$DATA")"
[[ -f "$DATA" ]] || { echo "Data file not found: $DATA" >&2; exit 2; }
if [[ -z "$OUT" ]]; then
  if ((${#QUICK[@]})); then OUT="$ROOT/outputs/dryrun_full10k"; else OUT="$ROOT/outputs/full10k"; fi
fi
mkdir -p -- "$OUT"
OUT="$(realpath -- "$OUT")"
PYTHON="$ROOT/.venv/bin/python"
[[ -x "$PYTHON" ]] || { echo "Missing interpreter: $PYTHON" >&2; exit 2; }
nohup "$PYTHON" -u "$ROOT/full10k_runner.py" "$DATA" --out "$OUT" "${QUICK[@]}" </dev/null >>"$OUT/launcher.log" 2>&1 &
PID=$!
printf 'Started PID %s; log: %s/run.log\n' "$PID" "$OUT"
