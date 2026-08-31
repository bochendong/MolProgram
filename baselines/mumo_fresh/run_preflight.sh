#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${MUMO_BASELINE_DIR:?MUMO_BASELINE_DIR must be exported}"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRAIN_JSONL="${MUMO_TRAIN_JSONL:?Set MUMO_TRAIN_JSONL to the indexed release}"
BASE="${MUMO_BASE_MODEL:?Set MUMO_BASE_MODEL to the local backbone path}"
PY="${MUMO_PYTHON_BIN:-python}"
OUT_ROOT="${MUMO_OUTPUT_ROOT:-$REPO_ROOT/outputs/baselines/mumo_fresh/seed_32002}"

module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11
for path in "$TRAIN_JSONL" "${TRAIN_JSONL%.jsonl}.idx" "$BASE/config.json"; do
  [[ -f "$path" ]] || { echo "missing required artifact: $path" >&2; exit 2; }
done
"$PY" -m pytest -q "$REPO_ROOT/tests/test_mumo_baseline_contract.py"
"$PY" -m compileall -q "$SCRIPT_DIR"
mkdir -p "$OUT_ROOT"
sha256sum "$TRAIN_JSONL" > "$OUT_ROOT/input_hash.sha256"
touch "$OUT_ROOT/PREFLIGHT_COMPLETE"
