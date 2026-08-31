#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${NEGATIVE_ABLATION_DIR:?NEGATIVE_ABLATION_DIR must be exported}"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SOURCE_ROOT="${NEGATIVE_SOURCE_ROOT:?Set NEGATIVE_SOURCE_ROOT to the frozen refinement-data root}"
GATE_ROOT="${NEGATIVE_GATE_ROOT:?Set NEGATIVE_GATE_ROOT to the frozen Raw@1 gates}"
BASE="${NEGATIVE_BASE_MODEL:?Set NEGATIVE_BASE_MODEL to the local backbone path}"
OUT_ROOT="${NEGATIVE_OUTPUT_ROOT:-$REPO_ROOT/outputs/ablations/negative_refinement/seed_2323}"
PY="${NEGATIVE_PYTHON_BIN:-python}"

module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11

for path in \
  "$SOURCE_ROOT/data/train.contrastive.jsonl" \
  "$SOURCE_ROOT/model/sft/adapter/adapter_model.safetensors" \
  "$SOURCE_ROOT/model/stage1_v2/adapter/adapter_model.safetensors" \
  "$GATE_ROOT/gate.denovo.jsonl" \
  "$GATE_ROOT/gate.edit.jsonl" \
  "$BASE/config.json"; do
  [[ -f "$path" ]] || { echo "missing required artifact: $path" >&2; exit 2; }
done

"$PY" -m pytest -q "$REPO_ROOT/tests/test_ablation_contract.py"
"$PY" -m compileall -q "$SCRIPT_DIR" "$REPO_ROOT/scripts" "$REPO_ROOT/src"
mkdir -p "$OUT_ROOT"
sha256sum \
  "$SOURCE_ROOT/data/train.contrastive.jsonl" \
  "$SOURCE_ROOT/model/sft/adapter/adapter_model.safetensors" \
  "$SOURCE_ROOT/model/stage1_v2/adapter/adapter_model.safetensors" \
  "$GATE_ROOT/gate.denovo.jsonl" \
  "$GATE_ROOT/gate.edit.jsonl" > "$OUT_ROOT/input_hashes.sha256"
touch "$OUT_ROOT/PREFLIGHT_COMPLETE"
