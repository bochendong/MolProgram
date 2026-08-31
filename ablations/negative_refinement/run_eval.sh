#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${NEGATIVE_ABLATION_DIR:?NEGATIVE_ABLATION_DIR must be exported}"
ARM="${NEGATIVE_ARM:?NEGATIVE_ARM must be exported}"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SOURCE_ROOT="${NEGATIVE_SOURCE_ROOT:?Set NEGATIVE_SOURCE_ROOT to the frozen refinement-data root}"
OUT_ROOT="${NEGATIVE_OUTPUT_ROOT:-$REPO_ROOT/outputs/ablations/negative_refinement/seed_2323}"
GATE_ROOT="${NEGATIVE_GATE_ROOT:?Set NEGATIVE_GATE_ROOT to the frozen Raw@1 gates}"
BASE="${NEGATIVE_BASE_MODEL:?Set NEGATIVE_BASE_MODEL to the local backbone path}"
PY="${NEGATIVE_PYTHON_BIN:-python}"
DEP="${NEGATIVE_DEP_OVERLAY:-$REPO_ROOT/src}"
ORACLE_DIR="${NEGATIVE_ASSAY_ORACLE_DIR:?Set NEGATIVE_ASSAY_ORACLE_DIR to the frozen assay models}"

if [[ "$ARM" == "semantic_plus_syntax" ]]; then
  ADAPTER="$SOURCE_ROOT/model/stage1_v2/adapter"
else
  ADAPTER="$OUT_ROOT/model/$ARM/adapter"
fi

module purge >/dev/null 2>&1 || true
module load StdEnv/2023 python/3.11 rdkit/2025.09.4 cuda/12.6
export PYTHONPATH="$DEP:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export SUCC_GSK3B_ORACLE_PATH="${SUCC_GSK3B_ORACLE_PATH:-$ORACLE_DIR/gsk3b_legacy_sklearn_compatible.pkl}"
export SUCC_DRD2_ORACLE_PATH="${SUCC_DRD2_ORACLE_PATH:-$ORACLE_DIR/drd2_graph2graph_svc_py36.pkl}"

"$PY" "$REPO_ROOT/scripts/evaluate_raw1.py" \
  --denovo-gate "$GATE_ROOT/gate.denovo.jsonl" \
  --edit-gate "$GATE_ROOT/gate.edit.jsonl" \
  --base-model "$BASE" --adapter-dir "$ADAPTER" \
  --arm "$ARM" --output-dir "$OUT_ROOT/eval/$ARM" \
  --batch-size 8 --seed 33151 \
  --protocol molprogram_negative_refinement_raw1_v1
