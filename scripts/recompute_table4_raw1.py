#!/usr/bin/env python3
"""Rescore frozen joint/specialist Raw@1 candidates for paper Table 4."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCRIPT_DIR.parent / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from molprogram import protocol  # noqa: E402
from molprogram.scoring import (  # noqa: E402
    PINNED_ORACLE_ENVS,
    STRICT_TOLERANCE,
    configured_oracle_provenance,
    property_count,
    score_response,
)


SHARED_EDIT_TASKS = frozenset(
    {
        "HBA:decrease+LogP:increase",
        "HBA:decrease+MW:decrease",
        "HBA:increase+MW:increase+QED:decrease",
        "MW:increase",
        "RB:decrease",
    }
)
ALL_EDIT_TASKS = frozenset(protocol.TABLE1_TASK_KEYS.values())
EDIT_ONLY_TASKS = ALL_EDIT_TASKS - SHARED_EDIT_TASKS
DE_NOVO_BUCKETS = tuple(f"{count}p" for count in range(2, 8))
CSV_FIELDS = (
    "examples_per_task",
    "joint_denovo_strict",
    "denovo_specialist_strict",
    "joint_edit_strict",
    "edit_specialist_strict",
    "joint_edit_shared5_strict",
    "edit_specialist_shared5_strict",
    "joint_edit_only5_strict",
    "edit_specialist_edit_only5_strict",
    "joint_denovo_valid",
    "denovo_specialist_valid",
    "joint_edit_valid",
    "edit_specialist_valid",
    "joint_edit_shared5_valid",
    "edit_specialist_shared5_valid",
    "joint_edit_only5_valid",
    "edit_specialist_edit_only5_valid",
    "decision",
)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def row_id(row: Mapping[str, object]) -> str:
    return str(row.get("condition_id") or row.get("sample_id") or "")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(values: Iterable[object]) -> float:
    materialized = [float(value) for value in values]
    if not materialized:
        raise ValueError("cannot average an empty group")
    return sum(materialized) / len(materialized)


def evaluate_arm(
    gate_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
) -> dict[str, list[dict[str, object]]]:
    references = {row_id(row): row for row in gate_rows}
    candidate_ids = [row_id(row) for row in candidate_rows]
    if len(references) != len(gate_rows):
        raise ValueError("gate condition IDs are empty or duplicated")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate condition IDs are empty or duplicated")
    if set(candidate_ids) != set(references):
        missing = sorted(set(references) - set(candidate_ids))[:5]
        extra = sorted(set(candidate_ids) - set(references))[:5]
        raise ValueError(f"candidate/gate ID mismatch: missing={missing}, extra={extra}")

    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for candidate in candidate_rows:
        reference = references[row_id(candidate)]
        _, details = score_response(reference, str(candidate.get("raw", "")))
        mode = str(reference.get("task_mode", ""))
        group = (
            f"{property_count(reference)}p"
            if mode == "de_novo"
            else str(reference.get("task_key", ""))
        )
        groups[group].append(details)
    return groups


def group_macro(
    groups: Mapping[str, Sequence[Mapping[str, object]]],
    keys: Iterable[str],
    metric: str,
) -> float:
    selected = list(keys)
    for key in selected:
        if not groups.get(key):
            raise ValueError(f"missing evaluation group: {key}")
    return mean(mean(row[metric] for row in groups[key]) for key in selected)


def decision(row: Mapping[str, object]) -> str:
    denovo_delta = float(row["joint_denovo_strict"]) - float(
        row["denovo_specialist_strict"]
    )
    edit_delta = float(row["joint_edit_strict"]) - float(
        row["edit_specialist_strict"]
    )
    valid_deltas = (
        float(row["joint_denovo_valid"]) - float(row["denovo_specialist_valid"]),
        float(row["joint_edit_valid"]) - float(row["edit_specialist_valid"]),
    )
    noninferior = min(denovo_delta, edit_delta, *valid_deltas) >= -0.02 - 1e-12
    positive = max(denovo_delta, edit_delta) >= 0.02 - 1e-12
    negative = min(denovo_delta, edit_delta) < -0.02
    if noninferior and positive:
        return "positive_transfer_pilot"
    if noninferior:
        return "parameter_efficiency_pilot"
    if negative:
        return "negative_transfer_pilot"
    return "inconclusive_pilot"


def parse_mapping(values: Sequence[str], option: str) -> dict[int, Path]:
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} expects EXAMPLES=PATH, got: {value}")
        scale, raw_path = value.split("=", 1)
        parsed[int(scale)] = Path(raw_path).expanduser().resolve()
    return parsed


def summarize_scale(scale: int, root: Path, edit_gate_override: Path | None):
    paths = {
        "denovo_gate": root / "data/gate.denovo.jsonl",
        "edit_gate": edit_gate_override or root / "data/gate.edit.jsonl",
        "joint_candidates": root / "eval/joint/candidates.jsonl",
        "denovo_candidates": root / "eval/denovo/candidates.jsonl",
        "edit_candidates": root / "eval/edit/candidates.jsonl",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing Table 4 input(s): " + ", ".join(missing))

    joint_candidates = read_jsonl(paths["joint_candidates"])
    wanted_ids = {row_id(row) for row in joint_candidates}
    available_gate = read_jsonl(paths["denovo_gate"]) + read_jsonl(paths["edit_gate"])
    gate = [row for row in available_gate if row_id(row) in wanted_ids]
    joint = evaluate_arm(gate, joint_candidates)
    denovo = evaluate_arm(gate, read_jsonl(paths["denovo_candidates"]))
    edit = evaluate_arm(gate, read_jsonl(paths["edit_candidates"]))

    row: dict[str, object] = {"examples_per_task": scale}
    row.update(
        {
            "joint_denovo_strict": group_macro(
                joint, DE_NOVO_BUCKETS, "strict"
            ),
            "denovo_specialist_strict": group_macro(
                denovo, DE_NOVO_BUCKETS, "strict"
            ),
            "joint_edit_strict": group_macro(joint, ALL_EDIT_TASKS, "strict"),
            "edit_specialist_strict": group_macro(edit, ALL_EDIT_TASKS, "strict"),
            "joint_edit_shared5_strict": group_macro(
                joint, SHARED_EDIT_TASKS, "strict"
            ),
            "edit_specialist_shared5_strict": group_macro(
                edit, SHARED_EDIT_TASKS, "strict"
            ),
            "joint_edit_only5_strict": group_macro(
                joint, EDIT_ONLY_TASKS, "strict"
            ),
            "edit_specialist_edit_only5_strict": group_macro(
                edit, EDIT_ONLY_TASKS, "strict"
            ),
            "joint_denovo_valid": group_macro(joint, DE_NOVO_BUCKETS, "valid"),
            "denovo_specialist_valid": group_macro(
                denovo, DE_NOVO_BUCKETS, "valid"
            ),
            "joint_edit_valid": group_macro(joint, ALL_EDIT_TASKS, "valid"),
            "edit_specialist_valid": group_macro(edit, ALL_EDIT_TASKS, "valid"),
            "joint_edit_shared5_valid": group_macro(
                joint, SHARED_EDIT_TASKS, "valid"
            ),
            "edit_specialist_shared5_valid": group_macro(
                edit, SHARED_EDIT_TASKS, "valid"
            ),
            "joint_edit_only5_valid": group_macro(
                joint, EDIT_ONLY_TASKS, "valid"
            ),
            "edit_specialist_edit_only5_valid": group_macro(
                edit, EDIT_ONLY_TASKS, "valid"
            ),
        }
    )
    row["decision"] = decision(row)
    provenance = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }
    return row, provenance


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scale-root",
        action="append",
        required=True,
        help="Frozen scale input as EXAMPLES=ROOT; repeat for each scale.",
    )
    parser.add_argument(
        "--edit-gate",
        action="append",
        default=[],
        help="Optional legacy edit gate override as EXAMPLES=PATH.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    roots = parse_mapping(args.scale_root, "--scale-root")
    edit_gates = parse_mapping(args.edit_gate, "--edit-gate")
    oracle_provenance = configured_oracle_provenance()
    missing_oracles = sorted(set(PINNED_ORACLE_ENVS) - set(oracle_provenance))
    if missing_oracles:
        raise SystemExit(
            "Table 4 rescoring requires pinned assay oracles: "
            + ", ".join(missing_oracles)
        )

    rows = []
    input_provenance = {}
    for scale, root in sorted(roots.items()):
        row, provenance = summarize_scale(scale, root, edit_gates.get(scale))
        rows.append(row)
        input_provenance[str(scale)] = provenance

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "joint_vs_specialists_scale_raw1.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "protocol": "paper_frozen_joint_vs_specialists_raw1_rescore_v1",
        "candidate_selection": "existing sampled-once Raw@1; no regeneration or reranking",
        "de_novo_metric": {
            "aggregation": "unweighted 2p--7p arity macro",
            "strict_tolerances": STRICT_TOLERANCE,
        },
        "editing_metric": {
            "aggregation": "unweighted task macro",
            "strict": "all directions satisfied and Morgan Tanimoto >= 0.65",
            "shared_tasks": sorted(SHARED_EDIT_TASKS),
            "edit_only_tasks": sorted(EDIT_ONLY_TASKS),
        },
        "oracles": oracle_provenance,
        "inputs": input_provenance,
        "rows": rows,
    }
    json_path = args.output_dir / "joint_vs_specialists_scale_raw1_rescore.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
