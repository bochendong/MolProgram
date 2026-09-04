#!/usr/bin/env python3
"""Freeze the target-blind 440/5000 headline gates for fresh evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence


DE_NOVO_COUNTS = {2: 100, 3: 100, 4: 100, 5: 100, 6: 20, 7: 20}
EDIT_COUNTS = {
    "DRD2:decrease+MW:decrease+SA:decrease": 500,
    "GSK3B:increase": 500,
    "HBA:decrease+LogP:increase": 500,
    "HBA:decrease+MW:decrease": 500,
    "HBA:decrease+SA:decrease": 500,
    "HBA:increase+MW:increase+QED:decrease": 500,
    "MW:increase": 500,
    "QED:increase+SA:decrease": 500,
    "RB:decrease": 500,
    "SA:decrease": 500,
}
FORBIDDEN_KEYS = {
    "target",
    "target_smiles",
    "target_canonical_smiles",
    "reference_smiles",
    "oracle_output",
}


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(row)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def user_payload(row: Mapping[str, object]) -> dict[str, object]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError("each frozen request must contain system and user messages")
    roles = [str(message.get("role", "")) for message in messages]
    if roles != ["system", "user"]:
        raise ValueError(f"unexpected message roles: {roles}")
    payload = json.loads(str(messages[1].get("content", "")))
    if not isinstance(payload, dict):
        raise ValueError("user message must serialize one request object")
    return payload


def assert_target_blind(row: Mapping[str, object]) -> None:
    present = FORBIDDEN_KEYS & set(map(str, row))
    if present:
        raise ValueError(f"target-bearing top-level keys: {sorted(present)}")
    for message in list(row.get("messages", [])):
        if str(message.get("role", "")) == "assistant":
            raise ValueError("frozen request contains an assistant completion")
    payload = user_payload(row)
    leaked = FORBIDDEN_KEYS & set(map(str, payload))
    if leaked:
        raise ValueError(f"target-bearing prompt keys: {sorted(leaked)}")


def validate_denovo(rows: Sequence[Mapping[str, object]]) -> dict[int, int]:
    counts: Counter[int] = Counter()
    identifiers: set[str] = set()
    for row in rows:
        assert_target_blind(row)
        identifier = str(row.get("condition_id", ""))
        if not identifier or identifier in identifiers:
            raise ValueError(f"missing or duplicate de novo condition_id: {identifier}")
        identifiers.add(identifier)
        if str(row.get("task_mode", "")) != "de_novo":
            raise ValueError(f"non-de-novo row in construction gate: {identifier}")
        payload = user_payload(row)
        if payload.get("source") != "<EMPTY>":
            raise ValueError(f"de novo source is not <EMPTY>: {identifier}")
        conditions = payload.get("conditions")
        if not isinstance(conditions, list):
            raise ValueError(f"missing conditions: {identifier}")
        counts[len(conditions)] += 1
    if dict(counts) != DE_NOVO_COUNTS:
        raise ValueError(f"de novo arity counts {dict(counts)} != {DE_NOVO_COUNTS}")
    return dict(sorted(counts.items()))


def validate_edit(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    identifiers: set[str] = set()
    for row in rows:
        assert_target_blind(row)
        identifier = str(row.get("condition_id", ""))
        if not identifier or identifier in identifiers:
            raise ValueError(f"missing or duplicate editing condition_id: {identifier}")
        identifiers.add(identifier)
        if str(row.get("task_mode", "")) != "edit":
            raise ValueError(f"non-edit row in editing gate: {identifier}")
        payload = user_payload(row)
        source = str(payload.get("source", ""))
        if not source or source == "<EMPTY>":
            raise ValueError(f"editing source is empty: {identifier}")
        if source != str(row.get("source_smiles", "")):
            raise ValueError(f"source mismatch: {identifier}")
        counts[str(row.get("task_key", ""))] += 1
    if dict(counts) != EDIT_COUNTS:
        raise ValueError("editing gate does not contain exactly 500 requests per task")
    return dict(sorted(counts.items()))


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def freeze(
    denovo_paths: Sequence[Path], editing_path: Path, output_dir: Path
) -> dict[str, object]:
    de_novo = [row for path in denovo_paths for row in read_jsonl(path)]
    editing = read_jsonl(editing_path)
    de_counts = validate_denovo(de_novo)
    edit_counts = validate_edit(editing)
    if {str(row["condition_id"]) for row in de_novo} & {
        str(row["condition_id"]) for row in editing
    }:
        raise ValueError("condition IDs overlap across modes")

    output_dir.mkdir(parents=True, exist_ok=True)
    de_path = output_dir / "gate.denovo.jsonl"
    edit_path = output_dir / "gate.edit.jsonl"
    write_jsonl(de_path, de_novo)
    write_jsonl(edit_path, editing)
    manifest = {
        "protocol": "fresh_balanced_headline_gates_v1",
        "target_blind": True,
        "assistant_completions_present": False,
        "de_novo": {
            "rows": len(de_novo),
            "arity_counts": {str(k): v for k, v in de_counts.items()},
            "sha256": sha256(de_path),
        },
        "editing": {
            "rows": len(editing),
            "task_counts": edit_counts,
            "sha256": sha256(edit_path),
        },
        "source_files": [
            {"label": f"de_novo_part_{index + 1}", "sha256": sha256(path)}
            for index, path in enumerate(denovo_paths)
        ]
        + [{"label": "editing_all10", "sha256": sha256(editing_path)}],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--denovo", action="append", required=True, type=Path)
    parser.add_argument("--editing", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if len(args.denovo) != 3:
        raise SystemExit("pass exactly three --denovo prompt files")
    manifest = freeze(args.denovo, args.editing, args.output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
