#!/usr/bin/env python3
"""Freeze a small task-covered train set and reuse the existing Raw@1 gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SOURCE_DIR = REPO_ROOT / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))
from molprogram import program_routing  # noqa: E402


EDIT_TASKS = (
    "DRD2:decrease+MW:decrease+SA:decrease",
    "GSK3B:increase",
    "HBA:decrease+LogP:increase",
    "HBA:decrease+MW:decrease",
    "HBA:decrease+SA:decrease",
    "HBA:increase+MW:increase+QED:decrease",
    "MW:increase",
    "QED:increase+SA:decrease",
    "RB:decrease",
    "SA:decrease",
)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def identity(row: Mapping[str, object]) -> str:
    return str(row.get("example_id", row.get("condition_id", row.get("sample_id", ""))))


def stable_key(row: Mapping[str, object], seed: int) -> str:
    value = identity(row) or json.dumps(row, sort_keys=True)
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def training_bucket(row: Mapping[str, object]) -> str | None:
    mode = program_routing.task_mode(row)
    if mode == "de_novo":
        arity = len(program_routing.condition_program(row))
        return f"de_novo:{arity}p" if 2 <= arity <= 7 else None
    task = str(row.get("task_key", ""))
    return f"edit:{task}" if task in EDIT_TASKS else None


def select_rows(
    rows: Sequence[dict[str, object]],
    *,
    seed: int,
    per_denovo_arity: int,
    per_edit_task: int,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        bucket = training_bucket(row)
        if bucket is not None:
            grouped[bucket].append(row)
    quotas = {
        **{f"de_novo:{arity}p": per_denovo_arity for arity in range(2, 8)},
        **{f"edit:{task}": per_edit_task for task in EDIT_TASKS},
    }
    selected_by_bucket: dict[str, list[dict[str, object]]] = {}
    for bucket, quota in quotas.items():
        candidates = sorted(grouped[bucket], key=lambda row: stable_key(row, seed))
        if len(candidates) < quota:
            raise ValueError(f"bucket {bucket} has {len(candidates)} rows; needs {quota}")
        selected_by_bucket[bucket] = candidates[:quota]
    de_novo = [
        row
        for arity in range(2, 8)
        for row in selected_by_bucket[f"de_novo:{arity}p"]
    ]
    editing = [
        row for task in EDIT_TASKS for row in selected_by_bucket[f"edit:{task}"]
    ]
    if len(de_novo) != len(editing):
        raise ValueError(
            f"matched pilot requires equal modes, found {len(de_novo)}/{len(editing)}"
        )
    joint: list[dict[str, object]] = []
    for de_row, edit_row in zip(de_novo, editing):
        joint.extend((de_row, edit_row))
    return joint, quotas


def node_histogram(rows: Sequence[Mapping[str, object]]) -> Counter[str]:
    return Counter(
        f"{program_routing.task_mode(row)}:{prop}"
        for row in rows
        for prop in program_routing.properties(row)
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-covered-train", required=True, type=Path)
    parser.add_argument("--gate-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--per-denovo-arity", type=int, default=640)
    parser.add_argument("--per-edit-task", type=int, default=384)
    parser.add_argument("--minimum-node-rows", type=int, default=64)
    parser.add_argument("--seed", type=int, default=33401)
    args = parser.parse_args(argv)

    source_rows = read_jsonl(args.task_covered_train)
    joint, quotas = select_rows(
        source_rows,
        seed=args.seed,
        per_denovo_arity=args.per_denovo_arity,
        per_edit_task=args.per_edit_task,
    )
    mode_counts = Counter(program_routing.task_mode(row) for row in joint)
    if mode_counts["de_novo"] != mode_counts["edit"]:
        raise ValueError(f"unbalanced mode counts: {mode_counts}")
    train_nodes = node_histogram(joint)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.joint.jsonl"
    write_jsonl(train_path, joint)
    gate_paths = {
        "denovo": args.gate_root / "data" / "gate.denovo.jsonl",
        "edit": args.gate_root / "data" / "gate.edit.jsonl",
    }
    for mode, source in gate_paths.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copyfile(source, args.output_dir / f"gate.{mode}.jsonl")
    gate_rows = read_jsonl(args.output_dir / "gate.denovo.jsonl") + read_jsonl(
        args.output_dir / "gate.edit.jsonl"
    )
    required_nodes = sorted(node_histogram(gate_rows))
    undercovered = {
        node: train_nodes[node]
        for node in required_nodes
        if train_nodes[node] < args.minimum_node_rows
    }
    if undercovered:
        raise ValueError(f"training coverage is insufficient for gate nodes: {undercovered}")

    manifest = {
        "protocol": "transfer_aware_program_routing_3840_per_mode_pilot_v1",
        "seed": args.seed,
        "source": str(args.task_covered_train.resolve()),
        "source_sha256": sha256(args.task_covered_train),
        "selection": "deterministic smallest identity hash per frozen bucket",
        "quotas": quotas,
        "row_counts": {**dict(mode_counts), "joint": len(joint)},
        "node_histogram": dict(sorted(train_nodes.items())),
        "required_gate_nodes": required_nodes,
        "minimum_node_rows": args.minimum_node_rows,
        "all_gate_nodes_covered": True,
        "sha256": {
            "train": sha256(train_path),
            "denovo_gate": sha256(args.output_dir / "gate.denovo.jsonl"),
            "edit_gate": sha256(args.output_dir / "gate.edit.jsonl"),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "DATA_COMPLETE").touch()
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
