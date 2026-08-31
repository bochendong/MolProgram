#!/usr/bin/env python3
"""Build an edit-matched joint set with de-novo replay from shared properties."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence


SHARED_PROPERTIES = frozenset({"MW", "LogP", "QED", "HBA", "RB"})
REPLAY_ARITIES = (2, 3)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def identity(row: Mapping[str, object]) -> str:
    return str(row.get("example_id", row.get("condition_id", row.get("sample_id", ""))))


def stable_score(row: Mapping[str, object], seed: int) -> int:
    return int(hashlib.sha256(f"{seed}:{identity(row)}".encode()).hexdigest(), 16)


def condition_program(row: Mapping[str, object]) -> list[dict[str, object]]:
    program = row.get("condition_program")
    if isinstance(program, list):
        return [item for item in program if isinstance(item, dict)]
    for message in list(row.get("messages", [])):
        if str(message.get("role")) != "user":
            continue
        payload = json.loads(str(message.get("content", "{}")))
        conditions = payload.get("conditions", []) if isinstance(payload, dict) else []
        return [item for item in conditions if isinstance(item, dict)]
    return []


def shared_replay_bucket(row: Mapping[str, object]) -> int | None:
    if str(row.get("task_mode", "")) != "de_novo":
        return None
    program = condition_program(row)
    properties = {str(item.get("property", "")) for item in program}
    if len(program) in REPLAY_ARITIES and properties and properties <= SHARED_PROPERTIES:
        return len(program)
    return None


def quotas(total: int) -> dict[int, int]:
    base, remainder = divmod(total, len(REPLAY_ARITIES))
    return {
        arity: base + int(index < remainder)
        for index, arity in enumerate(REPLAY_ARITIES)
    }


def select_shared_replay(
    paths: Sequence[Path], seed: int, total: int
) -> tuple[list[dict[str, object]], dict[int, int]]:
    requested = quotas(total)
    heaps: dict[int, list[tuple[int, int, dict[str, object]]]] = {
        arity: [] for arity in REPLAY_ARITIES
    }
    counter = 0
    for path in paths:
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                arity = shared_replay_bucket(row)
                if arity is None:
                    continue
                counter += 1
                item = (-stable_score(row, seed), counter, row)
                heap = heaps[arity]
                if len(heap) < requested[arity]:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
    selected: list[dict[str, object]] = []
    for arity in REPLAY_ARITIES:
        if len(heaps[arity]) != requested[arity]:
            raise ValueError(
                f"shared replay {arity}p has {len(heaps[arity])} rows; "
                f"needs {requested[arity]}"
            )
        selected.extend(item[2] for item in sorted(heaps[arity], reverse=True))
    return selected, requested


def interleave(
    editing: Sequence[dict[str, object]], replay: Sequence[dict[str, object]]
) -> list[dict[str, object]]:
    if len(editing) != len(replay):
        raise ValueError(
            f"matched joint training requires equal rows, found edit={len(editing)} "
            f"and replay={len(replay)}"
        )
    rows: list[dict[str, object]] = []
    for edit_row, replay_row in zip(editing, replay):
        rows.extend((edit_row, replay_row))
    return rows


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-data-dir", required=True, type=Path)
    parser.add_argument("--train-source", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--replay-total", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=33101)
    parser.add_argument("--protocol", default="shared_property_transfer_v1")
    args = parser.parse_args(argv)

    sources: list[Path] = []
    for source in args.train_source:
        sources.extend(sorted(source.glob("*.jsonl")) if source.is_dir() else [source])
    if not sources:
        raise ValueError("no de-novo training shards found")

    edit_source = args.baseline_data_dir / "train.edit.jsonl"
    editing = read_jsonl(edit_source)
    if len(editing) != args.replay_total:
        raise ValueError(
            f"baseline edit subset has {len(editing)} rows; expected {args.replay_total}"
        )
    if any(str(row.get("task_mode", "")) != "edit" for row in editing):
        raise ValueError("baseline edit subset is not task-pure")

    replay, requested = select_shared_replay(sources, args.seed, args.replay_total)
    joint = interleave(editing, replay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train.shared_property_joint.jsonl"
    write_jsonl(train_path, joint)
    for name in ("gate.denovo.jsonl", "gate.edit.jsonl"):
        shutil.copyfile(args.baseline_data_dir / name, args.output_dir / name)

    property_histogram = Counter(
        property_name
        for row in replay
        for property_name in sorted(
            str(item.get("property", "")) for item in condition_program(row)
        )
    )
    summary = {
        "protocol": args.protocol,
        "seed": args.seed,
        "selection": "deterministic smallest hash within 2p and 3p",
        "shared_properties": sorted(SHARED_PROPERTIES),
        "replay_arity_quotas": {f"{key}p": value for key, value in requested.items()},
        "replay_property_histogram": dict(sorted(property_histogram.items())),
        "rows": {"editing": len(editing), "de_novo_replay": len(replay), "joint": len(joint)},
        "baseline_edit_sha256": digest(edit_source),
        "output_edit_rows_sha256": hashlib.sha256(
            "".join(
                json.dumps(row, sort_keys=True) + "\n" for row in joint[::2]
            ).encode()
        ).hexdigest(),
        "gate_sha256": {
            name: digest(args.output_dir / name)
            for name in ("gate.denovo.jsonl", "gate.edit.jsonl")
        },
        "matched_edit_subset": True,
        "shared_property_replay_only": True,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "DATA_COMPLETE").touch()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
