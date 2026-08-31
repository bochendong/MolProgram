#!/usr/bin/env python3
"""Build an indexed MuMO training release with a deterministic per-task cap."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--per-task-cap", type=int, default=10_000)
    args = parser.parse_args()

    rows = json.loads(args.input_json.read_text(encoding="utf-8"))
    available = Counter(
        str(row["task"])
        for row in rows
        if str(row.get("split", "")) == "train"
    )
    selected = Counter()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    index_path = args.output_jsonl.with_suffix(".idx")
    with args.output_jsonl.open("wb") as output, index_path.open("wb") as index:
        for row in rows:
            if str(row.get("split", "")) != "train":
                continue
            task = str(row["task"])
            if selected[task] >= args.per_task_cap:
                continue
            compact = {
                "task": task,
                "source_smiles": str(row["source_smiles"]),
                "target_smiles": str(row["target_smiles"]),
                "instr_idx": int(row["instr_idx"]),
            }
            encoded = (json.dumps(compact, separators=(",", ":")) + "\n").encode()
            index.write(struct.pack("<Q", output.tell()))
            output.write(encoded)
            selected[task] += 1

    manifest = {
        "protocol": "mumo_fresh_indexed_cap_v1",
        "input_sha256": sha256(args.input_json),
        "raw_rows": len(rows),
        "per_task_cap": args.per_task_cap,
        "available_task_rows": dict(sorted(available.items())),
        "selected_task_rows": dict(sorted(selected.items())),
        "selected_tasks": len(selected),
        "selected_rows": sum(selected.values()),
        "output_sha256": sha256(args.output_jsonl),
    }
    args.output_jsonl.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
