#!/usr/bin/env python3
"""Split preregistered combined gates into immutable mode-specific inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--dev-jsonl", required=True, type=Path)
    parser.add_argument("--final-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    counts: dict[str, int] = {}
    for split, source in (("dev", args.dev_jsonl), ("final", args.final_jsonl)):
        rows = read_jsonl(source)
        for mode, suffix in (("de_novo", "denovo"), ("edit", "edit")):
            selected = [row for row in rows if row.get("task_mode") == mode]
            if not selected:
                raise ValueError(f"{source} has no {mode} rows")
            output = args.output_dir / f"{split}.{suffix}.jsonl"
            write_jsonl(output, selected)
            outputs[f"{split}_{suffix}"] = output
            counts[f"{split}_{suffix}"] = len(selected)

    manifest = {
        "protocol": "molprogram_safe_joint_raw1_frozen_inputs_v1",
        "train_jsonl": str(args.train_jsonl.resolve()),
        "source_gates": {
            "dev": str(args.dev_jsonl.resolve()),
            "final": str(args.final_jsonl.resolve()),
        },
        "source_sha256": {
            "train": sha256(args.train_jsonl),
            "dev": sha256(args.dev_jsonl),
            "final": sha256(args.final_jsonl),
        },
        "outputs": {name: str(path.resolve()) for name, path in outputs.items()},
        "output_sha256": {name: sha256(path) for name, path in outputs.items()},
        "row_counts": counts,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "INPUTS_FROZEN").touch()
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
