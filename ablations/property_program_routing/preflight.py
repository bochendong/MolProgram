#!/usr/bin/env python3
"""Validate the frozen 10k baseline inputs before routed-LoRA training."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SOURCE_DIR = REPO_ROOT / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))
from molprogram import program_routing  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-data-dir", required=True, type=Path)
    parser.add_argument("--routing-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--protocol", default="property_program_routed_lora_10k_v1")
    args = parser.parse_args(argv)

    layout = program_routing.load_layout(args.routing_config)
    paths = {
        "train": args.baseline_data_dir / "train.joint.jsonl",
        "denovo_gate": args.baseline_data_dir / "gate.denovo.jsonl",
        "edit_gate": args.baseline_data_dir / "gate.edit.jsonl",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    train = read_jsonl(paths["train"])
    de_gate = read_jsonl(paths["denovo_gate"])
    edit_gate = read_jsonl(paths["edit_gate"])
    counts = Counter(program_routing.task_mode(row) for row in train)
    if counts != {"de_novo": 10000, "edit": 10000}:
        raise ValueError(f"expected 10k rows per mode, found {counts}")
    if len(de_gate) != 120 or len(edit_gate) != 500:
        raise ValueError(
            f"expected 120/500 frozen gate rows, found {len(de_gate)}/{len(edit_gate)}"
        )
    all_rows = train + de_gate + edit_gate
    for row in all_rows:
        program_routing.route_values(row, layout)
    manifest = {
        "protocol": args.protocol,
        "baseline_data_dir": str(args.baseline_data_dir.resolve()),
        "row_counts": {
            "train": len(train),
            "train_by_mode": dict(counts),
            "denovo_gate": len(de_gate),
            "edit_gate": len(edit_gate),
        },
        "sha256": {name: sha256(path) for name, path in paths.items()},
        "routing_config_sha256": sha256(args.routing_config),
        "all_properties_routable": True,
        "matched_baseline_files_reused_without_resampling": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "PREFLIGHT_COMPLETE").touch()
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
