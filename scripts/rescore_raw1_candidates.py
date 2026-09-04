#!/usr/bin/env python3
"""Rescore existing sampled-once candidates without running generation again."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import evaluate_raw1 as evaluator
from molprogram.scoring import property_count, score_response


def candidate_id(row) -> str:
    return str(row.get("condition_id") or row.get("sample_id") or "")


def main(argv: Sequence[str] | None = None) -> int:
    from rdkit import RDLogger

    for channel in ("rdApp.error", "rdApp.warning", "rdApp.info", "rdApp.debug"):
        RDLogger.DisableLog(channel)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--denovo-gate", required=True, type=Path)
    parser.add_argument("--edit-gate", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--routing-config", type=Path)
    args = parser.parse_args(argv)

    de_novo = evaluator.read_jsonl(args.denovo_gate)
    editing = [
        row
        for row in evaluator.read_jsonl(args.edit_gate)
        if row["task_mode"] == "edit"
    ]
    evaluator.require_pinned_assay_oracles(editing)
    references = {candidate_id(row): row for row in [*de_novo, *editing]}
    candidates = evaluator.read_jsonl(args.candidates)
    if len(references) != len(de_novo) + len(editing):
        raise ValueError("gate condition IDs are empty or duplicated")
    if {candidate_id(row) for row in candidates} != set(references):
        raise ValueError("candidate condition IDs do not exactly match the frozen gates")

    rescored = []
    for candidate in candidates:
        reference = references[candidate_id(candidate)]
        reward, details = score_response(reference, str(candidate.get("raw", "")))
        rescored.append(
            {
                "condition_id": candidate_id(candidate),
                "task_mode": reference["task_mode"],
                "property_count": property_count(reference),
                "task_key": reference.get("task_key", ""),
                "raw": candidate.get("raw", ""),
                "reward": reward,
                **details,
            }
        )

    summary = evaluator.summarize_records(
        rescored,
        arm=args.arm,
        protocol_name=args.protocol,
        seed=args.seed,
        routing_config=args.routing_config,
    )
    summary["rescore_source"] = str(args.candidates.resolve())
    summary["regenerated"] = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "candidates.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rescored)
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "EVAL_COMPLETE").touch()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
