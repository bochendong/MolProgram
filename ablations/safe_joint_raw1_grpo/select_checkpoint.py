#!/usr/bin/env python3
"""Select one checkpoint per arm using only the frozen development gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = (
    "denovo_strict_macro",
    "denovo_valid_macro",
    "edit_strict_065_macro",
    "edit_relaxed_015_macro",
    "edit_valid_macro",
)


def comparison(baseline: dict[str, object], candidate: dict[str, object]):
    left = baseline["aggregate"]
    right = candidate["aggregate"]
    deltas = {metric: float(right[metric]) - float(left[metric]) for metric in METRICS}
    safety = {
        "edit_strict_noninferior_1pp": deltas["edit_strict_065_macro"] >= -0.01,
        "edit_relaxed_noninferior_1pp": deltas["edit_relaxed_015_macro"] >= -0.01,
        "denovo_valid_noninferior_1pp": deltas["denovo_valid_macro"] >= -0.01,
        "edit_valid_noninferior_1pp": deltas["edit_valid_macro"] >= -0.01,
    }
    return deltas, safety


def selection_score(record: dict[str, object]):
    deltas = record["deltas"]
    safe = all(record["safety"].values())
    return (
        int(safe),
        float(deltas["denovo_strict_macro"]),
        min(
            float(deltas["denovo_strict_macro"]),
            float(deltas["edit_strict_065_macro"]),
        ),
        -int(record["step"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--baseline-summary", required=True, type=Path)
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", nargs="+", type=int, default=[10, 20, 30])
    args = parser.parse_args()

    baseline = json.loads(args.baseline_summary.read_text())
    candidates = []
    for step in args.steps:
        summary_path = args.evaluation_root / f"step{step:03d}" / "summary.json"
        summary = json.loads(summary_path.read_text())
        deltas, safety = comparison(baseline, summary)
        candidates.append(
            {
                "step": step,
                "summary": str(summary_path),
                "deltas": deltas,
                "safety": safety,
            }
        )
    selected = max(candidates, key=selection_score)
    step = int(selected["step"])
    adapter = args.model_root / f"checkpoint-{step:03d}" / "adapter"
    if not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(adapter / "adapter_model.safetensors")
    result = {
        "protocol": "molprogram_safe_raw1_dev_selection_v1",
        "arm": args.arm,
        "selection_uses_final_gate": False,
        "selected_step": step,
        "selected_adapter": str(adapter),
        "selected_is_safety_eligible": all(selected["safety"].values()),
        "selection_score": selection_score(selected),
        "selected_comparison": selected,
        "all_candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
