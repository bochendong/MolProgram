#!/usr/bin/env python3
"""Collect matched negative-refinement Raw@1 deltas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ARMS = ("positive_only", "semantic_only", "semantic_plus_syntax")
METRICS = (
    "denovo_strict_macro",
    "denovo_valid_macro",
    "denovo_property_strict_macro",
    "edit_strict_065_macro",
    "edit_relaxed_015_macro",
    "edit_valid_macro",
    "edit_property_strict_macro",
    "edit_copy_macro",
    "edit_source_similarity_macro",
)


def subtract(left, right):
    return {
        metric: float(left[metric]) - float(right[metric])
        for metric in METRICS
        if left.get(metric) is not None and right.get(metric) is not None
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    summaries = {
        arm: json.loads(
            (args.output_root / "eval" / arm / "summary.json").read_text()
        )["aggregate"]
        for arm in ARMS
    }
    result = {
        "protocol": "molprogram_negative_refinement_matched_v1",
        "arms": summaries,
        "deltas": {
            "semantic_minus_positive": subtract(
                summaries["semantic_only"], summaries["positive_only"]
            ),
            "all_negatives_minus_positive": subtract(
                summaries["semantic_plus_syntax"], summaries["positive_only"]
            ),
            "syntax_increment": subtract(
                summaries["semantic_plus_syntax"], summaries["semantic_only"]
            ),
        },
        "interpretation": {
            "semantic_effect_is_isolated": True,
            "syntax_effect_isolated_by_semantic_plus_syntax_minus_semantic": True,
            "both_modes_reported": True,
        },
    }
    output = args.output_root / "result"
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    (output / "RESULT_COMPLETE").touch()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
