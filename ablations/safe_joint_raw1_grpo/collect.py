#!/usr/bin/env python3
"""Collect the held-out safe-joint GRPO result with paired uncertainty."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence


METRICS = (
    "denovo_strict_macro",
    "denovo_valid_macro",
    "edit_strict_065_macro",
    "edit_relaxed_015_macro",
    "edit_valid_macro",
)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def identity(row: Mapping[str, object]) -> tuple[str, str]:
    return str(row["task_mode"]), str(row["condition_id"])


def bucket(row: Mapping[str, object]) -> str:
    if str(row["task_mode"]) == "de_novo":
        return f"de_novo:{int(row['property_count'])}p"
    return f"edit:{row['task_key']}"


def paired_rows(left: Sequence[dict[str, object]], right: Sequence[dict[str, object]]):
    left_by_id = {identity(row): row for row in left}
    right_by_id = {identity(row): row for row in right}
    if set(left_by_id) != set(right_by_id):
        missing = sorted(set(left_by_id) ^ set(right_by_id))[:5]
        raise ValueError(f"evaluation conditions do not match: {missing}")
    return [(left_by_id[key], right_by_id[key]) for key in sorted(left_by_id)]


def row_metric(row: Mapping[str, object], metric: str) -> float:
    if metric.endswith("strict_macro") or metric == "edit_strict_065_macro":
        return float(bool(row["strict"]))
    if metric == "edit_relaxed_015_macro":
        return float(bool(row["relaxed"]))
    if metric.endswith("valid_macro"):
        return float(bool(row["valid"]))
    raise KeyError(metric)


def macro_delta(pairs, metric: str) -> float:
    grouped = defaultdict(list)
    for left, right in pairs:
        if metric.startswith("denovo") and str(left["task_mode"]) != "de_novo":
            continue
        if metric.startswith("edit") and str(left["task_mode"]) != "edit":
            continue
        grouped[bucket(left)].append(row_metric(right, metric) - row_metric(left, metric))
    if not grouped:
        raise ValueError(f"no rows for {metric}")
    return sum(sum(values) / len(values) for values in grouped.values()) / len(grouped)


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_bootstrap(pairs, *, replicates: int, seed: int):
    by_bucket = defaultdict(list)
    for pair in pairs:
        by_bucket[bucket(pair[0])].append(pair)
    rng = random.Random(seed)
    samples = {metric: [] for metric in METRICS}
    for _ in range(replicates):
        resampled = []
        for values in by_bucket.values():
            resampled.extend(values[rng.randrange(len(values))] for _ in values)
        for metric in METRICS:
            samples[metric].append(macro_delta(resampled, metric))
    return {
        metric: {
            "delta": macro_delta(pairs, metric),
            "ci95": [percentile(samples[metric], 0.025), percentile(samples[metric], 0.975)],
        }
        for metric in METRICS
    }


def compare(left_dir: Path, right_dir: Path, *, replicates: int, seed: int):
    left = read_jsonl(left_dir / "candidates.jsonl")
    right = read_jsonl(right_dir / "candidates.jsonl")
    return paired_bootstrap(
        paired_rows(left, right), replicates=replicates, seed=seed
    )


def deltas_only(result: Mapping[str, Mapping[str, object]]) -> dict[str, float]:
    return {metric: float(values["delta"]) for metric, values in result.items()}


def gate_vs_baseline(deltas: Mapping[str, float]) -> dict[str, bool]:
    return {
        "denovo_strict_gain_ge_2pp": deltas["denovo_strict_macro"] >= 0.02,
        "edit_strict_noninferior_1pp": deltas["edit_strict_065_macro"] >= -0.01,
        "edit_relaxed_noninferior_1pp": deltas["edit_relaxed_015_macro"] >= -0.01,
        "denovo_valid_noninferior_1pp": deltas["denovo_valid_macro"] >= -0.01,
        "edit_valid_noninferior_1pp": deltas["edit_valid_macro"] >= -0.01,
    }


def gate_vs_control(deltas: Mapping[str, float]) -> dict[str, bool]:
    return {
        "denovo_strict_gain_ge_1pp": deltas["denovo_strict_macro"] >= 0.01,
        "edit_strict_noninferior_1pp": deltas["edit_strict_065_macro"] >= -0.01,
        "edit_relaxed_noninferior_1pp": deltas["edit_relaxed_015_macro"] >= -0.01,
        "denovo_valid_noninferior_1pp": deltas["denovo_valid_macro"] >= -0.01,
        "edit_valid_noninferior_1pp": deltas["edit_valid_macro"] >= -0.01,
    }


def promotion_confirmed(*gates: Mapping[str, bool]) -> bool:
    return all(all(gate.values()) for gate in gates)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--control-dir", required=True, type=Path)
    parser.add_argument("--rl-dir", required=True, type=Path)
    parser.add_argument("--rl-selection", required=True, type=Path)
    parser.add_argument("--control-selection", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=37101)
    args = parser.parse_args()

    rl_vs_baseline = compare(
        args.baseline_dir,
        args.rl_dir,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    rl_vs_control = compare(
        args.control_dir,
        args.rl_dir,
        replicates=args.bootstrap_replicates,
        seed=args.seed + 1,
    )
    rl_selection = json.loads(args.rl_selection.read_text())
    control_selection = json.loads(args.control_selection.read_text())
    baseline_gate = gate_vs_baseline(deltas_only(rl_vs_baseline))
    control_gate = gate_vs_control(deltas_only(rl_vs_control))
    development_gate = {
        "rl_checkpoint_safety_eligible": bool(
            rl_selection.get("selected_is_safety_eligible", False)
        )
    }
    confirmed = promotion_confirmed(development_gate, baseline_gate, control_gate)
    result = {
        "protocol": "molprogram_safe_joint_raw1_grpo_result_v1",
        "decision": "PROMOTE_NATIVE_RAW1_TABLES" if confirmed else "STOP_AFTER_GATE",
        "confirmed_rl_specific_improvement": confirmed,
        "property_reranking": False,
        "evaluation_budget": 1,
        "rl_selection": rl_selection,
        "control_selection": control_selection,
        "rl_vs_initial_policy": rl_vs_baseline,
        "rl_vs_continued_sft": rl_vs_control,
        "gates": {
            "development_selection": development_gate,
            "vs_initial_policy": baseline_gate,
            "vs_continued_sft": control_gate,
        },
        "bootstrap": {
            "paired": True,
            "stratified_by_task_bucket": True,
            "replicates": args.bootstrap_replicates,
            "seed": args.seed,
            "used_for_gate": False,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    base = deltas_only(rl_vs_baseline)
    control = deltas_only(rl_vs_control)
    lines = [
        "# Safe joint Raw@1 GRPO result",
        "",
        f"**Decision:** `{result['decision']}`",
        "",
        "| Comparison | De novo strict | Edit strict | Edit relaxed |",
        "| --- | ---: | ---: | ---: |",
        f"| RL - initial fresh SFT | {100*base['denovo_strict_macro']:+.2f} pp | {100*base['edit_strict_065_macro']:+.2f} pp | {100*base['edit_relaxed_015_macro']:+.2f} pp |",
        f"| RL - continued SFT | {100*control['denovo_strict_macro']:+.2f} pp | {100*control['edit_strict_065_macro']:+.2f} pp | {100*control['edit_relaxed_015_macro']:+.2f} pp |",
        "",
        "All evaluations are target-blind Raw@1 with no property reranking.",
        "Training group size is exploration for the gradient only and is not an inference budget.",
        "",
    ]
    (args.output_dir / "RESULT.md").write_text("\n".join(lines))
    (args.output_dir / "RESULT_COMPLETE").touch()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
