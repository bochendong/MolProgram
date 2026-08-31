from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ABLATION = ROOT / "ablations" / "shared_property_transfer"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


prepare = load_module("shared_transfer_prepare", ABLATION / "prepare_data.py")
collect = load_module("shared_transfer_collect", ABLATION / "collect.py")


def row(mode: str, identity: str, properties: tuple[str, ...]):
    return {
        "example_id": identity,
        "task_mode": mode,
        "condition_program": [
            {"property": prop, "goal": {"around": 1.0}} for prop in properties
        ],
    }


def test_filter_accepts_only_shared_two_and_three_property_replay():
    assert prepare.shared_replay_bucket(row("de_novo", "a", ("MW", "HBA"))) == 2
    assert prepare.shared_replay_bucket(row("de_novo", "b", ("MW", "HBA", "QED"))) == 3
    assert prepare.shared_replay_bucket(row("de_novo", "c", ("MW",))) is None
    assert prepare.shared_replay_bucket(row("de_novo", "d", ("MW", "SA"))) is None
    assert prepare.shared_replay_bucket(row("edit", "e", ("MW", "HBA"))) is None


def test_interleave_preserves_all_edit_rows_and_equal_exposure():
    editing = [row("edit", f"e-{index}", ("MW",)) for index in range(3)]
    replay = [row("de_novo", f"d-{index}", ("MW", "HBA")) for index in range(3)]
    joint = prepare.interleave(editing, replay)
    assert [item["example_id"] for item in joint[::2]] == ["e-0", "e-1", "e-2"]
    assert [item["example_id"] for item in joint[1::2]] == ["d-0", "d-1", "d-2"]


def evaluation(strict: float, valid: float, similarity: float):
    buckets = {
        task: {
            "strict_rate": strict,
            "relaxed_rate": strict + 0.1,
            "valid_rate": valid,
            "property_strict_rate": strict + 0.2,
            "mean_source_similarity": similarity,
        }
        for task in collect.SHARED_TASKS
    }
    return {"edit_buckets": buckets, "aggregate": {"edit_valid_macro": valid}}


def test_collector_requires_strict_gain_and_validity_guardrail():
    result = collect.summarize(
        evaluation(0.25, 0.91, 0.61),
        evaluation(0.22, 0.92, 0.60),
        evaluation(0.21, 0.92, 0.62),
    )
    assert result["decision"] == "positive_transfer"
    assert result["deltas"]["strict_vs_edit_specialist"] == pytest.approx(0.04)
