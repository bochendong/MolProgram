from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ABLATION = ROOT / "ablations" / "joint_vs_specialists"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


prepare = load_module("ablation_prepare", ABLATION / "prepare_data.py")
collect = load_module("ablation_collect", ABLATION / "collect.py")
negative = load_module(
    "negative_refinement",
    ROOT / "ablations" / "negative_refinement" / "train_refinement.py",
)


def row(mode: str, identity: str, count: int = 2, task_key: str = ""):
    return {
        "example_id": identity,
        "task_mode": mode,
        "condition_program": [{} for _ in range(count)],
        "task_key": task_key,
        "messages": [
            {"role": "user", "content": "{}"},
            {"role": "assistant", "content": "x"},
        ],
    }


def test_joint_uses_exact_specialist_subsets():
    rows = []
    for count in range(2, 8):
        rows.extend(
            row("de_novo", f"d-{count}-{index}", count=count)
            for index in range(500)
        )
    for task in prepare.EDIT_TASKS:
        rows.extend(
            row("edit", f"e-{task}-{index}", count=1, task_key=task)
            for index in range(300)
        )
    de_novo, editing, joint, quotas = prepare.select_train(rows, 33001)
    assert len(de_novo) == len(editing) == 3000
    assert len(joint) == 6000
    joint_ids = {item["example_id"] for item in joint}
    assert {item["example_id"] for item in de_novo}.issubset(joint_ids)
    assert {item["example_id"] for item in editing}.issubset(joint_ids)
    assert set(quotas.values()) == {300, 500}


def test_collector_reports_transfer_and_parameter_efficiency():
    def evaluation(de=0.50, edit=0.60, de_valid=0.95, edit_valid=0.94):
        return {
            "aggregate": {
                "denovo_strict_macro": de,
                "denovo_valid_macro": de_valid,
                "edit_strict_065_macro": edit,
                "edit_valid_macro": edit_valid,
            }
        }

    evals = {
        "joint": evaluation(de=0.53, edit=0.60),
        "denovo": evaluation(de=0.50),
        "edit": evaluation(edit=0.60),
    }
    trains = {arm: {"trainable_parameters": 100} for arm in evals}
    result = collect.summarize(evals, trains)
    assert result["decision"] == "positive_transfer"
    assert result["efficiency"]["joint_over_two_specialists_parameter_ratio"] == 0.5


def test_negative_refinement_arms_change_only_registered_hinge_weights():
    assert negative.active_weight("positive_only", "source_copy", 0.1) == 0.0
    assert negative.active_weight("positive_only", "invalid_corruption", 0.2) == 0.0
    assert negative.active_weight("semantic_only", "source_copy", 0.1) == 0.1
    assert negative.active_weight("semantic_only", "invalid_corruption", 0.2) == 0.0
    assert (
        negative.active_weight(
            "semantic_plus_syntax", "invalid_corruption", 0.2
        )
        == 0.2
    )
