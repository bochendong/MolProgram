from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
from molprogram import program_routing


ABLATION = ROOT / "ablations" / "property_conditional_residual"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


prepare = load_module("conditional_residual_prepare", ABLATION / "prepare.py")
train = load_module("conditional_residual_train", ABLATION / "train_residual.py")
collect = load_module("conditional_residual_collect", ABLATION / "collect.py")
conditional = program_routing.load_layout(ABLATION / "conditional_layout.json")
always_on = program_routing.load_layout(ABLATION / "always_on_layout.json")


def row(mode: str, task: str, identifier: str = "x"):
    conditions = [
        {"property": clause.split(":", 1)[0], "goal": clause.split(":", 1)[1]}
        for clause in task.split("+")
    ]
    return {
        "condition_id": identifier,
        "task_mode": mode,
        "task_key": task,
        "condition_program": conditions,
    }


def endpoint(shared: float, private: float, valid: float = 0.90):
    buckets = {}
    for task_name in collect.hard.SHARED_TASKS:
        buckets[task_name] = {
            "strict_rate": shared,
            "valid_rate": valid,
            "property_strict_rate": shared,
            "mean_source_similarity": 0.7,
        }
    for task_name in collect.hard.EDIT_ONLY_TASKS:
        buckets[task_name] = {
            "strict_rate": private,
            "valid_rate": valid,
            "property_strict_rate": private,
            "mean_source_similarity": 0.7,
        }
    return {
        "aggregate": {
            "denovo_strict_macro": 0.10,
            "denovo_valid_macro": valid,
            "edit_strict_065_macro": (shared + private) / 2,
            "edit_valid_macro": valid,
        },
        "edit_buckets": buckets,
    }


def test_conditional_layout_activates_residual_only_for_edit_only_properties():
    de_novo = program_routing.route_values(row("de_novo", "MW:increase"), conditional)
    shared = program_routing.route_values(row("edit", "MW:increase"), conditional)
    private = program_routing.route_values(row("edit", "SA:decrease"), conditional)
    mixed = program_routing.route_values(
        row("edit", "HBA:decrease+SA:decrease"), conditional
    )
    assert de_novo == [1.0] * 16 + [0.0] * 4
    assert shared == [1.0] * 16 + [0.0] * 4
    assert private == [1.0] * 20
    assert mixed == [1.0] * 20
    assert program_routing.route_values(
        row("de_novo", "MW:increase"), always_on
    ) == [1.0] * 20


def test_prepare_selects_only_the_five_edit_only_tasks():
    rows = []
    for task_name in sorted(prepare.EDIT_ONLY_TASKS):
        rows.extend(row("edit", task_name, f"{task_name}-{i}") for i in range(2))
    rows.append(row("edit", "MW:increase", "shared"))
    rows.append(row("de_novo", "MW:increase+QED:increase", "de-novo"))
    selected, counts = prepare.select_rows(rows, expected_rows=10)
    assert len(selected) == 10
    assert set(counts) == prepare.EDIT_ONLY_TASKS
    assert set(counts.values()) == {2}


def test_shared_slice_copy_and_gradient_freeze():
    torch = pytest.importorskip("torch")
    source_a = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    destination_a = torch.full((5, 4), -1.0)
    train.copy_shared_slice(destination_a, source_a, kind="A", shared_rank=3)
    assert torch.equal(destination_a[:3], source_a)
    assert torch.equal(destination_a[3:], torch.full((2, 4), -1.0))
    frozen_a = train.freeze_shared_gradient(
        torch.ones_like(destination_a), kind="A", shared_rank=3
    )
    assert torch.count_nonzero(frozen_a[:3]) == 0
    assert torch.all(frozen_a[3:] == 1)

    source_b = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    destination_b = torch.full((4, 5), -1.0)
    train.copy_shared_slice(destination_b, source_b, kind="B", shared_rank=3)
    assert torch.equal(destination_b[:, :3], source_b)
    frozen_b = train.freeze_shared_gradient(
        torch.ones_like(destination_b), kind="B", shared_rank=3
    )
    assert torch.count_nonzero(frozen_b[:, :3]) == 0
    assert torch.all(frozen_b[:, 3:] == 1)


def test_saved_peft_key_mapping_removes_the_runtime_adapter_name():
    runtime = (
        "base_model.model.model.layers.0.self_attn.q_proj."
        "lora_A.default.weight"
    )
    saved = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": object()
    }
    key, tensor = train.source_tensor_for(runtime, saved)
    assert key in saved
    assert tensor is saved[key]


def test_collector_requires_gain_and_exact_structural_guardrails():
    baseline = endpoint(shared=0.30, private=0.20)
    candidate = endpoint(shared=0.30, private=0.24)
    always = endpoint(shared=0.25, private=0.24)
    inactive_identity = {"identical": True, "checked": 370, "mismatch_count": 0}
    active_identity = {"identical": True, "checked": 250, "mismatch_count": 0}
    result = collect.summarize(
        baseline,
        candidate,
        always,
        {
            "shared_slice_max_abs_delta": 0.0,
            "trainable_tensor_parameters": 125,
            "effectively_updated_residual_parameters": 25,
            "shared_rank": 16,
            "residual_rank": 4,
        },
        inactive_identity,
        active_identity,
    )
    assert result["decision"] == "supported"
    assert result["deltas_conditional_minus_frozen"][
        "edit_only5_strict_065_macro"
    ] == pytest.approx(0.04)


def test_raw_identity_checks_the_requested_subsets():
    base = [
        {**row("de_novo", "MW:increase", "d"), "raw": "D"},
        {**row("edit", "MW:increase", "s"), "raw": "S"},
        {**row("edit", "SA:decrease", "p"), "raw": "P"},
    ]
    changed_private = [dict(item) for item in base]
    changed_private[2]["raw"] = "PRIVATE_AFTER"
    assert collect.raw_identity(base, changed_private, subset="inactive")["identical"]
    assert not collect.raw_identity(base, changed_private, subset="active")["identical"]
