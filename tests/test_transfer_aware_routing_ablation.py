from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ABLATION = ROOT / "ablations" / "transfer_aware_program_routing"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


prepare = load_module("transfer_aware_prepare", ABLATION / "prepare.py")
collect = load_module("transfer_aware_collect", ABLATION / "collect.py")
probe = load_module("transfer_aware_probe", ABLATION / "probe_gradients.py")


def conditions(task: str):
    return [
        {"property": clause.split(":", 1)[0], "goal": clause.split(":", 1)[1]}
        for clause in task.split("+")
    ]


def training_rows():
    props = ("MW", "LogP", "QED", "HBA", "RB", "TPSA", "HBD")
    rows = []
    for arity in range(2, 8):
        for index in range(5):
            program = [
                {"property": props[offset], "goal": "increase"}
                for offset in range(arity)
            ]
            rows.append(
                {
                    "example_id": f"d-{arity}-{index}",
                    "task_mode": "de_novo",
                    "condition_program": program,
                    "task_key": "+".join(
                        f"{item['property']}:{item['goal']}" for item in program
                    ),
                }
            )
    for task in prepare.EDIT_TASKS:
        for index in range(3):
            rows.append(
                {
                    "example_id": f"e-{task}-{index}",
                    "task_mode": "edit",
                    "condition_program": conditions(task),
                    "task_key": task,
                }
            )
    return rows


def evaluation(shared: float, private: float, valid: float = 0.9):
    buckets = {}
    for task in collect.hard.SHARED_TASKS:
        buckets[task] = {
            "strict_rate": shared,
            "valid_rate": valid,
            "property_strict_rate": shared,
            "mean_source_similarity": 0.75,
        }
    for task in collect.hard.EDIT_ONLY_TASKS:
        buckets[task] = {
            "strict_rate": private,
            "valid_rate": valid,
            "property_strict_rate": private,
            "mean_source_similarity": 0.75,
        }
    return {
        "aggregate": {
            "denovo_strict_macro": 0.20,
            "denovo_valid_macro": valid,
            "edit_strict_065_macro": (shared + private) / 2,
            "edit_valid_macro": valid,
        },
        "edit_buckets": buckets,
    }


def test_prepare_freezes_equal_task_covered_modes():
    joint, quotas = prepare.select_rows(
        training_rows(), seed=7, per_denovo_arity=5, per_edit_task=3
    )
    assert len(joint) == 60
    assert sum(row["task_mode"] == "de_novo" for row in joint) == 30
    assert sum(row["task_mode"] == "edit" for row in joint) == 30
    assert set(quotas) == {
        *(f"de_novo:{arity}p" for arity in range(2, 8)),
        *(f"edit:{task}" for task in prepare.EDIT_TASKS),
    }
    nodes = prepare.node_histogram(joint)
    assert nodes["edit:SA"] > 0
    assert nodes["edit:GSK3B"] > 0
    assert nodes["edit:DRD2"] > 0


def test_collector_requires_private_gain_and_guardrails():
    routed = evaluation(shared=0.25, private=0.24)
    dense = evaluation(shared=0.26, private=0.20)
    result = collect.summarize(
        routed,
        dense,
        {"trainable_parameters": 10, "extra_trainable_routing_parameters": 0},
        {"trainable_parameters": 10, "extra_trainable_routing_parameters": 0},
        {"nodes": ["edit:SA"], "eigenvalues": [1.0]},
    )
    assert result["decision"] == "supported"
    assert result["deltas_transfer_aware_minus_dense"][
        "edit_only5_strict_065_macro"
    ] == pytest.approx(0.04)


def test_probe_keeps_disconnected_lora_coordinates_as_zero():
    torch = pytest.importorskip("torch")

    class Layer:
        def __init__(self):
            self.lora_A = {"default": torch.nn.Linear(3, 2, bias=False)}
            self.lora_B = {"default": torch.nn.Linear(2, 4, bias=False)}

    active = Layer()
    inactive = Layer()
    with torch.no_grad():
        active.lora_B["default"].weight.fill_(2.0)
    active.lora_B["default"].weight.grad = torch.ones_like(
        active.lora_B["default"].weight
    )

    class Model:
        def named_modules(self):
            return [("active", active), ("inactive", inactive)]

    labels, values = probe.rank_scaling_signature(Model())
    assert labels == [
        "active:default:r0",
        "active:default:r1",
        "inactive:default:r0",
        "inactive:default:r1",
    ]
    assert values == pytest.approx([8.0, 8.0, 0.0, 0.0])
