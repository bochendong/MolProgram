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


ABLATION = ROOT / "ablations" / "property_program_routing"
LAYOUT = program_routing.load_layout(ABLATION / "routing_layout.json")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


collect = load_module("property_routing_collect", ABLATION / "collect.py")


def row(mode: str, props: tuple[str, ...]):
    return {
        "task_mode": mode,
        "condition_program": [
            {"property": prop, "goal": "increase"} for prop in props
        ],
    }


def active(mask):
    return {index for index, value in enumerate(mask) if value != 0.0}


def test_layout_routes_shared_and_private_properties_to_declared_pools():
    shared = program_routing.route_values(row("edit", ("MW", "QED")), LAYOUT)
    denovo_private = program_routing.route_values(
        row("de_novo", ("MW", "TPSA")), LAYOUT
    )
    edit_private = program_routing.route_values(row("edit", ("SA",)), LAYOUT)
    assert active(shared) == set(range(8)) | {8, 9, 10, 11}
    assert active(denovo_private) == set(range(8)) | {8, 9, 12}
    assert active(edit_private) == set(range(8)) | {14}
    assert 14 not in active(denovo_private)
    assert 12 not in active(edit_private)


def test_rms_normalization_keeps_every_mask_at_rank16_energy():
    for request in (
        row("edit", ("SA",)),
        row("de_novo", ("MW", "TPSA")),
        row("edit", ("HBA", "LogP", "SA")),
    ):
        mask = program_routing.route_values(request, LAYOUT)
        assert sum(value * value for value in mask) == pytest.approx(16.0)


def test_lora_a_hook_applies_a_different_mask_to_each_batch_item():
    torch = pytest.importorskip("torch")

    class FakeLoraLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.ModuleDict(
                {"default": torch.nn.Linear(3, 4, bias=False)}
            )

        def forward(self, values):
            return self.lora_A["default"](values)

    model = FakeLoraLayer()
    count = program_routing.install_lora_rank_routing(model, rank=4)
    assert count == 1
    program_routing.set_lora_route_mask(
        model, torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    )
    output = model(torch.ones(2, 5, 3))
    assert torch.count_nonzero(output[0, :, 1::2]) == 0
    assert torch.count_nonzero(output[1, :, 0::2]) == 0


def evaluation(shared: float, private: float, de: float = 0.10, valid: float = 0.90):
    buckets = {}
    for task in collect.SHARED_TASKS:
        buckets[task] = {
            "strict_rate": shared,
            "valid_rate": valid,
            "property_strict_rate": shared + 0.1,
            "mean_source_similarity": 0.75,
        }
    for task in collect.EDIT_ONLY_TASKS:
        buckets[task] = {
            "strict_rate": private,
            "valid_rate": valid,
            "property_strict_rate": private + 0.1,
            "mean_source_similarity": 0.75,
        }
    return {
        "aggregate": {
            "denovo_strict_macro": de,
            "denovo_valid_macro": valid,
            "edit_strict_065_macro": (shared + private) / 2,
            "edit_valid_macro": valid,
        },
        "edit_buckets": buckets,
    }


def test_decision_requires_private_gain_guardrails_and_parameter_parity():
    candidate = evaluation(shared=0.25, private=0.23)
    vanilla = evaluation(shared=0.26, private=0.20)
    specialist = evaluation(shared=0.22, private=0.25)
    result = collect.summarize(
        candidate,
        vanilla,
        specialist,
        specialist,
        {"trainable_parameters": 100, "extra_trainable_routing_parameters": 0},
        {"trainable_parameters": 100},
    )
    assert result["decision"] == "supported"
    assert result["deltas"]["edit_only5_strict_vs_vanilla_joint"] == pytest.approx(0.03)
