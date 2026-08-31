from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "baselines" / "mumo_fresh"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


protocol = load_module("mumo_protocol", BASELINE / "protocol.py")
trainer = load_module("mumo_trainer", BASELINE / "train.py")


def test_direction_and_serialization_contract():
    text = protocol.user_text("CCO", "qed+mutagenicity", 5)
    assert "increase QED" in text
    assert "decrease Mutagenicity" in text
    response = protocol.messages("CCO", "qed", 0, "CCN")[-1]["content"]
    assert protocol.extract_smiles(response) == "CCN"


def test_stability_contract_excludes_output_head():
    assert "lm_head" not in trainer.LORA_TARGETS
    assert trainer.LORA_TARGETS == (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    assert trainer.DEFAULT_BATCH_SIZE == 1
    assert trainer.DEFAULT_GRADIENT_ACCUMULATION == 128
    assert trainer.DEFAULT_BATCH_SIZE * trainer.DEFAULT_GRADIENT_ACCUMULATION == 128
