from __future__ import annotations

import importlib.util
import json
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


def test_raw1_protocol_is_single_sample_target_blind_and_unranked():
    payload = json.loads((BASELINE / "raw1_protocol.json").read_text())
    generation = payload["generation"]
    assert payload["benchmark"]["frozen_conditions"] == 1992
    assert len(payload["benchmark"]["tasks"]) == 10
    assert generation["candidate_budget"] == 1
    assert generation["raw_at_1"] is True
    assert generation["generation_batch_size"] == 1
    assert generation["target_access"] is False
    assert generation["property_reranking"] is False
    assert generation["validity_repair"] is False


def test_raw1_pipeline_restores_instruction_index_and_orders_dependencies():
    generator = (BASELINE / "generate_raw1.py").read_text()
    submit = (BASELINE / "submit_raw1.sh").read_text()
    collector = (BASELINE / "collect_raw1.py").read_text()
    assert 'source_row["instr_idx"]' in generator
    assert 'num_return_sequences=1' in generator
    assert '"property_reranking": False' in generator
    assert '"target_access": False' in generator
    assert 'afterok:$preflight' in submit
    assert 'afterok:$generate' in submit
    assert 'afterok:$score' in submit
    assert 'candidate_rows' in collector and 'input_groups' in collector
