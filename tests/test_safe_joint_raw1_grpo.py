from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("rdkit")

from molprogram import protocol
from molprogram.safe_grpo import (
    CHANNEL_WEIGHTS,
    DE_NOVO_BUCKETS,
    EDIT_BUCKETS,
    balanced_bucket,
    decoupled_advantages,
    equal_norm_bisector,
    reward_channels,
    select_balanced_pairs,
)


ROOT = Path(__file__).resolve().parents[1]
ABLATION = ROOT / "ablations" / "safe_joint_raw1_grpo"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(mode: str, bucket: str, index: int) -> dict[str, object]:
    if mode == "de_novo":
        count = int(bucket.removeprefix("de_novo:").removesuffix("p"))
        conditions = [
            {"property": "MW", "goal": {"around": 100.0 + offset}}
            for offset in range(count)
        ]
        source = "<EMPTY>"
        task_key = ""
        answer = protocol.response("CCO", mode)
    else:
        conditions = [{"property": "MW", "goal": "increase"}]
        source = "CC"
        task_key = bucket.removeprefix("edit:")
        answer = protocol.response("CCC", mode)
    return {
        "example_id": f"{mode}-{bucket}-{index}",
        "condition_id": f"condition-{mode}-{bucket}-{index}",
        "task_mode": mode,
        "task_key": task_key,
        "messages": [
            {"role": "system", "content": protocol.SYSTEM},
            {
                "role": "user",
                "content": json.dumps({"source": source, "conditions": conditions}),
            },
            {"role": "assistant", "content": answer},
        ],
    }


def test_thirty_pairs_cover_all_six_and_ten_buckets_equally():
    rows = []
    for bucket in DE_NOVO_BUCKETS:
        rows.extend(row("de_novo", bucket, index) for index in range(5))
    for bucket in EDIT_BUCKETS:
        rows.extend(row("edit", bucket, index) for index in range(3))
    pairs = select_balanced_pairs(rows, 30, 37001)
    assert len(pairs) == 30
    de_counts = Counter(balanced_bucket(left) for left, _ in pairs)
    edit_counts = Counter(balanced_bucket(right) for _, right in pairs)
    assert set(de_counts) == set(DE_NOVO_BUCKETS)
    assert set(edit_counts) == set(EDIT_BUCKETS)
    assert set(de_counts.values()) == {5}
    assert set(edit_counts.values()) == {3}


def test_candidate_channels_are_target_blind_and_not_group_soft_or():
    request = row("de_novo", "de_novo:2p", 0)
    good, good_details = reward_channels(
        request, protocol.response("CCO", "de_novo")
    )
    bad, bad_details = reward_channels(request, "not-json")
    assert good_details["valid"] is True
    assert bad_details["valid"] is False
    assert good["validity"] == 1.0
    assert bad["validity"] == 0.0
    assert "target_smiles" not in json.dumps(request)
    assert set(good) == set(CHANNEL_WEIGHTS["de_novo"])


def test_channel_normalization_produces_candidate_level_signal():
    rows = [
        {"validity": 0.0, "property_strict": 0.0},
        {"validity": 1.0, "property_strict": 0.0},
        {"validity": 1.0, "property_strict": 1.0},
    ]
    advantages, record = decoupled_advantages(
        rows, {"validity": 0.5, "property_strict": 1.0}
    )
    assert advantages[2] > advantages[1] > advantages[0]
    assert record["zero_signal"] is False


def test_equal_norm_bisector_is_common_descent_for_orthogonal_modes():
    torch = pytest.importorskip("torch")
    merged, record = equal_norm_bisector(
        [torch.tensor([2.0, 0.0])], [torch.tensor([0.0, 5.0])]
    )
    assert record["common_descent"] is True
    assert merged[0][0] == pytest.approx(merged[0][1])


def test_preregistration_locks_fresh_non_aligned_raw1_contract():
    prereg = json.loads((ABLATION / "preregistration.json").read_text())
    training = prereg["training"]
    assert prereg["initialization"]["task_aligned_refresh"] is False
    assert prereg["initialization"]["historical_inherited_adapter"] is False
    assert training["paired_optimizer_steps"] == 30
    assert training["group_size"] == 16
    assert training["soft_or_across_candidates"] is False
    assert training["best_of_k_objective"] is False
    assert prereg["heldout_confirmation"]["evaluation_budget"] == 1
    assert prereg["heldout_confirmation"]["property_reranking"] is False


def test_slurm_chain_keeps_dev_selection_before_final_evaluation():
    submit = (ABLATION / "submit_slurm.sh").read_text()
    select = (ABLATION / "select_checkpoint.py").read_text()
    assert "afterok:$dev_eval" in submit
    assert "afterok:$select" in submit
    assert "--final" not in select
    assert '"selection_uses_final_gate": False' in select


def test_input_validation_rejects_split_overlap():
    validator = load(ABLATION / "validate_inputs.py", "safe_input_validator")
    with pytest.raises(ValueError, match="overlap"):
        validator.assert_disjoint(
            {"train": {"a", "b"}, "dev": {"c"}, "final": {"b"}}
        )


def test_combined_gates_are_split_and_hashed(tmp_path, monkeypatch):
    prepare = load(ABLATION / "prepare_frozen_inputs.py", "safe_input_prepare")
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps(row("de_novo", "de_novo:2p", 99)) + "\n")
    sources = []
    for split, offset in (("dev", 100), ("final", 200)):
        source = tmp_path / f"{split}.jsonl"
        source.write_text(
            json.dumps(row("de_novo", "de_novo:2p", offset))
            + "\n"
            + json.dumps(row("edit", EDIT_BUCKETS[0], offset + 1))
            + "\n"
        )
        sources.append(source)
    output = tmp_path / "frozen"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_frozen_inputs.py",
            "--train-jsonl",
            str(train),
            "--dev-jsonl",
            str(sources[0]),
            "--final-jsonl",
            str(sources[1]),
            "--output-dir",
            str(output),
        ],
    )
    assert prepare.main() == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["row_counts"] == {
        "dev_denovo": 1,
        "dev_edit": 1,
        "final_denovo": 1,
        "final_edit": 1,
    }
    assert (output / "INPUTS_FROZEN").is_file()


def test_promotion_requires_a_safe_development_checkpoint():
    collector = load(ABLATION / "collect.py", "safe_collector_gate")
    passing = {"metric": True}
    assert collector.promotion_confirmed(
        {"rl_checkpoint_safety_eligible": False}, passing, passing
    ) is False


def test_selector_prefers_safe_checkpoint_then_de_novo_gain(tmp_path):
    selector = load(ABLATION / "select_checkpoint.py", "safe_selector")
    baseline = {"aggregate": {metric: 0.5 for metric in selector.METRICS}}
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline))
    eval_root = tmp_path / "eval"
    model_root = tmp_path / "model"
    for step, de_gain, edit_gain in ((10, 0.02, 0.00), (20, 0.05, -0.02), (30, 0.03, 0.00)):
        aggregate = dict(baseline["aggregate"])
        aggregate["denovo_strict_macro"] += de_gain
        aggregate["edit_strict_065_macro"] += edit_gain
        path = eval_root / f"step{step:03d}" / "summary.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"aggregate": aggregate}))
        adapter = model_root / f"checkpoint-{step:03d}" / "adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_model.safetensors").touch()
    records = []
    for step in (10, 20, 30):
        summary = json.loads((eval_root / f"step{step:03d}" / "summary.json").read_text())
        deltas, safety = selector.comparison(baseline, summary)
        records.append({"step": step, "deltas": deltas, "safety": safety})
    assert max(records, key=selector.selection_score)["step"] == 30


def test_collector_uses_paired_condition_identity():
    collector = load(ABLATION / "collect.py", "safe_collector")
    baseline = [
        {
            "condition_id": "a",
            "task_mode": "de_novo",
            "property_count": 2,
            "task_key": "",
            "strict": False,
            "relaxed": False,
            "valid": True,
        }
    ]
    rl = [{**baseline[0], "strict": True}]
    pairs = collector.paired_rows(baseline, rl)
    assert collector.macro_delta(pairs, "denovo_strict_macro") == 1.0
