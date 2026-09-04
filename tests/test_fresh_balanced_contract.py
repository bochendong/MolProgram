import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "ablations" / "fresh_balanced"


def test_fresh_balanced_protocol_freezes_schedule_and_milestones():
    payload = json.loads((EXPERIMENT / "protocol.json").read_text())
    assert payload["initialization"]["input_adapter"] is None
    assert payload["training"]["effective_batch_size"] == 65
    assert payload["training"]["physical_batch_size"] == 1
    assert payload["training"]["gradient_accumulation"] == 65
    assert payload["training"]["learning_rate"] == 2e-5
    assert payload["training"]["optimizer_steps"] == 16_283
    assert payload["training"]["effective_examples"] == 1_058_395
    assert [row["step"] for row in payload["milestones"]] == [
        1539, 3077, 7693, 16283,
    ]


def test_fresh_balanced_runner_uses_new_lora_and_preserves_adapters():
    runner = (EXPERIMENT / "run_train.sh").read_text()
    assert "--fresh-lora" in runner
    assert "--input-adapter" not in runner
    assert "--sampler-mode balanced" in runner
    assert "--per-device-batch-size 1" in runner
    assert "--gradient-accumulation 65" in runner
    assert "--learning-rate 2e-5" in runner
    assert "--guard-every-microbatch" in runner
    assert "stable_v2_seed_36001" in runner
    for step in (1539, 3077, 7693, 16283):
        assert f"--milestone-step {step}" in runner


def test_indexed_trainer_supports_fresh_lora_and_milestone_adapters():
    trainer = (ROOT / "scripts" / "train_indexed_sft.py").read_text()
    assert "add_mutually_exclusive_group(required=True)" in trainer
    assert 'initialization.add_argument("--fresh-lora"' in trainer
    assert "peft.get_peft_model" in trainer
    assert "class MilestoneAdapterCallback" in trainer
    assert "class FiniteTrainingCallback" in trainer
    assert "non-finite microbatch loss" in trainer
    assert "nonfinite_gradient_count" in trainer
    assert "control.should_save = True" in trainer
    assert '"optimizer_state_preserved": False' in trainer


def test_slurm_chain_orders_smoke_before_full_and_supports_deferred_start():
    submit = (EXPERIMENT / "submit_slurm.sh").read_text()
    assert 'begin_args+=(--begin="$FRESH_BEGIN")' in submit
    assert 'afterok:$preflight' in submit
    assert 'afterok:$smoke' in submit
    assert "checkpoint-16283/adapter" in submit
    assert "stable_v2_seed_36001" in submit
    assert "--time=3-00:00:00" in submit


def test_fresh_evaluation_is_frozen_and_full_only():
    protocol = json.loads((EXPERIMENT / "evaluation_protocol.json").read_text())
    assert protocol["evaluation"]["de_novo_requests"] == 440
    assert protocol["evaluation"]["editing_requests"] == 5000
    assert protocol["evaluation"]["outputs_per_request"] == 1
    assert protocol["headline_rule"]["minimum_de_novo_strict_pooled"] == 53.45
    assert (
        protocol["headline_rule"]["minimum_editing_all10_strict_065_macro"]
        == 56.94
    )
    assert protocol["headline_rule"]["intermediate_checkpoint_selection_forbidden"]
    assert protocol["checkpoints"][-1]["role"] == "only headline-eligible checkpoint"

    runner = (EXPERIMENT / "run_evaluation.sh").read_text()
    assert "gate.denovo.jsonl" in runner
    assert "gate.edit.jsonl" in runner
    assert "SAFE_GRPO_RELEASED" in runner
    submit = (EXPERIMENT / "submit_evaluation_slurm.sh").read_text()
    assert 'afterok:$FRESH_EVAL_TRAIN_JOB' in submit
    assert "FRESH_EVAL_SAFE_GRPO_JOB" in submit
