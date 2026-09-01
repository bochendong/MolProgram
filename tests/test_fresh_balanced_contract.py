import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "ablations" / "fresh_balanced"


def test_fresh_balanced_protocol_freezes_schedule_and_milestones():
    payload = json.loads((EXPERIMENT / "protocol.json").read_text())
    assert payload["initialization"]["input_adapter"] is None
    assert payload["training"]["effective_batch_size"] == 65
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
    assert "--per-device-batch-size 5" in runner
    assert "--gradient-accumulation 13" in runner
    for step in (1539, 3077, 7693, 16283):
        assert f"--milestone-step {step}" in runner


def test_indexed_trainer_supports_fresh_lora_and_milestone_adapters():
    trainer = (ROOT / "scripts" / "train_indexed_sft.py").read_text()
    assert "add_mutually_exclusive_group(required=True)" in trainer
    assert 'initialization.add_argument("--fresh-lora"' in trainer
    assert "peft.get_peft_model" in trainer
    assert "class MilestoneAdapterCallback" in trainer
    assert "control.should_save = True" in trainer
    assert '"optimizer_state_preserved": False' in trainer


def test_slurm_chain_orders_smoke_before_full_and_supports_deferred_start():
    submit = (EXPERIMENT / "submit_slurm.sh").read_text()
    assert 'begin_args+=(--begin="$FRESH_BEGIN")' in submit
    assert 'afterok:$preflight' in submit
    assert 'afterok:$smoke' in submit
    assert "checkpoint-16283/adapter" in submit
