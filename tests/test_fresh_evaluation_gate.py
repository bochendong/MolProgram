from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "ablations" / "fresh_balanced"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


freeze_gate = load_module(
    "freeze_fresh_eval_gates", ROOT / "scripts" / "freeze_fresh_eval_gates.py"
)
collector = load_module(
    "collect_fresh_evaluations", EXPERIMENT / "collect_evaluations.py"
)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def request(
    mode: str, identity: str, conditions: list[dict[str, object]], task: str = ""
) -> dict[str, object]:
    source = "<EMPTY>" if mode == "de_novo" else "CCO"
    return {
        "condition_id": identity,
        "task_mode": mode,
        "source_smiles": "" if mode == "de_novo" else source,
        "task_key": task,
        "messages": [
            {"role": "system", "content": "Return one molecule."},
            {
                "role": "user",
                "content": json.dumps(
                    {"source": source, "conditions": conditions}, sort_keys=True
                ),
            },
        ],
    }


def make_source_gates(root: Path) -> tuple[list[Path], Path]:
    de_by_arity = {
        arity: [
            request(
                "de_novo",
                f"d-{arity}-{index}",
                [{"property": f"p-{offset}"} for offset in range(arity)],
            )
            for index in range(count)
        ]
        for arity, count in freeze_gate.DE_NOVO_COUNTS.items()
    }
    de_paths = [root / f"de-{part}.jsonl" for part in range(3)]
    write_jsonl(de_paths[0], de_by_arity[2] + de_by_arity[3] + de_by_arity[4])
    write_jsonl(de_paths[1], de_by_arity[5])
    write_jsonl(de_paths[2], de_by_arity[6] + de_by_arity[7])

    editing = [
        request(
            "edit",
            f"e-{task}-{index}",
            [{"property": task.split(":", 1)[0], "goal": "increase"}],
            task,
        )
        for task in freeze_gate.EDIT_COUNTS
        for index in range(500)
    ]
    edit_path = root / "edit.jsonl"
    write_jsonl(edit_path, editing)
    return de_paths, edit_path


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_freezer_writes_exact_target_blind_gate_and_hashes(tmp_path: Path):
    de_paths, edit_path = make_source_gates(tmp_path)
    output = tmp_path / "frozen"
    manifest = freeze_gate.freeze(de_paths, edit_path, output)

    assert manifest["de_novo"]["rows"] == 440
    assert manifest["editing"]["rows"] == 5000
    assert manifest["de_novo"]["sha256"] == file_hash(
        output / "gate.denovo.jsonl"
    )
    assert manifest["editing"]["sha256"] == file_hash(output / "gate.edit.jsonl")
    assert [item["sha256"] for item in manifest["source_files"]] == [
        file_hash(path) for path in [*de_paths, edit_path]
    ]
    assert manifest["target_blind"] is True
    for line in (output / "gate.edit.jsonl").read_text().splitlines():
        row = json.loads(line)
        assert [message["role"] for message in row["messages"]] == [
            "system",
            "user",
        ]


def test_freezer_rejects_target_leakage(tmp_path: Path):
    de_paths, edit_path = make_source_gates(tmp_path)
    rows = freeze_gate.read_jsonl(de_paths[0])
    rows[0]["target_smiles"] = "CC"
    write_jsonl(de_paths[0], rows)
    with pytest.raises(ValueError, match="target-bearing"):
        freeze_gate.freeze(de_paths, edit_path, tmp_path / "bad")


def evaluation_summary(protocol: str, de_novo: float, editing: float):
    edit_buckets = {
        task: {
            "strict_rate": editing,
            "valid_rate": 0.96,
            "property_strict_rate": editing + 0.1,
            "mean_source_similarity": 0.72,
        }
        for task in collector.SHARED_TASKS | collector.EDIT_ONLY_TASKS
    }
    return {
        "protocol": protocol,
        "rows": {"de_novo": 440, "edit": 5000},
        "aggregate": {
            "denovo_strict_pooled": de_novo,
            "denovo_valid_pooled": 0.95,
            "denovo_strict_macro": de_novo - 0.01,
            "denovo_valid_macro": 0.94,
            "denovo_property_strict_macro": de_novo + 0.1,
            "denovo_property_fraction_macro": 0.83,
            "edit_strict_065_macro": editing,
            "edit_valid_macro": 0.96,
            "edit_property_strict_macro": editing + 0.1,
            "edit_source_similarity_macro": 0.72,
        },
        "edit_buckets": edit_buckets,
    }


def test_collector_checks_exact_gate_and_applies_full_only_rule(tmp_path: Path):
    de_paths, edit_source = make_source_gates(tmp_path)
    gate_dir = tmp_path / "gate"
    freeze_gate.freeze(de_paths, edit_source, gate_dir)
    protocol_path = EXPERIMENT / "evaluation_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    evaluation_root = tmp_path / "evaluations"
    training_root = tmp_path / "training"

    frozen_requests = freeze_gate.read_jsonl(gate_dir / "gate.denovo.jsonl")
    frozen_requests += freeze_gate.read_jsonl(gate_dir / "gate.edit.jsonl")
    candidates = [
        {"task_mode": row["task_mode"], "condition_id": row["condition_id"]}
        for row in frozen_requests
    ]

    for checkpoint in protocol["checkpoints"]:
        label = checkpoint["label"]
        step = checkpoint["step"]
        adapter = (
            training_root / "full" / "milestones" / f"checkpoint-{step}" / "adapter"
        )
        adapter.mkdir(parents=True)
        (adapter / "adapter_model.safetensors").write_bytes(f"weights-{label}".encode())
        (adapter / "adapter_config.json").write_text(json.dumps({"rank": 16}))
        (adapter.parent / "milestone_manifest.json").write_text(
            json.dumps({"optimizer_step": step})
        )
        output = evaluation_root / label
        output.mkdir(parents=True)
        de_novo = 0.99 if label != "full" else 0.54
        editing = 0.99 if label != "full" else 0.58
        (output / "summary.json").write_text(
            json.dumps(evaluation_summary(protocol["protocol"], de_novo, editing))
        )
        write_jsonl(output / "candidates.jsonl", candidates)

    result, csv_rows = collector.collect(
        evaluation_root, training_root, gate_dir, protocol_path
    )
    assert result["decision"]["full_is_headline_eligible"] is True
    assert result["decision"]["safe_grpo_allowed"] is True
    assert result["evaluations"]["100k"]["metrics"][
        "de_novo_strict_pooled"
    ] == 0.99
    assert result["evaluations"]["full"]["metrics"][
        "de_novo_strict_pooled"
    ] == 0.54
    assert result["integrity"]["all_checks_pass"] is True
    assert result["integrity"]["candidate_sets"]["full"][
        "matches_frozen_gate"
    ] is True
    assert len(result["evaluations"]["full"]["checkpoint_sha256"]) == 64
    assert len(csv_rows) == 4
    assert "editing_shared5_source_similarity_macro" in csv_rows[0]


def test_candidate_integrity_rejects_same_size_wrong_gate(tmp_path: Path):
    expected = {("de_novo", f"d-{index}") for index in range(440)} | {
        ("edit", f"e-{index}") for index in range(5000)
    }
    wrong = [
        {"task_mode": mode, "condition_id": identity}
        for mode, identity in sorted(expected)
    ]
    wrong[-1]["condition_id"] = "not-in-frozen-gate"
    path = tmp_path / "candidates.jsonl"
    write_jsonl(path, wrong)
    integrity = collector.candidate_integrity(path, expected)
    assert integrity["rows"] == 5440
    assert integrity["matches_frozen_gate"] is False
    assert integrity["valid"] is False
