#!/usr/bin/env python3
"""Audit whether an editing policy has enough reward support for online RL."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCRIPT_DIR.parent / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from molprogram.support_audit import summarize_support  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stable_key(row: Mapping[str, object], seed: int) -> str:
    identity = row.get("example_id", row.get("sample_id", row.get("condition_id", "")))
    return hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()


def select_rows(rows, prompts_per_task: int, seed: int):
    from molprogram import protocol

    grouped = defaultdict(list)
    for row in rows:
        if str(row.get("task_mode", "")) != "edit":
            continue
        task = str(row.get("task_key", "") or "")
        if task in protocol.TABLE1_TASK_KEYS.values():
            grouped[task].append(row)
    missing = [
        task for task in sorted(protocol.TABLE1_TASK_KEYS.values())
        if len(grouped[task]) < prompts_per_task
    ]
    if missing:
        raise ValueError(f"insufficient editing audit rows: {missing}")
    selected = []
    for task in sorted(protocol.TABLE1_TASK_KEYS.values()):
        candidates = sorted(grouped[task], key=lambda row: stable_key(row, seed))
        selected.extend(candidates[:prompts_per_task])
    return selected


def audit_candidate(row: Mapping[str, object], raw: str) -> dict[str, object]:
    from molprogram.rewards import STRICT_SIMILARITY, hard_boundary_reward
    from molprogram.scoring import score_response

    soft_reward, details = score_response(row, raw)
    hard_reward = hard_boundary_reward({}, details, "edit")
    similarity = details.get("source_similarity")
    source_feasible = bool(
        details.get("valid")
        and not details.get("copy")
        and similarity is not None
        and float(similarity) >= STRICT_SIMILARITY
    )
    return {
        "raw": raw,
        "soft_reward": soft_reward,
        "hard_reward": hard_reward,
        "source_feasible": source_feasible,
        **details,
    }


def generate_group(model, tokenizer, row, group_size: int, seed: int):
    import torch

    prompt = tokenizer.apply_chat_template(
        row["messages"][:-1], tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    offset = encoded["input_ids"].shape[1]
    torch.manual_seed(seed)
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            num_return_sequences=group_size,
            pad_token_id=tokenizer.pad_token_id,
        )
    return [
        tokenizer.decode(ids[offset:], skip_special_tokens=True).strip()
        for ids in generated
    ]


def load_model(base_model: str, adapter_dir: Path):
    import peft
    import torch
    import transformers

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("editing support generation requires BF16 CUDA")
    config = transformers.AutoConfig.from_pretrained(base_model, local_files_only=True)
    if type(config) in transformers.AutoModelForCausalLM._model_mapping:
        loader = transformers.AutoModelForCausalLM
    elif type(config) in transformers.AutoModelForImageTextToText._model_mapping:
        loader = transformers.AutoModelForImageTextToText
    else:
        raise TypeError(f"unsupported config: {type(config).__name__}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model, use_fast=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = loader.from_pretrained(
        base_model, config=config, dtype=torch.bfloat16,
        low_cpu_mem_usage=True, local_files_only=True,
    )
    model = peft.PeftModel.from_pretrained(base, adapter_dir).cuda().eval()
    model.config.use_cache = True
    return model, tokenizer


def report_markdown(summary: Mapping[str, object]) -> str:
    aggregate = summary["aggregate"]
    candidate = aggregate["candidate"]
    group = aggregate["group"]
    gate = summary["gate"]
    lines = [
        "# MolProgram editing reward-support audit",
        "",
        f"**Decision:** `{gate['decision']}`",
        "",
        "The audit uses prompt-visible conditions and source molecules only. "
        "No target molecule is available to scoring.",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Groups | {aggregate['groups']} |",
        f"| Candidates | {aggregate['candidates']} |",
        f"| Valid candidate rate | {100 * candidate['valid_rate']:.2f}% |",
        f"| Source-feasible candidate rate | {100 * candidate['source_feasible_rate']:.2f}% |",
        f"| Strict candidate rate | {100 * candidate['strict_rate']:.2f}% |",
        f"| Strict Any@K | {100 * group['strict_any_at_k']:.2f}% |",
        f"| Mixed-strict groups | {100 * group['mixed_strict_group_rate']:.2f}% |",
        f"| Hard-reward informative groups | {100 * group['hard_informative_group_rate']:.2f}% |",
        f"| Supported tasks | {gate['supported_tasks']}/{gate['task_count']} |",
        "",
        "## Gate checks",
        "",
    ]
    lines.extend(
        f"- [{'x' if passed else ' '}] `{name}`"
        for name, passed in gate["checks"].items()
    )
    lines.extend([
        "", "## Per-task strict support", "",
        "| Task | Any@K | Strict candidate | Source feasible |",
        "| --- | ---: | ---: | ---: |",
    ])
    for task, values in summary["tasks"].items():
        lines.append(
            f"| {task} | {100 * values['group']['strict_any_at_k']:.2f}% | "
            f"{100 * values['candidate']['strict_rate']:.2f}% | "
            f"{100 * values['candidate']['source_feasible_rate']:.2f}% |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups-jsonl", type=Path)
    parser.add_argument("--train-jsonl", type=Path)
    parser.add_argument("--base-model")
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prompts-per-task", type=int, default=50)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=41001)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    live_path = args.output_dir / "groups.live.jsonl"
    if args.groups_jsonl:
        groups = read_jsonl(args.groups_jsonl)
    else:
        if not args.train_jsonl or not args.base_model or not args.adapter_dir:
            parser.error(
                "generation requires --train-jsonl, --base-model, and --adapter-dir"
            )
        selected = select_rows(
            read_jsonl(args.train_jsonl), args.prompts_per_task, args.seed
        )
        existing = read_jsonl(live_path) if live_path.is_file() else []
        by_id = {str(group["example_id"]): group for group in existing}
        model, tokenizer = load_model(args.base_model, args.adapter_dir)
        with live_path.open("a", encoding="utf-8") as handle:
            for index, row in enumerate(selected):
                example_id = str(
                    row.get("example_id", row.get("sample_id", row.get("condition_id", "")))
                )
                if not example_id:
                    raise ValueError("editing support row is missing a stable identity")
                if example_id in by_id:
                    continue
                raw_group = generate_group(
                    model, tokenizer, row, args.group_size,
                    args.seed * 100000 + index,
                )
                group = {
                    "example_id": example_id,
                    "task_key": row["task_key"],
                    "group_size": args.group_size,
                    "seed": args.seed * 100000 + index,
                    "candidates": [audit_candidate(row, raw) for raw in raw_group],
                }
                handle.write(json.dumps(group, sort_keys=True) + "\n")
                handle.flush()
                by_id[example_id] = group
                print(f"[support] {len(by_id)}/{len(selected)}", flush=True)
        groups = [by_id[str(
            row.get("example_id", row.get("sample_id", row.get("condition_id", "")))
        )] for row in selected]

    summary = summarize_support(groups)
    summary["sampling"] = {
        "source": str(args.groups_jsonl or args.train_jsonl),
        "seed": args.seed if not args.groups_jsonl else None,
        "prompts_per_task": args.prompts_per_task if not args.groups_jsonl else None,
        "group_size_requested": args.group_size if not args.groups_jsonl else None,
        "temperature": 0.8 if not args.groups_jsonl else None,
        "top_p": 0.95 if not args.groups_jsonl else None,
    }
    (args.output_dir / "support_report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "support_report.md").write_text(report_markdown(summary))
    (args.output_dir / "AUDIT_COMPLETE").touch()
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
