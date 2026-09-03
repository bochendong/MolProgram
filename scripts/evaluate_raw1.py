#!/usr/bin/env python3
"""Evaluate one joint or specialist adapter on frozen Raw@1 gates."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCRIPT_DIR.parent / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))
from molprogram import protocol  # noqa: E402
from molprogram import program_routing  # noqa: E402
from molprogram.scoring import property_count, score_response  # noqa: E402


def mean(values) -> float:
    values = list(values)
    return sum(float(value) for value in values) / max(len(values), 1)


def present_mean(items, key: str) -> float | None:
    values = [item[key] for item in items if item.get(key) is not None]
    return mean(values) if values else None


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def generate_records(
    model, tokenizer, rows, *, seed: int, batch_size: int, mode_offset: int,
    routing_layout=None,
):
    import torch

    records: list[dict[str, object]] = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=True
            )
            for row in batch
        ]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        if routing_layout is not None:
            route_mask = program_routing.route_matrix(batch, routing_layout).to(model.device)
            program_routing.set_lora_route_mask(model, route_mask)
        offset = encoded["input_ids"].shape[1]
        torch.manual_seed(seed + mode_offset + start)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )
        for row, ids in zip(batch, generated):
            raw = tokenizer.decode(ids[offset:], skip_special_tokens=True).strip()
            reward, details = score_response(row, raw)
            records.append(
                {
                    "condition_id": row.get("condition_id", row.get("sample_id", "")),
                    "task_mode": row["task_mode"],
                    "property_count": property_count(row),
                    "task_key": row.get("task_key", ""),
                    "raw": raw,
                    "reward": reward,
                    **details,
                }
            )
        print(f"[raw1] {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--denovo-gate", required=True, type=Path)
    parser.add_argument("--edit-gate", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--arm",
        required=True,
        help="Result label, for example joint, denovo, edit, or an ablation arm.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=33051)
    parser.add_argument(
        "--protocol", default="molprogram_raw1_v1"
    )
    parser.add_argument(
        "--routing-config",
        type=Path,
        help="Optional property-program LoRA rank-routing layout.",
    )
    args = parser.parse_args(argv)

    import peft
    import torch
    import transformers

    de_novo = read_jsonl(args.denovo_gate)
    editing = [row for row in read_jsonl(args.edit_gate) if row["task_mode"] == "edit"]
    config = transformers.AutoConfig.from_pretrained(args.base_model, local_files_only=True)
    loader = (
        transformers.AutoModelForCausalLM
        if type(config) in transformers.AutoModelForCausalLM._model_mapping
        else transformers.AutoModelForImageTextToText
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model, use_fast=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = loader.from_pretrained(
        args.base_model, config=config, dtype=torch.bfloat16,
        low_cpu_mem_usage=True, local_files_only=True,
    )
    model = peft.PeftModel.from_pretrained(base, args.adapter_dir).cuda().eval()
    model.config.use_cache = True
    routing_layout = None
    if args.routing_config is not None:
        routing_layout = program_routing.load_layout(args.routing_config)
        program_routing.install_lora_rank_routing(
            model, rank=int(routing_layout["rank"])
        )
    records = generate_records(
        model, tokenizer, de_novo,
        seed=args.seed, batch_size=args.batch_size, mode_offset=0,
        routing_layout=routing_layout,
    )
    records.extend(
        generate_records(
            model, tokenizer, editing,
            seed=args.seed, batch_size=args.batch_size, mode_offset=100000,
            routing_layout=routing_layout,
        )
    )

    de_groups: dict[int, list[dict[str, object]]] = defaultdict(list)
    edit_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        if record["task_mode"] == "de_novo":
            de_groups[int(record["property_count"])].append(record)
        else:
            edit_groups[str(record["task_key"])].append(record)
    de_buckets = {
        f"{count}p": {
            "rows": len(de_groups[count]),
            "strict_rate": mean(item["strict"] for item in de_groups[count]),
            "valid_rate": mean(item["valid"] for item in de_groups[count]),
            "property_strict_rate": mean(
                item["property_strict"] for item in de_groups[count]
            ),
            "mean_property_fraction": present_mean(
                de_groups[count], "property_fraction"
            ),
            "mean_satisfaction": present_mean(de_groups[count], "mean_satisfaction"),
        }
        for count in range(2, 8)
    }
    edit_buckets = {}
    for task in sorted(protocol.TABLE1_TASK_KEYS.values()):
        items = edit_groups[task]
        if not items:
            raise ValueError(f"missing editing gate task {task}")
        edit_buckets[task] = {
            "rows": len(items),
            "strict_rate": mean(item["strict"] for item in items),
            "relaxed_rate": mean(item["relaxed"] for item in items),
            "valid_rate": mean(item["valid"] for item in items),
            "property_strict_rate": mean(item["property_strict"] for item in items),
            "copy_rate": mean(item["copy"] for item in items),
            "mean_property_fraction": present_mean(items, "property_fraction"),
            "mean_source_similarity": present_mean(items, "source_similarity"),
        }
    summary = {
        "protocol": args.protocol,
        "arm": args.arm,
        "sampling": {"temperature": 0.8, "top_p": 0.95, "seed": args.seed},
        "property_reranking": False,
        "program_routing": (
            str(args.routing_config.resolve()) if args.routing_config else None
        ),
        "rows": {"de_novo": len(de_novo), "edit": len(editing)},
        "aggregate": {
            "denovo_strict_macro": mean(v["strict_rate"] for v in de_buckets.values()),
            "denovo_valid_macro": mean(v["valid_rate"] for v in de_buckets.values()),
            "denovo_property_strict_macro": mean(
                v["property_strict_rate"] for v in de_buckets.values()
            ),
            "denovo_property_fraction_macro": mean(
                v["mean_property_fraction"] for v in de_buckets.values()
            ),
            "edit_strict_065_macro": mean(v["strict_rate"] for v in edit_buckets.values()),
            "edit_relaxed_015_macro": mean(v["relaxed_rate"] for v in edit_buckets.values()),
            "edit_valid_macro": mean(v["valid_rate"] for v in edit_buckets.values()),
            "edit_property_strict_macro": mean(
                v["property_strict_rate"] for v in edit_buckets.values()
            ),
            "edit_copy_macro": mean(v["copy_rate"] for v in edit_buckets.values()),
            "edit_property_fraction_macro": mean(
                v["mean_property_fraction"] for v in edit_buckets.values()
            ),
            "edit_source_similarity_macro": present_mean(
                list(edit_buckets.values()), "mean_source_similarity"
            ),
        },
        "denovo_buckets": de_buckets,
        "edit_buckets": edit_buckets,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "candidates.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "EVAL_COMPLETE").touch()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
