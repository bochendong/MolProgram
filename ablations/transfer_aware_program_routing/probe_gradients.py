#!/usr/bin/env python3
"""Measure mode-property gradient signatures and compile a routing layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PUBLIC_SCRIPTS = REPO_ROOT / "scripts"
SOURCE_DIR = REPO_ROOT / "src"
for path in (PUBLIC_SCRIPTS, SOURCE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import train_sft as common  # noqa: E402
from molprogram import program_routing, transfer_graph  # noqa: E402


def identity(row: Mapping[str, object]) -> str:
    return str(row.get("example_id", row.get("condition_id", row.get("sample_id", ""))))


def stable_key(row: Mapping[str, object], seed: int) -> str:
    value = identity(row) or json.dumps(row, sort_keys=True)
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def group_rows(rows: Sequence[dict[str, object]]):
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        mode = program_routing.task_mode(row)
        for prop in program_routing.properties(row):
            grouped[f"{mode}:{prop}"].append(row)
    return grouped


def rank_scaling_signature(model: object) -> tuple[list[str], list[float]]:
    import torch

    labels: list[str] = []
    values: list[float] = []
    for module_name, layer in sorted(model.named_modules()):
        lora_a = getattr(layer, "lora_A", None)
        lora_b = getattr(layer, "lora_B", None)
        if lora_a is None or lora_b is None or not hasattr(lora_a, "items"):
            continue
        for adapter_name, projection_a in sorted(lora_a.items()):
            if adapter_name not in lora_b:
                raise ValueError(f"missing LoRA-B pair for {module_name}:{adapter_name}")
            projection_b = lora_b[adapter_name]
            a_weight = projection_a.weight
            b_weight = projection_b.weight
            if a_weight.shape[0] != b_weight.shape[1]:
                raise ValueError(f"rank mismatch for {module_name}:{adapter_name}")
            # Some adapters (notably Qwen-VL's visual tower for text-only
            # program examples) are deliberately disconnected from the loss.
            # Their derivative with respect to rank-component scaling is zero,
            # not an error.  Preserve the coordinates so every task signature
            # has the same labels; normalize_signatures still rejects a task
            # whose complete signature is zero.
            a_term = (
                (a_weight.grad * a_weight).sum(dim=1)
                if a_weight.grad is not None
                else torch.zeros(a_weight.shape[0], device=a_weight.device)
            )
            b_term = (
                (b_weight.grad * b_weight).sum(dim=0)
                if b_weight.grad is not None
                else torch.zeros(b_weight.shape[1], device=b_weight.device)
            )
            score = a_term + b_term
            for rank_index, value in enumerate(score.detach().float().cpu().tolist()):
                labels.append(f"{module_name}:{adapter_name}:r{rank_index}")
                values.append(float(value))
    if not values or not torch.isfinite(torch.tensor(values)).all():
        raise ValueError("failed to produce a finite LoRA gradient signature")
    return labels, values


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--samples-per-node", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=448)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--common-ranks", type=int, default=8)
    parser.add_argument("--inactive-floor", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=33401)
    args = parser.parse_args(argv)

    import peft
    import torch
    import transformers

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("gradient probing requires BF16 CUDA")
    rows = common.read_jsonl(args.train_jsonl)
    grouped = group_rows(rows)
    selected = {
        node: sorted(values, key=lambda row: stable_key(row, args.seed))[
            : args.samples_per_node
        ]
        for node, values in grouped.items()
    }
    if any(len(values) < args.samples_per_node for values in selected.values()):
        short = {node: len(values) for node, values in selected.items() if len(values) < args.samples_per_node}
        raise ValueError(f"insufficient probe rows: {short}")

    config = transformers.AutoConfig.from_pretrained(
        args.base_model, local_files_only=True
    )
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
    base = loader.from_pretrained(
        args.base_model,
        config=config,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    model = peft.PeftModel.from_pretrained(base, args.adapter_dir, is_trainable=True)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    model.cuda().eval()
    collator = common.CompletionCollator(tokenizer)

    signatures: dict[str, list[float]] = {}
    signature_labels: list[str] | None = None
    selected_ids: dict[str, list[str]] = {}
    for node in sorted(selected):
        model.zero_grad(set_to_none=True)
        selected_ids[node] = []
        for row in selected[node]:
            dataset = common.ChatDataset([row], tokenizer, args.max_length)
            batch = {
                key: value.cuda() for key, value in collator([dataset[0]]).items()
            }
            loss = model(**batch).loss / args.samples_per_node
            loss.backward()
            selected_ids[node].append(identity(row))
        labels, signature = rank_scaling_signature(model)
        if signature_labels is None:
            signature_labels = labels
        elif labels != signature_labels:
            raise ValueError("gradient signature labels changed between nodes")
        signatures[node] = signature
        print(f"[probe] {node}: {len(selected[node])} rows", flush=True)

    layout, evidence = transfer_graph.compile_signed_spectral_layout(
        signatures,
        rank=args.rank,
        common_ranks=args.common_ranks,
        inactive_floor=args.inactive_floor,
    )
    program_routing.validate_layout(layout)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "routing_layout.json").write_text(
        json.dumps(layout, indent=2, sort_keys=True) + "\n"
    )
    evidence.update(
        {
            "protocol": "transfer_aware_program_routing_3840_per_mode_pilot_v1",
            "signature": "per-layer per-rank derivative with respect to rank-component scaling",
            "samples_per_node": args.samples_per_node,
            "selected_ids": selected_ids,
            "signature_labels": signature_labels,
            "signatures": signatures,
        }
    )
    (args.output_dir / "transfer_graph.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "PROBE_COMPLETE").touch()
    print(json.dumps(layout, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
