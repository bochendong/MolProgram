#!/usr/bin/env python3
"""Matched continued-SFT control for safe joint Raw@1 GRPO."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

import train_safe_joint_raw1_grpo as grpo

from molprogram.safe_grpo import (
    balanced_bucket,
    equal_norm_bisector,
    select_balanced_pairs,
)


def load_control(base_model: str, adapter_source: Path):
    import peft
    import torch
    import transformers

    config = transformers.AutoConfig.from_pretrained(base_model, local_files_only=True)
    if type(config) in transformers.AutoModelForCausalLM._model_mapping:
        loader = transformers.AutoModelForCausalLM
        loader_kind = "causal_lm"
    elif type(config) in transformers.AutoModelForImageTextToText._model_mapping:
        loader = transformers.AutoModelForImageTextToText
        loader_kind = "image_text_to_text_text_only"
    else:
        raise TypeError(f"unsupported config: {type(config).__name__}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model, use_fast=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = loader.from_pretrained(
        base_model,
        config=config,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model = peft.PeftModel.from_pretrained(
        base, adapter_source, adapter_name="default", is_trainable=True
    ).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    return model, tokenizer, loader_kind


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--input-adapter", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--paired-steps", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1.5e-7)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=37001)
    args = parser.parse_args(argv)

    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("continued-SFT control requires BF16 CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    schedule = select_balanced_pairs(
        grpo.read_jsonl(args.train_jsonl), args.paired_steps, args.seed
    )
    checkpoints = grpo.complete_checkpoints(args.output_dir)
    start_step, history = 0, []
    adapter_source = args.input_adapter
    resume_checkpoint = checkpoints[-1] if checkpoints else None
    if resume_checkpoint is not None:
        state = json.loads((resume_checkpoint / "state.json").read_text())
        start_step = int(state["next_step"])
        history = list(state["history"])
        adapter_source = resume_checkpoint / "adapter"

    model, tokenizer, loader_kind = load_control(args.base_model, adapter_source)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for parameter in trainable:
        parameter.data = parameter.data.float()
    optimizer = torch.optim.AdamW(
        trainable, lr=float(args.learning_rate), weight_decay=0.0
    )
    if resume_checkpoint is not None:
        optimizer.load_state_dict(
            torch.load(resume_checkpoint / "optimizer.pt", map_location="cpu")
        )

    totals = Counter()
    for record in history:
        totals["paired_steps"] += 1
        totals["gradient_conflicts"] += int(record["gradient_conflict"])
        totals["common_descent_steps"] += int(record["common_descent"])
    live_log = args.output_dir / "training_history.live.jsonl"
    for step in range(start_step, len(schedule)):
        rows = schedule[step]
        gradients = []
        losses = []
        for row in rows:
            optimizer.zero_grad(set_to_none=True)
            model.train()
            loss = grpo.chosen_sft_loss(model, tokenizer, list(row["messages"]))
            loss.backward()
            gradients.append(grpo.capture_gradients(trainable))
            losses.append(float(loss.detach()))
        merged, gradient_record = equal_norm_bisector(gradients[0], gradients[1])
        optimizer.zero_grad(set_to_none=True)
        for parameter, gradient in zip(trainable, merged):
            parameter.grad = gradient
        unclipped_norm = torch.nn.utils.clip_grad_norm_(
            trainable, float(args.grad_clip)
        )
        optimizer.step()
        record = {
            "step": step,
            "de_novo_bucket": balanced_bucket(rows[0]),
            "edit_bucket": balanced_bucket(rows[1]),
            "de_novo_sft_loss": losses[0],
            "edit_sft_loss": losses[1],
            **gradient_record,
            "unclipped_gradient_norm": float(unclipped_norm),
        }
        history.append(record)
        totals["paired_steps"] += 1
        totals["gradient_conflicts"] += int(record["gradient_conflict"])
        totals["common_descent_steps"] += int(record["common_descent"])
        with live_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps({"stage": "paired_sft_step", **record}, sort_keys=True))
        next_step = step + 1
        if (
            next_step % args.checkpoint_every == 0
            or next_step == len(schedule)
        ):
            grpo.save_checkpoint(
                model, tokenizer, optimizer, args.output_dir, next_step, history
            )

    nonfinite = sum(
        int((~torch.isfinite(parameter)).sum().item()) for parameter in trainable
    )
    if nonfinite:
        raise FloatingPointError(f"non-finite trainable parameters: {nonfinite}")
    adapter = args.output_dir / "adapter"
    grpo.save_policy(model, adapter)
    tokenizer.save_pretrained(adapter)
    summary = {
        "protocol": "molprogram_matched_continued_sft_control_v1",
        "loader_kind": loader_kind,
        "base_model": args.base_model,
        "input_adapter": str(args.input_adapter),
        "output_adapter": str(adapter),
        "paired_steps": len(schedule),
        "same_prompts_as_rl": True,
        "same_optimizer_steps_as_rl": True,
        "same_checkpoint_selection_budget_as_rl": True,
        "rollout_compute_matched": False,
        "learning_rate": args.learning_rate,
        "gradient_merge": "equal_norm_bisector",
        "adapter_nonfinite_parameters": nonfinite,
        "totals": dict(sorted(totals.items())),
        "history": history,
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "history"},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
