#!/usr/bin/env python3
"""Train one matched joint or task-specialist LoRA arm."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PUBLIC_SCRIPTS = REPO_ROOT / "scripts"
if str(PUBLIC_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PUBLIC_SCRIPTS))
import train_sft as common  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--arm", choices=("joint", "denovo", "edit"), required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=448)
    parser.add_argument("--gradient-accumulation", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--seed", type=int, default=33001)
    parser.add_argument(
        "--protocol", default="molprogram_joint_vs_specialist_v1"
    )
    args = parser.parse_args(argv)

    import peft
    import torch
    import transformers

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("matched LoRA training requires BF16 CUDA")
    config = transformers.AutoConfig.from_pretrained(args.base_model, local_files_only=True)
    if type(config) in transformers.AutoModelForCausalLM._model_mapping:
        loader = transformers.AutoModelForCausalLM
        loader_kind = "causal_lm"
    elif type(config) in transformers.AutoModelForImageTextToText._model_mapping:
        loader = transformers.AutoModelForImageTextToText
        loader_kind = "image_text_to_text_text_only"
    else:
        raise TypeError(f"unsupported config: {type(config).__name__}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model, use_fast=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = loader.from_pretrained(
        args.base_model, config=config, dtype=torch.bfloat16,
        low_cpu_mem_usage=True, local_files_only=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = peft.get_peft_model(
        model,
        peft.LoraConfig(
            task_type=peft.TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        ),
    )
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rows = common.read_jsonl(args.train_jsonl)
    dataset = common.ChatDataset(rows, tokenizer, args.max_length)
    mode_counts = {
        mode: sum(str(row.get("task_mode", "")) == mode for row in rows)
        for mode in ("de_novo", "edit")
    }
    if args.arm == "joint" and (
        not mode_counts["de_novo"]
        or mode_counts["de_novo"] != mode_counts["edit"]
    ):
        raise ValueError(f"joint arm requires equal nonzero task counts, found {mode_counts}")
    if args.arm == "denovo" and mode_counts != {"de_novo": len(rows), "edit": 0}:
        raise ValueError(f"de-novo arm is not task-pure: {mode_counts}")
    if args.arm == "edit" and mode_counts != {"de_novo": 0, "edit": len(rows)}:
        raise ValueError(f"editing arm is not task-pure: {mode_counts}")
    training_ns = argparse.Namespace(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=1,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        logging_steps=5,
        seed=args.seed,
    )
    trainer = transformers.Trainer(
        model=model,
        args=common.training_arguments(transformers, training_ns, compute_dtype="bfloat16"),
        train_dataset=dataset,
        data_collator=common.CompletionCollator(tokenizer),
    )
    result = trainer.train()
    nonfinite = common.adapter_nonfinite_count(model)
    if nonfinite:
        raise FloatingPointError(f"adapter has {nonfinite} non-finite LoRA values")
    adapter = args.output_dir / "adapter"
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(adapter)
    summary = {
        "protocol": args.protocol,
        "arm": args.arm,
        "base_model": args.base_model,
        "fresh_lora_from_base": True,
        "input_adapter": None,
        "loader_kind": loader_kind,
        "train_rows": len(dataset),
        "train_rows_by_mode": mode_counts,
        "epochs": args.epochs,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "lora_rank": 16,
        "trainable_parameters": trainable_parameters,
        "adapter_nonfinite_parameters": nonfinite,
        "train_metrics": dict(result.metrics),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "TRAINING_COMPLETE").touch()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
