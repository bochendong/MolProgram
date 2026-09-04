#!/usr/bin/env python3
"""Train the 10k property-program routed LoRA candidate."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PUBLIC_SCRIPTS = REPO_ROOT / "scripts"
SOURCE_DIR = REPO_ROOT / "src"
for path in (PUBLIC_SCRIPTS, SOURCE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import train_sft as common  # noqa: E402
from molprogram import program_routing  # noqa: E402


class RoutedDataset:
    def __init__(self, rows, tokenizer, max_length: int, layout):
        base = common.ChatDataset(rows, tokenizer, max_length)
        if len(base) != len(rows):
            raise ValueError(
                f"routing requires one tokenized example per row; got {len(base)}/{len(rows)}"
            )
        self.examples = []
        for row, example in zip(rows, base.examples):
            routed = dict(example)
            routed["route_mask"] = program_routing.route_values(row, layout)
            self.examples.append(routed)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index: int):
        return self.examples[index]


class RoutedCollator:
    def __init__(self, tokenizer):
        self.base = common.CompletionCollator(tokenizer)

    def __call__(self, features):
        import torch

        route_mask = torch.tensor(
            [item["route_mask"] for item in features], dtype=torch.float32
        )
        plain = [
            {key: value for key, value in item.items() if key != "route_mask"}
            for item in features
        ]
        batch = self.base(plain)
        batch["route_mask"] = route_mask
        return batch


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--routing-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--input-adapter-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=448)
    parser.add_argument("--gradient-accumulation", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--seed", type=int, default=33101)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--expected-per-mode", type=int, default=10000)
    parser.add_argument("--protocol", default="property_program_routed_lora_10k_v1")
    args = parser.parse_args(argv)

    import peft
    import torch
    import transformers

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("routed LoRA training requires BF16 CUDA")
    layout = program_routing.load_layout(args.routing_config)
    rank = int(layout["rank"])
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
    if args.input_adapter_dir is None:
        model = peft.get_peft_model(
            model,
            peft.LoraConfig(
                task_type=peft.TaskType.CAUSAL_LM,
                r=rank,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
            ),
        )
    else:
        if not args.input_adapter_dir.joinpath("adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"missing input adapter: {args.input_adapter_dir}")
        model = peft.PeftModel.from_pretrained(
            model, args.input_adapter_dir, is_trainable=True
        )
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    routed_projections = program_routing.install_lora_rank_routing(model, rank=rank)
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rows = common.read_jsonl(args.train_jsonl)
    mode_counts = Counter(program_routing.task_mode(row) for row in rows)
    expected_counts = {
        "de_novo": args.expected_per_mode,
        "edit": args.expected_per_mode,
    }
    if mode_counts != expected_counts:
        raise ValueError(
            f"expected exact balanced counts {expected_counts}, found {mode_counts}"
        )
    dataset = RoutedDataset(rows, tokenizer, args.max_length, layout)
    active_rank_histogram = Counter(
        sum(value != 0.0 for value in item["route_mask"])
        for item in dataset.examples
    )
    training_ns = argparse.Namespace(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=1,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        logging_steps=5,
        seed=args.seed,
    )
    training_args = common.training_arguments(
        transformers, training_ns, compute_dtype="bfloat16"
    )
    training_args.max_steps = args.max_steps

    class RoutedTrainer(transformers.Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            route_mask = inputs.pop("route_mask")
            program_routing.set_lora_route_mask(model, route_mask)
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs, **kwargs
            )

    trainer = RoutedTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=RoutedCollator(tokenizer),
    )
    result = trainer.train()
    nonfinite = common.adapter_nonfinite_count(model)
    if nonfinite:
        raise FloatingPointError(f"adapter has {nonfinite} non-finite LoRA values")
    adapter = args.output_dir / "adapter"
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(adapter)
    shutil.copyfile(args.routing_config, adapter / "program_routing.json")
    summary = {
        "protocol": args.protocol,
        "base_model": args.base_model,
        "fresh_lora_from_base": args.input_adapter_dir is None,
        "input_adapter": (
            str(args.input_adapter_dir.resolve()) if args.input_adapter_dir else None
        ),
        "loader_kind": loader_kind,
        "train_rows": len(dataset),
        "train_rows_by_mode": dict(mode_counts),
        "expected_rows_per_mode": args.expected_per_mode,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "lora_rank": rank,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "trainable_parameters": trainable_parameters,
        "extra_trainable_routing_parameters": 0,
        "routed_lora_a_projections": routed_projections,
        "active_rank_histogram": dict(sorted(active_rank_histogram.items())),
        "routing_config": str((adapter / "program_routing.json").resolve()),
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
