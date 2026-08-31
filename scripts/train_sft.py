#!/usr/bin/env python3
"""LoRA fine-tune a common instruction LLM on ConstraintIR-to-action chats."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Mapping, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--validation-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--input-adapter-dir", type=Path, default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--compute-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1701)
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    count = 0
    for left_id, right_id in zip(left, right):
        if int(left_id) != int(right_id):
            break
        count += 1
    return count


def input_id_list(value: object) -> list[int]:
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"Unsupported tokenizer output: {type(value).__name__}")
    return [int(item) for item in value]


class ChatDataset:
    def __init__(self, rows: Sequence[dict[str, object]], tokenizer: object, max_length: int):
        self.examples = []
        for row in rows:
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                continue
            full_ids = input_id_list(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                )
            )
            prompt_ids = input_id_list(
                tokenizer.apply_chat_template(
                    messages[:-1],
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )
            eos_id = getattr(tokenizer, "eos_token_id", None)
            if eos_id is not None and (not full_ids or full_ids[-1] != eos_id):
                full_ids.append(int(eos_id))
            full_ids = full_ids[:max_length]
            mask_length = min(common_prefix_length(full_ids, prompt_ids), len(full_ids))
            labels = [-100] * mask_length + full_ids[mask_length:]
            if not any(label != -100 for label in labels):
                continue
            self.examples.append(
                {
                    "input_ids": full_ids,
                    "attention_mask": [1] * len(full_ids),
                    "labels": labels,
                }
            )
        if not self.examples:
            raise ValueError("No tokenized assistant targets were produced")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


class CompletionCollator:
    def __init__(self, tokenizer: object):
        self.pad_token_id = int(tokenizer.pad_token_id)

    def __call__(self, features: Sequence[dict[str, list[int]]]):
        import torch

        max_length = max(len(item["input_ids"]) for item in features)
        input_ids = []
        attention_mask = []
        labels = []
        for item in features:
            padding = max_length - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append(item["attention_mask"] + [0] * padding)
            labels.append(item["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def training_arguments(transformers: object, args: argparse.Namespace, *, compute_dtype: str):
    values = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "warmup_steps": 20,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "logging_steps": args.logging_steps,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "bf16": compute_dtype == "bfloat16",
        "fp16": False,
        "tf32": compute_dtype == "float32",
        "gradient_checkpointing": True,
        "optim": "adamw_torch",
        "report_to": [],
        "logging_nan_inf_filter": False,
        "remove_unused_columns": False,
        "seed": args.seed,
        "data_seed": args.seed,
    }
    signature = inspect.signature(transformers.TrainingArguments.__init__)
    filtered = {key: value for key, value in values.items() if key in signature.parameters}
    return transformers.TrainingArguments(**filtered)


def adapter_nonfinite_count(model: object) -> int:
    import torch

    return sum(
        int((~torch.isfinite(parameter.detach().float())).sum().item())
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import peft
        import torch
        import transformers
    except ImportError as exc:
        raise SystemExit(f"Missing common-LLM dependency: {exc}") from exc
    if not torch.cuda.is_available():
        raise SystemExit("Common-LLM LoRA training requires a CUDA GPU")
    if args.compute_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise SystemExit("--compute-dtype bfloat16 requires GPU bfloat16 support")
    model_dtype = torch.float32 if args.compute_dtype == "float32" else torch.bfloat16
    if args.compute_dtype == "float32":
        torch.backends.cuda.matmul.allow_tf32 = True

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=model_dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if args.input_adapter_dir is not None:
        if not args.input_adapter_dir.joinpath("adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"Missing input adapter: {args.input_adapter_dir}")
        model = peft.PeftModel.from_pretrained(
            model,
            args.input_adapter_dir,
            is_trainable=True,
        )
    else:
        lora_config = peft.LoraConfig(
            task_type=peft.TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = peft.get_peft_model(model, lora_config)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    model.print_trainable_parameters()

    class FiniteAdapterCallback(transformers.TrainerCallback):
        def on_step_end(self, args, state, control, model=None, **kwargs):
            if model is not None and (state.global_step <= 5 or state.global_step % 10 == 0):
                nonfinite = adapter_nonfinite_count(model)
                if nonfinite:
                    raise FloatingPointError(
                        f"Detected {nonfinite} non-finite trainable adapter parameters at step {state.global_step}"
                    )
            return control

    train_rows = read_jsonl(args.train_jsonl)
    train_dataset = ChatDataset(train_rows, tokenizer, args.max_length)
    trainer = transformers.Trainer(
        model=model,
        args=training_arguments(transformers, args, compute_dtype=args.compute_dtype),
        train_dataset=train_dataset,
        data_collator=CompletionCollator(tokenizer),
        callbacks=[FiniteAdapterCallback()],
    )
    result = trainer.train()
    nonfinite = adapter_nonfinite_count(model)
    if nonfinite:
        raise FloatingPointError(f"Refusing to save adapter with {nonfinite} non-finite parameters")
    adapter_dir = args.output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    summary = {
        "protocol": "unified_constraint_common_llm_lora_v1",
        "base_model": args.base_model,
        "input_adapter_dir": str(args.input_adapter_dir) if args.input_adapter_dir else None,
        "train_jsonl": str(args.train_jsonl),
        "validation_jsonl": str(args.validation_jsonl) if args.validation_jsonl else None,
        "tokenized_train_rows": len(train_dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "compute_dtype": args.compute_dtype,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "train_metrics": dict(result.metrics),
        "adapter_dir": str(adapter_dir),
        "adapter_nonfinite_parameters": nonfinite,
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
