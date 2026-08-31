#!/usr/bin/env python3
"""Train a numerically guarded fresh LoRA on indexed MuMO rows."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PUBLIC_SCRIPTS = REPO_ROOT / "scripts"
if str(PUBLIC_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PUBLIC_SCRIPTS))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import train_sft as common  # noqa: E402
from protocol import messages  # noqa: E402


LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class IndexedMumoDataset:
    def __init__(self, path: Path, tokenizer, max_length: int):
        self.path = path
        self.offsets = np.memmap(path.with_suffix(".idx"), dtype="<u8", mode="r")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._handles: dict[int, object] = {}

    def __len__(self):
        return len(self.offsets)

    def _row(self, index: int):
        process_id = os.getpid()
        handle = self._handles.get(process_id)
        if handle is None:
            handle = self.path.open("rb")
            self._handles[process_id] = handle
        handle.seek(int(self.offsets[index]))
        return json.loads(handle.readline())

    def __getitem__(self, index: int):
        row = self._row(index)
        chat = messages(
            str(row["source_smiles"]),
            str(row["task"]),
            int(row["instr_idx"]),
            str(row["target_smiles"]),
        )
        full = common.input_id_list(
            self.tokenizer.apply_chat_template(
                chat, tokenize=True, add_generation_prompt=False
            )
        )
        prompt = common.input_id_list(
            self.tokenizer.apply_chat_template(
                chat[:-1], tokenize=True, add_generation_prompt=True
            )
        )
        eos = self.tokenizer.eos_token_id
        if eos is not None and (not full or full[-1] != eos):
            full.append(int(eos))
        full = full[: self.max_length]
        mask_length = min(common.common_prefix_length(full, prompt), len(full))
        labels = [-100] * mask_length + full[mask_length:]
        if not any(value != -100 for value in labels):
            raise ValueError(f"assistant target truncated at row {index}")
        return {
            "input_ids": full,
            "attention_mask": [1] * len(full),
            "labels": labels,
        }


def nonfinite_gradients(model) -> int:
    import torch

    return sum(
        int((~torch.isfinite(parameter.grad.detach().float())).sum().item())
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-length", type=int, default=448)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=32)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--save-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=32002)
    parser.add_argument("--run-kind", choices=("smoke", "full"), required=True)
    args = parser.parse_args()

    import peft
    import torch
    import transformers

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("MuMO fresh training requires a BF16 CUDA GPU")
    if not args.train_jsonl.is_file() or not args.train_jsonl.with_suffix(
        ".idx"
    ).is_file():
        raise FileNotFoundError("indexed MuMO training data is incomplete")
    if args.batch_size * args.gradient_accumulation != 128:
        raise ValueError("effective batch size must remain exactly 128")
    if "lm_head" in LORA_TARGETS:
        raise AssertionError("lm_head must not be adapted in the stable baseline")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model, use_fast=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = transformers.AutoConfig.from_pretrained(
        args.base_model, local_files_only=True
    )
    if type(config) in transformers.AutoModelForCausalLM._model_mapping:
        loader = transformers.AutoModelForCausalLM
        loader_kind = "causal_lm"
    elif type(config) in transformers.AutoModelForImageTextToText._model_mapping:
        loader = transformers.AutoModelForImageTextToText
        loader_kind = "image_text_to_text_text_only"
    else:
        raise TypeError(f"unsupported model config: {type(config).__name__}")
    base = loader.from_pretrained(
        args.base_model,
        config=config,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model = peft.get_peft_model(
        base,
        peft.LoraConfig(
            task_type=peft.TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=list(LORA_TARGETS),
        ),
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()

    dataset = IndexedMumoDataset(args.train_jsonl, tokenizer, args.max_length)
    steps_per_epoch = math.ceil(
        math.ceil(len(dataset) / args.batch_size) / args.gradient_accumulation
    )
    values = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "warmup_steps": 50,
        "lr_scheduler_type": "cosine",
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "logging_steps": 5,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": 3,
        "bf16": True,
        "fp16": False,
        "tf32": True,
        "gradient_checkpointing": True,
        "optim": "adamw_torch",
        "report_to": [],
        "logging_nan_inf_filter": False,
        "remove_unused_columns": False,
        "dataloader_num_workers": 4,
        "dataloader_persistent_workers": True,
        "seed": args.seed,
        "data_seed": args.seed,
    }
    signature = inspect.signature(transformers.TrainingArguments.__init__)
    training_args = transformers.TrainingArguments(
        **{key: value for key, value in values.items() if key in signature.parameters}
    )

    class FiniteTrainer(transformers.Trainer):
        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            outputs = model(**inputs)
            loss = outputs.loss
            if loss is None or not torch.isfinite(loss.detach()).all():
                raise FloatingPointError(
                    f"non-finite microbatch loss at optimizer step {self.state.global_step}"
                )
            return (loss, outputs) if return_outputs else loss

    class FiniteCallback(transformers.TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            bad = nonfinite_gradients(model)
            if bad:
                raise FloatingPointError(
                    f"{bad} non-finite gradient values before step {state.global_step + 1}"
                )
            return control

        def on_step_end(self, args, state, control, model=None, **kwargs):
            bad = common.adapter_nonfinite_count(model)
            if bad:
                raise FloatingPointError(
                    f"{bad} non-finite adapter values after step {state.global_step}"
                )
            return control

    trainer = FiniteTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=common.CompletionCollator(tokenizer),
        callbacks=[FiniteCallback()],
    )
    result = trainer.train()
    nonfinite = common.adapter_nonfinite_count(model)
    if nonfinite:
        raise FloatingPointError(f"non-finite final adapter values: {nonfinite}")
    adapter = args.output_dir / "adapter"
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(adapter)
    summary = {
        "protocol": "mumo_fresh_stable_lora_v1",
        "run_kind": args.run_kind,
        "base_model": args.base_model,
        "fresh_adapter": True,
        "resume_checkpoint": None,
        "loader_kind": loader_kind,
        "train_rows": len(dataset),
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "steps_per_epoch": steps_per_epoch,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "effective_batch_size": 128,
        "learning_rate": args.learning_rate,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_targets": list(LORA_TARGETS),
        "lm_head_adapted": False,
        "finite_guard_each_step": True,
        "adapter_nonfinite_parameters": nonfinite,
        "train_metrics": dict(result.metrics),
        "adapter": str(adapter),
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
