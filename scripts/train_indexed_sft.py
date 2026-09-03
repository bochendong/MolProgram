#!/usr/bin/env python3
"""Memory-bounded continuation SFT over indexed MolProgramInstruct shards."""

from __future__ import annotations

import argparse
import array
import bisect
import heapq
import json
import math
import os
import sys
from pathlib import Path
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import train_sft as common  # noqa: E402


class IndexedChatDataset:
    """Map-style JSONL dataset backed by uint64 byte-offset sidecars."""

    def __init__(self, release_root: Path, tokenizer: object, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.paths: list[Path] = []
        self.indices: list[object] = []
        self.cumulative = [0]
        self.bucket_indices: dict[str, array.array] = {}
        self._handles: dict[str, object] = {}
        for mode in ("de_novo", "edit"):
            for path in sorted((release_root / mode).glob("*.jsonl")):
                index = path.with_suffix(".idx")
                if not index.is_file() or index.stat().st_size % 8:
                    raise ValueError(f"missing or invalid index: {index}")
                import numpy as np

                offsets = np.memmap(index, dtype="<u8", mode="r")
                self.paths.append(path)
                self.indices.append(offsets)
                base_index = self.cumulative[-1]
                with path.open(encoding="utf-8") as handle:
                    for local_index, line in enumerate(handle):
                        row = json.loads(line)
                        bucket = f"{row['task_mode']}:{row['property_count']}p"
                        self.bucket_indices.setdefault(bucket, array.array("Q")).append(
                            base_index + local_index
                        )
                self.cumulative.append(self.cumulative[-1] + len(offsets))
        if not self.paths or not self.cumulative[-1]:
            raise ValueError(f"no indexed release shards under {release_root}")
        expected = {
            *{f"de_novo:{count}p" for count in range(2, 8)},
            *{f"edit:{count}p" for count in range(1, 8)},
        }
        if set(self.bucket_indices) != expected:
            raise ValueError(
                f"task buckets differ: expected={sorted(expected)} actual={sorted(self.bucket_indices)}"
            )

    def __len__(self) -> int:
        return self.cumulative[-1]

    def _row(self, index: int) -> dict[str, object]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard = bisect.bisect_right(self.cumulative, index) - 1
        local_index = index - self.cumulative[shard]
        path = self.paths[shard]
        key = f"{os.getpid()}:{path}"
        handle = self._handles.get(key)
        if handle is None:
            handle = path.open("rb")
            self._handles[key] = handle
        handle.seek(int(self.indices[shard][local_index]))
        return json.loads(handle.readline())

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self._row(index)
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            raise ValueError(f"invalid messages at dataset index {index}")
        full_ids = common.input_id_list(
            self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False,
            )
        )
        prompt_ids = common.input_id_list(
            self.tokenizer.apply_chat_template(
                messages[:-1], tokenize=True, add_generation_prompt=True,
            )
        )
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_id is not None and (not full_ids or full_ids[-1] != eos_id):
            full_ids.append(int(eos_id))
        full_ids = full_ids[: self.max_length]
        mask_length = min(common.common_prefix_length(full_ids, prompt_ids), len(full_ids))
        labels = [-100] * mask_length + full_ids[mask_length:]
        if not any(label != -100 for label in labels):
            raise ValueError(f"assistant target truncated at dataset index {index}")
        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
            "row_index": index,
        }


class IndexedCompletionCollator:
    """Pad language-model inputs while retaining source rows for diagnostics."""

    def __init__(self, tokenizer: object):
        self.completion_collator = common.CompletionCollator(tokenizer)

    def __call__(self, features: Sequence[dict[str, object]]):
        import torch

        batch = self.completion_collator(features)
        batch["_row_indices"] = torch.tensor(
            [int(item["row_index"]) for item in features], dtype=torch.long
        )
        return batch


def nonfinite_gradient_count(model: object) -> int:
    """Count non-finite trainable-gradient values without changing gradients."""

    import torch

    return sum(
        int((~torch.isfinite(parameter.grad.detach().float())).sum().item())
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    )


class TaskBalancedSampler:
    """Deterministic, task-homogeneous batches with equal task counts."""

    def __init__(self, dataset: IndexedChatDataset, seed: int, batch_size: int = 1):
        self.dataset = dataset
        self.seed = seed
        self.batch_size = batch_size
        self.keys = sorted(dataset.bucket_indices)
        available_per_bucket = min(len(dataset.bucket_indices[key]) for key in self.keys)
        self.per_bucket = available_per_bucket - (available_per_bucket % self.batch_size)
        if not self.per_bucket:
            raise ValueError("physical batch size exceeds the smallest task bucket")

    def __len__(self) -> int:
        return self.per_bucket * len(self.keys)

    def __iter__(self):
        import torch

        generator = torch.Generator()
        generator.manual_seed(self.seed)
        permutations = {
            key: torch.randperm(len(self.dataset.bucket_indices[key]), generator=generator)[: self.per_bucket]
            for key in self.keys
        }
        for start in range(0, self.per_bucket, self.batch_size):
            for key in self.keys:
                for position in range(start, start + self.batch_size):
                    local = int(permutations[key][position])
                    yield int(self.dataset.bucket_indices[key][local])


class ProportionalOnePassSampler:
    """Deterministically interleave every release row exactly once.

    Each task bucket is shuffled independently.  A small merge heap then
    spreads bucket positions over the unit interval, so the emitted prefix
    tracks the frozen bucket proportions without leaving a de-novo-only tail.
    """

    def __init__(self, dataset: IndexedChatDataset, seed: int):
        self.dataset = dataset
        self.seed = seed
        self.keys = sorted(dataset.bucket_indices)
        self.counts = {
            key: len(dataset.bucket_indices[key])
            for key in self.keys
        }
        if any(count < 1 for count in self.counts.values()):
            raise ValueError("proportional sampler requires non-empty task buckets")

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        import numpy as np

        generator = np.random.default_rng(self.seed)
        permutations = {
            key: generator.permutation(self.counts[key])
            for key in self.keys
        }
        schedule: list[tuple[float, str, int]] = []
        for key in self.keys:
            heapq.heappush(schedule, (0.5 / self.counts[key], key, 0))
        while schedule:
            _, key, position = heapq.heappop(schedule)
            local = int(permutations[key][position])
            yield int(self.dataset.bucket_indices[key][local])
            next_position = position + 1
            if next_position < self.counts[key]:
                location = (next_position + 0.5) / self.counts[key]
                heapq.heappush(schedule, (location, key, next_position))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    initialization = parser.add_mutually_exclusive_group(required=True)
    initialization.add_argument("--input-adapter", type=Path)
    initialization.add_argument("--fresh-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=448)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument(
        "--milestone-step", action="append", type=int, default=[],
        help="Optimizer step whose adapter must be preserved outside rotating checkpoints.",
    )
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=24003)
    parser.add_argument(
        "--sampler-mode", choices=("balanced", "proportional_one_pass"),
        default="balanced",
    )
    parser.add_argument("--expected-train-rows", type=int)
    parser.add_argument("--resume-from-checkpoint", action="store_true")
    parser.add_argument(
        "--guard-every-microbatch",
        action="store_true",
        help="Audit gradients after every microbatch (intended for smoke tests).",
    )
    args = parser.parse_args(argv)
    if args.per_device_batch_size < 1:
        raise SystemExit("--per-device-batch-size must be positive")
    if args.gradient_accumulation < 1:
        raise SystemExit("--gradient-accumulation must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        raise SystemExit("--max-steps must be positive when provided")
    if args.num_train_epochs <= 0:
        raise SystemExit("--num-train-epochs must be positive")
    if args.lora_r < 1 or args.lora_alpha < 1:
        raise SystemExit("LoRA rank and alpha must be positive")
    if not 0.0 <= args.lora_dropout < 1.0:
        raise SystemExit("--lora-dropout must be in [0, 1)")
    milestone_steps = sorted(set(args.milestone_step))
    if any(step < 1 for step in milestone_steps):
        raise SystemExit("--milestone-step values must be positive")
    if args.max_steps is not None and any(step > args.max_steps for step in milestone_steps):
        raise SystemExit("--milestone-step cannot exceed --max-steps")
    if args.sampler_mode == "balanced" and args.max_steps is None:
        raise SystemExit("balanced training requires --max-steps")
    if (
        args.sampler_mode == "proportional_one_pass"
        and args.max_steps is None
        and args.num_train_epochs != 1.0
    ):
        raise SystemExit("formal proportional one-pass training requires exactly one epoch")

    import inspect
    import peft
    import torch
    import transformers

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("indexed MolProgram training requires BF16 CUDA")
    if not args.release_root.joinpath("RELEASE_COMPLETE").is_file():
        raise FileNotFoundError(f"release is not frozen: {args.release_root}")
    if (
        args.input_adapter is not None
        and not args.input_adapter.joinpath("adapter_model.safetensors").is_file()
    ):
        raise FileNotFoundError(f"missing input adapter: {args.input_adapter}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model, use_fast=True, local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = transformers.AutoConfig.from_pretrained(args.base_model, local_files_only=True)
    if type(config) in transformers.AutoModelForCausalLM._model_mapping:
        loader = transformers.AutoModelForCausalLM
        loader_kind = "causal_lm"
    elif type(config) in transformers.AutoModelForImageTextToText._model_mapping:
        loader = transformers.AutoModelForImageTextToText
        loader_kind = "image_text_to_text_text_only"
    else:
        raise TypeError(f"unsupported config: {type(config).__name__}")
    base = loader.from_pretrained(
        args.base_model, config=config, dtype=torch.bfloat16,
        low_cpu_mem_usage=True, local_files_only=True,
    )
    if args.input_adapter is not None:
        model = peft.PeftModel.from_pretrained(base, args.input_adapter, is_trainable=True)
    else:
        model = peft.get_peft_model(
            base,
            peft.LoraConfig(
                task_type=peft.TaskType.CAUSAL_LM,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
            ),
        )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()

    dataset = IndexedChatDataset(args.release_root, tokenizer, args.max_length)
    if args.expected_train_rows is not None and len(dataset) != args.expected_train_rows:
        raise ValueError(
            f"release rows differ: expected={args.expected_train_rows} actual={len(dataset)}"
        )
    balanced_rows_per_task = (
        min(map(len, dataset.bucket_indices.values()))
        - min(map(len, dataset.bucket_indices.values())) % args.per_device_batch_size
    )
    sampler_rows = (
        balanced_rows_per_task * len(dataset.bucket_indices)
        if args.sampler_mode == "balanced"
        else len(dataset)
    )
    physical_batches = math.ceil(sampler_rows / args.per_device_batch_size)
    expected_optimizer_steps = math.ceil(physical_batches / args.gradient_accumulation)
    values = {
        "output_dir": str(args.output_dir),
        "max_steps": args.max_steps if args.max_steps is not None else -1,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "learning_rate": args.learning_rate, "warmup_steps": args.warmup_steps,
        "weight_decay": 0.01, "max_grad_norm": 1.0,
        "logging_steps": args.logging_steps, "save_strategy": "steps",
        "save_steps": args.save_steps, "save_total_limit": 2,
        "bf16": True, "fp16": False, "gradient_checkpointing": True,
        "optim": "adamw_torch", "report_to": [],
        "logging_nan_inf_filter": False, "remove_unused_columns": False,
        "seed": args.seed, "data_seed": args.seed,
    }
    signature = inspect.signature(transformers.TrainingArguments.__init__)
    training_args = transformers.TrainingArguments(
        **{key: value for key, value in values.items() if key in signature.parameters}
    )
    class BalancedTrainer(transformers.Trainer):
        guarded_microbatches = 0
        current_row_indices: list[int] = []
        current_microbatch_loss = float("nan")

        def _get_train_sampler(self, train_dataset=None):
            selected = train_dataset if train_dataset is not None else self.train_dataset
            if args.sampler_mode == "balanced":
                return TaskBalancedSampler(selected, args.seed, args.per_device_batch_size)
            return ProportionalOnePassSampler(selected, args.seed)

        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            row_indices = inputs.pop("_row_indices")
            self.current_row_indices = [int(value) for value in row_indices.tolist()]
            outputs = model(**inputs)
            loss = outputs.loss
            if loss is None or not torch.isfinite(loss.detach()).all():
                raise FloatingPointError(
                    "non-finite microbatch loss at optimizer step "
                    f"{self.state.global_step}; rows={self.current_row_indices}"
                )
            self.current_microbatch_loss = float(loss.detach().float().item())
            return (loss, outputs) if return_outputs else loss

        def training_step(self, model, inputs, num_items_in_batch=None):
            loss = super().training_step(model, inputs, num_items_in_batch)
            self.guarded_microbatches += 1
            if (
                args.guard_every_microbatch
                or self.guarded_microbatches <= args.gradient_accumulation
            ):
                bad = nonfinite_gradient_count(model)
                if bad:
                    raise FloatingPointError(
                        f"{bad} non-finite gradient values after microbatch "
                        f"{self.guarded_microbatches}; rows={self.current_row_indices}; "
                        f"loss={self.current_microbatch_loss:.8g}"
                    )
            return loss

    class FiniteTrainingCallback(transformers.TrainerCallback):
        """Stop before one bad gradient can poison the complete LoRA adapter."""

        def on_pre_optimizer_step(self, training_args, state, control, model=None, **kwargs):
            if model is not None:
                bad = nonfinite_gradient_count(model)
                if bad:
                    raise FloatingPointError(
                        f"{bad} non-finite gradient values before step "
                        f"{state.global_step + 1}"
                    )
            return control

        def on_step_end(self, training_args, state, control, model=None, **kwargs):
            if model is not None:
                bad = common.adapter_nonfinite_count(model)
                if bad:
                    raise FloatingPointError(
                        f"{bad} non-finite adapter values after step {state.global_step}"
                    )
            return control

    milestone_root = args.output_dir / "milestones"

    class MilestoneAdapterCallback(transformers.TrainerCallback):
        """Preserve adapter-only checkpoints even when Trainer rotates state."""

        def on_step_end(self, training_args, state, control, **kwargs):
            if int(state.global_step) in milestone_steps:
                control.should_save = True
            return control

        def on_save(self, training_args, state, control, model=None, **kwargs):
            step = int(state.global_step)
            if step not in milestone_steps or model is None:
                return control
            adapter_dir = milestone_root / f"checkpoint-{step}" / "adapter"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(adapter_dir))
            tokenizer.save_pretrained(adapter_dir)
            effective_examples = (
                step * args.per_device_batch_size * args.gradient_accumulation
            )
            bucket_count = len(dataset.bucket_indices)
            per_bucket = (
                effective_examples // bucket_count
                if args.sampler_mode == "balanced" and effective_examples % bucket_count == 0
                else None
            )
            manifest = {
                "optimizer_step": step,
                "effective_examples": effective_examples,
                "sampler_mode": args.sampler_mode,
                "task_bucket_count": bucket_count,
                "examples_per_bucket": per_bucket,
                "de_novo_examples": per_bucket * 6 if per_bucket is not None else None,
                "editing_examples": per_bucket * 7 if per_bucket is not None else None,
                "adapter": str(adapter_dir),
                "evaluation_only": True,
                "optimizer_state_preserved": False,
            }
            (adapter_dir.parent / "milestone_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            return control

    trainer = BalancedTrainer(
        model=model, args=training_args, train_dataset=dataset,
        data_collator=IndexedCompletionCollator(tokenizer),
        callbacks=[FiniteTrainingCallback(), MilestoneAdapterCallback()],
    )
    resume: bool | str = False
    if args.resume_from_checkpoint:
        checkpoints = sorted(
            args.output_dir.glob("checkpoint-*"),
            key=lambda path: int(path.name.rsplit("-", 1)[-1]),
        )
        resume = str(checkpoints[-1]) if checkpoints else False
    result = trainer.train(resume_from_checkpoint=resume)
    nonfinite = common.adapter_nonfinite_count(model)
    if nonfinite:
        raise FloatingPointError(f"non-finite trainable adapter parameters: {nonfinite}")
    adapter = args.output_dir / "adapter"
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(adapter)
    summary = {
        "protocol": "molprogram_indexed_sft_v1",
        "base_model": args.base_model,
        "input_adapter": str(args.input_adapter) if args.input_adapter is not None else None,
        "fresh_lora_from_base": args.fresh_lora,
        "lora": {
            "rank": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
        },
        "loader_kind": loader_kind, "train_rows": len(dataset),
        "task_bucket_rows": {
            key: len(value) for key, value in sorted(dataset.bucket_indices.items())
        },
        "sampler_mode": args.sampler_mode,
        "sampler_rows_per_epoch": sampler_rows,
        "sampler_expected_optimizer_steps_per_epoch": expected_optimizer_steps,
        "sampler_exact_no_replacement": args.sampler_mode == "proportional_one_pass",
        "sampler_interleave": (
            "smooth proportional merge of independently shuffled task buckets"
            if args.sampler_mode == "proportional_one_pass"
            else "equal task round robin"
        ),
        "balanced_sampler_rows_per_task": balanced_rows_per_task,
        "balanced_sampler_dropped_rows_per_task": (
            min(map(len, dataset.bucket_indices.values())) % args.per_device_batch_size
        ),
        "sampler_physical_batches_task_homogeneous": args.sampler_mode == "balanced",
        "max_steps": args.max_steps,
        "num_train_epochs": args.num_train_epochs,
        "per_device_batch_size": args.per_device_batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "effective_batch_size": args.per_device_batch_size * args.gradient_accumulation,
        "effective_examples": (
            args.max_steps * args.per_device_batch_size * args.gradient_accumulation
            if args.max_steps is not None else sampler_rows
        ),
        "formal_one_pass_contract": bool(
            args.sampler_mode == "proportional_one_pass"
            and args.max_steps is None
            and args.num_train_epochs == 1.0
        ),
        "learning_rate": args.learning_rate, "resume_checkpoint": resume,
        "finite_guard_each_optimizer_step": True,
        "guard_every_microbatch": args.guard_every_microbatch,
        "milestone_steps": milestone_steps,
        "milestone_root": str(milestone_root),
        "adapter_nonfinite_parameters": nonfinite,
        "train_metrics": dict(result.metrics), "adapter": str(adapter),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "TRAINING_COMPLETE").write_text(
        hashlib_sha256(adapter / "adapter_model.safetensors") + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def hashlib_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
