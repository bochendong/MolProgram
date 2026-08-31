#!/usr/bin/env python3
"""Run a compute-matched positive or structured-negative refinement arm."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PUBLIC_SCRIPTS = REPO_ROOT / "scripts"
if str(PUBLIC_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PUBLIC_SCRIPTS))
import train_sft as common  # noqa: E402


SEMANTIC_TYPES = {
    "source_copy",
    "condition_mismatch",
    "opposite_program_target",
    "partial_program_target",
}
SYNTAX_TYPE = "invalid_corruption"


def active_weight(arm: str, negative_type: str, registered_weight: float) -> float:
    if arm == "positive_only":
        return 0.0
    if arm == "semantic_only":
        return registered_weight if negative_type in SEMANTIC_TYPES else 0.0
    if arm == "semantic_plus_syntax":
        return registered_weight
    raise ValueError(f"unsupported arm: {arm}")


def encode_completion(tokenizer, messages, max_length: int) -> dict[str, list[int]]:
    full = common.input_id_list(
        tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
    )
    prompt = common.input_id_list(
        tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True
        )
    )
    eos = tokenizer.eos_token_id
    if eos is not None and (not full or full[-1] != eos):
        full.append(int(eos))
    full = full[:max_length]
    mask_length = min(common.common_prefix_length(full, prompt), len(full))
    labels = [-100] * mask_length + full[mask_length:]
    if not any(value != -100 for value in labels):
        raise ValueError("completion was truncated away")
    return {"input_ids": full, "attention_mask": [1] * len(full), "labels": labels}


class PairDataset:
    def __init__(
        self,
        rows: Sequence[Mapping[str, object]],
        tokenizer,
        max_length: int,
        arm: str,
    ):
        self.examples = []
        for row in rows:
            messages = list(row["messages"])
            chosen = encode_completion(tokenizer, messages, max_length)
            rejected_messages = [
                *messages[:-1],
                {"role": "assistant", "content": str(row["rejected_assistant"])},
            ]
            rejected = encode_completion(tokenizer, rejected_messages, max_length)
            negative_type = str(row["negative_type"])
            self.examples.append(
                {
                    **chosen,
                    "rejected_input_ids": rejected["input_ids"],
                    "rejected_attention_mask": rejected["attention_mask"],
                    "rejected_labels": rejected["labels"],
                    "chosen_ce_weight": float(row["chosen_ce_weight"]),
                    "negative_weight": active_weight(
                        arm, negative_type, float(row["negative_weight"])
                    ),
                    "margin": float(row["margin"]),
                }
            )
        if not self.examples:
            raise ValueError("no contrastive pairs")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class PairCollator:
    def __init__(self, tokenizer):
        self.pad = int(tokenizer.pad_token_id)

    def _pad(self, rows, prefix: str):
        import torch

        keys = tuple(
            f"{prefix}{suffix}"
            for suffix in ("input_ids", "attention_mask", "labels")
        )
        width = max(len(row[keys[0]]) for row in rows)
        return {
            keys[0]: torch.tensor(
                [
                    row[keys[0]] + [self.pad] * (width - len(row[keys[0]]))
                    for row in rows
                ],
                dtype=torch.long,
            ),
            keys[1]: torch.tensor(
                [
                    row[keys[1]] + [0] * (width - len(row[keys[1]]))
                    for row in rows
                ],
                dtype=torch.long,
            ),
            keys[2]: torch.tensor(
                [
                    row[keys[2]] + [-100] * (width - len(row[keys[2]]))
                    for row in rows
                ],
                dtype=torch.long,
            ),
        }

    def __call__(self, rows):
        import torch

        batch = self._pad(rows, "")
        batch.update(self._pad(rows, "rejected_"))
        for key in ("chosen_ce_weight", "negative_weight", "margin"):
            batch[key] = torch.tensor(
                [row[key] for row in rows], dtype=torch.float32
            )
        return batch


def per_example_nll(logits, labels):
    import torch

    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    valid = shift_labels.ne(-100)
    token_loss = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)
    return (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--input-adapter", required=True, type=Path)
    parser.add_argument(
        "--arm",
        choices=("positive_only", "semantic_only", "semantic_plus_syntax"),
        required=True,
    )
    parser.add_argument("--epochs", type=float, default=0.25)
    parser.add_argument("--max-length", type=int, default=448)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=2323)
    args = parser.parse_args(argv)

    import peft
    import torch
    import transformers

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("negative-refinement training requires BF16 CUDA")
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
        raise TypeError(f"unsupported config: {type(config).__name__}")
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
    model = peft.PeftModel.from_pretrained(
        base, args.input_adapter, is_trainable=True
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()

    raw_rows = common.read_jsonl(args.train_jsonl)
    dataset = PairDataset(raw_rows, tokenizer, args.max_length, args.arm)
    training_ns = argparse.Namespace(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=1,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        logging_steps=20,
        seed=args.seed,
    )

    class RefinementTrainer(transformers.Trainer):
        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            ce_weight = inputs.pop("chosen_ce_weight")
            negative_weight = inputs.pop("negative_weight")
            margin = inputs.pop("margin")
            rejected = {
                "input_ids": inputs.pop("rejected_input_ids"),
                "attention_mask": inputs.pop("rejected_attention_mask"),
                "labels": inputs.pop("rejected_labels"),
            }
            chosen_out = model(**inputs)
            chosen_nll = per_example_nll(chosen_out.logits, inputs["labels"])
            rejected_out = model(**rejected)
            rejected_nll = per_example_nll(
                rejected_out.logits, rejected["labels"]
            )
            device = chosen_nll.device
            loss = (
                chosen_nll * ce_weight.to(device)
                + torch.relu(
                    margin.to(device) + chosen_nll - rejected_nll
                )
                * negative_weight.to(device)
            ).mean()
            return (loss, chosen_out) if return_outputs else loss

    trainer = RefinementTrainer(
        model=model,
        args=common.training_arguments(
            transformers, training_ns, compute_dtype="bfloat16"
        ),
        train_dataset=dataset,
        data_collator=PairCollator(tokenizer),
    )
    result = trainer.train()
    nonfinite = common.adapter_nonfinite_count(model)
    if nonfinite:
        raise FloatingPointError(
            f"non-finite trainable adapter parameters: {nonfinite}"
        )
    adapter = args.output_dir / "adapter"
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(adapter)
    counts = Counter(str(row["negative_type"]) for row in raw_rows)
    active_counts = Counter(
        str(row["negative_type"])
        for row in raw_rows
        if active_weight(
            args.arm,
            str(row["negative_type"]),
            float(row["negative_weight"]),
        )
        > 0
    )
    summary = {
        "protocol": "molprogram_negative_refinement_matched_v1",
        "arm": args.arm,
        "base_model": args.base_model,
        "input_adapter": str(args.input_adapter),
        "loader_kind": loader_kind,
        "logical_pair_instances": len(dataset),
        "registered_negative_instances": dict(sorted(counts.items())),
        "active_negative_instances": dict(sorted(active_counts.items())),
        "unique_example_ids": len({row["example_id"] for row in raw_rows}),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "gradient_accumulation": args.gradient_accumulation,
        "chosen_ce_weight_per_unique_row": 1.0,
        "compute_matched_rejected_forwards": True,
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
