#!/usr/bin/env python3
"""Expand a frozen rank-16 LoRA and train only four residual rank slices."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PUBLIC_SCRIPTS = REPO_ROOT / "scripts"
for path in (PUBLIC_SCRIPTS, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import train_sft as common  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def saved_key_candidates(parameter_name: str) -> tuple[str, ...]:
    normalized = parameter_name.replace(".default.weight", ".weight")
    candidates = [normalized]
    for prefix in ("base_model.model.", "base_model."):
        if normalized.startswith(prefix):
            candidates.append(normalized[len(prefix) :])
    return tuple(dict.fromkeys(candidates))


def source_tensor_for(
    parameter_name: str, source: Mapping[str, object]
) -> tuple[str, object]:
    for candidate in saved_key_candidates(parameter_name):
        if candidate in source:
            return candidate, source[candidate]
    suffix = parameter_name.replace(".default.weight", ".weight")
    matches = [
        key
        for key in source
        if suffix.endswith(str(key)) or str(key).endswith(suffix)
    ]
    if len(matches) == 1:
        return matches[0], source[matches[0]]
    raise KeyError(f"cannot uniquely map adapter parameter {parameter_name}")


def copy_shared_slice(destination, source, *, kind: str, shared_rank: int) -> None:
    if kind == "A":
        if tuple(source.shape) != (shared_rank, destination.shape[1]):
            raise ValueError(f"LoRA-A shape mismatch: {source.shape} -> {destination.shape}")
        destination[:shared_rank].copy_(source.to(destination))
    elif kind == "B":
        if tuple(source.shape) != (destination.shape[0], shared_rank):
            raise ValueError(f"LoRA-B shape mismatch: {source.shape} -> {destination.shape}")
        destination[:, :shared_rank].copy_(source.to(destination))
    else:
        raise ValueError(f"unsupported LoRA tensor kind: {kind}")


def freeze_shared_gradient(gradient, *, kind: str, shared_rank: int):
    result = gradient.clone()
    if kind == "A":
        result[:shared_rank].zero_()
    elif kind == "B":
        result[:, :shared_rank].zero_()
    else:
        raise ValueError(f"unsupported LoRA tensor kind: {kind}")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--shared-adapter-dir", required=True, type=Path)
    parser.add_argument("--conditional-layout", required=True, type=Path)
    parser.add_argument("--baseline-output-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shared-rank", type=int, default=16)
    parser.add_argument("--residual-rank", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=448)
    parser.add_argument("--gradient-accumulation", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--seed", type=int, default=33501)
    parser.add_argument("--expected-rows", type=int, default=1920)
    args = parser.parse_args(argv)

    import peft
    import torch
    import transformers
    from safetensors.torch import load_file

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("conditional residual training requires BF16 CUDA")
    source_weights = args.shared_adapter_dir / "adapter_model.safetensors"
    source_config_path = args.shared_adapter_dir / "adapter_config.json"
    if not source_weights.is_file() or not source_config_path.is_file():
        raise FileNotFoundError(f"incomplete shared adapter: {args.shared_adapter_dir}")
    source_config = json.loads(source_config_path.read_text())
    if int(source_config["r"]) != args.shared_rank:
        raise ValueError("shared adapter rank does not match --shared-rank")
    source_scale = float(source_config["lora_alpha"]) / int(source_config["r"])
    rank = args.shared_rank + args.residual_rank
    alpha = int(round(source_scale * rank))
    if alpha / rank != source_scale:
        raise ValueError("expanded LoRA cannot preserve the source scaling exactly")

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
    model = loader.from_pretrained(
        args.base_model,
        config=config,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = peft.get_peft_model(
        model,
        peft.LoraConfig(
            task_type=peft.TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=float(source_config.get("lora_dropout", 0.05)),
            bias="none",
            target_modules=sorted(source_config["target_modules"]),
        ),
    )
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()

    source_state = load_file(str(source_weights), device="cpu")
    copied_keys: set[str] = set()
    routed_parameters = []
    residual_parameters = 0
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if ".lora_A." in name:
                kind = "A"
                residual_parameters += args.residual_rank * parameter.shape[1]
            elif ".lora_B." in name:
                kind = "B"
                residual_parameters += parameter.shape[0] * args.residual_rank
            else:
                raise ValueError(f"unexpected trainable parameter: {name}")
            key, source_tensor = source_tensor_for(name, source_state)
            copy_shared_slice(
                parameter, source_tensor, kind=kind, shared_rank=args.shared_rank
            )
            copied_keys.add(key)
            routed_parameters.append((name, parameter, kind, source_tensor))
    if copied_keys != set(source_state):
        missing = sorted(set(source_state) - copied_keys)
        raise ValueError(f"source adapter tensors were not copied: {missing[:5]}")

    args.baseline_output_dir.mkdir(parents=True, exist_ok=True)
    baseline_adapter = args.baseline_output_dir / "adapter"
    model.save_pretrained(str(baseline_adapter), safe_serialization=True)
    tokenizer.save_pretrained(baseline_adapter)
    shutil.copyfile(args.conditional_layout, baseline_adapter / "program_routing.json")

    for _name, parameter, kind, _source in routed_parameters:
        parameter.register_hook(
            lambda gradient, kind=kind: freeze_shared_gradient(
                gradient, kind=kind, shared_rank=args.shared_rank
            )
        )

    rows = common.read_jsonl(args.train_jsonl)
    if len(rows) != args.expected_rows:
        raise ValueError(f"expected {args.expected_rows} rows, found {len(rows)}")
    dataset = common.ChatDataset(rows, tokenizer, args.max_length)
    if len(dataset) != len(rows):
        raise ValueError(f"tokenized {len(dataset)} of {len(rows)} training rows")
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
    training_args.weight_decay = 0.0
    training_args.save_strategy = "no"
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=common.CompletionCollator(tokenizer),
    )
    result = trainer.train()

    shared_max_delta = 0.0
    with torch.no_grad():
        for _name, parameter, kind, source_tensor in routed_parameters:
            shared = (
                parameter[: args.shared_rank]
                if kind == "A"
                else parameter[:, : args.shared_rank]
            )
            delta = (shared.detach().cpu() - source_tensor.float()).abs().max().item()
            shared_max_delta = max(shared_max_delta, float(delta))
    if shared_max_delta != 0.0:
        raise FloatingPointError(
            f"frozen shared LoRA slices changed by {shared_max_delta}"
        )
    nonfinite = common.adapter_nonfinite_count(model)
    if nonfinite:
        raise FloatingPointError(f"adapter has {nonfinite} non-finite values")

    adapter = args.output_dir / "adapter"
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(adapter)
    shutil.copyfile(args.conditional_layout, adapter / "program_routing.json")
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    summary = {
        "protocol": "property_conditional_residual_rank4_pilot_v1",
        "base_model": args.base_model,
        "loader_kind": loader_kind,
        "source_adapter": str(args.shared_adapter_dir.resolve()),
        "source_adapter_sha256": sha256(source_weights),
        "train_jsonl": str(args.train_jsonl.resolve()),
        "train_rows": len(dataset),
        "epochs": args.epochs,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "weight_decay": 0.0,
        "shared_rank": args.shared_rank,
        "residual_rank": args.residual_rank,
        "expanded_rank": rank,
        "lora_alpha": alpha,
        "lora_scaling": alpha / rank,
        "trainable_tensor_parameters": trainable_parameters,
        "effectively_updated_residual_parameters": residual_parameters,
        "shared_slice_max_abs_delta": shared_max_delta,
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
