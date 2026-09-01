#!/usr/bin/env python3
"""Train one shared policy with mode-balanced, target-blind Raw@1 GRPO."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCRIPT_DIR.parent / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from molprogram.safe_grpo import (  # noqa: E402
    CHANNEL_WEIGHTS,
    balanced_bucket,
    decoupled_advantages,
    equal_norm_bisector,
    group_record,
    reward_channels,
    scalar_rewards,
    select_balanced_pairs,
)
from molprogram.support_audit import validate_rl_authorization  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def chosen_sft_loss(model, tokenizer, messages: Sequence[Mapping[str, str]]):
    import torch

    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    answer = str(messages[-1]["content"]) + (tokenizer.eos_token or "")
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    answer_ids = tokenizer(
        answer, add_special_tokens=False, return_tensors="pt"
    )["input_ids"]
    ids = torch.cat((prompt_ids, answer_ids), dim=1).to(model.device)
    labels = ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100
    return model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels).loss.float()


def generate_group(
    model,
    tokenizer,
    messages,
    *,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
):
    import torch

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    offset = encoded["input_ids"].shape[1]
    torch.manual_seed(int(seed))
    model.eval()
    model.config.use_cache = True
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            num_return_sequences=int(group_size),
            pad_token_id=tokenizer.pad_token_id,
        )
    model.config.use_cache = False
    return encoded["input_ids"].detach().cpu(), [
        tokenizer.decode(ids[offset:], skip_special_tokens=True).strip()
        for ids in generated
    ]


def completion_token_logprobs(model, tokenizer, prompt_ids, candidate: str):
    import torch

    suffix = str(candidate) + (tokenizer.eos_token or "")
    answer_ids = tokenizer(
        suffix, add_special_tokens=False, return_tensors="pt"
    )["input_ids"]
    if answer_ids.numel() == 0:
        answer_ids = torch.tensor([[int(tokenizer.eos_token_id)]], dtype=torch.long)
    ids = torch.cat((prompt_ids.to(dtype=torch.long), answer_ids), dim=1).to(model.device)
    logits = model(
        input_ids=ids, attention_mask=torch.ones_like(ids)
    ).logits[:, :-1].float()
    targets = ids[:, 1:]
    token_logprobs = torch.log_softmax(logits, dim=-1).gather(
        -1, targets.unsqueeze(-1)
    ).squeeze(-1)
    positions = torch.arange(targets.shape[1], device=ids.device) + 1
    return token_logprobs[0, positions.ge(prompt_ids.shape[1])]


def reference_kl_loss(model, tokenizer, prompt_ids, candidate: str):
    """Return policy log-probability and a non-negative sampled KL estimate."""
    import torch

    model.set_adapter("reference")
    with torch.no_grad():
        reference = completion_token_logprobs(
            model, tokenizer, prompt_ids, candidate
        ).detach()
    model.set_adapter("default")
    policy = completion_token_logprobs(model, tokenizer, prompt_ids, candidate)
    log_ratio = policy - reference
    # Schulman's k3 estimator of KL(policy || reference).
    kl = (torch.exp(-log_ratio) + log_ratio - 1.0).mean()
    return policy.mean(), kl


def capture_gradients(parameters):
    import torch

    return [
        parameter.grad.detach().clone()
        if parameter.grad is not None
        else torch.zeros_like(parameter)
        for parameter in parameters
    ]


def complete_checkpoints(output_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in output_dir.glob("checkpoint-*")
        if (path / "CHECKPOINT_COMPLETE").is_file()
    )


def save_policy(model, path: Path) -> None:
    model.save_pretrained(path, selected_adapters=["default"])


def save_checkpoint(
    model, tokenizer, optimizer, output_dir: Path, next_step: int, history
) -> Path:
    import torch

    checkpoint = output_dir / f"checkpoint-{next_step:03d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    save_policy(model, checkpoint / "adapter")
    tokenizer.save_pretrained(checkpoint / "adapter")
    torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
    (checkpoint / "state.json").write_text(
        json.dumps({"next_step": next_step, "history": history}, indent=2, sort_keys=True)
        + "\n"
    )
    (checkpoint / "CHECKPOINT_COMPLETE").touch()
    return checkpoint


def backward_mode(
    model,
    tokenizer,
    row,
    prompt_ids,
    candidates,
    advantages,
    *,
    anchor_weight: float,
    reference_kl_weight: float,
):
    model.eval()
    policy_loss = 0.0
    kl_values: list[float] = []
    for candidate, advantage in zip(candidates, advantages):
        mean_logprob, kl = reference_kl_loss(
            model, tokenizer, prompt_ids, candidate
        )
        loss = (
            -float(advantage) * mean_logprob / len(candidates)
            + float(reference_kl_weight) * kl / len(candidates)
        )
        loss.backward()
        policy_loss += float(loss.detach())
        kl_values.append(float(kl.detach()))
    model.set_adapter("default")
    model.train()
    anchor = chosen_sft_loss(model, tokenizer, list(row["messages"]))
    (float(anchor_weight) * anchor).backward()
    return {
        "policy_plus_kl_loss": policy_loss,
        "reference_kl_mean": sum(kl_values) / max(len(kl_values), 1),
        "sft_anchor_loss": float(anchor.detach()),
    }


def load_policy(base_model: str, adapter_source: Path, reference_adapter: Path):
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
    model.load_adapter(reference_adapter, adapter_name="reference", is_trainable=False)
    model.set_adapter("default")
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
    parser.add_argument("--editing-support-report", required=True, type=Path)
    parser.add_argument("--paired-steps", type=int, default=30)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=1.5e-7)
    parser.add_argument("--denovo-anchor-weight", type=float, default=1.5)
    parser.add_argument("--edit-anchor-weight", type=float, default=1.5)
    parser.add_argument("--reference-kl-weight", type=float, default=0.10)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=37001)
    args = parser.parse_args(argv)

    support_bytes = args.editing_support_report.read_bytes()
    support_report = json.loads(support_bytes)
    support_decision = validate_rl_authorization(support_report)
    support_sha256 = hashlib.sha256(support_bytes).hexdigest()

    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("safe joint GRPO requires BF16 CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    schedule = select_balanced_pairs(
        read_jsonl(args.train_jsonl), args.paired_steps, args.seed
    )
    checkpoints = complete_checkpoints(args.output_dir)
    start_step, history = 0, []
    adapter_source = args.input_adapter
    resume_checkpoint = checkpoints[-1] if checkpoints else None
    if resume_checkpoint is not None:
        state = json.loads((resume_checkpoint / "state.json").read_text())
        start_step = int(state["next_step"])
        history = list(state["history"])
        adapter_source = resume_checkpoint / "adapter"

    model, tokenizer, loader_kind = load_policy(
        args.base_model, adapter_source, args.input_adapter
    )
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
        totals["common_descent_steps"] += int(record["common_descent"])
        totals["gradient_conflicts"] += int(record["gradient_conflict"])
        totals["de_novo_zero_signal_groups"] += int(
            record["de_novo"]["advantage"]["zero_signal"]
        )
        totals["edit_zero_signal_groups"] += int(
            record["edit"]["advantage"]["zero_signal"]
        )
    live_log = args.output_dir / "training_history.live.jsonl"
    for step in range(start_step, len(schedule)):
        denovo_row, edit_row = schedule[step]
        sampled_groups = []
        for mode_index, row in enumerate((denovo_row, edit_row)):
            model.set_adapter("default")
            prompt_ids, candidates = generate_group(
                model,
                tokenizer,
                list(row["messages"][:-1]),
                group_size=args.group_size,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed * 10000 + step * 2 + mode_index,
            )
            scored = [reward_channels(row, candidate) for candidate in candidates]
            channels = [item[0] for item in scored]
            details = [item[1] for item in scored]
            mode = str(row["task_mode"])
            advantages, advantage_record = decoupled_advantages(
                channels, CHANNEL_WEIGHTS[mode]
            )
            rewards = scalar_rewards(channels, CHANNEL_WEIGHTS[mode])
            sampled_groups.append(
                (
                    row,
                    prompt_ids,
                    candidates,
                    rewards,
                    details,
                    advantages,
                    advantage_record,
                )
            )

        mode_gradients = []
        mode_losses = []
        for mode_index, group in enumerate(sampled_groups):
            row, prompt_ids, candidates, _, _, advantages, _ = group
            optimizer.zero_grad(set_to_none=True)
            mode_losses.append(
                backward_mode(
                    model,
                    tokenizer,
                    row,
                    prompt_ids,
                    candidates,
                    advantages,
                    anchor_weight=(
                        args.denovo_anchor_weight
                        if mode_index == 0
                        else args.edit_anchor_weight
                    ),
                    reference_kl_weight=args.reference_kl_weight,
                )
            )
            mode_gradients.append(capture_gradients(trainable))

        merged, gradient_record = equal_norm_bisector(
            mode_gradients[0], mode_gradients[1]
        )
        optimizer.zero_grad(set_to_none=True)
        for parameter, gradient in zip(trainable, merged):
            parameter.grad = gradient
        unclipped_norm = torch.nn.utils.clip_grad_norm_(
            trainable, float(args.grad_clip)
        )
        optimizer.step()

        record = {
            "step": step,
            "de_novo": group_record(
                denovo_row,
                sampled_groups[0][3],
                sampled_groups[0][4],
                sampled_groups[0][6],
            ),
            "edit": group_record(
                edit_row,
                sampled_groups[1][3],
                sampled_groups[1][4],
                sampled_groups[1][6],
            ),
            "de_novo_loss": mode_losses[0],
            "edit_loss": mode_losses[1],
            **gradient_record,
            "unclipped_gradient_norm": float(unclipped_norm),
        }
        history.append(record)
        totals["paired_steps"] += 1
        totals["common_descent_steps"] += int(record["common_descent"])
        totals["gradient_conflicts"] += int(record["gradient_conflict"])
        totals["de_novo_zero_signal_groups"] += int(
            sampled_groups[0][6]["zero_signal"]
        )
        totals["edit_zero_signal_groups"] += int(
            sampled_groups[1][6]["zero_signal"]
        )
        with live_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps({"stage": "paired_step", **record}, sort_keys=True), flush=True)
        next_step = step + 1
        if (
            next_step % args.checkpoint_every == 0
            or next_step == len(schedule)
        ):
            checkpoint = save_checkpoint(
                model, tokenizer, optimizer, args.output_dir, next_step, history
            )
            print(
                json.dumps({"stage": "checkpoint", "path": str(checkpoint)}),
                flush=True,
            )

    model.set_adapter("default")
    nonfinite = sum(
        int((~torch.isfinite(parameter)).sum().item()) for parameter in trainable
    )
    if nonfinite:
        raise FloatingPointError(f"non-finite trainable parameters: {nonfinite}")
    adapter = args.output_dir / "adapter"
    save_policy(model, adapter)
    tokenizer.save_pretrained(adapter)
    bucket_counts = Counter()
    for denovo_row, edit_row in schedule:
        bucket_counts[balanced_bucket(denovo_row)] += 1
        bucket_counts[balanced_bucket(edit_row)] += 1
    summary = {
        "protocol": "molprogram_safe_joint_raw1_grpo_v1",
        "loader_kind": loader_kind,
        "base_model": args.base_model,
        "input_adapter": str(args.input_adapter),
        "output_adapter": str(adapter),
        "paired_steps": len(schedule),
        "group_size": args.group_size,
        "max_new_tokens": args.max_new_tokens,
        "sampling": {"temperature": args.temperature, "top_p": args.top_p},
        "training_group_only": True,
        "evaluation_contract": "Raw@1; no reranking and no best-of-K",
        "learning_rate": args.learning_rate,
        "channel_weights": CHANNEL_WEIGHTS,
        "advantage": "per_prompt_per_channel_group_zscore",
        "gradient_merge": "equal_norm_bisector",
        "denovo_anchor_weight": args.denovo_anchor_weight,
        "edit_anchor_weight": args.edit_anchor_weight,
        "reference_kl_weight": args.reference_kl_weight,
        "reference_kl": "token_level_sampled_k3",
        "editing_support_report": str(args.editing_support_report),
        "editing_support_report_sha256": support_sha256,
        "editing_support_decision": support_decision,
        "reward_target_smiles_access": False,
        "sft_anchor_uses_training_positive": True,
        "bucket_group_counts": dict(sorted(bucket_counts.items())),
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
