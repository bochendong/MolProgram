#!/usr/bin/env python3
"""Target-blind group-relative RL for the unified MolProgram policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCRIPT_DIR.parent / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from molprogram import protocol  # noqa: E402
from molprogram.rewards import hard_boundary_reward  # noqa: E402
from molprogram.scoring import score_response  # noqa: E402
from molprogram.support_audit import validate_rl_authorization  # noqa: E402


TARGET_EDIT_TASKS = frozenset(protocol.TABLE1_TASK_KEYS.values())
TARGET_BUCKETS = (
    "de_novo:5p", "de_novo:6p", "de_novo:7p",
    *(f"edit:{task}" for task in sorted(TARGET_EDIT_TASKS)),
)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prompt_payload(row: Mapping[str, object]) -> dict[str, object]:
    for message in list(row["messages"]):
        if str(message.get("role")) == "user":
            payload = json.loads(str(message["content"]))
            if isinstance(payload, dict):
                return payload
    raise ValueError(f"missing structured user payload for {row.get('example_id')}")


def property_count(row: Mapping[str, object]) -> int:
    conditions = prompt_payload(row).get("conditions", [])
    return len(conditions) if isinstance(conditions, list) else 0


def target_bucket(row: Mapping[str, object]) -> str:
    mode = str(row.get("task_mode", ""))
    if mode == "de_novo":
        count = property_count(row)
        return f"de_novo:{count}p" if count in {5, 6, 7} else ""
    if mode == "edit":
        task = str(row.get("task_key", ""))
        return f"edit:{task}" if task in TARGET_EDIT_TASKS else ""
    return ""


def stable_key(row: Mapping[str, object], seed: int) -> str:
    identity = row.get("example_id", row.get("sample_id", row.get("condition_id", "")))
    return hashlib.sha256(f"{seed}:{identity}".encode()).hexdigest()


def select_training_rows(
    rows: Sequence[dict[str, object]], rounds: int, seed: int
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        bucket = target_bucket(row)
        if bucket:
            grouped[bucket].append(row)
    missing = [bucket for bucket in TARGET_BUCKETS if len(grouped[bucket]) < rounds]
    if missing:
        raise ValueError(f"insufficient targeted training rows: {missing}")
    for bucket, values in grouped.items():
        values.sort(key=lambda item: stable_key(item, seed))
    selected: list[dict[str, object]] = []
    for round_index in range(rounds):
        buckets = list(TARGET_BUCKETS)
        random.Random(seed + round_index).shuffle(buckets)
        selected.extend(grouped[bucket][round_index] for bucket in buckets)
    return selected


def scorer_row(payload: Mapping[str, object], mode: str) -> dict[str, str]:
    source = str(payload.get("source", "") or "")
    if source == "<EMPTY>":
        source = ""
    conditions = payload.get("conditions", [])
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("property program is missing conditions")
    result: dict[str, str] = {"source_smiles": source, "task_mode": mode}
    selected: list[str] = []
    instruction: list[dict[str, str]] = []
    for item in conditions:
        if not isinstance(item, Mapping):
            continue
        prop = str(item.get("property", "") or "")
        if prop not in protocol.PROPERTIES:
            continue
        goal = item.get("goal")
        selected.append(prop)
        result[f"{prop}_active"] = "True"
        if isinstance(goal, str) and goal in {"increase", "decrease", "preserve"}:
            result[f"{prop}_direction"] = goal
            instruction.append({"property": prop, "direction": goal})
        elif isinstance(goal, Mapping) and "around" in goal:
            result[f"target_{prop}"] = str(float(goal["around"]))
    if not selected:
        raise ValueError("property program has no supported properties")
    result["condition_properties"] = ",".join(selected)
    if instruction:
        result["instruction_tasks"] = json.dumps(instruction, separators=(",", ":"))
    return result


def reward_response(
    row: Mapping[str, object], raw: str, editing_reward_mode: str = "hard_boundary"
) -> tuple[float, dict[str, object]]:
    """Score only prompt-visible conditions and source; never read a target molecule."""
    soft_reward, details = score_response(row, raw)
    if str(row.get("task_mode", "")) == "edit":
        if editing_reward_mode == "hard_boundary":
            return hard_boundary_reward({}, details, "edit"), details
        if editing_reward_mode != "soft":
            raise ValueError(f"unsupported editing reward mode: {editing_reward_mode}")
    return soft_reward, details


def group_advantages(rewards: Sequence[float], clip: float = 3.0) -> list[float]:
    center = sum(float(value) for value in rewards) / max(len(rewards), 1)
    variance = sum((float(value) - center) ** 2 for value in rewards) / max(len(rewards), 1)
    if variance < 1e-12:
        return [0.0 for _ in rewards]
    scale = variance**0.5
    return [max(-clip, min(clip, (float(value) - center) / scale)) for value in rewards]


def completion_mean_logprob(model, tokenizer, prompt_ids, answer: str):
    import torch

    suffix = str(answer) + (tokenizer.eos_token or "")
    answer_ids = tokenizer(suffix, add_special_tokens=False, return_tensors="pt")["input_ids"]
    if answer_ids.numel() == 0:
        answer_ids = torch.tensor([[int(tokenizer.eos_token_id)]], dtype=torch.long)
    ids = torch.cat((prompt_ids.to(dtype=torch.long), answer_ids), dim=1).to(model.device)
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[:, :-1].float()
    targets = ids[:, 1:]
    token_logprob = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    positions = torch.arange(targets.shape[1], device=ids.device) + 1
    mask = positions.ge(prompt_ids.shape[1]).to(token_logprob.dtype).unsqueeze(0)
    return (token_logprob * mask).sum() / mask.sum().clamp_min(1.0)


def chosen_sft_loss(model, tokenizer, messages: Sequence[Mapping[str, str]]):
    import torch

    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    answer = str(messages[-1]["content"]) + (tokenizer.eos_token or "")
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False, return_tensors="pt")["input_ids"]
    ids = torch.cat((prompt_ids, answer_ids), dim=1).to(model.device)
    labels = ids.clone()
    labels[:, : prompt_ids.shape[1]] = -100
    return model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels).loss.float()


def generate_group(
    model, tokenizer, messages, group_size: int, max_new_tokens: int,
    temperature: float, top_p: float, seed: int,
):
    import torch

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    offset = encoded["input_ids"].shape[1]
    torch.manual_seed(int(seed))
    model.eval()
    model.config.use_cache = True
    with torch.no_grad():
        generated = model.generate(
            **encoded, max_new_tokens=int(max_new_tokens), do_sample=True,
            temperature=float(temperature), top_p=float(top_p),
            num_return_sequences=int(group_size), pad_token_id=tokenizer.pad_token_id,
        )
    model.config.use_cache = False
    return encoded["input_ids"].detach().cpu(), [
        tokenizer.decode(ids[offset:], skip_special_tokens=True).strip() for ids in generated
    ]


def initial_adapter_penalty(trainable_named, initial_parameters):
    import torch

    terms = [
        (parameter.float() - initial_parameters[name]).pow(2).mean()
        for name, parameter in trainable_named
    ]
    return torch.stack(terms).mean()


def complete_checkpoints(output_dir: Path) -> list[Path]:
    return sorted(
        path for path in output_dir.glob("checkpoint-*")
        if (path / "CHECKPOINT_COMPLETE").is_file()
    )


def save_checkpoint(model, tokenizer, optimizer, output_dir: Path, next_index: int, history) -> Path:
    import torch

    checkpoint = output_dir / f"checkpoint-{next_index:03d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint / "adapter")
    tokenizer.save_pretrained(checkpoint / "adapter")
    torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
    (checkpoint / "state.json").write_text(json.dumps({
        "next_index": next_index, "history": history,
    }, indent=2, sort_keys=True) + "\n")
    (checkpoint / "CHECKPOINT_COMPLETE").touch()
    return checkpoint


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--input-adapter", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--editing-support-report", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--sft-anchor-weight", type=float, default=1.0)
    parser.add_argument("--initial-adapter-weight", type=float, default=0.05)
    parser.add_argument(
        "--editing-reward-mode",
        choices=("hard_boundary", "soft"),
        default="hard_boundary",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=13)
    parser.add_argument("--seed", type=int, default=2525)
    args = parser.parse_args(argv)

    support_report_bytes = args.editing_support_report.read_bytes()
    support_report = json.loads(support_report_bytes)
    support_decision = validate_rl_authorization(support_report)
    support_report_sha256 = hashlib.sha256(support_report_bytes).hexdigest()

    import peft
    import torch
    import transformers

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("MolProgram RL requires BF16 CUDA")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = select_training_rows(read_jsonl(args.train_jsonl), args.rounds, args.seed)
    checkpoints = complete_checkpoints(args.output_dir)
    start_index, history = 0, []
    adapter_source = args.input_adapter
    resume_checkpoint = checkpoints[-1] if checkpoints else None
    if resume_checkpoint is not None:
        state = json.loads((resume_checkpoint / "state.json").read_text())
        start_index = int(state["next_index"])
        history = list(state["history"])
        adapter_source = resume_checkpoint / "adapter"

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
    base = loader.from_pretrained(
        args.base_model, config=config, dtype=torch.bfloat16,
        low_cpu_mem_usage=True, local_files_only=True,
    )
    model = peft.PeftModel.from_pretrained(base, adapter_source, is_trainable=True).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    trainable_named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    for _, parameter in trainable_named:
        parameter.data = parameter.data.float()
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable_named],
        lr=float(args.learning_rate), weight_decay=0.0,
    )
    if resume_checkpoint is not None:
        optimizer.load_state_dict(torch.load(resume_checkpoint / "optimizer.pt", map_location="cpu"))

    initial_path = args.output_dir / "initial_adapter_parameters.pt"
    if initial_path.is_file():
        initial_parameters = torch.load(initial_path, map_location="cpu")
    else:
        if resume_checkpoint is not None:
            raise FileNotFoundError("resume checkpoint exists without initial adapter snapshot")
        initial_parameters = {
            name: parameter.detach().float().cpu().clone() for name, parameter in trainable_named
        }
        torch.save(initial_parameters, initial_path)
    missing_initial = sorted(set(name for name, _ in trainable_named) - set(initial_parameters))
    if missing_initial:
        raise ValueError(f"initial adapter snapshot misses parameters: {missing_initial[:3]}")
    initial_parameters = {
        name: value.to(model.device) for name, value in initial_parameters.items()
    }

    totals = Counter()
    live_log = args.output_dir / "training_history.live.jsonl"
    for index in range(start_index, len(selected)):
        row = selected[index]
        prompt_messages = list(row["messages"][:-1])
        prompt_ids, candidates = generate_group(
            model, tokenizer, prompt_messages, args.group_size, args.max_new_tokens,
            args.temperature, args.top_p, args.seed * 1000 + index,
        )
        scored = [
            reward_response(row, candidate, args.editing_reward_mode)
            for candidate in candidates
        ]
        rewards = [item[0] for item in scored]
        details = [item[1] for item in scored]
        advantages = group_advantages(rewards)
        optimizer.zero_grad(set_to_none=True)
        model.train()
        policy_loss_value = 0.0
        for candidate, advantage in zip(candidates, advantages):
            if advantage == 0.0:
                continue
            logprob = completion_mean_logprob(model, tokenizer, prompt_ids, candidate)
            loss = -float(advantage) * logprob / max(len(candidates), 1)
            loss.backward()
            policy_loss_value += float(loss.detach())
        anchor = chosen_sft_loss(model, tokenizer, list(row["messages"]))
        (float(args.sft_anchor_weight) * anchor).backward()
        trust_penalty = initial_adapter_penalty(trainable_named, initial_parameters)
        (float(args.initial_adapter_weight) * trust_penalty).backward()
        torch.nn.utils.clip_grad_norm_([parameter for _, parameter in trainable_named], float(args.grad_clip))
        optimizer.step()

        mode = str(row["task_mode"])
        bucket = target_bucket(row)
        mean_reward = sum(rewards) / len(rewards)
        reward_std = (sum((value - mean_reward) ** 2 for value in rewards) / len(rewards)) ** 0.5
        record = {
            "index": index, "example_id": row["example_id"], "mode": mode,
            "bucket": bucket, "property_count": property_count(row),
            "mean_reward": mean_reward, "reward_std": reward_std,
            "informative_group": reward_std >= 1e-6,
            "valid_fraction": sum(bool(item["valid"]) for item in details) / len(details),
            "property_strict_fraction": sum(bool(item["property_strict"]) for item in details) / len(details),
            "strict_fraction": sum(bool(item["strict"]) for item in details) / len(details),
            "source_feasible_fraction": sum(
                bool(item["valid"])
                and not bool(item["copy"])
                and float(item.get("source_similarity") or 0.0) >= 0.65
                for item in details
            ) / len(details) if mode == "edit" else None,
            "relaxed_fraction": sum(bool(item["relaxed"]) for item in details) / len(details),
            "policy_loss": policy_loss_value,
            "sft_anchor_loss": float(anchor.detach()),
            "initial_adapter_penalty": float(trust_penalty.detach()),
        }
        history.append(record)
        totals["groups"] += 1
        totals["informative_groups"] += int(record["informative_group"])
        with live_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"attempt": len(checkpoints), **record}, sort_keys=True) + "\n")
        print(json.dumps({"stage": "group", **record}, sort_keys=True), flush=True)
        next_index = index + 1
        if next_index % args.checkpoint_every == 0 or next_index == len(selected):
            checkpoint = save_checkpoint(
                model, tokenizer, optimizer, args.output_dir, next_index, history
            )
            print(json.dumps({"stage": "checkpoint", "path": str(checkpoint)}), flush=True)

    nonfinite = sum(
        int((~torch.isfinite(parameter)).sum().item()) for _, parameter in trainable_named
    )
    if nonfinite:
        raise FloatingPointError(f"non-finite trainable adapter parameters: {nonfinite}")
    adapter = args.output_dir / "adapter"
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    bucket_counts = Counter(target_bucket(row) for row in selected)
    summary = {
        "protocol": "molprogram_group_relative_rl_v1",
        "loader_kind": loader_kind,
        "base_model": args.base_model,
        "input_adapter": str(args.input_adapter),
        "output_adapter": str(adapter),
        "prompts": len(selected),
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "group_size": args.group_size,
        "editing_reward_mode": args.editing_reward_mode,
        "editing_support_report": str(args.editing_support_report),
        "editing_support_report_sha256": support_report_sha256,
        "editing_support_decision": support_decision,
        "learning_rate": args.learning_rate,
        "sft_anchor_weight": args.sft_anchor_weight,
        "initial_adapter_weight": args.initial_adapter_weight,
        "critic": False,
        "reward_target_smiles_access": False,
        "sft_anchor_uses_training_positive": True,
        "adapter_nonfinite_parameters": nonfinite,
        "totals": dict(sorted(totals.items())),
        "history": history,
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
