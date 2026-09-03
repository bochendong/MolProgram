#!/usr/bin/env python3
"""Generate exactly one target-blind sampled MuMO completion per frozen input."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from protocol import extract_smiles, messages  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def joined_instruction(
    benchmark_row: Mapping[str, str], source_rows: Sequence[Mapping[str, object]]
) -> tuple[str, int]:
    raw_index = int(benchmark_row["external_source_row_index"])
    source_row = source_rows[raw_index]
    source_smiles = str(source_row["source_smiles"])
    task = str(source_row["task"])
    if source_smiles != str(benchmark_row["source_smiles"]):
        raise ValueError(f"source mismatch at raw row {raw_index}")
    if task != str(benchmark_row["external_task_key"]):
        raise ValueError(f"task mismatch at raw row {raw_index}")
    instruction_index = int(source_row["instr_idx"])
    return task, instruction_index


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_progress(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-csv", required=True, type=Path)
    parser.add_argument("--source-json", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=32021)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if not args.adapter_dir.joinpath("adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"missing adapter: {args.adapter_dir}")

    benchmark_rows = read_csv(args.rows_csv)
    source_rows = json.loads(args.source_json.read_text(encoding="utf-8"))
    if not isinstance(source_rows, list):
        raise ValueError("MuMO source JSON must contain a list")
    if len(benchmark_rows) != 1992:
        raise ValueError(f"frozen MuMO gate must contain 1992 rows, found {len(benchmark_rows)}")
    task_counts: dict[str, int] = {}
    joined: list[tuple[str, int]] = []
    for row in benchmark_rows:
        task, instruction_index = joined_instruction(row, source_rows)
        joined.append((task, instruction_index))
        task_id = str(row["external_task_id"])
        task_counts[task_id] = task_counts.get(task_id, 0) + 1
    if len(task_counts) != 10:
        raise ValueError(f"frozen MuMO gate must contain 10 tasks, found {task_counts}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "progress.jsonl"
    predictions_path = args.output_dir / "predictions.csv"
    completed = read_progress(progress_path)
    if len(completed) > len(benchmark_rows):
        raise ValueError("Raw@1 progress has more rows than the benchmark")
    for index, item in enumerate(completed):
        if str(item.get("condition_id")) != str(benchmark_rows[index]["condition_id"]):
            raise ValueError(f"Raw@1 progress is not a benchmark prefix at row {index}")
    if len(completed) == len(benchmark_rows):
        write_csv(predictions_path, completed)
        return 0

    import peft
    import torch
    import transformers
    from rdkit import Chem

    config = transformers.AutoConfig.from_pretrained(args.base_model, local_files_only=True)
    if type(config) in transformers.AutoModelForCausalLM._model_mapping:
        loader = transformers.AutoModelForCausalLM
    elif type(config) in transformers.AutoModelForImageTextToText._model_mapping:
        loader = transformers.AutoModelForImageTextToText
    else:
        raise TypeError(f"unsupported config: {type(config).__name__}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model, use_fast=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = loader.from_pretrained(
        args.base_model, config=config, dtype=torch.bfloat16,
        low_cpu_mem_usage=True, local_files_only=True,
    )
    model = peft.PeftModel.from_pretrained(base, args.adapter_dir).cuda().eval()
    model.config.use_cache = True

    start_index = len(completed)
    with progress_path.open("a", encoding="utf-8") as progress:
        for start in range(start_index, len(benchmark_rows), args.batch_size):
            batch = benchmark_rows[start : start + args.batch_size]
            batch_joined = joined[start : start + args.batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    messages(row["source_smiles"], task, instruction_index),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for row, (task, instruction_index) in zip(batch, batch_joined)
            ]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            prompt_length = int(encoded["input_ids"].shape[1])
            torch.manual_seed(args.seed + start)
            with torch.no_grad():
                sampled = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                )
            new_rows = []
            for offset, (row, (_task, instruction_index)) in enumerate(
                zip(batch, batch_joined)
            ):
                raw = tokenizer.decode(
                    sampled[offset][prompt_length:], skip_special_tokens=True
                ).strip()
                proposed = extract_smiles(raw)
                molecule = Chem.MolFromSmiles(proposed) if proposed else None
                canonical = (
                    Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
                    if molecule is not None else ""
                )
                item: dict[str, object] = dict(row)
                item.update(
                    {
                        "generated_smiles": canonical,
                        "candidate_rank": 1,
                        "candidate_budget": 1,
                        "candidate_selected": True,
                        "raw_completion": raw,
                        "strict_parse": molecule is not None,
                        "valid_smiles": molecule is not None,
                        "external_instruction_index": instruction_index,
                        "method": "mumo_fresh_stable_v2_raw1",
                        "sampling_seed": args.seed,
                        "property_reranking": False,
                        "target_access": False,
                    }
                )
                new_rows.append(item)
                progress.write(json.dumps(item, sort_keys=True) + "\n")
            progress.flush()
            os.fsync(progress.fileno())
            completed.extend(new_rows)
            print(f"[mumo-raw1] {len(completed)}/{len(benchmark_rows)}", flush=True)

    write_csv(predictions_path, completed)
    summary = {
        "protocol": "mumo_fresh_official_raw1_v1",
        "conditions": len(completed),
        "task_counts": dict(sorted(task_counts.items())),
        "candidate_budget": 1,
        "raw_at_1": True,
        "property_reranking": False,
        "target_access": False,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "rows_sha256": sha256(args.rows_csv),
        "source_json_sha256": sha256(args.source_json),
        "adapter_sha256": sha256(args.adapter_dir / "adapter_model.safetensors"),
        "predictions_sha256": sha256(predictions_path),
    }
    (args.output_dir / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "GENERATION_COMPLETE").touch()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
