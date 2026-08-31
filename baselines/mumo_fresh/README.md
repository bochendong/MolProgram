# MuMO fresh-LoRA baseline

This baseline measures what an editing-only model learns from MuMOInstruct
without inheriting any MolProgram adapter. It starts from the same unadapted
Qwen backbone with a fresh rank-16 LoRA and retains the original MuMO/GeLLM
prompt and `<SMILES>...</SMILES>` response contract.

The indexed training release contains 228,076 rows across all 63 non-empty
combinations of six MuMO properties, capped at 10,000 rows per task.

## Stability correction

An earlier internal run became non-finite during its first few optimizer steps
but continued for a full epoch because it checked parameters only at the end.
The corrected baseline changes the numerical contract before rerunning:

- excludes `lm_head` from LoRA targets and uses the same seven projection
  modules as the stable MolProgram fresh-LoRA comparisons;
- reduces the learning rate from `1e-4` to `2e-5`;
- uses physical batch 1 with accumulation 128, preserving effective batch 128;
  this follows the previously validated fresh-LoRA path and avoids the
  variable-length multi-example batches implicated by the first repaired
  smoke test;
- checks every microbatch loss, every backward pass in the first effective
  batch, every optimizer-step gradient, and every optimizer-step trainable
  parameter for finite values; the first-batch guard reports exact source rows;
- disables NaN/Inf log filtering so numerical failures cannot appear as zero
  loss;
- runs a fresh 50-step smoke test before the full one-epoch job is released.

The smoke model is a safety gate only. The full arm starts again from the clean
base model and never resumes from the smoke or failed adapter.

## Submit

Prepare the indexed JSONL with `prepare_data.py`, or point to a frozen release
containing both `.jsonl` and `.idx` files. Then run:

```bash
export MUMO_TRAIN_JSONL=/path/to/mumo_train.jsonl
export MUMO_BASE_MODEL=/path/to/Qwen2.5-VL-7B-Instruct
bash baselines/mumo_fresh/submit.sh
```

Outputs default to `outputs/baselines/mumo_fresh/stable_v2_seed_32002`. The full job is
submitted with an `afterok` dependency on the smoke test, and a final CPU job
independently verifies that every saved adapter tensor is finite.
