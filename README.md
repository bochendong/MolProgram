# MolProgram

MolProgram is one language-model policy for two molecular design modes:

- **de novo construction:** a property program is mapped to a new molecule;
- **source-conditioned editing:** the same policy modifies a source molecule
  while satisfying directional property constraints.

The two modes share one prompt contract, one LoRA adapter, and one structured
output format. This repository contains the dataset builder, supervised
training, matched joint-versus-specialist comparison, target-blind Raw@1
evaluation, and group-relative reinforcement learning used in our experiments.

> Research release in preparation. Dataset and checkpoint download links will
> be added after the corresponding artifacts are frozen. The code paths in this
> repository are executable and do not depend on our internal experiment names.

## Method at a glance

```text
optional source SMILES + property program
                    |
                    v
           shared Qwen + LoRA policy
                    |
                    v
       {"plan": "BUILD|MODIFY", "smiles": "..."}
                    |
          +---------+---------+
          |                   |
   supervised tuning   target-blind RL
                       property + source reward
```

The model never receives the target molecule during generation or RL reward
calculation. Editing success requires both the requested property changes and
source similarity; copying the source is tracked separately.

## Repository layout

```text
src/molprogram/
  protocol.py              shared prompt and output contract
  scoring.py               target-blind property and similarity scoring
  rewards.py               hard-boundary editing reward
scripts/
  build_dataset.py         audited, leakage-checked dataset construction
  freeze_fresh_eval_gates.py  freeze target-blind headline requests and hashes
  train_sft.py             standard LoRA supervised fine-tuning
  train_indexed_sft.py     memory-bounded full-corpus continuation
  audit_editing_reward_support.py  no-training RL support gate
  train_rl.py              group-relative online RL with SFT anchoring
  evaluate_raw1.py         matched target-blind Raw@1 evaluation
ablations/
  fresh_balanced/          fresh headline training and exposure checkpoints
  joint_vs_specialists/    matched shared-policy comparison
  negative_refinement/     positive-only versus structured negatives
  safe_joint_raw1_grpo/    paired shared-policy Raw@1 GRPO method gate
audits/
  editing_reward_support/  preregistered online-RL go/no-go gate
baselines/
  mumo_fresh/              fresh editing-only LoRA on MuMOInstruct
recipes/                   reproducible command-line entrypoints
tests/                     protocol and reward contract tests
```

## Installation

Python 3.10 or newer is required. Training requires a CUDA GPU with BF16
support.

```bash
git clone https://github.com/bochendong/MolProgram.git
cd MolProgram
python -m venv .venv
source .venv/bin/activate
pip install -e '.[train,oracle,test]'
pytest -q
```

## Unified request format

De novo construction uses an empty source:

```json
{
  "source": "<EMPTY>",
  "conditions": [
    {"property": "MW", "goal": {"around": 320.0}},
    {"property": "QED", "goal": {"around": 0.75}}
  ]
}
```

Editing uses a source molecule and directional goals:

```json
{
  "source": "CCOc1ccc(C(=O)O)cc1",
  "conditions": [
    {"property": "QED", "goal": "increase"},
    {"property": "SA", "goal": "decrease"}
  ]
}
```

The only accepted responses are:

```json
{"plan":"BUILD","smiles":"CANONICAL_SMILES"}
{"plan":"MODIFY","smiles":"CANONICAL_SMILES"}
```

## Training

For ordinary JSONL data:

```bash
python scripts/train_sft.py \
  --train-jsonl data/train.jsonl \
  --output-dir outputs/sft \
  --base-model Qwen/Qwen2.5-VL-7B-Instruct \
  --compute-dtype bfloat16
```

For the full indexed MolProgramInstruct release:

```bash
bash recipes/train_full_sft.sh
```

Run target-blind group-relative RL from an SFT adapter:

```bash
bash recipes/audit_editing_support.sh
bash recipes/train_rl.sh
```

The editing support audit is a required scientific gate, not a training stage.
It samples the current policy at `K=32`, measures source-feasible and strict
support, checks whether the hard reward ranks strict candidates correctly, and
writes `support_report.json`. Do not launch editing online RL when its decision
is `DO_NOT_RUN_ONLINE_RL_SUPPORT_TOO_LOW`. The RL recipe reads this report from
`SUPPORT_REPORT` and stops before model loading unless the gate authorizes a
small pilot.

## Matched joint-versus-specialist comparison

The comparison uses the same base checkpoint, LoRA rank, examples per task,
optimizer settings, and frozen evaluation prompts. The joint system has one
adapter; the separate system has one construction adapter and one editing
adapter.

```bash
bash ablations/joint_vs_specialists/run.sh
```

This comparison is the primary test of positive transfer and parameter
efficiency from sharing a policy. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)
for the evaluation contract and [ablations/](ablations/) for the complete
ablation index and historical protocols.

The fresh reconstruction of the balanced headline schedule, including the
100k, 200k, 500k, and full-exposure adapters, is defined in
[ablations/fresh_balanced/](ablations/fresh_balanced/).
Its checked-in gate under `benchmarks/fresh_balanced_raw1/` is directly readable
by the evaluator, so the fresh run does not depend on an older repository.

The follow-up RL method gate is defined in
[ablations/safe_joint_raw1_grpo/](ablations/safe_joint_raw1_grpo/). It starts
from the final fresh, non-task-aligned balanced adapter, uses 16 candidates
only for training-time advantage estimation, and evaluates one target-blind
candidate per request. Its matched continued-SFT control receives the same
paired prompts, optimizer steps, and checkpoint-selection budget.

## Data

MolProgramInstruct contains 2,000,000 unique de novo targets and 569,919 unique
source-target editing pairs, balanced across property-count buckets. Large data
files are intentionally not committed to Git. The release manifest records
counts, hashes, seeds, duplicate audits, and frozen-evaluation exclusions.

See [docs/DATASET.md](docs/DATASET.md) for the schema and build procedure.

## Citation

The paper citation, model cards, and dataset links will be added when the
corresponding artifacts are frozen.

## License

Released under the [Apache License 2.0](LICENSE).
