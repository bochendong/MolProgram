# Transfer-aware property-program routing

This pilot replaces the failed hand-written hard rank mask with an evidence-
compiled soft residual route. It is explicitly positioned as a DSN-inspired,
rank-wise LoRA pilot rather than a claim that rank gating itself is new.

The experiment first freezes a task-covered 3,840/3,840 de novo/edit training
set. Both arms start from the same 16-step dense warm-up. On that adapter, the
probe measures a signed gradient signature for every `mode:property` node. The
cosine transfer graph is decomposed into four signed spectral factors; their
positive and negative sides address eight residual LoRA ranks. Requests compose
node routes with an elementwise maximum.

Eight common ranks remain fully active. Every residual rank has an activation
floor of 0.25, so sparse routing cannot recreate the hard-v1 failure in which
untrained private ranks replaced trained capacity. All masks are RMS-normalized.
The matched dense arm shares the same data, warm-up adapter, rank, optimizer,
and Raw@1 gates.

Run `submit_slurm.sh` only with a task-aligned source that contains all ten
editing tasks. `prepare.py` fails before GPU allocation if an evaluation
mode-property node has fewer than 64 training examples.

```bash
export MODEL=/path/to/node-local/model
export TASK_COVERED_TRAIN=/path/to/task-aligned/train.sft.jsonl
export GATE_ROOT=/path/to/completed/frozen-scale-output
bash ablations/transfer_aware_program_routing/submit_slurm.sh
```

The primary endpoint is Edit-only-5 strict Raw@1 success at source similarity
0.65. All-10 must not regress; Shared-5, de novo, and validity each have a 2 pp
guardrail. The result remains a pilot regardless of outcome.
