# Property-program routed LoRA: 10k pilot

This preregistered pilot compares one fresh property-routed rank-16 LoRA with
the completed fresh 10k vanilla joint adapter and its two specialists. It
reuses the baseline's byte-identical 20,000 training rows and frozen 120/500
Raw@1 gates.

The adapter is a standard LoRA with no extra trainable parameters. Eight ranks
are common. The remaining ranks are deterministically addressed by the
prompt-visible property names: ranks 8--11 cover properties shared across de
novo and editing, ranks 12--13 cover de-novo-only properties, and ranks 14--15
cover edit-only properties. A request activates the union of its addressed
ranks, with RMS normalization. Training and generation apply the same mask.

The primary question is whether routing recovers at least two points on the
Edit-only-5 strict Raw@1 endpoint while keeping Shared-5, complete de novo, and
both validity rates within two points of vanilla joint training. All endpoints
are reported even if the gate fails.

## Submit

```bash
export MODEL=/path/to/Qwen2.5-VL-7B-Instruct
export BASELINE_ROOT=/path/to/completed/fresh-10k
bash ablations/property_program_routing/submit_slurm.sh
```

Outputs are written to `outputs/property-program-routing-10k` by default.

## Post-hoc dense-inference diagnostic

The frozen hard-routing result activates untrained private ranks for properties
absent from the 10k scale-sweep training data and fails its primary endpoint.
`dense_inference_diagnostic.json` therefore registers a mechanism diagnostic:
reuse the completed routed adapter and Raw@1 gates, but activate all LoRA ranks
at inference. This diagnostic requires no additional training and cannot be
reported as an independent confirmatory result.

```bash
export MODEL=/path/to/Qwen2.5-VL-7B-Instruct
export BASELINE_ROOT=/path/to/completed/fresh-10k
bash ablations/property_program_routing/submit_dense_diagnostic.sh
```
