# Property-conditional residual adapter

This pilot tests a narrower response to the observed transfer pattern than
rank-routing the whole shared adapter. It freezes the completed dense rank-16
adapter and adds four residual LoRA ranks. The residual is active only for
editing requests containing an edit-only property: SA, GSK3B, or DRD2.

The shared rank-16 slices are copied exactly into a rank-20 adapter and are
gradient-frozen. The new rank-4 slices are trained on the 1,920 edit-only rows
from the existing task-covered 3,840/3,840 pilot release. LoRA scaling is kept
at 2.0 by changing alpha from 32/16 to 40/20.

Three Raw@1 evaluations isolate the effects:

- `frozen_shared`: the expanded adapter before residual training;
- `conditional_residual`: the trained residual is active only for Edit-only-5;
- `always_on_residual`: the same trained adapter activates the residual for
  every request.

The conditional and always-on arms use byte-identical adapter weights. Their
only difference is the inference routing layout. The collector requires exact
Raw@1 output identity between `frozen_shared` and `conditional_residual` on de
novo and Shared-5 requests, and between `conditional_residual` and
`always_on_residual` on Edit-only-5 requests.

The primary endpoint is the Edit-only-5 strict Raw@1 delta at source similarity
0.65. The pilot is supported only if this delta is at least +2 percentage
points, All-10 does not regress, inactive outputs are exactly preserved, and
editing validity remains within 2 points.

This is a small mechanistic pilot. It does not establish that conditional
adapters or mixture-of-experts routing are new in general.

## Slurm

The run reuses the completed transfer-aware pilot's dense adapter, frozen
training subset, and Raw@1 gates:

```bash
export MODEL=/path/to/Qwen2.5-VL-7B-Instruct
export SOURCE_WORK=/path/to/completed/transfer-aware-program-routing-10k
export ASSAY_ORACLE_DIR=/path/to/frozen/assay-models
bash ablations/property_conditional_residual/submit_slurm.sh
```

GPU jobs default to `--nice=10000`, leaving the fresh balanced run at higher
scheduler priority while allowing this short experiment to backfill.
