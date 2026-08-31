#!/usr/bin/env python3
"""Independently audit every saved MuMO adapter tensor for finite values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    from safetensors.torch import load_file
    import torch

    path = args.adapter_dir / "adapter_model.safetensors"
    tensors = load_file(path)
    nonfinite_by_tensor = {
        key: int((~torch.isfinite(value.float())).sum().item())
        for key, value in tensors.items()
    }
    nonfinite_by_tensor = {
        key: count for key, count in nonfinite_by_tensor.items() if count
    }
    result = {
        "adapter": str(args.adapter_dir),
        "tensor_count": len(tensors),
        "nonfinite_values": sum(nonfinite_by_tensor.values()),
        "nonfinite_by_tensor": nonfinite_by_tensor,
        "valid": not nonfinite_by_tensor,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if nonfinite_by_tensor:
        raise FloatingPointError(json.dumps(result, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
