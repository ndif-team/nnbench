"""Export the J maps a jacobian_collect run produced into the fitted-lens .pt layout.

The run file's collect-task output is [n_sources, d, d] (source layers 0..n_sources-1, in
order). The export writes {"J": {layer_index: [d, d]}, "source_layers": [...]}, the same
layout as the upstream jacobian-lens checkpoints, so the result drops into any
jacobian_lens transport param as "file:<out>".

    python scripts/export_jacobian.py runs/inbox/jacobian_collect_gpt2-hf.pt runs/artifacts/gpt2-J.pt
"""
import sys
from pathlib import Path

import torch


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    run_path, out_path = sys.argv[1], sys.argv[2]
    rf = torch.load(run_path, map_location="cpu", weights_only=False)
    stacks = [(k, v) for k, v in rf["outputs"].items()
              if k != ("__meta__",) and v is not None and v.dim() == 3 and v.shape[1] == v.shape[2]]
    if len(stacks) != 1:
        raise SystemExit(f"expected exactly one [n_sources, d, d] output, found {[k for k, _ in stacks]}")
    key, J = stacks[0]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"J": {i: J[i].clone() for i in range(J.shape[0])},
                "source_layers": list(range(J.shape[0])),
                "provenance": {"run": run_path, "task": key}}, out)
    print(f"{J.shape[0]} layer maps ({J.shape[1]}x{J.shape[2]}) from {key} -> {out}")


if __name__ == "__main__":
    main()
