"""Diff two debug companion files layer by layer.

    python scripts/debug_compare.py runs/inbox/debug/<cand>-debug.pt runs/inbox/debug/<ref>-debug.pt

Prints, per prompt: the relative residual difference at every layer, the first layer past the
drift floor, and whether the final top token survives. The companions are the instrumented
traces `--debug` runs write (isb/debugtrace.py); the release run files stay the only source of
verdicts.
"""
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from isb.debugtrace import DRIFT_FLOOR, compare_debug  # noqa: E402


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    cand = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    ref = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
    print(f"candidate: {sys.argv[1]}\nreference: {sys.argv[2]}")
    for row in compare_debug(cand["outputs"], ref["outputs"]):
        if "error" in row:
            print(f"\nprompt {row['prompt']}: {row['error']}")
            continue
        print(f"\nprompt {row['prompt']}: "
              f"top token {'SAME' if row['top1_match'] else 'DIFFERENT'}, "
              f"final-distribution difference {row['logit_tv']:.4f}")
        first = row["first_drift_layer"]
        print(f"  drift starts at layer {first}" if first is not None
              else f"  no layer drifts past the floor ({DRIFT_FLOOR})")
        print("  layer | relative residual difference")
        for L, v in enumerate(row["per_layer_rel"]):
            bar = "#" * min(60, int(v * 200))
            print(f"  {L:5d} | {v:8.5f} {bar}")


if __name__ == "__main__":
    main()
