"""Effect-size guard + Workload-validation tests (isb/sweep/) — no GPU; torch only."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isb.sweep.guards import compute_effect_size  # noqa: E402
from isb.sweep.spec import Workload  # noqa: E402

V = 8


def test_strong_when_top1_flips():
    base = torch.zeros(1, V); base[0, 2] = 9.0     # argmax 2
    pert = torch.zeros(1, V); pert[0, 5] = 9.0     # argmax 5 -> top-1 flipped
    eff = compute_effect_size(base, pert)
    assert eff["strong"]
    assert eff["top1_agree"] == 0.0


def test_weak_when_no_effect():
    base = torch.zeros(1, V); base[0, 2] = 9.0
    pert = base.clone(); pert[0, 2] = 9.001        # same argmax, negligible shift
    eff = compute_effect_size(base, pert)
    assert not eff["strong"]
    assert eff["top1_agree"] == 1.0


def test_strong_via_distribution_shift_same_argmax():
    base = torch.zeros(1, V); base[0, 2] = 9.0     # sharp at 2
    pert = torch.full((1, V), 3.0); pert[0, 2] = 4.0   # argmax still 2 but much flatter -> large TV
    eff = compute_effect_size(base, pert)
    assert eff["top1_agree"] == 1.0                # top-1 unchanged
    assert eff["tv"] > 0.2 and eff["strong"]       # caught by the distribution-shift floor


def test_workload_validation():
    Workload("interactive", ["x"])                 # ok
    Workload("batched", ["a", "b"])                # ok
    for bad in (lambda: Workload("generation", ["x"]),            # new_tokens==0 stub
                lambda: Workload("bogus", ["x"])):
        raised = False
        try:
            bad()
        except ValueError:
            raised = True
        assert raised


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
