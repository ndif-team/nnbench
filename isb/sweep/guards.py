"""Shared non-vacuity guard (design.md §8.3) — the single implementation of the effect-size check
copy-pasted across smoke_steering/_patching/_ablation.

A write methodology's SUPPORTED/SILENTLY_WRONG verdict is only meaningful if the intervention
actually moves the control's output. `compute_effect_size` measures TV(control baseline output,
control perturbed output) and reports whether it is strong enough that the downstream verdict is not
vacuous. Pure (no model / no GPU) so it is unit-testable: the driver supplies the two already-run
control outputs.
"""
from __future__ import annotations

from ..oracle.equivalence import compare


def compute_effect_size(baseline_value, perturbed_value, *, tv_floor: float = 0.2,
                        top1_ceiling: float = 0.5) -> dict:
    """Strong iff the perturbation flips the control's top-1 (top1_agree < top1_ceiling) OR shifts
    the distribution well past the oracle tolerance (tv > tv_floor). A weak result means a backend
    that silently dropped the write could pass the oracle vacuously — flag it."""
    m = compare(baseline_value, perturbed_value)
    strong = m["top1_agree"] < top1_ceiling or m["tv"] > tv_floor
    return {"top1_agree": m["top1_agree"], "tv": m["tv"], "strong": strong}
