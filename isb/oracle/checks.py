"""Correctness checks: how the scorer compares a row's check output with the reference's.

A check is `check(expected, actual, case) -> (passed, metrics)`: `expected` and `actual` are the
reference's and the candidate's check outputs for the row, and `case` gives the row's regime and
label and both jobs' full check outputs, for a check that needs other rows of the job. A
methodology registers its own check with `@check_of(methodology)`;
every other methodology gets `default`, the equivalence oracle's top-1 and total-variation gate.

A timed check relates a timed call's result to the check call's output for the same units, for a
methodology whose cells return the user's result when timed (`check=False`) and a scored output
when checked: `timed_check(timed, checked) -> (passed, metrics)`. A failure means the timed call did
not compute what the check call did, a contract violation. The shared forms below cover a token
against logits, generated tokens against per-step logits, and a read against its scored copy.
"""
from __future__ import annotations

from .equivalence import compare, is_equivalent

CHECKS = {}   # methodology -> check
TIMED = {}    # methodology -> timed check

NEAR_TIE = 0.02   # of the top logit's magnitude: separate calls may order logits this close differently


def default(expected, actual, case):
    metrics = compare(expected, actual)
    return is_equivalent(metrics), metrics


def tokens_match(timed, checked):
    """A timed call's greedy next tokens `[B, 1]` against the check call's next-token logits
    `[B, vocab]` for the same units: each token is the check's argmax, or its logit is within
    NEAR_TIE of the top one. A token outside the vocabulary or a nonfinite logit fails, with no
    gap to report."""
    tokens = timed.reshape(-1).long()
    logits = checked.reshape(len(tokens), -1).float()
    if int(tokens.max()) >= logits.shape[-1] or not bool(logits.isfinite().all()):
        return False, {"tokens_matched": None, "max_gap": None, "units": len(tokens)}
    top = logits.max(-1).values
    gap = top - logits.gather(-1, tokens[:, None]).squeeze(-1)
    return bool((gap <= NEAR_TIE * top.abs()).all()), {
        "tokens_matched": float((gap == 0).float().mean()), "max_gap": float(gap.max()),
        "units": len(tokens)}


def generated_tokens_match(timed, checked):
    """A timed generation's tokens `[steps, B, 1]` against the check call's per-step logits
    `[steps, B, vocab]`: per unit, each step's token is the check's argmax up to the first step
    where it is not, whose token must be within NEAR_TIE of the top logit; after that step the two
    calls continue different prefixes and are not compared."""
    import torch

    steps, units = checked.shape[0], checked.shape[-2]
    tokens = timed.reshape(steps, units).long()
    logits = checked.reshape(steps, units, -1).float()
    if int(tokens.max()) >= logits.shape[-1] or not bool(logits.isfinite().all()):
        return False, {"steps_matched": None, "max_gap": None, "units": units}
    top = logits.max(-1).values                                          # [steps, B]
    gap = top - logits.gather(-1, tokens[..., None]).squeeze(-1)         # [steps, B]
    flipped = gap > 0
    flips = flipped.any(0)
    first = torch.where(flips, flipped.int().argmax(0), torch.full((units,), steps - 1))
    compared = torch.arange(steps)[:, None] <= first[None, :]            # up to the first flip
    passed = bool(((gap <= NEAR_TIE * top.abs()) | ~compared).all())
    exact = torch.where(flips, first, torch.full((units,), steps))       # steps before a flip
    return passed, {"steps_matched": float(exact.float().mean() / steps),
                    "max_gap": float(gap[compared].max()), "units": units}


def read_matches(timed, checked):
    """A read's timed tensor, in the dtype the system returned it, against the check call's copy
    of the same units under the equivalence gate."""
    metrics = compare(checked, timed.float())
    return is_equivalent(metrics), metrics


def probabilities_match(timed, checked):
    """Timed attention probabilities against the check call's log-probabilities: the timed values'
    log, a zero probability floored at its dtype's smallest normal so its log is finite, under the
    equivalence gate."""
    import torch

    logs = torch.log(timed.clamp_min(torch.finfo(timed.dtype).tiny)).float()
    metrics = compare(checked, logs)
    return is_equivalent(metrics), metrics


def _register(table, methodology):
    def register(fn):
        table[methodology] = fn
        return fn
    return register


def check_of(methodology):
    """Register the check for `methodology`'s rows."""
    return _register(CHECKS, methodology)


def timed_check_of(methodology):
    """Register the timed check for `methodology`'s cells."""
    return _register(TIMED, methodology)


def _registered():
    """Import the methodology modules, which register their checks beside their cells."""
    import isb.methodologies  # noqa: F401


def check_for(methodology):
    _registered()
    return CHECKS.get(methodology, default)


def timed_check_for(methodology):
    _registered()
    if methodology not in TIMED:
        raise LookupError(f"{methodology} cells return a separate result when timed, but no timed "
                          f"check relates it to their check output (timed_check_of)")
    return TIMED[methodology]
