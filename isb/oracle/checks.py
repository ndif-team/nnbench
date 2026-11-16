"""Correctness checks: how the scorer compares a row's check output with the reference's.

A check is `check(expected, actual, case) -> (passed, metrics)`: `expected` and `actual` are the
reference's and the candidate's check outputs for the row, and `case` gives the row's regime and
label and both jobs' full check outputs, for a check that needs other rows of the job. A
methodology registers its own check with `@check_of(methodology)`;
every other methodology gets `default`, the equivalence oracle's top-1 and total-variation gate.

A timed check relates a timed call's result to the check call's output for the same units, for a
methodology whose cells return the user's result when timed (`check=False`) and a scored output
when checked: `timed_check(timed, checked) -> (passed, metrics)`, and a check that takes `stops`
also receives the engine's stop tokens (`stop_tokens`). A failure means the timed call did not
compute what the check call did, a contract violation. The shared forms below cover a token
against logits, generated tokens against per-step logits, and a read against its scored copy.
"""
from __future__ import annotations

import functools

from .equivalence import compare, is_equivalent

CHECKS = {}   # methodology -> check
TIMED = {}    # methodology -> timed check

NEAR_TIE = 0.02   # of the top logit's magnitude: separate calls may order logits this close differently


def default(expected, actual, case):
    metrics = compare(expected, actual)
    return is_equivalent(metrics), metrics


@functools.lru_cache(maxsize=None)
def _config_stops(repo):
    from transformers import GenerationConfig

    try:
        eos = GenerationConfig.from_pretrained(repo).eos_token_id
    except OSError:                                      # a checkpoint without generation_config.json
        return ()
    return tuple(eos) if isinstance(eos, (list, tuple)) else (eos,)


def stop_tokens(tokenizer, repo):
    """The token ids an engine masks while it holds a generation to its minimum length (vLLM's
    `min_tokens`, HF's `min_new_tokens`): the tokenizer's end-of-sequence id and the end ids of
    the checkpoint's generation config."""
    return frozenset(i for i in (tokenizer.eos_token_id, *_config_stops(repo)) if i is not None)


def _gaps(logits, tokens, stops):
    """Each token's logit below the top logit, and that top: over the whole vocabulary for a stop
    token, and over the other tokens for any other token, since an engine holding a generation to
    its minimum length samples with the stop tokens masked."""
    import torch

    top = logits.max(-1).values
    stops = [i for i in stops if i < logits.shape[-1]]
    if stops:
        masked = logits.clone()
        masked[..., stops] = -torch.inf
        top = torch.where(torch.isin(tokens, torch.tensor(stops)), top, masked.max(-1).values)
    return top - logits.gather(-1, tokens[..., None]).squeeze(-1), top


def tokens_match(timed, checked, stops=()):
    """A timed call's greedy next tokens `[B, 1]` against the check call's next-token logits
    `[B, vocab]` for the same units: each token is the check's argmax (among the tokens that are
    not `stops`, for a token that is not one), or its logit is within NEAR_TIE of that top. A token
    outside the vocabulary or a nonfinite logit fails, with no gap to report."""
    tokens = timed.reshape(-1).long()
    logits = checked.reshape(len(tokens), -1).float()
    if int(tokens.max()) >= logits.shape[-1] or not bool(logits.isfinite().all()):
        return False, {"tokens_matched": None, "max_gap": None, "units": len(tokens)}
    gap, top = _gaps(logits, tokens, stops)
    return bool((gap <= NEAR_TIE * top.abs()).all()), {
        "tokens_matched": float((gap == 0).float().mean()), "max_gap": float(gap.max()),
        "units": len(tokens)}


def generated_tokens_match(timed, checked, stops=()):
    """A timed generation's tokens `[steps, B, 1]` against the check call's per-step logits
    `[steps, B, vocab]`: per unit, each step's token is the check's argmax (as `tokens_match`
    takes it) up to the first step where it is not, whose token must be within NEAR_TIE of the
    top logit; after that step the two calls continue different prefixes and are not compared."""
    import torch

    steps, units = checked.shape[0], checked.shape[-2]
    tokens = timed.reshape(steps, units).long()
    logits = checked.reshape(steps, units, -1).float()
    if int(tokens.max()) >= logits.shape[-1] or not bool(logits.isfinite().all()):
        return False, {"steps_matched": None, "max_gap": None, "units": units}
    gap, top = _gaps(logits, tokens, stops)                              # [steps, B]
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
