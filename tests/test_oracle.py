"""Oracle unit tests (design.md §8.1, §8.3) — no GPU, CPU tensors.

Locks the review must-fix: the equivalence gate must NOT pass a top-1-preserving but
genuinely divergent distribution (false SUPPORTED), and must NOT condemn a correct
result when there's no reference.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from isb.oracle.checks import (  # noqa: E402
    generated_tokens_match, probabilities_match, read_matches,
)
from isb.oracle.equivalence import compare, is_equivalent  # noqa: E402


def test_generated_tokens_are_compared_up_to_the_first_flip():
    """Each step's timed token must be the check's argmax until the first step where it is not;
    that step's token must be a near tie, and later steps continue a different prefix."""
    logits = torch.tensor([[1.0, 50.0, 0.0, 0.0],      # step 0: token 1, far ahead
                           [0.0, 0.0, 10.0, 9.9],      # step 1: tokens 2 and 3 a near tie
                           [7.0, 0.0, 0.0, 0.0]])[:, None, :]       # [steps, B, vocab]
    passed, metrics = generated_tokens_match(torch.tensor([1, 3, 2])[:, None, None], logits)
    assert passed and abs(metrics["steps_matched"] - 1 / 3) < 1e-6   # flipped at step 1
    assert generated_tokens_match(torch.tensor([1, 2, 0])[:, None, None], logits)[0]
    assert not generated_tokens_match(torch.tensor([1, 0, 0])[:, None, None], logits)[0]
    assert not generated_tokens_match(torch.tensor([0, 2, 0])[:, None, None], logits)[0]


def test_a_timed_read_and_attention_probabilities_match_their_check_forms():
    """A read returned in the model's dtype matches its fp32 check copy; timed attention
    probabilities match the check call's log-probabilities; a different read fails either way."""
    lens = torch.randn(3, 2, 50)
    assert read_matches(lens.bfloat16(), lens.bfloat16().float())[0]
    assert not read_matches(lens.bfloat16(), torch.randn(3, 2, 50))[0]
    probs = torch.softmax(torch.randn(2, 1, 4, 6), dim=-1)          # [layers, B, heads, keys]
    assert probabilities_match(probs, torch.log(probs))[0]
    assert not probabilities_match(probs, torch.log(probs.flip(-1)))[0]


def test_identical_is_equivalent():
    x = torch.randn(12, 1, 50257)
    m = compare(x, x.clone())
    assert m["top1_agree"] == 1.0 and m["tv"] < 1e-6 and m["shape_match"]
    assert is_equivalent(m)


def test_vocab_padding_aligned():
    ref = torch.randn(12, 1, 50257)
    got = torch.cat([ref, torch.full((12, 1, 47), -1e9)], dim=-1)  # padded junk
    m = compare(ref, got)
    assert m["shape_match"] and m["top1_agree"] == 1.0 and is_equivalent(m)


def test_uniform_shift_is_equivalent_but_maxabs_large():
    ref = torch.randn(12, 1, 1000)
    got = ref + 5.0  # uniform logit shift: softmax & argmax unchanged
    m = compare(ref, got)
    assert m["top1_agree"] == 1.0 and m["tv"] < 1e-5
    assert is_equivalent(m)        # correct: a shift does not change the lens prediction
    assert m["max_abs"] >= 4.9     # max_abs is large -> proves it is NOT the gate


def test_top1_preserving_divergence_is_caught():
    # same argmax everywhere, but a genuinely different distribution -> high TV.
    ref = torch.zeros(4, 1, 1000)
    ref[..., 0] = 10.0             # sharply peaked on token 0
    got = torch.zeros(4, 1, 1000)
    got[..., 0] = 0.2             # token 0 still top-1, but nearly flat
    m = compare(ref, got)
    assert m["top1_agree"] == 1.0  # top-1 agrees ...
    assert m["tv"] > 0.05          # ... but distributions differ
    assert not is_equivalent(m)    # caught by TV gate, NOT a false SUPPORTED


def test_unrelated_is_not_equivalent():
    m = compare(torch.randn(12, 1, 1000), torch.randn(12, 1, 1000))
    assert not is_equivalent(m)


def test_missing_reference_is_not_equivalent():
    m = compare(None, torch.randn(4, 1, 10))
    assert not m["has_ref"] and not is_equivalent(m)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
