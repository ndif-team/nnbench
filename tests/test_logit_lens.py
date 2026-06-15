"""logit-lens residual-extraction tests (isb/methodologies/logit_lens.py) — no GPU; torch only.

`_resid` is load-bearing for the vLLM-Llama SILENTLY_WRONG finding (the fused-residual denotation
mismatch — a plain read is silently wrong): vLLM's fused-residual
RMSNorm layers return `(hidden, residual)` whose SUM is the real residual stream. `plain` (the
GPT-2 idiom) reads only `hidden` -> silently wrong; `fused` reconstructs the sum. These tests pin
that distinction without a model.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isb.methodologies.logit_lens import _resid, _untuple  # noqa: E402


def test_plain_takes_first_element_or_bare():
    h, r = torch.ones(2, 4), torch.full((2, 4), 9.0)
    assert torch.equal(_resid((h, r), "plain"), h)        # ignores residual -> the GPT-2 idiom
    assert torch.equal(_resid(h, "plain"), h)             # bare tensor passes through


def test_fused_sums_hidden_and_residual():
    h, r = torch.ones(2, 4), torch.full((2, 4), 9.0)
    out = _resid((h, r), "fused")
    assert torch.equal(out, h + r)                        # the real residual stream = hidden+residual
    # the whole point: fused != plain when residual is non-zero, which is why plain is silently wrong
    assert not torch.equal(out, _resid((h, r), "plain"))


def test_fused_falls_back_when_not_a_2tuple():
    h = torch.ones(2, 4)
    assert torch.equal(_resid(h, "fused"), h)             # bare tensor -> plain (safe on HF)
    assert torch.equal(_resid((h,), "fused"), h)          # 1-tuple -> plain (HF single-output block)


def test_fused_safe_on_hf_kv_tuple_via_plain_only():
    # An HF block may return (hidden, past_kv). We must NEVER fuse those (garbage); the HF cell
    # uses residual="plain", so verify plain ignores element 1 entirely.
    hidden, fake_kv = torch.ones(2, 4), "DynamicCache"
    assert torch.equal(_resid((hidden, fake_kv), "plain"), hidden)


def test_untuple_helper():
    h = torch.ones(3)
    assert torch.equal(_untuple((h, "x")), h)
    assert torch.equal(_untuple(h), h)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
