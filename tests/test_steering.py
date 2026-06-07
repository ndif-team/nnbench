"""Steering-methodology tests (design.md §12; isb/methodologies/steering.py) — no GPU; torch only.

These verify the WRITE mechanics in isolation (the part that makes vLLM diverge): in-place mutates
the live residual buffer, replace swaps in a NEW tensor leaving the original untouched, and
alpha=0 is a true no-op baseline. The cross-backend verdict itself is exercised by the GPU smoke;
here we pin the semantics a backend must honor for that verdict to mean anything.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import isb.methodologies  # noqa: F401,E402  (registers cells)
from isb.methodologies.registry import CELLS, get_cell  # noqa: E402
from isb.methodologies.steering import _resolve_token, _steer_and_read  # noqa: E402

VOCAB, HID = 11, 4


class _Block:
    """Minimal Envoy stand-in: `.output` is a readable/writable attribute (tuple or tensor)."""

    def __init__(self, out):
        self.output = out


class _Head:
    def __init__(self):
        self.weight = torch.eye(VOCAB, HID)  # rows are unit-ish directions; row i picks dim i


def _last(t):
    return t[-1:, :]


def _blocks(steer_out, final_hidden):
    # block 0 = the steer target; block -1 = the readout (kept distinct, like layer 8 vs 11)
    return [_Block(steer_out), _Block(final_hidden)]


def test_cells_registered():
    assert ("steering", "gpt2", "hf") in CELLS
    assert ("steering", "gpt2", "vllm_async") in CELLS
    assert get_cell("steering", "gpt2", "hf") is not None


def test_inplace_mutates_live_buffer():
    hidden = torch.ones(2, HID)  # non-zero: relative scale = ‖residual‖ must be > 0 to steer
    before = hidden.clone()
    blocks = _blocks((hidden, "kv"), torch.ones(2, HID))
    ptr_before = hidden.data_ptr()
    _steer_and_read(blocks, lambda x: x, _Head(),
                    layer=0, token_id=1, alpha=3.0, mode="inplace", last_fn=_last)
    # same storage, values now changed -> the write landed in the existing buffer (vLLM-fragile form)
    assert blocks[0].output[0].data_ptr() == ptr_before
    assert not torch.equal(blocks[0].output[0], before)
    assert blocks[0].output[1] == "kv"  # tuple tail preserved


def test_replace_swaps_new_tensor_leaving_original():
    hidden = torch.ones(2, HID)
    before = hidden.clone()
    blocks = _blocks((hidden, "kv"), torch.ones(2, HID))
    _steer_and_read(blocks, lambda x: x, _Head(),
                    layer=0, token_id=1, alpha=3.0, mode="replace", last_fn=_last)
    assert blocks[0].output[0].data_ptr() != hidden.data_ptr()   # a NEW tensor was assigned
    assert torch.equal(hidden, before)                           # original buffer untouched
    assert not torch.equal(blocks[0].output[0], before)          # replacement carries the steer
    assert blocks[0].output[1] == "kv"


def test_alpha_zero_is_true_noop():
    hidden = torch.full((2, HID), 5.0)
    blocks = _blocks((hidden.clone(), "kv"), torch.ones(2, HID))
    before = blocks[0].output[0].clone()
    _steer_and_read(blocks, lambda x: x, _Head(),
                    layer=0, token_id=1, alpha=0.0, mode="inplace", last_fn=_last)
    assert torch.equal(blocks[0].output[0], before)  # no write at all


def test_bare_tensor_output_supported():
    """vLLM blocks may yield a bare tensor (not a tuple); replace must still work."""
    hidden = torch.ones(3, HID)
    before = hidden.clone()
    blocks = [_Block(hidden), _Block(torch.ones(3, HID))]
    out = _steer_and_read(blocks, lambda x: x, _Head(),
                          layer=0, token_id=2, alpha=2.0, mode="replace", last_fn=_last)
    assert not isinstance(blocks[0].output, tuple)
    assert not torch.equal(blocks[0].output, before)
    assert out.shape == (1, VOCAB)


def test_readout_reflects_final_block_last_token():
    blocks = _blocks((torch.zeros(2, HID), "kv"), torch.tensor([[0.0, 0, 0, 0], [9.0, 0, 0, 0]]))
    out = _steer_and_read(blocks, lambda x: x, _Head(),
                          layer=0, token_id=1, alpha=1.0, mode="replace", last_fn=_last)
    # identity norm + eye head -> logits == final hidden; last token's dim-0 is the 9.0 row
    assert out.shape == (1, VOCAB)
    assert out[0, 0].item() == 9.0


def test_scale_is_relative_to_residual_norm():
    """alpha is dimensionless: doubling the residual norm doubles the applied vec (self-calibration)."""
    head = _Head()

    def delta(magnitude):
        hidden = torch.full((2, HID), magnitude)
        before = hidden.clone()
        blocks = _blocks((hidden, "kv"), torch.ones(2, HID))
        _steer_and_read(blocks, lambda x: x, head,
                        layer=0, token_id=1, alpha=2.0, mode="inplace", last_fn=_last)
        return blocks[0].output[0] - before

    d1 = delta(1.0)   # row-norm 2
    d2 = delta(2.0)   # row-norm 4 -> vec must double
    assert torch.allclose(d2, 2 * d1, atol=1e-5)
    # steer lies along the token's unembed direction (eye head row 1 -> dim 1 only)
    assert d1[:, 1].abs().sum() > 0
    assert d1[:, 0].abs().sum() == 0


def test_steer_layer_cannot_equal_readout_block():
    """layer == final block would be a same-module write-then-read; must raise, not silently risk it."""
    blocks = [_Block((torch.ones(2, HID), "kv")), _Block(torch.ones(2, HID))]  # last index = 1
    for bad in (1, -1):                                  # positive and negative spellings of "last"
        raised = False
        try:
            _steer_and_read(blocks, lambda x: x, _Head(),
                            layer=bad, token_id=1, alpha=2.0, mode="inplace", last_fn=_last)
        except ValueError:
            raised = True
        assert raised, f"steering the read-out block (layer={bad}) must raise"


def test_resolve_token_takes_last_piece():
    class _Tok:
        def __call__(self, s, add_special_tokens=False):
            return {"input_ids": [10, 20, 30]}

    assert _resolve_token(_Tok(), "anything") == 30


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
