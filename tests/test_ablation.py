"""Ablation-methodology tests (isb/methodologies/ablation.py) — no GPU; torch only.

Pin the knockout mechanics: the target submodule's output is zeroed by REPLACEMENT (tensor or
tuple-stream-0), `target="none"` leaves it untouched (the baseline), and the readout is the final
block's last-token logits.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import isb.methodologies  # noqa: F401,E402  (registers cells)
from isb.methodologies.ablation import _ablate_and_read, _target_module  # noqa: E402
from isb.methodologies.registry import CELLS, get_cell  # noqa: E402

HID = 4


class _Mod:
    def __init__(self, out):
        self.output = out


class _Block:
    def __init__(self, mlp_out, attn_out):
        self.mlp = _Mod(mlp_out)
        self.attn = _Mod(attn_out)


class _Head:
    def __init__(self):
        self.weight = torch.eye(HID, HID)


def _last(t):
    return t[-1:, :]


def test_cells_registered():
    assert ("ablation", "gpt2", "hf") in CELLS
    assert ("ablation", "gpt2", "vllm_async") in CELLS
    assert get_cell("ablation", "gpt2", "hf") is not None


def test_target_routing():
    blk = _Block(torch.ones(2, HID), (torch.ones(2, HID), "w"))
    assert _target_module(blk, "mlp") is blk.mlp
    assert _target_module(blk, "attn") is blk.attn
    assert _target_module(blk, "none") is blk.mlp  # placeholder; never read for "none"
    raised = False
    try:
        _target_module(blk, "bogus")
    except ValueError:
        raised = True
    assert raised


def test_ablate_zeros_tensor_output():
    mlp = _Mod(torch.ones(2, HID))
    final = _Mod(torch.ones(2, HID))
    blocks = [final]  # blocks[-1] is the readout block
    _ablate_and_read(mlp, blocks, lambda x: x, _Head(),
                     target="mlp", residual="plain", last_fn=_last)
    assert torch.count_nonzero(mlp.output) == 0  # knocked out


def test_ablate_zeros_tuple_stream0_keeps_tail():
    attn = _Mod((torch.ones(2, HID), "weights"))
    blocks = [_Mod(torch.ones(2, HID))]
    _ablate_and_read(attn, blocks, lambda x: x, _Head(),
                     target="attn", residual="plain", last_fn=_last)
    assert torch.count_nonzero(attn.output[0]) == 0   # stream 0 zeroed
    assert attn.output[1] == "weights"                # tail preserved


def test_target_none_is_baseline_noop():
    mlp = _Mod(torch.full((2, HID), 5.0))
    before = mlp.output.clone()
    blocks = [_Mod(torch.ones(2, HID))]
    _ablate_and_read(mlp, blocks, lambda x: x, _Head(),
                     target="none", residual="plain", last_fn=_last)
    assert torch.equal(mlp.output, before)            # no write at all


def test_readout_is_final_block_last_token():
    final = _Mod(torch.tensor([[0.0, 0, 0, 0], [3.0, 0, 0, 0]]))
    out = _ablate_and_read(_Mod(torch.ones(2, HID)), [final], lambda x: x, _Head(),
                           target="none", residual="plain", last_fn=_last)
    assert out.shape == (1, HID)
    assert out[0, 0].item() == 3.0


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
