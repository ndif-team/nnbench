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
    """The model's head: `.output` is its [B, S, vocab] logits (the HF read site)."""

    def __init__(self, output=None):
        self.output = torch.zeros(1, 2, HID) if output is None else output


class _Model:
    """`.logits` is the engine's [tokens, vocab] logits (the vLLM read site)."""

    def __init__(self, logits=None):
        self.logits = torch.zeros(2, HID) if logits is None else logits


def test_cells_registered():
    # family-generic (§12.8): registered once under family="*", resolved per profiled family
    assert ("ablation", "*", "hf") in CELLS
    assert ("ablation", "*", "vllm_async") in CELLS
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
    _ablate_and_read(mlp, _Model(), _Head(),
                     target="mlp", site="head")
    assert torch.count_nonzero(mlp.output) == 0  # knocked out


def test_ablate_zeros_tuple_stream0_keeps_tail():
    attn = _Mod((torch.ones(2, HID), "weights"))
    blocks = [_Mod(torch.ones(2, HID))]
    _ablate_and_read(attn, _Model(), _Head(),
                     target="attn", site="head")
    assert torch.count_nonzero(attn.output[0]) == 0   # stream 0 zeroed
    assert attn.output[1] == "weights"                # tail preserved


def test_target_none_is_baseline_noop():
    mlp = _Mod(torch.full((2, HID), 5.0))
    before = mlp.output.clone()
    blocks = [_Mod(torch.ones(2, HID))]
    _ablate_and_read(mlp, _Model(), _Head(),
                     target="none", site="head")
    assert torch.equal(mlp.output, before)            # no write at all


def test_readout_is_the_documented_site_last_token():
    logits = torch.tensor([[[0.0, 0, 0, 0], [3.0, 0, 0, 0]]])            # [B=1, S=2, vocab]
    out = _ablate_and_read(_Mod(torch.ones(2, HID)), _Model(), _Head(logits),
                           target="none", site="head")                  # HF: lm_head.output
    assert out.shape == (1, HID)
    assert out[0, 0].item() == 3.0
    engine = _Model(torch.tensor([[0.0, 0, 0, 0], [0.0, 7.0, 0, 0]]))   # vLLM: model.logits
    out = _ablate_and_read(_Mod(torch.ones(2, HID)), engine, _Head(), target="none", site="engine")
    assert out.shape == (1, HID)
    assert out[0, 1].item() == 7.0


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
