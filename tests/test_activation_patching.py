"""Activation-patching tests (isb/methodologies/activation_patching.py) — no GPU; torch only.

Pin the transplant mechanics in isolation: capture snapshots an INDEPENDENT clone of the clean
residual; the patch REPLACES the target layer's residual with it (whole-tuple, vLLM-safe); a
clean/corrupted length mismatch fails LOUD rather than broadcasting into a silently-wrong patch.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import isb.methodologies  # noqa: F401,E402  (registers cells)
from isb.methodologies.activation_patching import _capture, _patch_and_read  # noqa: E402
from isb.methodologies.registry import CELLS, get_cell  # noqa: E402

HID = 4


class _Block:
    def __init__(self, out):
        self.output = out


class _Head:
    def __init__(self):
        self.weight = torch.eye(HID, HID)


def _last(t):
    return t[-1:, :]


def test_cells_registered():
    assert ("activation_patching", "gpt2", "hf") in CELLS
    assert ("activation_patching", "gpt2", "vllm_async") in CELLS
    assert get_cell("activation_patching", "gpt2", "hf") is not None


def test_capture_clones_independent_snapshot():
    hidden = torch.ones(2, HID)
    snap = _capture([_Block((hidden, "kv"))], 0, "plain")
    hidden[:] = 9.0                                  # mutate the live buffer after capture
    assert not torch.equal(snap, hidden)             # snapshot unaffected -> it was cloned
    assert snap.shape == (2, HID)


def test_capture_fused_sums_hidden_and_residual():
    h, r = torch.ones(2, HID), torch.full((2, HID), 3.0)
    snap = _capture([_Block((h, r))], 0, "fused")
    assert torch.equal(snap, h + r)                  # fused reconstruction (vLLM-Llama, F-7)


def test_patch_replaces_target_layer_residual():
    blocks = [_Block((torch.zeros(2, HID), "kv")), _Block(torch.ones(2, HID))]  # patch 0, read 1
    clean = torch.full((2, HID), 5.0)
    _patch_and_read(blocks, lambda x: x, _Head(),
                    layer=0, clean_act=clean, residual="plain", last_fn=_last)
    assert torch.equal(blocks[0].output[0], clean)   # transplanted
    assert blocks[0].output[1] == "kv"               # tuple tail preserved


def test_patch_reads_final_block_logits():
    final = torch.tensor([[0.0, 0, 0, 0], [7.0, 0, 0, 0]])
    blocks = [_Block((torch.zeros(2, HID), "kv")), _Block(final)]
    out = _patch_and_read(blocks, lambda x: x, _Head(),
                          layer=0, clean_act=torch.ones(2, HID), residual="plain", last_fn=_last)
    assert out.shape == (1, HID)
    assert out[0, 0].item() == 7.0                   # identity norm + eye head -> last-token row


def test_patch_shape_mismatch_raises():
    blocks = [_Block((torch.zeros(2, HID), "kv")), _Block(torch.ones(2, HID))]
    clean = torch.zeros(3, HID)                       # clean/corrupted length mismatch
    raised = False
    try:
        _patch_and_read(blocks, lambda x: x, _Head(),
                        layer=0, clean_act=clean, residual="plain", last_fn=_last)
    except ValueError:
        raised = True
    assert raised, "a clean/corrupted length mismatch must fail loud, not silently broadcast"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
