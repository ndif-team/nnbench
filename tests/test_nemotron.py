"""Nemotron family tests (NVIDIA Nemotron 3 Nano 30B-A3B — hybrid Mamba/MoE) — no GPU; torch only.

Pins the nemotron-specific logic without a 30B model: the single-op-per-layer ablation target
(`block.mixer`, not GPT-2/Llama's within-block `mlp`/`attn`), and that the cells read the NemotronH
module tree (`model.backbone.layers` / `.norm_f` / `lm_head`) rather than any other family's names.
The residual read/write math itself is the shared `_lens_proxy`/`_steer_and_read`/`_ablate_and_read`
already covered by the logit_lens/steering/ablation tests — nemotron only rewires the module paths,
so the fake model below deliberately carries ONLY NemotronH names (no `model.layers`, no
`transformer.h`): a cell that silently assumed another family would AttributeError here.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import isb.methodologies  # noqa: F401,E402  (registers cells)
from isb.methodologies.ablation import _ablate_and_read, _target_module_nemotron  # noqa: E402
from isb.methodologies.registry import CELLS, get_cell  # noqa: E402

HID = 4


class _Mod:
    def __init__(self, out):
        self.output = out


class _NemoBlock:
    """A NemotronH block: ONE op exposed as `.mixer` (Mamba | attention | MLP | MoE) and a residual
    output. Deliberately has NO `.mlp`/`.attn`/`.self_attn` — the hybrid stack has no within-block
    component split."""
    def __init__(self, resid_out, block_type="mamba"):
        self.output = resid_out
        self.block_type = block_type
        self.mixer = _Mod(torch.ones(2, HID))


class _Norm:
    def __call__(self, x):
        return x                                   # identity stand-in for RMSNorm


class _Head:
    def __init__(self):
        self.weight = torch.eye(HID, HID)

    def __call__(self, x):                         # idiomatic unembed path (unembed="module")
        return torch.nn.functional.linear(x, self.weight)


class _Backbone:
    def __init__(self, layers):
        self.layers = layers
        self.norm_f = _Norm()


class _NemoModel:
    """A NemotronH-shaped model carrying ONLY the NemotronH names."""
    def __init__(self, n_layers=6):
        # distinct per-layer residuals so the lens stack is non-degenerate
        layers = [_NemoBlock(torch.full((2, HID), float(i))) for i in range(n_layers)]
        self.backbone = _Backbone(layers)
        self.lm_head = _Head()


class _FakeBE:
    """Runs the cell's build() closure with no trace/GPU — the no-GPU cell harness."""
    def run(self, model, prompts, build):
        return build()

    def last(self, t):
        return t[-1:, :]                           # [seq, vocab] -> [1, vocab]


def test_cells_registered_for_both_backends():
    for methodology in ("logit_lens", "steering", "ablation"):
        for backend in ("hf", "vllm_async"):
            assert (methodology, "nemotron", backend) in CELLS, (methodology, backend)
            assert get_cell(methodology, "nemotron", backend) is not None
    # every vllm_* variant falls back to the vllm_async nemotron cell (generalized routing)
    assert get_cell("logit_lens", "nemotron", "vllm_pp") is get_cell("logit_lens", "nemotron", "vllm_async")


def test_target_module_is_the_single_mixer():
    blk = _NemoBlock(torch.ones(2, HID), block_type="moe")
    assert _target_module_nemotron(blk, "mixer") is blk.mixer
    assert _target_module_nemotron(blk, "none") is blk.mixer    # placeholder; never read for "none"


def test_target_module_rejects_within_block_component_targets():
    # GPT-2/Llama targets ("mlp"/"attn") are NOT valid for a hybrid single-op block — must raise,
    # not silently fall through to a wrong submodule.
    blk = _NemoBlock(torch.ones(2, HID))
    for bad in ("mlp", "attn", "self_attn", "bogus"):
        raised = False
        try:
            _target_module_nemotron(blk, bad)
        except ValueError:
            raised = True
        assert raised, bad


def test_ablate_makes_layer_identity():
    # zeroing the mixer output -> hidden = residual + 0; the mechanic just zeroes target.output
    mixer = _Mod(torch.ones(2, HID))
    final = _Mod(torch.full((2, HID), 2.0))
    _ablate_and_read(mixer, [final], _Norm(), _Head(),
                     target="mixer", residual="plain", last_fn=lambda t: t[-1:, :])
    assert torch.count_nonzero(mixer.output) == 0


def test_logit_lens_reads_backbone_tree():
    model = _NemoModel(n_layers=6)
    cell = get_cell("logit_lens", "nemotron", "hf")
    out = cell(_FakeBE(), model, ["p"], unembed="weight", residual="plain")
    assert out.shape == (6, 1, HID)                # [n_layers, last-token=1, vocab]; identity norm+eye head
    # layer i's residual is constant i -> its last-token row is all-i (eye head, identity norm)
    assert torch.equal(out[3, 0], torch.full((HID,), 3.0))


def test_logit_lens_module_unembed_uses_head_call():
    model = _NemoModel(n_layers=3)
    cell = get_cell("logit_lens", "nemotron", "hf")
    out = cell(_FakeBE(), model, ["p"], unembed="module")   # exercises head(normed), not the weight matmul
    assert out.shape == (3, 1, HID)


def test_cell_does_not_assume_other_family_names():
    # the fake model has backbone.* only; a cell reaching for model.model.layers / transformer.h would
    # AttributeError. Success of the run above already proves backbone; assert the negative explicitly.
    model = _NemoModel(n_layers=2)
    assert not hasattr(model, "model")
    assert not hasattr(model, "transformer")
    get_cell("logit_lens", "nemotron", "hf")(_FakeBE(), model, ["p"], unembed="weight")  # must not raise


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
