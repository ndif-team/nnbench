"""Jacobian-lens methodology + spec tests (isb/methodologies/jacobian_lens.py) — no GPU; torch only.

Pins: the seeded transport map (deterministic + orthogonal, so both backends build the SAME matrix
and the oracle compares like with like), the identity-vs-transport distinction, the per-position
read on a fake model, and that the spec's workload really is the upstream multi-hop dataset.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import isb.methodologies  # noqa: F401,E402  (registers cells)
from isb.methodologies.jacobian_lens import _orthogonal  # noqa: E402
from isb.methodologies.registry import CELLS, get_cell  # noqa: E402
from isb.specs import SPECS, default_specs  # noqa: E402

HID = 6


def test_orthogonal_map_is_deterministic_and_orthogonal():
    q1, q2 = _orthogonal(HID, seed=0), _orthogonal(HID, seed=0)
    assert torch.equal(q1, q2)                                   # same seed -> byte-identical
    assert not torch.equal(q1, _orthogonal(HID, seed=1))         # seed matters
    assert torch.allclose(q1 @ q1.T, torch.eye(HID), atol=1e-5)  # full-rank, norm-preserving


def test_registered_generic_and_resolves_per_family():
    assert ("jacobian_lens", "*", "hf") in CELLS
    assert ("jacobian_lens", "*", "vllm_async") in CELLS
    for fam in ("gpt2", "llama"):
        assert get_cell("jacobian_lens", fam, "hf") is not None
        assert get_cell("jacobian_lens", fam, "vllm_async") is not None


class _Mod:
    def __init__(self, out):
        self.output = out


class _Head:
    def __init__(self):
        self.weight = torch.eye(HID, HID)


class _GPT2ish:
    """gpt2-tree fake: transformer.h / transformer.ln_f / lm_head; [B, S, D] outputs."""
    def __init__(self, n_layers=3, seq=4):
        blocks = [_Mod(torch.randn(1, seq, HID)) for _ in range(n_layers)]
        self.transformer = SimpleNamespace(h=blocks, ln_f=lambda x: x)
        self.lm_head = _Head()
        self.tokenizer = None            # position="last" never touches it


class _FakeBE:
    name = "hf"

    def run(self, model, prompts, build):
        return build()

    def last(self, t):
        return t[:, -1, :]


def test_cell_reads_position_and_transport_changes_readout():
    model = _GPT2ish(n_layers=3)
    fn = get_cell("jacobian_lens", "gpt2", "hf")
    ident = fn(_FakeBE(), model, ["p"], unembed="weight", transport=None)
    assert ident.shape == (3, 1, HID)                            # [band, row, vocab]
    # identity transport at position -1 == the raw last-position residual through eye head
    assert torch.allclose(ident[0, 0], model.transformer.h[0].output[0, -1, :])
    moved = fn(_FakeBE(), model, ["p"], unembed="weight", transport=0)
    assert moved.shape == ident.shape
    assert not torch.allclose(moved, ident)                      # the J-matmul actually applied
    again = fn(_FakeBE(), model, ["p"], unembed="weight", transport=0)
    assert torch.equal(moved, again)                             # deterministic across calls


def test_per_layer_mapping_transport_applies_each_layers_own_map():
    # the fitted lens is one map PER LAYER: layer 0 gets J0, layer 1 gets J1; a band layer with no
    # map must raise (KeyError), never silently fall back to identity
    model = _GPT2ish(n_layers=2)
    fn = get_cell("jacobian_lens", "gpt2", "hf")
    J0, J1 = torch.eye(HID), 2.0 * torch.eye(HID)                # identity vs doubling
    out = fn(_FakeBE(), model, ["p"], unembed="weight", transport={0: J0, 1: J1})
    ident = fn(_FakeBE(), model, ["p"], unembed="weight", transport=None)
    assert torch.allclose(out[0], ident[0])                      # layer 0: J0 = identity
    assert torch.allclose(out[1], 2.0 * ident[1])                # layer 1: doubled by J1
    try:
        fn(_FakeBE(), model, ["p"], unembed="weight", transport={0: J0})   # layer 1 missing
        raise AssertionError("missing per-layer map must raise, not no-op")
    except KeyError:
        pass


def test_hub_transport_spec_parses_and_rejects_malformed():
    from isb.methodologies.jacobian_lens import _parse_hub
    repo, rev, fname = _parse_hub("hub:neuronpedia/jacobian-lens@qwen-n1000:a/b/lens.pt")
    assert (repo, rev, fname) == ("neuronpedia/jacobian-lens", "qwen-n1000", "a/b/lens.pt")
    for bad in ("hub:no-revision:file.pt", "hub:repo@rev", "hub:@rev:f.pt"):
        try:
            _parse_hub(bad)
            raise AssertionError(f"malformed hub spec must raise: {bad!r}")
        except ValueError:
            pass


def test_spec_workload_is_the_upstream_multihop_dataset():
    spec = SPECS["jacobian_lens_gpt2"]
    assert spec.name in default_specs()                          # gpt2-scale -> in the default sweep
    prompts = spec.workloads[0].prompts
    assert len(prompts) == 93                                    # the shipped multihop item count
    assert any("Carnival" in p for p in prompts)                 # really the upstream data
    labels = [label for _, label in spec.tasks]
    assert any("identity" in l for l in labels) and any("orthogonal" in l for l in labels)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
