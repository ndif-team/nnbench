"""Activation harvesting cells (isb/methodologies/activation_harvest.py): what each engine's export
contains. No GPU; fake backends serve the block output each engine serves, on a family whose module
names differ from every bundled one."""
import pytest
import torch

import isb.methodologies  # noqa: F401  (registers the cells)
from isb.methodologies.registry import CELLS
from isb.profiles import ModelProfile

PROFILE = ModelProfile("probe", "trunk.decoder_blocks", "trunk.final_norm",
                       head_path="output_projection",
                       residual_denotation={"hf": "plain", "vllm": "fused"})
HF = CELLS[("activation_harvest", "*", "hf")]
VLLM = CELLS[("activation_harvest", "*", "vllm_async")]
D = 4


def _acts(prompt):
    """What the fake forward computes: one distinct row per whitespace token."""
    return torch.tensor([[len(w), j, sum(map(ord, w)) % 97, 1.0]
                         for j, w in enumerate(prompt.split())])


class _Tokenizer:
    def __init__(self, padding_side):
        self.padding_side = padding_side

    def __call__(self, text):
        return {"input_ids": text.split()}


class _Model:
    def __init__(self, padding_side="left"):
        self.trunk = type("Trunk", (), {})()
        self.trunk.decoder_blocks = [type("Block", (), {})() for _ in range(2)]
        self.tokenizer = _Tokenizer(padding_side)


class _HF:
    """One padded batch on the tokenizer's padding side (nnsight left-pads a causal decoder); pad
    positions hold values."""
    name = "hf"

    def __init__(self, extra_pad=0):
        self.extra_pad = extra_pad

    def run(self, model, prompts, build):
        acts = [_acts(p) for p in prompts]
        width = max(len(a) for a in acts) + self.extra_pad
        pad = lambda a: torch.full((width - len(a), D), 1e4)    # noqa: E731
        left = model.tokenizer.padding_side == "left"
        batch = torch.stack([torch.cat([pad(a), a] if left else [a, pad(a)]) for a in acts])
        model.trunk.decoder_blocks[1].output = (batch, "attention weights")
        return build().detach().float().cpu()


class _VLLM:
    """One request: `[tokens, d]` with no batch axis, a fused-residual layer's (hidden, residual)."""
    name = "vllm_async"

    def run(self, model, prompts, build):
        (prompt,) = prompts
        hidden = _acts(prompt)
        model.trunk.decoder_blocks[1].output = (hidden, 10 * hidden + 3)
        return build().detach().float().cpu()


def test_hf_exports_every_valid_token_in_prompt_order():
    prompts = ["the cat sat", "dogs", "a quick brown fox"]
    batched = HF(_HF(), _Model(), PROFILE, prompts, layer=1)
    assert torch.equal(batched, torch.cat([_acts(p) for p in prompts]))      # [8, d], no pad row
    alone = torch.cat([HF(_HF(), _Model(), PROFILE, [p], layer=1) for p in prompts])
    assert torch.equal(batched, alone)                     # the per-prompt reference's layout


def test_hf_refuses_a_batch_its_token_counts_do_not_describe():
    with pytest.raises(ValueError, match="padded to 5 positions"):
        HF(_HF(extra_pad=1), _Model(), PROFILE, ["the cat sat", "a quick brown fox"], layer=1)


def test_hf_refuses_a_right_padded_batch():
    prompts = ["the cat sat", "a quick brown fox"]
    with pytest.raises(ValueError, match="pads on the right"):
        HF(_HF(), _Model(padding_side="right"), PROFILE, prompts, layer=1)
    alone = HF(_HF(), _Model(padding_side="right"), PROFILE, prompts[:1], layer=1)
    assert torch.equal(alone, _acts(prompts[0]))           # one prompt has no padding to locate


def test_vllm_exports_the_summed_stream_on_a_fused_residual_family():
    hidden = _acts("the cat sat")
    out = VLLM(_VLLM(), _Model(), PROFILE, ["the cat sat"], layer=1)
    assert torch.equal(out, hidden + (10 * hidden + 3))    # hidden + residual at every token
    naive = VLLM(_VLLM(), _Model(), PROFILE, ["the cat sat"], layer=1, residual="plain")
    assert torch.equal(naive, hidden)                      # an explicit plain read stays possible
