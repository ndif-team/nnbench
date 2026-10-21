"""Per-step capture cells (isb/methodologies/gen_capture.py): which row each decode step exports.
No GPU; a fake decode loop serves the block output each engine serves per step, on a family whose
module names differ from every bundled one."""
import pytest
import torch

import isb.methodologies  # noqa: F401  (registers the cells)
from isb.methodologies.registry import CELLS
from isb.profiles import ModelProfile

PROFILE = ModelProfile("probe", "trunk.decoder_blocks", "trunk.final_norm",
                       head_path="output_projection",
                       residual_denotation={"hf": "plain", "vllm": "fused"})
D = 4


class _Model:
    def __init__(self):
        self.trunk = type("Trunk", (), {})()
        self.trunk.decoder_blocks = [type("Block", (), {})() for _ in range(2)]


class _Generate:
    """Fake be.generate. The prefill step serves every prompt token, each decode step one: HF as
    `[1, seq, d]` with the stream in `[0]`, vLLM as `[tokens, d]` (hidden, residual) pairs."""

    def __init__(self, name):
        self.name, self.bounded, self.expected = name, [], []

    def last(self, t):
        return t[:, -1, :] if self.name == "hf" else t[-1:, :]

    def generate(self, model, prompts, build_step, *, new_tokens, bounded):
        self.bounded.append(bounded)
        (prompt,) = prompts
        rows = []
        for step in range(new_tokens):
            n = len(prompt.split()) if step == 0 else 1
            hidden = torch.arange(n * D, dtype=torch.float).reshape(n, D) + 100 * step
            if self.name == "hf":
                model.trunk.decoder_blocks[1].output = (hidden[None], "attention weights")
                self.expected.append(hidden[-1:])
            else:
                residual = 2 * hidden - 5
                model.trunk.decoder_blocks[1].output = (hidden, residual)
                self.expected.append((hidden + residual)[-1:])
            rows.append(build_step())
        return torch.cat(rows)


@pytest.mark.parametrize("backend", ["hf", "vllm_async"])
@pytest.mark.parametrize("bound", ["bounded", "unbounded"])
def test_each_step_exports_the_newest_row_of_the_stream(backend, bound):
    be = _Generate(backend)
    out = CELLS[("gen_capture", "*", backend)](be, _Model(), PROFILE, ["the cat sat"], layer=1,
                                               new_tokens=4, bound=bound)
    assert out.shape == (4, D)
    assert torch.equal(out, torch.cat(be.expected))   # vLLM: hidden + residual, newest row
    assert be.bounded == [bound == "bounded"]
