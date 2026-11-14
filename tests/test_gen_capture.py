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
        self.name, self.bounded, self.expected, self.per_step = name, [], [], []

    def last(self, t):
        return t[:, -1, :] if self.name == "hf" else t[-1:, :]

    def generate(self, model, prompts, build_step, *, new_tokens, bounded, keep_dtype=False,
                 tokens=False):
        self.bounded.append(bounded)
        self.per_step.append(build_step is not None)
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
            row = None if build_step is None else build_step(step)
            rows.append(model.samples if tokens else row)
        return torch.stack(rows)                      # [new_tokens, 1, d] or [new_tokens, 1, 1]


@pytest.mark.parametrize("backend", ["hf", "vllm_async"])
@pytest.mark.parametrize("bound", ["bounded", "unbounded"])
def test_each_step_exports_the_newest_row_of_the_stream(backend, bound):
    be = _Generate(backend)
    out = CELLS[("gen_capture", "*", backend)](be, _Model(), PROFILE, ["the cat sat"], layer=1,
                                               new_tokens=4, bound=bound)
    assert out.shape == (4, 1, D)
    assert torch.equal(out[:, 0], torch.cat(be.expected))   # vLLM: hidden + residual, newest row
    assert be.bounded == [bound == "bounded"]


@pytest.mark.parametrize("backend", ["hf", "vllm_async"])
def test_baseline_is_a_decode_with_no_per_step_work_returning_its_tokens(backend):
    # the export=False baseline is the user's plain decode: no per-step read, the tokens it produced
    model = _Model()                                          # carries no `logits` to read
    model.samples = torch.tensor([[9]], dtype=torch.int32)
    be = _Generate(backend)
    out = CELLS[("gen_capture", "*", backend)](be, model, PROFILE, ["the cat sat"], layer=1,
                                               new_tokens=3, export=False)
    assert be.per_step == [False]
    assert out.dtype == torch.int32 and out.tolist() == [[[9]]] * 3
