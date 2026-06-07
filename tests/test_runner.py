"""Runner / per-family-control tests (design.md §12.2) — no GPU, needs torch."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from isb.runner.run import CellResult, evaluate  # noqa: E402
from isb.states import AppState  # noqa: E402


def _cell(family, backend, value):
    c = CellResult("logit_lens", family, backend, "", "RAN")
    c.value = value
    return c


def test_control_is_per_family_not_global():
    """vLLM-Llama must be judged against HF-Llama, never HF-GPT2 (§12.2)."""
    gpt2_ref = torch.randn(4, 1, 64)
    llama_ref = torch.randn(4, 1, 64)        # different from gpt2_ref
    cells = [
        _cell("gpt2", "hf", gpt2_ref),
        _cell("gpt2", "vllm_async", gpt2_ref + 1e-3),   # matches gpt2-hf
        _cell("llama", "hf", llama_ref),
        _cell("llama", "vllm_async", gpt2_ref),         # equals GPT-2's ref, NOT llama's
    ]
    evaluate(cells)
    st = {(c.family, c.backend): c.state for c in cells}
    assert st[("gpt2", "hf")] == AppState.SUPPORTED
    assert st[("gpt2", "vllm_async")] == AppState.SUPPORTED
    assert st[("llama", "hf")] == AppState.SUPPORTED
    # If control were global (first HF = gpt2), llama-vllm (==gpt2_ref) would wrongly pass.
    # With per-family control it is compared to llama-hf and is caught:
    assert st[("llama", "vllm_async")] == AppState.SILENTLY_WRONG


def test_no_reference_when_control_failed():
    cells = [
        _cell("gpt2", "vllm_async", torch.randn(4, 1, 8)),  # ran, no HF control present
    ]
    cells[0]  # vllm ran but there is no hf cell -> NO_REFERENCE
    evaluate(cells)
    assert cells[0].state == AppState.NO_REFERENCE


def test_error_cells_are_not_overwritten():
    err = CellResult("logit_lens", "gpt2", "vllm_async", "", AppState.ERROR, error="boom")
    hf = _cell("gpt2", "hf", torch.randn(4, 1, 8))
    evaluate([hf, err])
    assert err.state == AppState.ERROR           # preserved, not rescored
    assert hf.state == AppState.SUPPORTED


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
