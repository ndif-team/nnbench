"""Runner / per-family-control tests (design.md §12.2) — no GPU, needs torch."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from isb.runner.run import CellResult, disambiguate_precision, evaluate  # noqa: E402
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


def test_dtype_control_reclassifies_precision_not_bug():
    """A near-tie SILENTLY_WRONG that matches the control at the control's dtype -> SUPPORTED_DEGRADED;
    one that still diverges at the control's dtype stays SILENTLY_WRONG."""
    ref = torch.zeros(1, 8); ref[0, 3] = 5.0           # control argmax = index 3
    near = ref.clone(); near[0, 3] = 4.9; near[0, 2] = 4.95   # bf16-ish near-tie: argmax flipped to 2
    far = torch.zeros(1, 8); far[0, 7] = 9.0            # genuinely different distribution

    degraded = _cell("gpt2", "vllm_async", None); degraded.state = AppState.SILENTLY_WRONG
    realbug = _cell("gpt2", "vllm_async", None); realbug.state = AppState.SILENTLY_WRONG
    control = _cell("gpt2", "hf", ref); control.state = AppState.SUPPORTED

    # rerun-at-control-dtype: the degraded cell becomes ref-equal at fp32; the real bug stays `far`
    fp32 = {id(degraded): ref.clone(), id(realbug): far}
    disambiguate_precision([control, degraded, realbug], ref, lambda c: fp32[id(c)])

    assert degraded.state == AppState.SUPPORTED_DEGRADED
    assert realbug.state == AppState.SILENTLY_WRONG
    assert control.state == AppState.SUPPORTED          # control untouched


def test_dtype_control_noop_without_control_value():
    cell = _cell("gpt2", "vllm_async", None); cell.state = AppState.SILENTLY_WRONG
    called = []
    disambiguate_precision([cell], None, lambda c: called.append(c) or None)
    assert cell.state == AppState.SILENTLY_WRONG        # unchanged
    assert called == []                                 # rerun never invoked


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
