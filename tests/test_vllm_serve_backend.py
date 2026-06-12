"""vllm_serve backend tests — no GPU.

Covers the backend's own logic: registration, the registry fallback (serve reuses the in-process vLLM
cell), health polling, the batched-not-yet guard, and the pure tensor/teardown helpers. The full
serialize -> POST -> push round-trip needs a meta VLLM model (which requires the CPU platform, i.e. a
GPU-less container — it crashes on a bare-metal GPU host) plus a live server, so it is exercised by the
containerized end-to-end test, not here.
"""
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import isb.methodologies  # noqa: F401,E402  (registers cells)
from isb.backends import IMPLS, VLLMServeBackend  # noqa: E402
from isb.methodologies.registry import get_cell  # noqa: E402
from isb.sweep.driver import _default_backend  # noqa: E402
from isb.specs import SPECS  # noqa: E402


def test_registered_in_impls():
    assert IMPLS.get("vllm_serve") is VLLMServeBackend
    assert VLLMServeBackend.name == "vllm_serve"


def test_registry_fallback_serve_reuses_async_cell():
    # No explicit vllm_serve cell is registered; get_cell must fall back to the vllm_async cell (the
    # serve backend runs the same vLLM model via the same intervention code, differing only in `be`).
    for method in ("logit_lens",):
        for family in ("gpt2", "llama"):
            assert (
                get_cell(method, family, "vllm_serve")
                is get_cell(method, family, "vllm_async")
                is not None
            )
    # a genuinely missing (method, family) is still None, not a spurious fallback
    assert get_cell("logit_lens", "no-such-family", "vllm_serve") is None


def test_init_strips_trailing_slash_and_keeps_dtype():
    be = VLLMServeBackend("http://server:6677/", dtype="float32")
    assert be.host == "http://server:6677"
    assert be.dtype == "float32"
    assert VLLMServeBackend("http://server:6677").dtype is None


def test_last_returns_final_token_row():
    # server returns flat [tokens, vocab]; last() takes the final row as [1, vocab]
    t = torch.arange(6.0).reshape(3, 2)
    out = VLLMServeBackend("http://x").last(t)
    assert out.shape == (1, 2)
    assert torch.equal(out, t[-1:, :])


def test_run_rejects_batched_before_touching_model():
    be = VLLMServeBackend("http://x")
    sentinel_model = object()  # must never be touched — the guard fires before model.trace
    raised = False
    try:
        be.run(sentinel_model, ["p1", "p2"], build=None)
    except NotImplementedError as e:
        raised = True
        assert "batched" in str(e).lower()
    assert raised, "multi-prompt serve must raise NotImplementedError, not silently drop prompts"


def test_health_wait_returns_on_200():
    be = VLLMServeBackend("http://server:6677/")
    calls = []

    class _Resp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False

    orig = urllib.request.urlopen
    urllib.request.urlopen = lambda url, timeout=None: calls.append(url) or _Resp()
    try:
        be._wait_for_health()  # returns without raising
    finally:
        urllib.request.urlopen = orig
    assert calls == ["http://server:6677/health"]


def test_health_wait_raises_after_timeout():
    be = VLLMServeBackend("http://server:6677")
    be.HEALTH_TIMEOUT = 0.05  # instance override so the test is fast

    def _boom(url, timeout=None):
        raise urllib.error.URLError("refused")

    orig_open, orig_sleep = urllib.request.urlopen, time.sleep
    urllib.request.urlopen = _boom
    time.sleep = lambda *_a, **_k: None  # don't actually sleep between retries
    try:
        raised = False
        try:
            be._wait_for_health()
        except RuntimeError as e:
            raised = True
            assert "not healthy" in str(e)
        assert raised, "an unreachable server must raise RuntimeError after the timeout"
    finally:
        urllib.request.urlopen, time.sleep = orig_open, orig_sleep


def test_teardown_is_clean_noop():
    # client holds no engine; teardown must not raise and must not require CUDA
    VLLMServeBackend("http://x").teardown(model=object())


def test_default_backend_factory_and_no_host_guard():
    spec = next(iter(SPECS.values()))
    be = _default_backend("vllm_serve", spec, serve_host="http://server:6677")
    assert isinstance(be, VLLMServeBackend) and be.host == "http://server:6677"
    raised = False
    try:
        _default_backend("vllm_serve", spec, serve_host=None)
    except ValueError:
        raised = True
    assert raised, "vllm_serve without a server URL must raise"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
