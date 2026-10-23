"""The vLLM backends share one engine-config signature (isb/backends/) — no GPU.

The driver splats a spec's `vllm_kwargs` (e.g. trust_remote_code for the NemotronH auto_map repo) into
ANY vLLM backend constructor, so all three must accept the same engine-config. The bug this guards:
trust_remote_code reached only the async backend, so VLLMSyncBackend(**{"trust_remote_code": True})
(in _fp32_rerun / vllm_sync / vllm_serve runs) raised TypeError and silently no-op'd disambiguation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isb.backends import VLLMAsyncBackend, VLLMServeBackend, VLLMSyncBackend  # noqa: E402
from isb.backends.vllm_base import VLLMBackend  # noqa: E402


def _vllm_backends(**kw):
    # exactly how the driver builds them: serve takes a positional host, all take the shared config.
    return [VLLMAsyncBackend(**kw), VLLMSyncBackend(**kw), VLLMServeBackend("http://x", **kw)]


def test_all_vllm_backends_accept_the_shared_engine_config():
    for be in _vllm_backends(dtype="float32", trust_remote_code=True):
        assert isinstance(be, VLLMBackend)
        assert be.dtype == "float32"
        assert be.trust_remote_code is True


def test_trust_remote_code_splat_does_not_raise_on_any_backend():
    # the regression: driver does Backend(**spec.vllm_kwargs) with {"trust_remote_code": True}.
    for be in _vllm_backends(trust_remote_code=True):
        assert be.trust_remote_code is True


def test_engine_kwargs_built_only_from_set_config():
    assert VLLMAsyncBackend()._engine_kwargs() == {}                      # default GPT-2 spec: no overrides
    assert VLLMSyncBackend(dtype="float32")._engine_kwargs() == {"dtype": "float32"}
    assert VLLMSyncBackend(trust_remote_code=True)._engine_kwargs() == {"trust_remote_code": True}


def test_async_teardown_drains_pending_handlers():
    import asyncio
    from types import SimpleNamespace

    backend = VLLMAsyncBackend()
    loop = backend._loop = asyncio.new_event_loop()
    finished = []

    async def handler():
        try:
            await asyncio.Future()
        finally:
            finished.append(True)

    task = loop.create_task(handler())
    loop.run_until_complete(asyncio.sleep(0))
    backend.teardown(SimpleNamespace(vllm_entrypoint=SimpleNamespace(shutdown=lambda: None)))
    assert task.done() and finished == [True]
    assert loop.is_closed() and backend._loop is None


def test_concurrent_session_overlaps_requests_submitted_from_several_threads():
    """Inside a concurrent session the loop runs on its own thread and client threads submit to
    it, so two 50 ms requests issued together finish together, as they must for the open loop."""
    import asyncio
    import threading
    import time

    from isb.backends.async_loop import AsyncLoopMixin

    class Loop(AsyncLoopMixin):
        pass

    be = Loop()
    assert be.concurrent_requests

    async def request():
        await asyncio.sleep(0.05)
        return threading.current_thread().name

    with be.concurrent_session():
        results = []
        threads = [threading.Thread(target=lambda: results.append(be._run_coro(request())))
                   for _ in range(2)]
        start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.perf_counter() - start
    assert len(results) == 2 and all(name == "isb-vllm-loop" for name in results)
    assert elapsed < 0.09                                  # overlapped, not 2 x 50 ms
    assert be._loop_thread is None
    be._close_loop()
    assert be._loop is None


def test_serve_and_interp_engine_take_overlapping_requests_without_a_session():
    """Both interfaces take overlapping requests by construction (an HTTP request per trace; one
    loop thread the sync facade submits to), so they declare it and their session opens nothing."""
    import importlib.util
    from pathlib import Path

    from isb.backends.vllm_serve import VLLMServeBackend

    path = Path(__file__).resolve().parents[1] / "backends/interp-engine-vllm/backend.py"
    spec = importlib.util.spec_from_file_location("interp_engine_backend", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for backend in (VLLMServeBackend("http://server:6677"),
                    module.InterpEngineBackend.__new__(module.InterpEngineBackend)):
        assert backend.concurrent_requests
        with backend.concurrent_session():
            pass


def test_plain_async_config_records_its_entrypoint_and_mode():
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace

    path = Path(__file__).resolve().parents[1] / "backends/vllm-plain/async/backend.py"
    spec = importlib.util.spec_from_file_location("plain_async_backend", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    backend, run = module.create_backend(SimpleNamespace(vllm_kwargs={"enforce_eager": True}))
    assert backend.concurrent_requests and backend.name == "vllm_plain"
    assert run.engine.mode == "plain" and run.engine.params["entrypoint"] == "AsyncLLM"
    assert run.engine.params["enforce_eager"] is True


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
