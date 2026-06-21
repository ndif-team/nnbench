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


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
