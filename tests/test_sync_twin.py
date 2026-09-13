"""The batched regime of an async vLLM run executes on a sync-mode twin engine
(`sweep/execute._load_sync_twin`): nnsight's async engine runs one request per trace and
refuses several invokes, so the multi-invoke batched pattern needs the sync engine."""
from isb.backends import VLLMAsyncBackend, VLLMSyncBackend
from isb.sweep.execute import _load_sync_twin


def test_twin_mirrors_the_engine_config(monkeypatch):
    loaded = {}

    def fake_load(self, repo, gpu_memory_utilization=0.2):
        loaded["repo"] = repo
        loaded["gpu"] = gpu_memory_utilization
        return "MODEL"

    monkeypatch.setattr(VLLMSyncBackend, "load", fake_load)
    be = VLLMAsyncBackend(dtype="float32", trust_remote_code=True, max_model_len=128,
                          tokenizer="tok", gpu_memory_utilization=0.35)

    twin, model = _load_sync_twin(be, "openai-community/gpt2")

    assert isinstance(twin, VLLMSyncBackend)
    assert model == "MODEL"
    assert (twin.dtype, twin.trust_remote_code, twin.max_model_len, twin.tokenizer) == (
        "float32", True, 128, "tok")
    assert loaded == {"repo": "openai-community/gpt2", "gpu": 0.35}


def test_no_twin_for_parallel_runs():
    # The sync constructor takes no parallelism; TP/PP runs keep the async engine and
    # record its refusal per cell.
    assert _load_sync_twin(VLLMAsyncBackend(tensor_parallel_size=2), "r") is None
    assert _load_sync_twin(VLLMAsyncBackend(pipeline_parallel_size=2), "r") is None


def test_no_twin_for_other_backends():
    assert _load_sync_twin(object(), "r") is None
