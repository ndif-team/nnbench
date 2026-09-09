"""RunConfig layer tests (isb/runs.py); no GPU for the mapping/merge logic, host-resolving for
the provenance gates (this box has the editable nnsight checkout + CUDA, so the gates are
exercised against real resolution, with impossible pins to trigger the refusals)."""
import sys
from pathlib import Path
from types import SimpleNamespace
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isb.backends import HFBackend, VLLMAsyncBackend, VLLMServeBackend, VLLMSyncBackend  # noqa: E402
from isb.runs import (  # noqa: E402
    DeploymentConfig,
    EngineConfig,
    RunConfig,
    cell_interface,
    make_backend,
    resolve_provenance,
)

SPEC = SimpleNamespace(hf_kwargs={}, vllm_kwargs={"max_model_len": 1024})


def _run(engine_kind, mode="async", dep_kind="local", host=None, params=None, **kw):
    return RunConfig(engine=EngineConfig(engine_kind, mode=mode, params=params or {}),
                     deployment=DeploymentConfig(kind=dep_kind, host=host), **kw)


def test_executor_mapping_covers_the_grid():
    assert isinstance(make_backend(_run("transformers"), SPEC), HFBackend)
    assert isinstance(make_backend(_run("vllm"), SPEC), VLLMAsyncBackend)
    assert isinstance(make_backend(_run("vllm", mode="sync"), SPEC), VLLMSyncBackend)
    serve = make_backend(_run("vllm", dep_kind="serve", host="http://s:6677"), SPEC)
    assert isinstance(serve, VLLMServeBackend)


def test_ndif_is_a_reserved_deployment_not_an_engine():
    # ndif composes with either engine; the executor slot is reserved, loudly
    for engine in ("transformers", "vllm"):
        try:
            make_backend(_run(engine, dep_kind="ndif"), SPEC)
            raise AssertionError("ndif executor must be explicitly not-implemented, not silent")
        except NotImplementedError:
            pass


def test_spec_required_params_merge_and_conflict_loudly():
    # run params merge over the spec's required fragment
    be = make_backend(_run("vllm", params={"dtype": "float32"}), SPEC)
    assert be.max_model_len == 1024 and be.dtype == "float32"
    # same key, same value: fine
    make_backend(_run("vllm", params={"max_model_len": 1024}), SPEC)
    # same key, different value: the experiment must not silently override a model requirement
    try:
        make_backend(_run("vllm", params={"max_model_len": 2048}), SPEC)
        raise AssertionError("conflicting engine param must raise")
    except ValueError:
        pass


def test_cell_interface_is_engine_level_only():
    assert cell_interface(_run("transformers")) == "hf"
    assert cell_interface(_run("vllm")) == "vllm_async"
    assert cell_interface(_run("vllm", mode="sync")) == "vllm_sync"
    assert cell_interface(_run("vllm", dep_kind="serve", host="h")) == "vllm_serve"
    # topology params do NOT change the interface (tp2 runs the same cells)
    assert cell_interface(_run("vllm", params={"tensor_parallel_size": 2})) == "vllm_async"


def test_provenance_records_declared_next_to_resolved():
    # provenance is descriptive: a declared value that does not match reality is RECORDED (visible
    # drift), never a refusal — the record must exist precisely when things drift
    prov = resolve_provenance(_run("transformers", nnsight="0000000dead", gpu_model="X9000"))
    assert set(prov) == {"executed", "declared", "client", "deployment", "engine", "host"}
    assert prov["declared"]["nnsight"] == "0000000dead"           # kept verbatim
    assert prov["declared"]["gpu_model"] == "X9000"
    from importlib.util import find_spec
    # A lightweight host need not install any inference stack. Absence is provenance too.
    installed = find_spec("nnsight")
    assert bool(prov["client"]["nnsight"]["path"]) == bool(installed)
    assert prov["client"]["nnsight"].get("commit") != "0000000dead"
    assert prov["host"]["hostname"]


def test_git_wheel_provenance_keeps_full_commit(monkeypatch):
    import isb.runs as runs

    info = {"url": "https://github.com/ndif-team/nnsight.git",
            "vcs_info": {"vcs": "git", "commit_id": "a" * 40}}
    monkeypatch.setattr(runs.metadata, "distribution", lambda _: SimpleNamespace(
        read_text=lambda _: json.dumps(info)))
    assert runs._installed_git_identity("nnsight")["commit"] == "a" * 40
    info.pop("vcs_info")
    assert runs._installed_git_identity("nnsight") == {}


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
