"""Whole-system overhead: rows are divided by plain vLLM whose engine ran the same version, compile
and CUDA-graph mode and tensor-parallel size (design §12.14)."""
from isb.jobs.score import attach_plain_vllm, engine_run


def _run(version, compiled=True, graphs=True, tp=1):
    return {"vllm": version, "torch_compile": compiled, "cuda_graphs": graphs,
            "tensor_parallel_size": tp}


def _result(mode, run, vanilla_ms=None):
    calls = ([{"key": ["__vanilla__", "interactive"], "record": {"median_latency_ms": vanilla_ms}}]
             if vanilla_ms else [])
    return {"provenance": {"engine": {"mode": mode}}, "auxiliary_calls": calls, "engine_run": run}


def _rows(results, backends, ms=32.0):
    rows = [{"backend": b, "workload": "interactive", "median_latency_ms": ms} for b in backends]
    attach_plain_vllm(rows, results)
    return {r["backend"]: r for r in rows}


def test_rows_divide_by_plain_vllm_at_the_same_engine_version():
    results = {"plain-old": _result("plain", _run("0.19.1"), 20.0),
               "plain-new": _result("plain", _run("0.28.0"), 10.0),
               "lens-on-old": _result("hooks", _run("0.19.1")),
               "engine-on-new": _result("async", _run("0.28.0")),
               "hf-reference": _result("async", None)}
    rows = [{"backend": b, "workload": "interactive", "median_latency_ms": 40.0}
            for b in ("lens-on-old", "engine-on-new", "hf-reference")]
    rows.append({"backend": "engine-on-new", "workload": "generation", "median_latency_ms": 40.0})
    attach_plain_vllm(rows, results)
    by = {(r["backend"], r["workload"]): r for r in rows}
    assert by[("lens-on-old", "interactive")]["plain_vllm"] == "plain-old"
    assert by[("lens-on-old", "interactive")]["overhead_vs_plain_vllm"] == 2.0
    assert by[("engine-on-new", "interactive")]["overhead_vs_plain_vllm"] == 4.0
    assert "plain_vllm" not in by[("hf-reference", "interactive")]        # not a vLLM engine
    assert "plain_vllm" not in by[("engine-on-new", "generation")]        # no plain timing there


def test_eager_rows_divide_by_eager_plain_and_also_report_the_default_engine():
    eager = _run("0.19.1", compiled=False, graphs=False)
    results = {"plain-compiled": _result("plain", _run("0.19.1"), 10.0),
               "plain-eager": _result("plain", eager, 16.0),
               "eager-system": _result("async", eager),
               "compiled-system": _result("transformer-lens", _run("0.19.1"))}
    by = _rows(results, ("eager-system", "compiled-system"))
    eager_row, compiled = by["eager-system"], by["compiled-system"]
    assert eager_row["plain_vllm"] == "plain-eager" and eager_row["overhead_vs_plain_vllm"] == 2.0
    assert eager_row["plain_vllm_default"] == "plain-compiled"
    assert eager_row["overhead_vs_plain_vllm_default"] == 3.2
    assert compiled["plain_vllm"] == "plain-compiled" and compiled["overhead_vs_plain_vllm"] == 3.2
    assert "plain_vllm_default" not in compiled                          # already the default mode


def test_an_eager_system_without_an_eager_plain_job_gets_only_the_default_ratio():
    results = {"plain-compiled": _result("plain", _run("0.19.1"), 10.0),
               "eager-system": _result("async", _run("0.19.1", compiled=False, graphs=False))}
    row = _rows(results, ("eager-system",), ms=30.0)["eager-system"]
    assert "overhead_vs_plain_vllm" not in row
    assert row["overhead_vs_plain_vllm_default"] == 3.0


def test_floors_match_tensor_parallel_size_and_graphs_without_compile_have_no_floor(tmp_path):
    # the mode comes from each engine's own startup line, not from the declared enforce_eager
    log = tmp_path / "execution.log"
    log.write_text("(EngineCore pid=7) INFO Initializing a V1 LLM engine (v0.27.1) with config: "
                   "model='m', tensor_parallel_size=2, enforce_eager=False, compilation_config="
                   "{'mode': <CompilationMode.NONE: 0>, 'cudagraph_mode': "
                   "<CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>}\n")
    breakable = engine_run(log)
    assert breakable == _run("0.27.1", compiled=False, graphs=True, tp=2)
    results = {"plain-tp1": _result("plain", _run("0.27.1", tp=1), 10.0),
               "plain-tp2": _result("plain", _run("0.27.1", tp=2), 8.0),
               "plain-eager-tp2": _result("plain", _run("0.27.1", False, False, tp=2), 16.0),
               "system-tp1": _result("async", _run("0.27.1", tp=1)),
               "system-tp2": _result("async", _run("0.27.1", tp=2)),
               "graphs-tp2": _result("taps", breakable)}
    by = _rows(results, ("system-tp1", "system-tp2", "graphs-tp2"))
    assert by["system-tp1"]["plain_vllm"] == "plain-tp1"                   # neither floor overwritten
    assert by["system-tp2"]["plain_vllm"] == "plain-tp2"
    assert by["system-tp2"]["overhead_vs_plain_vllm"] == 4.0
    graphs = by["graphs-tp2"]
    assert "overhead_vs_plain_vllm" not in graphs                         # no stock engine runs this
    assert graphs["plain_vllm_default"] == "plain-tp2"
    assert graphs["overhead_vs_plain_vllm_default"] == 4.0
