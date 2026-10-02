"""Whole-system overhead: rows are divided by plain vLLM at the same engine version (design §12.14)."""
from isb.jobs.score import attach_plain_vllm


def _result(kind, mode, version, vanilla_ms=None):
    calls = ([{"key": ["__vanilla__", "interactive"], "record": {"median_latency_ms": vanilla_ms}}]
             if vanilla_ms else [])
    return {"provenance": {"engine": {"kind": kind, "mode": mode}, "client": {"vllm": version}},
            "auxiliary_calls": calls}


def test_rows_divide_by_plain_vllm_at_the_same_engine_version():
    results = {"plain-old": _result("vllm", "plain", "0.19.1", 20.0),
               "plain-new": _result("vllm", "plain", "0.28.0", 10.0),
               "lens-on-old": _result("vllm", "hooks", "0.19.1"),
               "engine-on-new": _result("vllm", "async", "0.28.0"),
               "hf-reference": _result("transformers", "async", None)}
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


def _eager_result(mode, version, eager, vanilla_ms=None):
    result = _result("vllm", mode, version, vanilla_ms)
    result["provenance"]["engine"]["params"] = {"enforce_eager": eager}
    return result


def test_eager_rows_divide_by_eager_plain_and_also_report_the_default_engine():
    results = {"plain-compiled": _eager_result("plain", "0.19.1", False, 10.0),
               "plain-eager": _eager_result("plain", "0.19.1", True, 16.0),
               "eager-system": _eager_result("async", "0.19.1", True),
               "compiled-system": _eager_result("transformer-lens", "0.19.1", False)}
    rows = [{"backend": b, "workload": "interactive", "median_latency_ms": 32.0}
            for b in ("eager-system", "compiled-system")]
    attach_plain_vllm(rows, results)
    eager, compiled = rows
    assert eager["plain_vllm"] == "plain-eager" and eager["overhead_vs_plain_vllm"] == 2.0
    assert eager["plain_vllm_default"] == "plain-compiled"
    assert eager["overhead_vs_plain_vllm_default"] == 3.2
    assert compiled["plain_vllm"] == "plain-compiled" and compiled["overhead_vs_plain_vllm"] == 3.2
    assert "plain_vllm_default" not in compiled                          # already the default mode


def test_an_eager_system_without_an_eager_plain_job_gets_only_the_default_ratio():
    results = {"plain-compiled": _eager_result("plain", "0.19.1", False, 10.0),
               "eager-system": _eager_result("async", "0.19.1", True)}
    rows = [{"backend": "eager-system", "workload": "interactive", "median_latency_ms": 30.0}]
    attach_plain_vllm(rows, results)
    assert "overhead_vs_plain_vllm" not in rows[0]
    assert rows[0]["overhead_vs_plain_vllm_default"] == 3.0
