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
