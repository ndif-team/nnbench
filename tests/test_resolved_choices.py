"""Bound arguments and cell-reported choices have separate, scoped records."""
from types import SimpleNamespace

import pytest

from isb.methodologies.observations import capture_choices, record_resolved
from isb.sweep.execute import _bind_case
from isb.sweep.spec import BaselineSpec, CellConfig, ExecutionRegime, TaskSpec


@pytest.mark.parametrize("method", ["logit_lens", "steering", "jacobian_lens"])
@pytest.mark.parametrize("family,expected", [("gpt2", "plain"), ("llama", "fused")])
@pytest.mark.parametrize("selector", [None, "plain"])
def test_real_cells_report_residual_choice_at_resolution_site(method, family, expected, selector):
    # Stop at the backend boundary: the actual cell and family profile resolve the selector.
    backend = SimpleNamespace(name="vllm_async", run=lambda model, prompts, build: "trace reached")
    model = SimpleNamespace(tokenizer=lambda *args, **kwargs: {"input_ids": [1]})
    spec = CellConfig("choice", method, family, "repo",
                      [ExecutionRegime("interactive", ["prompt"])],
                      [TaskSpec("auto", {"residual": selector})], BaselineSpec({}))
    record = {}
    call = _bind_case(spec, "vllm_async", spec.tasks[0].params, record)
    assert "resolved_params" not in record
    for _ in range(2):
        assert call.prepare(backend, model, ["prompt"])() == "trace reached"
    assert record["params"]["residual"] == selector
    assert record["realization"]["residual"] == selector
    assert record["resolved_params"] == {"residual": expected if selector is None else selector}
    assert spec.tasks[0].params == {"residual": selector}


def test_reporting_owns_values_and_restores_context_after_error():
    outer, inner = {}, {}
    values = [1]
    with capture_choices({"selector": None}, outer):
        record_resolved(selector=values)
        values.append(2)
        assert outer["resolved_params"] == {"selector": [1]}
        with pytest.raises(RuntimeError):
            with capture_choices({"selector": None}, inner):
                record_resolved(selector="inner")
                raise RuntimeError("cell failed")
        record_resolved(selector=[1])
    record_resolved(selector="outside execution")
    assert inner["resolved_params"] == {"selector": "inner"}
    assert outer["resolved_params"] == {"selector": [1]}


@pytest.mark.parametrize("choices,match", [
    ({"selector": "different"}, "changed across"),
    ({"unknown": "value"}, "absent from"),
    ({"selector": float("nan")}, "JSON compliant"),
])
def test_invalid_or_inconsistent_reports_fail_clearly(choices, match):
    record = {}
    with capture_choices({"selector": None}, record):
        record_resolved(selector="original")
        with pytest.raises(ValueError, match=match):
            record_resolved(**choices)
    assert record["resolved_params"] == {"selector": "original"}
