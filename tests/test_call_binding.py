"""Python call semantics at the benchmark's cell boundary."""
import functools
import inspect

import pytest

from isb.methodologies.registry import CELLS, get_cell
from isb.sweep.execute import BoundCell, _bind_case, _effective_params, _task_params
from isb.sweep.spec import BaselineSpec, CellConfig, ExecutionRegime
from isb.specs import SPECS


def test_binding_includes_named_defaults_and_variadic_keywords():
    def cell(be, model, prompts, option=2, *, alpha=6, **kwargs):
        return option, alpha, kwargs

    params = _effective_params(cell, {"custom": {"layers": [1]}})
    assert params == {"option": 2, "alpha": 6, "custom": {"layers": [1]}}
    assert cell(None, None, None, **params) == (2, 6, {"custom": {"layers": [1]}})


def test_binding_rejects_missing_required_and_unknown_arguments():
    def cell(be, model, prompts, *, required):
        pass

    with pytest.raises(TypeError, match="required"):
        _effective_params(cell, {})
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        _effective_params(cell, {"required": 1, "unknown": 2})


def test_optional_positional_only_defaults_replay_positionally():
    def cell(be, model, prompts, option=2, /, *, alpha=6):
        return option, alpha

    params = _effective_params(cell, {"alpha": 3})
    assert params == {"option": 2, "alpha": 3}
    assert BoundCell(cell, params).prepare(None, None, [])() == (2, 3)
    with pytest.raises(TypeError, match="option"):
        _effective_params(cell, {"option": 4})


def test_generic_cell_preserves_optional_positional_only_arguments(monkeypatch):
    def generic(be, model, profile, prompts, option=2, /, *, alpha=6):
        return prompts, option, alpha

    monkeypatch.setitem(CELLS, ("binding_test", "*", "hf"), generic)
    fn = get_cell("binding_test", "gpt2", "hf")
    params = _effective_params(fn, {"alpha": 3})
    assert BoundCell(fn, params).prepare(None, None, ["hello"])() == (["hello"], 2, 3)


def test_partial_binding_and_mutable_defaults_are_owned():
    defaults = [1]

    def cell(be, model, prompts, *, layers=defaults, alpha=6):
        layers.append(99)
        return list(layers), alpha

    fn = functools.partial(cell, alpha=2)
    params = _effective_params(fn, {})
    call = BoundCell(fn, params)
    for _ in range(2):
        assert call.prepare(None, None, [])() == ([1, 99], 2)
    assert defaults == [1] and params == {"layers": [1], "alpha": 2}


def test_profile_bound_signature_and_backend_fallback(monkeypatch):
    def generic(be, model, profile, prompts, *, alpha=6):
        return profile, prompts, alpha

    monkeypatch.setitem(CELLS, ("binding_test", "*", "vllm_async"), generic)
    bound = get_cell("binding_test", "gpt2", "vllm_sync")
    assert list(inspect.signature(bound).parameters) == ["be", "model", "prompts", "alpha"]
    params = _effective_params(bound, {})
    _, prompts, alpha = bound(None, None, ["hello"], **params)
    assert prompts == ["hello"] and alpha == 6

    def exact(be, model, prompts, *, alpha=3):
        return alpha

    monkeypatch.setitem(CELLS, ("binding_test", "gpt2", "vllm_sync"), exact)
    assert _effective_params(get_cell("binding_test", "gpt2", "vllm_sync"), {}) == {"alpha": 3}


def test_trial_preparation_reuses_the_bound_parameter_layout(monkeypatch):
    def cell(be, model, prompts, option=2, /):
        return option

    call = BoundCell(cell, _effective_params(cell, {}))

    def forbidden(*args, **kwargs):
        raise AssertionError("signature inspection belongs to case binding")

    monkeypatch.setattr(inspect, "signature", forbidden)
    for _ in range(3):
        assert call.prepare(None, None, [])() == 2


def test_dataset_task_and_generation_precedence():
    regime = ExecutionRegime("generation", ["hello"], new_tokens=20,
                             data_knobs={"alpha": 1, "new_tokens": 2})
    assert _task_params(regime, {"alpha": 6, "new_tokens": 5}) == {"alpha": 6, "new_tokens": 20}


def test_classification_failure_preserves_attempted_and_effective_settings(monkeypatch):
    from isb.protocol import InterventionSpec

    spec = CellConfig("binding_test", "binding_test", "gpt2", "gpt2",
                      [ExecutionRegime("interactive", ["hello"])], [({}, "case")], BaselineSpec({}),
                      protocol=InterventionSpec(semantic_params=("alpha",)))

    def cell(be, model, prompts, *, alpha=6, new_option=1):
        pass

    monkeypatch.setitem(CELLS, ("binding_test", "gpt2", "hf"), cell)
    record = {}
    with pytest.raises(ValueError, match="unclassified"):
        _bind_case(spec, "hf", {"alpha": 2}, record)
    assert record["attempted_params"] == {"alpha": 2}
    assert record["params"] == {"alpha": 2, "new_option": 1}
    assert record["error_stage"] == "classify"


@pytest.mark.parametrize("name", sorted(SPECS))
def test_existing_specs_bind_with_matching_semantics_across_backends(name):
    spec = SPECS[name]
    params = [task.params for task in spec.tasks] + [spec.baseline.params]
    if spec.effect is not None:
        params += [spec.effect.baseline_params, spec.effect.perturbed_params]
    for regime in spec.regimes:
        for declared in params:
            records = []
            for interface in ("hf", "vllm_async"):
                record = {}
                _bind_case(spec, interface, _task_params(regime, declared), record)
                records.append(record)
            # Backend realization defaults can differ; the scientific parameters must agree.
            assert records[0]["semantics"] == records[1]["semantics"], (name, regime.kind, declared)
