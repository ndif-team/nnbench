"""CPU integration of authored jobs, real execution, saved artifacts, and JSON-only browsing."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from isb.jobs import contract, worker
from isb.manager import Collection, dispatch, export_html
from isb.methodologies import registry, requirements
from isb.perf import timing
from isb.protocol import InterventionSpec
from isb.runs import EngineConfig, RunConfig
from isb.sweep.spec import BaselineSpec, CellConfig, EffectSpec, ExecutionRegime, TaskSpec


@pytest.mark.parametrize("custom_description", [True, False], ids=["case-hook", "template-fallback"])
@pytest.mark.parametrize("invalid_effect", [False, True], ids=["valid-effect", "nonfinite-effect"])
def test_real_worker_execution_preserves_calls_through_artifact_and_manager(
        tmp_path, monkeypatch, custom_description, invalid_effect):
    import isb.runs

    method, backend_name, interface = "cpu_contract_fixture", "independent-cpu", "cpu-fixture"
    monkeypatch.setenv("ISB_BACKEND", backend_name)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(timing, "force_gc", lambda: None)
    monkeypatch.setattr(isb.runs, "resolve_provenance", lambda run: {
        "engine": {"kind": run.engine.kind}, "client": {}, "host": {}, "deployment": {}})
    monkeypatch.setattr(registry, "CELLS", dict(registry.CELLS))
    monkeypatch.setattr(requirements, "CASE_DESCRIPTIONS", dict(requirements.CASE_DESCRIPTIONS))
    invocations, lifecycle = [], []

    @registry.cell(method, family="gpt2", backend=interface)
    def intervention(be, model, prompts, *, alpha=0, settings=None, mode="fixture", **extra):
        # Every warmup, measured call, aggregate/reference call, and effect invocation owns
        # its nested settings, even when a custom description hook also mutates its input.
        assert not extra
        assert settings == {"layers": [1, 2]}
        invocations.append((alpha, len(prompts), mode))
        settings["layers"].append(99)
        if alpha == 7:
            return torch.full((len(prompts), 2), float("nan"))
        return torch.tensor([[4.0 * alpha, -4.0 * alpha]]).repeat(len(prompts), 1)

    if custom_description:
        @requirements.case_description(method)
        def describe(params, *, template, family):
            assert family == "gpt2"
            params["settings"]["layers"].append(-7)
            if params["alpha"] == 0:
                return replace(template, components=("lm_head",), operations=("read",),
                               mechanisms=(), write_components=())
            return template

    class Backend:
        name = interface

        def load(self, repo):
            assert repo == "test/model"
            lifecycle.append("load")
            return SimpleNamespace(config=SimpleNamespace(_commit_hash="fake-cpu-revision"),
                                   tokenizer=SimpleNamespace(get_vocab=lambda: {"a": 0, "b": 1}))

        def teardown(self, model):
            lifecycle.append("teardown")

    template = InterventionSpec(
        components=("block_output", "lm_head"), operations=("read", "write"),
        mechanisms=("add_scaled",), write_components=("block_output",),
        semantic_params=("alpha", "settings"), realization_params=("mode",))
    effect_alpha = 7 if invalid_effect else 3
    spec = CellConfig(
        "cpu-integration", method, "gpt2", "test/model",
        [ExecutionRegime(kind, ["one", "two"], data_name="fixture-data",
                         data_knobs={"alpha": 1, "settings": {"layers": [1, 2]}})
         for kind in ("interactive", "batched")],
        [TaskSpec("control", {"alpha": 0}), TaskSpec("write", {"alpha": 3})],
        BaselineSpec({"alpha": 0}), EffectSpec({"alpha": 0}, {"alpha": effect_alpha}),
        warmup=1, n_trials=2, protocol=template)
    root = tmp_path / "attempt"
    job = root / "experiments" / "exp"
    experiment = contract.prepare(spec, job, seed=17)
    original_job = (job / "experiment.json").read_bytes()
    output = job / backend_name
    restored_specs = []

    def create_backend(restored):
        restored_specs.append(restored)
        return Backend(), RunConfig(engine=EngineConfig("transformers"))

    # Both worker.main and execute_run are the production implementations. Only the model,
    # intervention function, and environment-information collection are CPU fixtures.
    worker.main(create_backend, job=job, output=output)
    result = contract.validate_result(output, experiment, backend_name)
    assert result["status"] == "completed"
    assert lifecycle == ["load", "teardown"]
    assert (job / "experiment.json").read_bytes() == original_job
    assert restored_specs[0].protocol_source == "restored"
    assert all(regime.data_knobs["settings"] == {"layers": [1, 2]}
               for regime in (*spec.regimes, *restored_specs[0].regimes))
    assert {count for _, count, _ in invocations} == {1, 2}
    assert all(mode == "fixture" for _, _, mode in invocations)
    assert len(result["cells"]) == 4
    assert all(cell["state"] == "RAN" for cell in result["cells"])
    assert result["provenance"]["model_identity"]["revision"] == "fake-cpu-revision"
    assert result["provenance"]["model_identity"]["vocab_size"] == 2
    expected_scope = "case" if custom_description else "template"

    for row in result["cells"]:
        alpha = 0 if row["label"] == "control" else 3
        assert row["params"] == {"alpha": alpha, "settings": {"layers": [1, 2]}, "mode": "fixture"}
        assert row["semantics"] == {"alpha": alpha, "settings": {"layers": [1, 2]}}
        assert row["realization"] == {"mode": "fixture"}
        assert row["protocol_scope"] == expected_scope
        assert row["protocol_coverage"] == {"status": "described"}
        assert ("write" in row["protocol"]["operations"]) == (alpha != 0 or not custom_description)
        assert row["error"] is None and row["error_stage"] is None

    artifact = torch.load(output / "result.pt", map_location="cpu", weights_only=False)
    saved_meta = artifact["outputs"][("__meta__",)]
    auxiliary = {tuple(item["key"]): item["record"] for item in result["auxiliary_calls"]}
    assert auxiliary == {key: value for key, value in saved_meta.items()
                         if key not in contract.expected_cells(experiment)}
    assert artifact["provenance"]["coordinates"] == result["provenance"]["coordinates"]
    for kind in ("interactive", "batched"):
        baseline = saved_meta[("__baseline__", kind)]
        assert baseline["params"]["alpha"] == 0
        assert baseline["params"]["settings"] == {"layers": [1, 2]}
        assert baseline["protocol_scope"] == expected_scope
        assert baseline["error"] is None
    for label, alpha in (("control", 0), ("write", 3)):
        reference = saved_meta[("batched_perprompt", label)]
        assert reference["params"]["alpha"] == alpha
        assert reference["protocol_scope"] == expected_scope
        assert reference["error"] is None
        assert artifact["outputs"][("batched_perprompt", label)].shape == (2, 2)
    effect = saved_meta[("__effect__", "interactive")]
    if invalid_effect:
        assert effect["strong"] is False
        assert "finite tensor" in effect["error"]
        assert effect["error_stage"] == "effect"
        assert "tv" not in effect and "top1_agree" not in effect
    else:
        assert effect["strong"] and effect["error"] is None
    for side, alpha in (("baseline", 0), ("perturbed", effect_alpha)):
        assert effect[side]["params"]["alpha"] == alpha
        assert effect[side]["params"]["settings"] == {"layers": [1, 2]}
        assert effect[side]["protocol_scope"] == expected_scope
        assert effect[side]["error"] is None

    contract.write_json(root / "plan.json", {
        "version": contract.PLAN_VERSION, "experiments": ["exp"], "backends": [backend_name],
        "reference": None, "comparison": "correctness", "status": "completed"})
    contract.write_json(output / "execution.json", {
        "backend": backend_name, "experiment_id": experiment["id"], "status": "completed",
        "image_id": "sha256:cpu-fixture"})

    def forbidden(*args, **kwargs):
        raise AssertionError("manager browsing must use saved JSON only")

    monkeypatch.setattr(torch, "load", forbidden)
    collection = Collection(str(root))
    key = f"attempt/exp/{backend_name}"
    viewed_outputs, viewed_provenance = collection.entries[key]
    assert viewed_provenance["coordinates"] == result["provenance"]["coordinates"]
    assert viewed_provenance["auxiliary_calls"] == result["auxiliary_calls"]
    assert all(viewed_outputs[("__meta__",)][key] == record for key, record in auxiliary.items())
    assert viewed_provenance["submitted_coordinates"]["inputs_sha256"] == experiment["inputs_sha256"]
    assert all(cell.state == "RAN" for cell in collection.results[key])
    for row in result["cells"]:
        viewed = viewed_outputs[("__meta__",)][(row["workload"], row["label"])]
        for field in ("params", "semantics", "realization", "protocol", "protocol_scope", "protocol_coverage"):
            assert viewed[field] == row[field]
    page = dispatch(collection, f"/run/{key}")
    exported = export_html(str(root), items_per_source=0)
    assert backend_name in page and backend_name in exported
    if invalid_effect:
        assert "finite tensor" in page and "finite tensor" in exported
