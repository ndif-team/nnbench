"""Workloads, models and suites (design.md §12.16) — no GPU."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isb.models import FAMILIES, MODELS  # noqa: E402
from isb.suites import SUITES, suite_cases  # noqa: E402
from isb.sweep.spec import (Asset, Depth, NotApplicable, Regime, Row, Workload,  # noqa: E402
                            instantiate)
from isb.workloads import WORKLOADS, case  # noqa: E402


def test_one_workload_per_methodology_named_by_it():
    assert all(w.name == w.methodology for w in WORKLOADS.values())


def test_workloads_reject_realization_params():
    with pytest.raises(ValueError, match="realization"):
        Workload("steering", "steering", (Row("x", {"layer": 1, "mode": "inplace"}),),
                 (Regime("interactive", ["p"]),), baseline={"layer": 1, "alpha": 0.0})


def test_depth_resolves_against_the_model_and_names_are_generated():
    assert Depth(1 / 2).resolve(12) == 6 and Depth(1.0).resolve(12) == 11
    spec, skipped = instantiate(WORKLOADS["steering"], "qwen2.5-7b")
    assert spec.name == "steering/qwen2.5-7b" and not skipped
    assert sorted({t.params["layer"] for t in spec.tasks}) == [7, 14]
    assert spec.repo == MODELS["qwen2.5-7b"].repo and spec.n_layers == 28
    assert spec.effect.perturbed_params["layer"] == 14


def test_rows_a_model_cannot_express_are_reported_not_dropped_silently():
    spec, skipped = instantiate(WORKLOADS["ablation"], "nemotron3-nano-4b")
    assert {t.params["target"] for t in spec.tasks} == {"mixer"}
    assert len(skipped) == 4 and all("block component" in s["reason"] for s in skipped)
    assert spec.effect.perturbed_params["target"] == "mixer"
    spec, skipped = instantiate(WORKLOADS["jacobian_lens"], "gpt2")
    assert [s["label"] for s in skipped] == ["fitted transport (the model's published lens)"]
    spec, skipped = instantiate(WORKLOADS["jacobian_lens"], "qwen3.5-4b")
    fitted = spec.tasks[-1].params
    assert fitted["transport"] == MODELS["qwen3.5-4b"].assets["jlens"]["transport"] and not skipped


def test_absolute_layers_are_checked_against_the_model():
    w = Workload("steering", "steering", (Row("deep", {"layer": 20, "target": " Rome"}),),
                 (Regime("interactive", ["p"]),), baseline={"layer": 1, "target": " Rome"})
    with pytest.raises(ValueError, match="out of range"):
        instantiate(w, "gpt2")


def test_assets_require_the_model_to_carry_them():
    w = Workload("jacobian_lens", "jacobian_lens", (Row("fitted", {"transport": Asset("jlens", "transport")}),),
                 (Regime("interactive", ["p"]),), baseline={"transport": None})
    with pytest.raises(NotApplicable):
        instantiate(w, "gpt2")


def test_models_name_known_families_and_family_components():
    assert all(m.family in FAMILIES for m in MODELS.values())
    assert FAMILIES["gpt2"].components == {"attn", "mlp"}
    assert FAMILIES["nemotron"].components == {"mixer"}


def test_every_suite_instantiates_and_smoke_stays_small():
    for name in SUITES:
        assert suite_cases(name), name
    assert {s.repo for s in suite_cases("smoke")} == {MODELS["gpt2"].repo, MODELS["smollm2-135m"].repo}
    assert {s.name for s in suite_cases("smoke")} >= {f"{w}/gpt2" for w in WORKLOADS}


def test_case_names_resolve_and_historical_names_do_not():
    assert case("logit_lens/gpt2").name == "logit_lens/gpt2"
    assert case("logit_lens_gpt2") is None and case("cmp_steering") is None


def test_nnsight_cells_declare_realizations_and_variants_inherit_them():
    import isb.methodologies  # noqa: F401
    from isb.methodologies.registry import cell, get_realizations

    assert get_realizations("steering", "llama", "vllm_async") == ({"mode": "inplace"},)
    assert get_realizations("steering", "llama", "vllm_sync") == ({"mode": "inplace"},)
    assert {"residual": "plain"} in get_realizations("logit_lens", "llama", "vllm_async")
    assert get_realizations("steering", "llama", "vllm_lens") == ()   # a foreign system inherits none
    with pytest.raises(ValueError, match="realization params"):
        cell("steering", "gpt2", "hf_fixture", realizations=[{"alpha": 1.0}])
