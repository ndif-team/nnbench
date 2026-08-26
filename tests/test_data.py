"""Data registry + spec rebinding tests (isb/data.py, spec_with_data); no GPU.

Pins the data/spec split: sources resolve by name with their per-dataset knobs, specs bind data
by reference, rebinding swaps the feed without touching the procedure, and unit-kind mismatches
are loud. The knob-injection path (dataset knob -> cell params, task param wins) is pinned at the
execute layer's merge point.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isb.data import SOURCES, DataRef, load_data, unit_kind  # noqa: E402
from isb.sweep.execute import _task_params  # noqa: E402
from isb.sweep.spec import Workload, spec_with_data  # noqa: E402


def test_sources_resolve_with_counts_and_knobs():
    units, knobs = load_data(DataRef("jlens/multihop"))
    assert len(units) == 93 and isinstance(units[0], str)
    assert knobs == {"position": "last"}
    units, knobs = load_data(DataRef("jlens/poetry"))
    assert len(units) == 98
    assert knobs == {"position": "last_newline"}          # the per-dataset readout rule rides along
    units, _ = load_data(DataRef("jlens/poetry", n=5))
    assert len(units) == 5                                # file sources slice on demand
    pairs, _ = load_data(DataRef("ioi_pairs", 7))
    assert len(pairs) == 7 and isinstance(pairs[0], tuple)


def test_mib_ioi_views_share_the_snapshot():
    pairs, knobs = load_data(DataRef("mib/ioi"))
    assert len(pairs) == 1000 and knobs == {}
    clean, corrupted = pairs[0]
    assert isinstance(pairs[0], tuple) and clean != corrupted
    # the pair is a minimal name-swap: same whitespace-token count, same tail
    assert len(clean.split()) == len(corrupted.split())
    prompts, _ = load_data(DataRef("mib/ioi_prompts", 10))
    assert prompts == [p[0] for p in pairs[:10]]          # prompt view = clean side of the pairs
    assert unit_kind("mib/ioi") == "pair" and unit_kind("mib/ioi_prompts") == "prompt"


def test_counterfact_source():
    prompts, knobs = load_data(DataRef("counterfact"))
    assert len(prompts) == 1000 and knobs == {}
    assert all(isinstance(p, str) and p == p.rstrip() for p in prompts)  # pre-answer form, no tail space
    assert unit_kind("counterfact") == "prompt"
    assert len(load_data(DataRef("counterfact", 16))[0]) == 16


def test_unknown_source_and_missing_size_are_loud():
    try:
        load_data(DataRef("no-such-set"))
        raise AssertionError("unknown source must raise with the available list")
    except KeyError as e:
        assert "available" in str(e)
    try:
        load_data(DataRef("factual"))                     # generated sources require a size
        raise AssertionError("generated source without n must raise")
    except ValueError:
        pass


def test_workload_binds_a_dataref_and_carries_its_knobs():
    w = Workload("interactive", DataRef("jlens/poetry", n=4))
    assert len(w.prompts) == 4 and w.data_name == "jlens/poetry"
    assert w.data_knobs == {"position": "last_newline"}
    # literal lists keep working untouched (the bespoke single-trace specs)
    w2 = Workload("interactive", ["p1", "p2"])
    assert w2.prompts == ["p1", "p2"] and w2.data_knobs == {} and w2.data_name is None


def test_driver_injects_knobs_under_task_params():
    w = Workload("interactive", DataRef("jlens/poetry", n=2))
    assert _task_params(w, {"unembed": "weight"}) == {"position": "last_newline", "unembed": "weight"}
    # an explicit task param always wins over the dataset default
    assert _task_params(w, {"position": "last"})["position"] == "last"


def test_spec_with_data_swaps_feed_not_procedure():
    from isb.specs import SPECS
    spec = SPECS["jacobian_lens_gpt2"]
    rebound = spec_with_data(spec, DataRef("jlens/poetry"))
    assert rebound.name == "jacobian_lens_gpt2@jlens-poetry"      # refs/banners never collide
    assert len(rebound.workloads[0].prompts) == 98
    assert rebound.workloads[0].data_knobs == {"position": "last_newline"}
    assert rebound.tasks == spec.tasks                            # procedure untouched
    assert len(SPECS["jacobian_lens_gpt2"].workloads[0].prompts) == 93        # original untouched
    # unit-kind mismatch is loud: pair data cannot feed a prompt procedure
    try:
        spec_with_data(spec, DataRef("ioi_pairs", 8))
        raise AssertionError("pair data bound to a prompt procedure must raise")
    except ValueError:
        pass


def test_every_registered_source_declares_a_unit_kind():
    for name in SOURCES:
        assert unit_kind(name) in ("prompt", "pair", "pair_labeled"), name


def test_labeled_pair_source_and_kind_check():
    units, _ = load_data(DataRef("mib/ioi_labeled", 5))
    assert all(len(u) == 3 for u in units)
    clean, corrupted, answers = units[0]
    assert isinstance(answers, tuple) and len(answers) == 2
    assert all(a.startswith(" ") for a in answers)        # leading-space token form
    pairs, _ = load_data(DataRef("mib/ioi", 5))
    assert [(u[0], u[1]) for u in units] == pairs          # same snapshot, answers appended
    from isb.specs import attribution_patching as ap
    spec = ap.attribution_patching_gpt2
    try:                                                   # labeled procedure rejects plain pairs
        spec_with_data(spec, DataRef("mib/ioi", 5))
        raise AssertionError("pair data bound to a labeled-pair procedure must raise")
    except ValueError:
        pass
    rebound = spec_with_data(spec, DataRef("mib/ioi_labeled", 5))
    assert len(rebound.workloads[0].prompts) == 5


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()


def test_wikitext_source_and_jacobian_valid_slice():
    prompts, knobs = load_data(DataRef("wikitext"))
    assert len(prompts) == 100 and knobs == {}
    assert all(len(p) >= 150 for p in prompts)            # long enough to clear the sink skip
    from isb.methodologies.jacobian_collect import valid_slice
    assert valid_slice(50, 16) == slice(16, 49)           # sink prefix out, final position out
    try:
        valid_slice(17, 16)
        raise AssertionError("too-short prompt must raise")
    except ValueError:
        pass


def test_debug_companion_compare_and_prompts():
    import torch
    from isb.debugtrace import compare_debug, debug_prompts
    from isb.specs import activation_patching, logit_lens
    assert all(isinstance(p, str) for p in debug_prompts(activation_patching.activation_patching_gpt2))
    assert all(isinstance(p, str) for p in debug_prompts(logit_lens.logit_lens_gpt2))

    g = torch.Generator().manual_seed(0)
    resid = torch.randn(4, 8, generator=g)
    logits = torch.randn(50, generator=g)
    ref = {("debug_resid", 0): resid, ("debug_logits", 0): logits}
    drifted = resid.clone(); drifted[2:] += 0.1 * torch.randn(2, 8, generator=g)
    cand = {("debug_resid", 0): drifted, ("debug_logits", 0): torch.cat([logits, torch.full((14,), -1e4)])}
    (row,) = compare_debug(cand, ref)
    assert row["first_drift_layer"] == 2                  # layers 0-1 identical, drift enters at 2
    assert row["top1_match"] and row["logit_tv"] < 1e-6   # padded vocab columns are ignored
    bad = {("debug_resid", 0): torch.randn(5, 8), ("debug_logits", 0): logits}
    (row,) = compare_debug(bad, ref)
    assert "error" in row                                 # shape mismatch reported, not crashed
