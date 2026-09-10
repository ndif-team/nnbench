"""Manager tests (isb/manager.py), no GPU: directory model, one-baseline resolution,
archive/discard as file moves, and the rendered pages."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_execute_score import _execute as execute_fixture, _spec  # noqa: E402

import isb.manager as manager  # noqa: E402
from isb.specs import SPECS  # noqa: E402


def _execute(path, name, engine, **kwargs):
    execute_fixture(path, name, engine, **kwargs)


def _import(path):
    from isb.manager.model import import_legacy
    import_legacy(str(path))


def _collection(tmp_path):
    """A directory with one transformers baseline and one vllm run that is wrong on one prompt."""
    _execute(tmp_path, "base", "transformers")
    _execute(tmp_path, "cand", "vllm", wrong_on="p1")
    _import(tmp_path)
    return manager.load_dir(str(tmp_path))


def test_legacy_import_never_guesses_precision_control(tmp_path):
    from unittest.mock import patch
    _execute(tmp_path, "base", "transformers")
    _execute(tmp_path, "candidate", "vllm")
    _execute(tmp_path, "unrelated-fp32", "vllm")
    with patch.dict(SPECS, {"xs": _spec()}), patch("isb.sweep.score.score_runs", return_value=[]) as score:
        _import(tmp_path)
    assert score.call_count == 3
    for call in score.call_args_list:
        assert len(call.args) == 4 and "ctl" not in call.kwargs


def test_import_keeps_incomplete_history_browsable_without_rescoring(tmp_path):
    from unittest.mock import patch
    from isb.runfile import save_run
    from isb.runs import spec_coordinates

    coords = spec_coordinates(_spec(), "hf")
    coords.pop("inputs_sha256")
    coords.pop("config")
    coords.pop("identity_complete")
    coords["schema"] = 2
    save_run(str(tmp_path), "historical", {}, {"coordinates": coords, "engine": {"kind": "transformers"}})
    with patch.dict(SPECS, {"xs": _spec()}), patch("isb.sweep.score.score_runs") as score:
        _import(tmp_path)
    assert not score.called
    col = manager.Collection(str(tmp_path))
    assert "historical" in col.entries and col.results["historical"] is None
    assert any("identity is incomplete" in warning for warning in col.warnings)


def test_import_does_not_score_with_a_changed_current_spec(tmp_path):
    from dataclasses import replace
    from unittest.mock import patch
    from isb.runfile import save_run
    from isb.runs import spec_coordinates

    spec = _spec()
    save_run(str(tmp_path), "saved", {}, {"coordinates": spec_coordinates(spec, "hf"),
                                           "engine": {"kind": "transformers"}})
    with patch.dict(SPECS, {"xs": replace(spec, n_trials=spec.n_trials + 1)}), \
         patch("isb.sweep.score.score_runs") as score:
        _import(tmp_path)
    assert not score.called
    col = manager.Collection(str(tmp_path))
    assert any("current spec does not match" in warning for warning in col.warnings)


def test_changed_legacy_artifact_requires_explicit_reimport(tmp_path):
    _collection(tmp_path)
    with (tmp_path / "cand.pt").open("ab") as stream:
        stream.write(b"changed")
    col = manager.Collection(str(tmp_path))
    assert "cand" not in col.entries
    assert any("changed after import" in warning for warning in col.warnings)


def test_baseline_single_transformers_run_and_marker_override(tmp_path):
    entries = _collection(tmp_path)
    assert manager.baselines_of(str(tmp_path), entries) == {"xs": "base"}   # only tf run of xs
    (tmp_path / "BASELINE").write_text("cand\n")
    assert manager.baselines_of(str(tmp_path), entries) == {"xs": "cand"}   # marker wins
    (tmp_path / "BASELINE").write_text("missing\n")
    assert manager.baselines_of(str(tmp_path), entries) == {}               # names no file


def test_overview_cards_carry_rollup_of_the_scored_run(tmp_path):
    _collection(tmp_path)
    SPECS["xs"] = _spec()
    try:
        _import(tmp_path)
        page = manager.render_overview(str(tmp_path), inbox=str(tmp_path / "inbox"))
    finally:
        del SPECS["xs"]
    # every cell of cand is wrong on p1, so nothing passes -> the rollup is SILENTLY_WRONG
    # (assert the rendered chip, not the stylesheet: the CSS also contains state names)
    assert ">SILENTLY_WRONG</span>" in page
    assert "/method/m" in page                                     # card links the methodology
    assert "data -" not in page                                    # no run coordinates on a card


def test_method_page_is_the_model_x_backend_matrix(tmp_path):
    _collection(tmp_path)
    SPECS["xs"] = _spec()
    try:
        _import(tmp_path)
        page = manager.render_method(str(tmp_path), "m", inbox=str(tmp_path / "inbox"))
    finally:
        del SPECS["xs"]
    assert "Model × backend" in page
    assert "/spec/xs" in page                                      # matrix row links the model
    assert ">SILENTLY_WRONG</span>" in page                        # the model x backend rollup


def test_spec_page_shows_states_metrics_and_stack(tmp_path):
    _collection(tmp_path)
    SPECS["xs"] = _spec()
    try:
        _import(tmp_path)
        page = manager.render_spec(str(tmp_path), "xs", inbox=str(tmp_path / "inbox"))
    finally:
        del SPECS["xs"]
    assert ">SILENTLY_WRONG</span>" in page                        # per-cell state chips
    assert "top1=" in page and "tv=" in page                       # oracle metrics per cell
    assert "c-cand" in page                                        # stack commit shown
    # the baseline run leads with its own section: stack + latencies, no verdicts
    assert ">BASELINE</span>" in page and "c-base" in page
    assert page.index("c-base") < page.index("c-cand")


def test_spec_page_has_operating_point_figure(tmp_path):
    _collection(tmp_path)
    SPECS["xs"] = _spec()
    try:
        _import(tmp_path)
        page = manager.render_spec(str(tmp_path), "xs", inbox=str(tmp_path / "inbox"))
    finally:
        del SPECS["xs"]
    assert "<svg" in page and "median latency (ms)" in page
    # cand has two timed cells (interactive + batched) -> two points
    assert page.count("<circle") >= 2 + 1                          # points + one legend swatch
    assert "overhead" in page


def test_perf_points_include_baseline_and_skip_errored_cells(tmp_path):
    entries = _collection(tmp_path)
    pts = manager.perf_points(entries, list(entries))
    assert {p["run"] for p in pts} == {"base", "cand"}             # baseline is a config too
    assert all(p["lat"] > 0 and p["ovh"] > 0 for p in pts)


def test_rollup_verdicts(tmp_path=None):
    assert manager.rollup(["SUPPORTED", "SUPPORTED_DEGRADED"]) == "SUPPORTED"
    assert manager.rollup(["SUPPORTED", "ERROR"]) == "LIMITED"
    assert manager.rollup(["ERROR", "ERROR"]) == "ERROR"
    assert manager.rollup(["SILENTLY_WRONG", "ERROR"]) == "SILENTLY_WRONG"
    assert manager.rollup(["SUPPORTED", "SILENTLY_WRONG"]) == "LIMITED"


def _construct_run(dirpath, name="micro-async", engine_kind="vllm"):
    from isb.runfile import save_run
    iface = "vllm_async" if engine_kind == "vllm" else "hf"
    meta = {("probe", "input_boundary"): {"state": "SUPPORTED", "note": "x==y", "latency_s": 1.0},
            ("probe", "barrier"): {"state": "ERROR", "note": "not shared", "latency_s": 2.0}}
    prov = {"client": {"nnsight": {"commit": "c-micro"}, "vllm": "0.19.1", "transformers": None},
            "deployment": {"kind": "local"},
            "engine": {"kind": engine_kind, "mode": "async", "params": {}},
            "host": {"hostname": "testbox", "gpus": []},
            "coordinates": {"spec": "constructs", "methodology": "constructs", "family": "gpt2",
                            "repo": "gpt2", "data": [], "regimes": [],
                            "tasks": ["input_boundary", "barrier"], "interface": iface}}
    save_run(str(dirpath), name, {("__meta__",): meta}, prov)
    _import(dirpath)


def test_lone_construct_run_is_never_the_fallback_baseline(tmp_path):
    _construct_run(tmp_path, name="constructs-hf", engine_kind="transformers")
    entries = manager.load_dir(str(tmp_path))
    assert manager.baselines_of(str(tmp_path), entries) == {}      # probe runs self-check


def test_overview_matches_the_reviewed_design(tmp_path):
    _collection(tmp_path)
    SPECS["xs"] = _spec()
    try:
        _import(tmp_path)
        page = manager.render_overview(str(tmp_path), inbox=str(tmp_path / "inbox"))
    finally:
        del SPECS["xs"]
    assert "How to read the map" in page                           # legend panel
    assert "setup of the runs in this directory" in page           # backends table
    assert "HF-nnsight" in page and "vllm-nnsight" in page


def test_backend_page_shows_construct_support_from_construct_run(tmp_path):
    _construct_run(tmp_path, name="micro-async")
    page = manager.render_backend("vllm_async", str(tmp_path))
    assert "Construct support" in page and "barrier" in page
    assert ">ERROR</span>" in page and "not shared" in page


def test_construct_runs_stay_off_the_method_cards(tmp_path):
    """Construct probes surface as construct support on backend pages; they get no methodology
    card, and 'micro' names only the perf op-cost benchmark."""
    _collection(tmp_path)
    _construct_run(tmp_path, name="constructs-async")
    page = manager.render_overview(str(tmp_path), inbox=str(tmp_path / "inbox"))
    assert "/spec/constructs" not in page                          # no method card
    assert "perf micro" in page                                    # op-cost pointer instead
    backend = manager.render_backend("vllm_async", str(tmp_path))
    assert "Construct support" in page or "Construct support" in backend
    run_page = manager.render_run(str(tmp_path), "constructs-async", inbox=str(tmp_path / "inbox"))
    assert "SUPPORTED · x==y" in run_page


def test_perf_micro_page_is_op_cost(tmp_path):
    _collection(tmp_path)
    page = manager.render_micro(str(tmp_path), inbox=str(tmp_path / "inbox"))
    assert "Primitive op cost" in page and "read, steer, and qk" in page
    assert "No perf-micro runs" in page                            # none archived yet


def _perf_run(dirpath, name="perf-read-q"):
    from isb.runfile import save_run
    rows = [
        {"system": "pure_vllm", "op": "none", "footprint": "one", "phase": "decode",
         "median_total_lat_s": 1.0, "median_tok_per_s": 256.0, "overhead_pct": 0.0,
         "error": None, "cell": "pure_vllm.none", "repo": "q"},
        {"system": "nnsight_vllm", "op": "read", "footprint": "all", "phase": "decode",
         "median_total_lat_s": 1.5, "median_tok_per_s": 170.0, "overhead_pct": 33.6,
         "error": None, "cell": "nnsight_vllm.read", "repo": "q"},
        {"system": "vllm_hook", "op": "read", "footprint": "all", "phase": "decode",
         "median_total_lat_s": None, "median_tok_per_s": None, "overhead_pct": None,
         "error": "HANG > 900s", "cell": "vllm_hook.read", "repo": "q"},
    ]
    prov = {"client": {"nnsight": {"commit": "c-perf"}, "vllm": "0.19.1", "transformers": None},
            "deployment": {"kind": "local"},
            "engine": {"kind": "vllm", "mode": "async", "params": {}},
            "host": {"hostname": "t", "gpus": []},
            "coordinates": {"spec": name, "methodology": "perf_micro", "family": "-",
                            "repo": "q", "data": [], "regimes": [], "tasks": [],
                            "interface": "perf"}}
    save_run(str(dirpath), name, {("perf_rows",): rows, ("__meta__",): {}}, prov)
    _import(dirpath)


def test_perf_micro_runs_render_op_cost_tables(tmp_path):
    """A perf_micro run file renders per-op tables on /micro (baseline row + measured rows +
    error rows) and stays off the overview's method cards and backends table."""
    _collection(tmp_path)
    _perf_run(tmp_path)
    page = manager.render_micro(str(tmp_path), inbox=str(tmp_path / "inbox"))
    assert "pure vLLM (baseline)" in page and "+33.6%" in page
    assert "HANG" in page and ">ERROR</span>" in page
    overview = manager.render_overview(str(tmp_path), inbox=str(tmp_path / "inbox"))
    assert "/method/perf_micro" not in overview                    # no method card
    assert "perf-read-q" not in overview                           # not a backend setup either


def test_run_without_registered_spec_shows_no_verdict(tmp_path):
    entries = _collection(tmp_path)
    baselines = manager.baselines_of(str(tmp_path), entries)
    results = manager.dir_results(str(tmp_path), entries, baselines)  # "xs" not in SPECS
    assert results == {"base": None, "cand": None}
    page = manager.render_overview(str(tmp_path), inbox=str(tmp_path / "inbox"))
    assert "no scored runs" in page


def test_archive_and_discard_are_file_moves(tmp_path):
    inbox, dest = tmp_path / "inbox", tmp_path / "col"
    _execute(inbox, "fresh", "vllm")
    _execute(inbox, "junk", "vllm")
    manager.archive("fresh", str(dest), inbox=str(inbox))
    assert (dest / "fresh.pt").exists() and not (inbox / "fresh.pt").exists()
    manager.discard("junk", inbox=str(inbox))
    assert not (inbox / "junk.pt").exists()


def test_run_and_inbox_pages_render(tmp_path):
    _collection(tmp_path)
    inbox = tmp_path / "inbox"
    _execute(inbox, "fresh", "vllm")
    run_page = manager.render_run(str(tmp_path), "base", inbox=str(inbox))
    assert "base" in run_page and "coordinates" in run_page
    inbox_page = manager.render_inbox(str(tmp_path), inbox=str(inbox))
    assert "fresh" in inbox_page and "archive" in inbox_page and "discard" in inbox_page


def test_runs_index_marks_the_baseline(tmp_path):
    _collection(tmp_path)
    page = manager.render_runs(str(tmp_path), inbox=str(tmp_path / "inbox"))
    assert ">BASELINE</span>" in page and "cand" in page
    assert "xs" in page                                            # spec column


def test_repo_pages_render(tmp_path=None):
    idx = manager.render_data_index()
    assert "factual" in idx and "jlens/multihop" in idx
    src = manager.render_data_source("factual")
    assert "unit: prompt" in src and "/data/factual/0" in src
    item = manager.render_data_item("factual", 0)
    assert "factual · 000" in item
    models = manager.render_models()
    assert "llama" in models and "qwen3_5" in models
    model = manager.render_model("llama")
    assert "residual denotation" in model and "fused" in model
    backends = manager.render_backends()
    assert "nnsight-vllm" in backends
    backend = manager.render_backend("vllm_async")
    assert "vLLM async engine, in-process" in backend              # the setup line
    assert "micro.py --backend vllm_async" in backend              # no micro run in this dir
    findings = manager.render_doc("Findings", "docs/findings.md")
    assert "<h2" in findings and "<table>" in findings             # headings + tables render
    assert "the-batched-hf-verdict" in findings                    # anchored heading id


def test_export_is_selfcontained_and_link_rewritten(tmp_path):
    _collection(tmp_path)
    SPECS["xs"] = _spec()
    try:
        _import(tmp_path)
        page = manager.export_html(str(tmp_path), inbox=str(tmp_path / "inbox"))
    finally:
        del SPECS["xs"]
    assert "/spec/xs" in page and "/model/llama" in page and "/data/factual" in page
    assert "href='/" not in page                                   # every link is hash-routed
    assert "<form" not in page                                     # read-only export
    assert "hashchange" in page


def _run_all():
    import inspect
    import tempfile
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            with tempfile.TemporaryDirectory() as d:
                fn(Path(d))
            print(f"ok {name}")


if __name__ == "__main__":
    _run_all()
