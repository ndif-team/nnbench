"""Manager tests (isb/manager), no GPU: the pages over a job bundle, imported construct-probe and
performance run files, and archive/discard as file moves."""
from isb.jobs.contract import prepare, write_json
from isb.jobs.schema import PLAN_VERSION, REPORT_VERSION, WIRE_VERSION
from isb.manager import Collection, archive, discard, dispatch, export_html, perf_points, rollup
from isb.manager.model import import_run_files
from isb.runfile import save_run
from isb.runs import run_coordinates
from isb.sweep.spec import BaselineSpec, CellConfig, ExecutionRegime

REFERENCE, CANDIDATE = "nnsight-hf/default", "nnsight-vllm/default"
GROUP = "run/exp"


def _bundle(parent, name="run"):
    """One experiment on two backends: the reference ran; the candidate is a numerical mismatch
    in both regimes."""
    root = parent / name
    spec = CellConfig("lens", "logit_lens", "gpt2", "gpt2", [ExecutionRegime("interactive", ["a"]),
                      ExecutionRegime("batched", ["a"])], [({}, "task")], BaselineSpec({}))
    experiment = prepare(spec, root / "experiments" / "exp")
    write_json(root / "plan.json", {"version": PLAN_VERSION, "experiments": ["exp"], "backends": [REFERENCE, CANDIDATE],
                                    "reference": REFERENCE, "comparison": "correctness",
                                    "status": "completed"})
    for backend, kind, commit, latency in ((REFERENCE, "transformers", "c-base", 2.), (CANDIDATE, "vllm", "c-cand", 1.)):
        directory = root / "experiments" / "exp" / backend
        directory.mkdir(parents=True)
        (directory / "result.pt").write_bytes(b"never loaded by the manager")
        write_json(directory / "execution.json", {"status": "completed", "backend": backend,
                                                  "experiment_id": experiment["id"], "image_id": "sha256:x"})
        write_json(directory / "result.json", {
            "version": WIRE_VERSION, "status": "completed", "backend": backend,
            "experiment_id": experiment["id"], "inputs_sha256": experiment["inputs_sha256"],
            "cells": [{"regime": regime, "label": "task", "state": "RAN", "error": None,
                       "median_latency_ms": latency, "latency_kind": "call",
                       "overhead_vs_baseline": 1.5}
                      for regime in ("interactive", "batched")],
            "provenance": {"engine": {"kind": kind, "mode": "sync"}, "deployment": {},
                           "client": {"nnsight": {"commit": commit}}, "coordinates": {"interface": "hf"}}})
    write_json(root / "report.json", {"version": REPORT_VERSION, "run": name, "experiments": [{
        "experiment_id": experiment["id"], "reference": REFERENCE, "comparison": "correctness",
        "cells": [{"backend": backend, "regime": regime, "label": "task", "state": state,
                   "metrics": {"top1_agree": .5, "tv": .2, "max_abs": 1.}}
                  for backend, state in ((REFERENCE, "RAN"), (CANDIDATE, "NUMERICAL_MISMATCH"))
                  for regime in ("interactive", "batched")]}]})
    return root


def _construct_run(directory, name="constructs-async"):
    meta = {("probe", "input_boundary"): {"state": "SUPPORTED", "note": "x==y", "latency_s": 1.0},
            ("probe", "barrier"): {"state": "ERROR", "note": "not shared", "latency_s": 2.0}}
    prov = {"client": {"nnsight": {"commit": "c-probe"}, "vllm": "0.19.1", "transformers": None},
            "deployment": {"kind": "local"}, "engine": {"kind": "vllm", "mode": "async", "params": {}},
            "host": {"hostname": "testbox", "gpus": []},
            "coordinates": run_coordinates(spec="constructs", methodology="constructs", family="gpt2",
                                           repo="gpt2", interface="vllm_async",
                                           cases=[{"label": label} for label in ("input_boundary", "barrier")])}
    save_run(str(directory), name, {("__meta__",): meta}, prov)
    import_run_files(str(directory))


def _perf_run(directory, name="perf-read-q"):
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
            "deployment": {"kind": "local"}, "engine": {"kind": "vllm", "mode": "async", "params": {}},
            "host": {"hostname": "t", "gpus": []},
            "coordinates": run_coordinates(spec=name, methodology="perf_micro", family="-",
                                           repo="q", interface="perf")}
    save_run(str(directory), name, {("perf_rows",): rows, ("__meta__",): {}}, prov)
    import_run_files(str(directory))


def _page(tmp_path, route):
    return dispatch(Collection(str(tmp_path), str(tmp_path / "inbox")), route)


def test_overview_cards_carry_the_rollup(tmp_path):
    _bundle(tmp_path)
    page = _page(tmp_path, "/")
    # assert the rendered chip, not the stylesheet: the CSS also contains state names
    assert ">NUMERICAL_MISMATCH</span>" in page
    assert "/method/logit_lens" in page                            # card links the methodology
    assert "How to read the map" in page                           # legend panel
    assert "setup of the runs in this directory" in page           # backends table
    assert REFERENCE in page and CANDIDATE in page


def test_method_page_is_the_model_x_backend_matrix(tmp_path):
    _bundle(tmp_path)
    page = _page(tmp_path, "/method/logit_lens")
    assert "Model × backend" in page
    assert f"/spec/{GROUP}" in page
    assert ">NUMERICAL_MISMATCH</span>" in page


def test_spec_page_shows_states_metrics_stack_and_figure(tmp_path):
    _bundle(tmp_path)
    page = _page(tmp_path, f"/spec/{GROUP}")
    assert ">NUMERICAL_MISMATCH</span>" in page and " ms per call</span>" in page
    assert "top1=" in page and "tv=" in page
    # the reference leads with its own section
    assert ">BASELINE</span>" in page and page.index("c-base") < page.index("c-cand")
    assert "<svg" in page and "median latency (ms)" in page
    assert page.count("<circle") >= 4 + 1                          # four timed cells + a legend swatch


def test_perf_points_cover_every_timed_cell(tmp_path):
    _bundle(tmp_path)
    col = Collection(str(tmp_path))
    points = perf_points(col.entries, list(col.entries))
    assert {p["run"] for p in points} == {f"{GROUP}/{REFERENCE}", f"{GROUP}/{CANDIDATE}"}
    assert {p["regime"] for p in points} == {"interactive", "batched"}


def test_rollup_verdicts():
    assert rollup(["SUPPORTED", "SUPPORTED_DEGRADED"]) == "SUPPORTED"
    assert rollup(["SUPPORTED", "ERROR"]) == "LIMITED"
    assert rollup(["ERROR", "ERROR"]) == "ERROR"
    assert rollup(["SILENTLY_WRONG", "ERROR"]) == "SILENTLY_WRONG"
    assert rollup(["SUPPORTED", "SILENTLY_WRONG"]) == "LIMITED"


def test_changed_run_file_requires_explicit_reimport(tmp_path):
    _construct_run(tmp_path)
    with (tmp_path / "constructs-async.pt").open("ab") as stream:
        stream.write(b"changed")
    col = Collection(str(tmp_path))
    assert "constructs-async" not in col.entries
    assert any("changed after import" in warning for warning in col.warnings)


def test_import_skips_run_files_that_are_neither_probes_nor_performance(tmp_path):
    save_run(str(tmp_path), "method", {("__meta__",): {("interactive", "t"): {"error": None}}},
             {"engine": {"kind": "vllm"}})
    import_run_files(str(tmp_path))
    assert not (tmp_path / "method.summary.json").exists()


def test_construct_runs_show_on_backend_pages_not_method_cards(tmp_path):
    _bundle(tmp_path)
    _construct_run(tmp_path)
    overview = _page(tmp_path, "/")
    assert "/spec/constructs" not in overview and "perf micro" in overview
    backend = _page(tmp_path, "/backend/vllm_async")
    assert "Construct support" in backend and "barrier" in backend
    assert ">ERROR</span>" in backend and "not shared" in backend
    assert "SUPPORTED · x==y" in _page(tmp_path, "/run/constructs-async")


def test_perf_micro_page_is_op_cost(tmp_path):
    _bundle(tmp_path)
    page = _page(tmp_path, "/micro")
    assert "Primitive op cost" in page and "read, steer, and qk" in page
    assert "No perf-micro runs" in page
    _perf_run(tmp_path)
    page = _page(tmp_path, "/micro")
    assert "pure vLLM (baseline)" in page and "+33.6%" in page
    assert "HANG" in page and ">ERROR</span>" in page
    overview = _page(tmp_path, "/")
    assert "/method/perf_micro" not in overview and "perf-read-q" not in overview


def test_run_file_archive_and_discard_are_file_moves(tmp_path):
    inbox, dest = tmp_path / "inbox", tmp_path / "col"
    _construct_run(inbox, "fresh")
    _construct_run(inbox, "junk")
    archive("fresh", str(dest), inbox=str(inbox))
    assert (dest / "fresh.pt").exists() and (dest / "fresh.summary.json").exists()
    assert not (inbox / "fresh.pt").exists()
    discard("junk", inbox=str(inbox))
    assert not (inbox / "junk.pt").exists()


def test_run_runs_and_inbox_pages_render(tmp_path):
    _bundle(tmp_path)
    _bundle(tmp_path / "inbox", "fresh")
    run_page = _page(tmp_path, f"/run/{GROUP}/{REFERENCE}")
    assert "c-base" in run_page and "coordinates" in run_page
    runs = _page(tmp_path, "/runs")
    assert ">BASELINE</span>" in runs and CANDIDATE in runs
    inbox = _page(tmp_path, "/inbox")
    assert "fresh" in inbox and "archive" in inbox and "discard" in inbox


def test_repo_pages_render():
    idx = dispatch(None, "/data")
    assert "factual" in idx and "jlens/multihop" in idx
    src = dispatch(None, "/data/factual")
    assert "unit: prompt" in src and "/data/factual/0" in src
    assert "factual · 000" in dispatch(None, "/data/factual/0")
    assert "llama" in dispatch(None, "/models") and "qwen3_5" in dispatch(None, "/models")
    model = dispatch(None, "/model/llama")
    assert "residual denotation" in model and "fused" in model
    assert "nnsight-vllm/default" in dispatch(None, "/backends")
    findings = dispatch(None, "/findings")
    assert "<h2" in findings and "<table>" in findings
    assert "the-batched-hf-verdict" in findings                    # anchored heading id


def test_export_is_selfcontained_and_link_rewritten(tmp_path):
    _bundle(tmp_path)
    page = export_html(str(tmp_path), str(tmp_path / "inbox"))
    assert f"/spec/{GROUP}" in page and "/model/gpt2" in page and "/data/factual" in page
    assert "href='/" not in page                                   # every link is hash-routed
    assert "<form" not in page                                     # read-only export
    assert "hashchange" in page
