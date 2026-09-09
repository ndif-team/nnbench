"""Metadata-only result browsing, safe routing, and complete-run archival."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from isb.jobs.contract import prepare, write_json
from isb.manager import Collection, dispatch, export_html
from isb.manager.model import archive, discard, _rename_exclusive
from isb.sweep.spec import BaselineSpec, CellConfig, Workload


def bundle(parent, name="attempt", backends=("nnsight-hf", "custom-backend"), with_report=True):
    root = parent / name
    spec = CellConfig("lens", "logit_lens", "gpt2", "model", [Workload("interactive", ["a"])],
                      [({}, "task")], BaselineSpec({}))
    experiment = prepare(spec, root / "experiments" / "exp")
    plan = {"version": 1, "experiments": ["exp"], "backends": list(backends), "reference": backends[0],
            "comparison": "correctness", "status": "completed"}
    write_json(root / "plan.json", plan)
    for backend in backends:
        directory = root / "experiments" / "exp" / backend
        directory.mkdir()
        (directory / "result.pt").write_bytes(b"intentionally not a pickle: viewer must never load it")
        write_json(directory / "execution.json", {"status": "completed", "backend": backend,
            "experiment_id": experiment["id"], "image_id": "sha256:fixture"})
        write_json(directory / "result.json", {"version": 1, "status": "completed", "backend": backend,
            "experiment_id": experiment["id"], "inputs_sha256": experiment["inputs_sha256"],
            "cells": [{"workload": "interactive", "label": "task", "state": "RAN", "median_latency_ms": 1.}],
            "provenance": {"engine": {"kind": "transformers" if backend == backends[0] else "vllm"},
                           "client": {}, "coordinates": {"interface": "hf"}, "deployment": {}}})
    if with_report:
        write_json(root / "report.json", {"version": 1, "run": name, "experiments": [{
            "experiment_id": experiment["id"], "reference": backends[0], "comparison": "correctness",
            "cells": [{"backend": backend, "workload": "interactive", "label": "task",
                       "state": "RAN" if backend == backends[0] else "SILENTLY_WRONG",
                       "metrics": {"top1_agree": .5, "tv": .2, "max_abs": 1.}}
                      for backend in backends]}]})
    return root


def test_new_results_use_saved_verdicts_without_torch_load_or_scoring(tmp_path):
    root = bundle(tmp_path)
    with patch("torch.load", side_effect=AssertionError("must not load tensors")), \
         patch("isb.jobs.score.score_run", side_effect=AssertionError("must not rescore")), \
         patch("isb.sweep.score.score_runs", side_effect=AssertionError("must not rescore")):
        col = Collection(str(root), str(tmp_path / "inbox"))
        assert len(col.entries) == 2
        assert col.results["attempt/exp/custom-backend"][0].state == "SILENTLY_WRONG"
        assert ">SILENTLY_WRONG</span>" in dispatch(col, "/run/attempt/exp/custom-backend")
        assert "sha256:fixture" in dispatch(col, "/run/attempt/exp/custom-backend")
        assert dispatch(col, "/backend/custom-backend") is not None
        page = export_html(str(root), str(tmp_path / "inbox"), items_per_source=1)
        assert "SILENTLY_WRONG" in page and "custom-backend" in page
        assert "<form" not in page


def test_history_preserves_experiment_and_backend_identities(tmp_path):
    bundle(tmp_path, "first")
    bundle(tmp_path, "second")
    col = Collection(str(tmp_path))
    assert len(col.by_spec()) == 2
    assert len(col.baselines) == 2
    page = dispatch(col, "/method/logit_lens")
    assert "first/exp" in page and "second/exp" in page and "custom-backend" in page
    assert "not a latest-run selector" in dispatch(col, "/")


def test_without_report_never_invents_comparison(tmp_path):
    root = bundle(tmp_path, with_report=False)
    col = Collection(str(root))
    assert col.results["attempt/exp/custom-backend"][0].state == "RAN"
    assert col.warnings


def test_failed_execution_overrides_old_passing_report(tmp_path):
    root = bundle(tmp_path)
    path = root / "experiments/exp/custom-backend/execution.json"
    data = json.loads(path.read_text())
    data.update(status="failed", error="timeout")
    write_json(path, data)
    col = Collection(str(root))
    cell = col.results["attempt/exp/custom-backend"][0]
    assert cell.state == "JOB_FAILED" and cell.error == "timeout"


def test_job_level_scoring_failure_is_visible(tmp_path):
    root = bundle(tmp_path)
    path = root / "report.json"
    data = json.loads(path.read_text())
    data["experiments"][0]["cells"][1] = {"backend": "custom-backend", "state": "JOB_FAILED", "error": "corrupt tensor"}
    write_json(path, data)
    assert Collection(str(root)).results["attempt/exp/custom-backend"][0].state == "JOB_FAILED"


def test_missing_output_or_mismatched_result_is_visible(tmp_path):
    root = bundle(tmp_path)
    (root / "experiments/exp/custom-backend/result.pt").unlink()
    assert Collection(str(root)).results["attempt/exp/custom-backend"][0].state == "JOB_FAILED"
    path = root / "experiments/exp/nnsight-hf/result.json"
    data = json.loads(path.read_text())
    data["inputs_sha256"] = "wrong"
    write_json(path, data)
    assert Collection(str(root)).results["attempt/exp/nnsight-hf"][0].state == "JOB_FAILED"


def test_invalid_manifest_does_not_hide_other_runs(tmp_path):
    bundle(tmp_path, "good")
    bad = bundle(tmp_path, "bad")
    (bad / "plan.json").write_text("broken")
    col = Collection(str(tmp_path))
    assert len(col.entries) == 2 and col.warnings


@pytest.mark.parametrize("route", ["/run/../../outside", "/run//tmp/outside", "/run/missing",
                                    "/backend/missing", "/data/missing", "/data/factual/9999"])
def test_unknown_and_traversal_routes_are_404(tmp_path, route):
    with patch("isb.manager.model.load_run", side_effect=AssertionError("no pickle access")):
        assert dispatch(Collection(str(tmp_path)), route) is None


def test_symlink_manifest_escape_is_refused(tmp_path):
    outside = bundle(tmp_path / "outside")
    collection = tmp_path / "collection"
    collection.mkdir()
    (collection / "alias").symlink_to(outside, target_is_directory=True)
    assert not Collection(str(collection)).entries


def test_archive_moves_entire_bundle_and_refuses_collision(tmp_path):
    inbox, dest = tmp_path / "inbox", tmp_path / "archive"
    source = bundle(inbox)
    stored = bundle(dest)
    original = (stored / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        archive("attempt", str(dest), str(inbox))
    assert source.exists() and (stored / "report.json").read_bytes() == original
    moved = Path(archive("attempt", str(tmp_path / "other"), str(inbox)))
    assert not source.exists() and (moved / "experiments/exp/custom-backend/result.pt").exists()
    assert len(Collection(str(moved)).entries) == 2


def test_exclusive_rename_cannot_overwrite_even_after_precheck(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    src.write_text("new")
    dest.write_text("old")
    with pytest.raises(FileExistsError):
        _rename_exclusive(src, dest)
    assert src.read_text() == "new" and dest.read_text() == "old"


def test_discard_is_recoverable_and_not_discovered(tmp_path):
    source = bundle(tmp_path)
    saved = Path(discard("attempt", str(tmp_path)))
    assert not source.exists() and (saved / "plan.json").exists()
    assert ".trash" in saved.parts
    assert not Collection(str(tmp_path)).entries


def test_inbox_links_and_details_use_namespaced_ids(tmp_path):
    bundle(tmp_path / "inbox")
    col = Collection(str(tmp_path / "archive"), str(tmp_path / "inbox"))
    assert "/inbox-run/attempt" in dispatch(col, "/inbox")
    assert "/run/inbox/attempt/exp/custom-backend" in dispatch(col, "/inbox-run/attempt")
    assert "SILENTLY_WRONG" in dispatch(col, "/run/inbox/attempt/exp/custom-backend")


def test_legacy_files_are_not_loaded_implicitly(tmp_path):
    (tmp_path / "untrusted.pt").write_bytes(b"bad pickle")
    with patch("isb.manager.model.load_run", side_effect=AssertionError("no implicit import")):
        col = Collection(str(tmp_path))
        assert not col.entries
        assert "--import-legacy" in col.warnings[0]


def test_empty_collection_does_not_scan_working_directory(tmp_path, monkeypatch):
    bundle(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert not Collection("").entries


@pytest.mark.parametrize("bad", [None, "bad", {"state": "SUPPORTED", "metrics": "bad"}])
def test_malformed_report_rows_fall_back_to_execution(tmp_path, bad):
    root = bundle(tmp_path)
    path = root / "report.json"
    data = json.loads(path.read_text())
    data["experiments"][0]["cells"][1] = bad
    write_json(path, data)
    col = Collection(str(root))
    assert col.warnings
    assert col.results["attempt/exp/custom-backend"][0].state == "RAN"
    assert dispatch(col, "/run/attempt/exp/custom-backend") is not None


def test_failed_reference_remains_visible_in_method_matrix(tmp_path):
    root = bundle(tmp_path)
    (root / "experiments/exp/nnsight-hf/result.pt").unlink()
    assert ">JOB_FAILED</span>" in dispatch(Collection(str(root)), "/method/logit_lens")


def test_http_routes_and_archive_action_security(tmp_path):
    from io import BytesIO
    from types import SimpleNamespace
    from urllib.parse import urlencode
    import re
    from scripts.manager import make_handler

    inbox, archive_dir = tmp_path / "inbox", tmp_path / "archive"
    source = bundle(inbox)
    handler = make_handler(str(archive_dir), str(inbox))

    class Connection:
        def __init__(self, data):
            self.input, self.output = BytesIO(data), bytearray()

        def makefile(self, *args):
            return self.input

        def sendall(self, data):
            self.output.extend(data)

    def request(path, form=None, host="127.0.0.1:6688", origin=None):
        body = urlencode(form).encode() if form is not None else b""
        headers = [f"{'POST' if form is not None else 'GET'} {path} HTTP/1.0", f"Host: {host}"]
        if form is not None:
            headers += [f"Content-Length: {len(body)}", "Content-Type: application/x-www-form-urlencoded"]
        if origin:
            headers += [f"Origin: {origin}"]
        connection = Connection(("\r\n".join(headers) + "\r\n\r\n").encode() + body)
        handler(connection, ("127.0.0.1", 12345), SimpleNamespace(server_port=6688))
        response = connection.output.decode()
        return int(response.split()[1]), response

    assert request("/?query=yes")[0] == 200
    assert request("/run/%2e%2e/%2e%2e/outside")[0] == 404
    assert request("/", host="attacker.example:6688")[0] == 421
    status, page = request("/inbox")
    assert status == 200
    token = re.search(r"name='csrf' value='([^']+)'", page)[1]
    assert request("/archive", {"name": "attempt"})[0] == 403
    assert request("/archive", {"name": "attempt", "csrf": "é"})[0] == 403
    form = {"name": "attempt", "csrf": token}
    assert request("/archive", form, origin="https://attacker.example")[0] == 403
    assert request("/unexpected", form)[0] == 404
    assert request("/archive", {**form, "name": "../attempt"})[0] == 404
    assert source.exists()
    assert request("/archive", form)[0] == 303
    assert not source.exists() and (archive_dir / "attempt/report.json").exists()
    bundle(inbox)
    assert request("/archive", form)[0] == 409
    assert source.exists()
    assert request("/discard", form)[0] == 303
    assert not source.exists() and list(inbox.glob(".trash/*/attempt/report.json"))


def test_partial_report_never_looks_complete(tmp_path):
    root = bundle(tmp_path)
    path = root / "report.json"
    data = json.loads(path.read_text())
    data["experiments"][0]["cells"].pop()
    write_json(path, data)
    col = Collection(str(root))
    assert col.results["attempt/exp/custom-backend"][0].state == "RAN"
    assert any("does not cover" in warning for warning in col.warnings)


def test_provenance_cannot_create_executable_commit_link(tmp_path):
    root = bundle(tmp_path)
    path = root / "experiments/exp/custom-backend/result.json"
    data = json.loads(path.read_text())
    data["provenance"]["client"]["nnsight"] = {"commit": "123456", "remote": "javascript:alert('github.com')"}
    write_json(path, data)
    page = dispatch(Collection(str(root)), "/run/attempt/exp/custom-backend")
    assert page is not None and "href=\"javascript:" not in page
