"""Failed and incomplete effect checks remain browsable and exportable using saved JSON."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_manager_jobs import bundle  # noqa: E402
from test_sweep import _execute  # noqa: E402

from isb.jobs.contract import write_json  # noqa: E402
from isb.manager import Collection, dispatch, export_html  # noqa: E402
from isb.manager.pages import _effect_summary  # noqa: E402
from isb.runfile import load_run  # noqa: E402
from isb.sweep.spec import BaselineSpec, CellConfig, EffectSpec, ExecutionRegime  # noqa: E402


def _assert_pages_render(tmp_path, effect, expected):
    root = bundle(tmp_path / "bundles")
    path = root / "experiments/exp/custom-backend/result.json"
    result = json.loads(path.read_text())
    result["auxiliary_calls"] = [{"key": ["__effect__", "interactive"], "record": effect}]
    write_json(path, result)
    inbox = str(tmp_path / "inbox")
    with patch("torch.load", side_effect=AssertionError("browsing must use JSON only")):
        collection = Collection(str(root), inbox)
        pages = [dispatch(collection, route) for route in (
            "/run/attempt/exp/custom-backend", "/spec/attempt/exp")]
        pages.append(export_html(str(root), inbox, items_per_source=0))
    for page in pages:
        assert "effect guard" in page
        assert expected in page
        assert "<effect>" not in page


def test_real_failed_effect_record_renders_on_run_spec_and_export_pages(tmp_path, monkeypatch):
    from isb.perf import timing

    monkeypatch.setattr(timing, "force_gc", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    spec = CellConfig(
        "failed_effect", "fixture", "fixture", "fixture",
        [ExecutionRegime("interactive", ["one", "two"])], [({"side": "task"}, "task")],
        BaselineSpec({"side": "baseline"}), EffectSpec({"side": "baseline"}, {"side": "perturbed"}),
        warmup=0, n_trials=1, protocol_absence_reason="Effect failure fixture")

    def cells(*args):
        def cell(be, model, prompts, *, side):
            if side == "perturbed":
                raise RuntimeError("boom <effect>")
            return torch.ones(1, 2)
        return cell

    _execute(tmp_path / "artifacts", "failed", "transformers", spec, cells)
    outputs, _ = load_run(str(tmp_path / "artifacts"), "failed")
    effect = outputs[("__meta__",)][("__effect__", "interactive")]
    assert "top1_agree" not in effect and "tv" not in effect
    assert effect["error"] and effect["perturbed"]["error"]
    _assert_pages_render(tmp_path, effect, "boom &lt;effect&gt;")


@pytest.mark.parametrize("effect,expected", [
    ({}, "effect metrics unavailable"),
    ({"strong": True}, "effect metrics unavailable"),
    ({"top1_agree": None, "tv": 0.4}, "effect metrics unavailable"),
    ({"perturbed": {"error": "boom <effect>"}}, "boom &lt;effect&gt;"),
    ({"error": "legacy <effect> failure"}, "legacy &lt;effect&gt; failure"),
    ({"top1_agree": 1.0, "tv": 0.1, "strong": False}, "top1=1.00, tv=0.100, weak"),
    ({"top1_agree": 0.0, "tv": 0.8, "strong": True}, "top1=0.00, tv=0.800, strong"),
])
def test_partial_legacy_and_completed_effect_records_render_on_every_page(tmp_path, effect, expected):
    _assert_pages_render(tmp_path, effect, expected)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "unknown", True])
def test_non_numeric_or_nonfinite_historical_effect_metrics_are_unavailable(value):
    assert _effect_summary({"top1_agree": 1.0, "tv": value}) == "effect metrics unavailable"


def test_absent_legacy_effect_metadata_is_unavailable():
    assert _effect_summary(None) == "effect metrics unavailable"
