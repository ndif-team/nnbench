"""Perf-microbench framework tests. No GPU: covers Config, prompts, footprint, the overhead/
baseline math, and the full sweep runner via the _echo stub (incl. the timeout/HANG path).

Run with pytest, or directly: `python tests/test_perf.py`.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isb.perf.core import (  # noqa: E402
    Config, attach_overhead, layer_indices, make_prompts, row_from, throughput,
)


def test_config_roundtrip():
    c = Config(system="nnsight_vllm", op="read", footprint="half", prompt_len=128)
    assert Config.from_json(c.to_json()) == c
    assert "read" in c.cell_id() and "fp-half" in c.cell_id()


def test_make_prompts_exact_and_deterministic():
    c = Config(system="x", prompt_len=37, batch=5, seed=1)
    p1, p2 = make_prompts(c), make_prompts(c)
    assert len(p1) == 5 and all(len(x) == 37 for x in p1)
    assert p1 == p2


def test_layer_indices():
    assert layer_indices(Config(system="x", footprint="one"), 12) == [6]
    assert layer_indices(Config(system="x", footprint="all"), 4) == [0, 1, 2, 3]
    assert layer_indices(Config(system="x", footprint="half"), 6) == [0, 2, 4]


def test_eff_new_tokens_and_throughput():
    assert Config(system="x", phase="prefill", new_tokens=64).eff_new_tokens() == 1
    dec = Config(system="x", phase="decode", batch=4, new_tokens=64)
    assert throughput(dec, 0.1) == 4 * 64 / 0.1


def test_overhead_matches_pure_vllm_baseline():
    kw = dict(repo="m", phase="decode", prompt_len=64, batch=4, new_tokens=64)
    base = row_from(Config(system="pure_vllm", op="none", **kw), {"median_gen_lat_s": 0.1})
    read = row_from(Config(system="nnsight_vllm", op="read", **kw), {"median_gen_lat_s": 0.2})
    attach_overhead([base, read])
    assert read["baseline_tok_per_s"] == base["median_tok_per_s"]
    assert read["overhead_pct"] == 50.0
    assert base["overhead_pct"] == 0.0


def test_overhead_none_without_baseline():
    read = row_from(Config(system="nnsight_vllm", op="read", repo="m"), {"median_gen_lat_s": 0.2})
    attach_overhead([read])
    assert read["overhead_pct"] is None


def test_runner_sweep_with_echo(tmp_path: Path):
    plan = [
        asdict(Config(system="_echo", op="none", footprint="one")),
        asdict(Config(system="_echo", op="read", footprint="all")),
        asdict(Config(system="_echo", op="hang", timeout_s=1.0)),   # exercises the timeout/HANG path
    ]
    planf, outf = tmp_path / "plan.json", tmp_path / "out.json"
    planf.write_text(json.dumps(plan))
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "perf.py"), "sweep",
         "--plan", str(planf), "--out", str(outf)],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stderr
    rows = json.loads(outf.read_text())
    assert len(rows) == 3
    by_op = {row["op"]: row for row in rows}
    assert by_op["none"]["median_tok_per_s"] > 0
    assert by_op["read"]["median_tok_per_s"] > 0
    assert "HANG" in (by_op["hang"]["error"] or "")


def test_env_for_mapping():
    spec = importlib.util.spec_from_file_location("_perf_runner", ROOT / "scripts" / "perf.py")
    perf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(perf)
    # unmapped (e.g. the _echo stub) -> current interpreter, no extra env
    py, extra = perf._env_for("_echo")
    assert py == sys.executable and extra == {}
    # nnsight carries PYTHONPATH to the dev src
    py, extra = perf._env_for("nnsight_vllm")
    assert "nnsight-vllm" in py and "PYTHONPATH" in extra
    # ISB_PY_<system> overrides the python
    os.environ["ISB_PY_vllm_hook"] = "/custom/python"
    try:
        assert perf._env_for("vllm_hook")[0] == "/custom/python"
    finally:
        del os.environ["ISB_PY_vllm_hook"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            if "tmp_path" in fn.__code__.co_varnames[: fn.__code__.co_argcount]:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print(f"ok  {name}")
    print("all perf framework tests passed")
