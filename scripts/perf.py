"""Perf-microbench runner (docs/perf-micro-design.md).

Two modes:
  worker --config-file F   run ONE cell in this process, print `RESULT_JSON {...}`.
  sweep  --plan P --out O   launch one subprocess per cell (timeout watchdog), collect, attach
                            overhead, write O. One system per process (vllm-lens / vllm-hook /
                            nnsight never share an interpreter; vLLM EngineCore uses spawn).

Each cell runs in its own conda env (per-system, see _SYSTEM_ENV). The sweep launcher can be any
python that imports isb.perf.core; each cell subprocess is launched with its system's env python.
Example:
  CUDA_VISIBLE_DEVICES=5 /disk/u/zikai/anaconda3/envs/nnsight-vllm/bin/python scripts/perf.py sweep \
      --plan plans/read_footprint.json --out results/read_footprint.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isb.perf.core import Config, ROW_CFG_FIELDS, attach_overhead, row_from  # noqa: E402

_PREFIX = "RESULT_JSON "

# Per-system env python + extra env vars (machine defaults; override via ISB_PY_<system>=/path).
# The auto-registering plugins (vllm-lens / vllm-hook) must run in isolated envs; nnsight / pure /
# native share nnsight-vllm (nnsight is opt-in import, no auto-registration). The worker puts isb on
# sys.path itself, so the env python does not need interp-serve-bench installed.
_ENVS = "/disk/u/zikai/anaconda3/envs"
_NNSIGHT_SRC = "/disk/u/zikai/nnsight/src"
_SYSTEM_ENV: dict[str, tuple[str, dict]] = {
    "nnsight_vllm": (f"{_ENVS}/nnsight-vllm/bin/python", {"PYTHONPATH": _NNSIGHT_SRC}),
    "pure_vllm":    (f"{_ENVS}/nnsight-vllm/bin/python", {}),
    "native_eagle": (f"{_ENVS}/nnsight-vllm/bin/python", {}),
    "vllm_lens":    (f"{_ENVS}/bench-vllm-lens/bin/python", {}),
    "vllm_hook":    (f"{_ENVS}/bench-vllm-hook/bin/python", {}),
}


def _env_for(system: str) -> tuple[str, dict]:
    """(python, extra_env) for a system. ISB_PY_<system> overrides the python; unmapped systems
    (e.g. the _echo test stub) fall back to the current interpreter so tests run without the envs."""
    py, extra = _SYSTEM_ENV.get(system, (sys.executable, {}))
    return os.environ.get(f"ISB_PY_{system}", py), extra


def _error_row(cfg: Config, msg: str) -> dict:
    row = {f: getattr(cfg, f) for f in ROW_CFG_FIELDS}
    row.update({"cell": cfg.cell_id(), "error": msg, "median_tok_per_s": None})
    return row


def worker(cfg: Config) -> None:
    try:
        mod = importlib.import_module(f"isb.perf.systems.{cfg.system}")
        metrics = mod.run(cfg)
        row = row_from(cfg, metrics)
        row["error"] = None
    except Exception as e:  # clean per-cell error; the runner records it, never crashes the sweep
        tail = str(e).strip().splitlines()
        row = _error_row(cfg, tail[-1] if tail else type(e).__name__)
    print(_PREFIX + json.dumps(row), flush=True)


def _run_cell(cfg: Config) -> dict:
    py, extra = _env_for(cfg.system)
    if not os.path.exists(py):
        return _error_row(cfg, f"env python not found for {cfg.system!r}: {py} "
                               f"(set ISB_PY_{cfg.system}=/path/to/python)")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        f.write(cfg.to_json())
        cfgfile = f.name
    env = os.environ.copy()
    for k, v in extra.items():
        # prepend so dev nnsight (PYTHONPATH) shadows the env's editable install
        env[k] = v + os.pathsep + env[k] if (k == "PYTHONPATH" and env.get(k)) else v
    cmd = [py, str(Path(__file__).resolve()), "worker", "--config-file", cfgfile]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=cfg.timeout_s, env=env)
    except subprocess.TimeoutExpired:
        # HANG: the stuck process cannot be trusted; one-process-per-cell means its exit reclaims GPU.
        return _error_row(cfg, f"HANG > {cfg.timeout_s:.0f}s")
    finally:
        os.unlink(cfgfile)
    for line in reversed(p.stdout.splitlines()):
        if line.startswith(_PREFIX):
            return json.loads(line[len(_PREFIX):])
    return _error_row(cfg, f"no result (stderr tail: {p.stderr[-300:].strip()})")


def _teardown_note(row: dict) -> None:
    # ponytail: rely on per-cell subprocess exit to reclaim GPU; no active orphan kill. If leaks are
    # observed across cells, add an NVML free-memory delta check or a targeted pkill of EngineCore here.
    print(f"    -> {row.get('median_tok_per_s')} tok/s  peak={row.get('peak_mem_mb')}MB  "
          f"err={row.get('error')}", file=sys.stderr, flush=True)


def sweep(plan_path: str, out_path: str) -> None:
    plan = [Config(**c) for c in json.load(open(plan_path))]
    rows = []
    for i, cfg in enumerate(plan, 1):
        print(f"[{i}/{len(plan)}] {cfg.cell_id()}", file=sys.stderr, flush=True)
        row = _run_cell(cfg)
        _teardown_note(row)
        rows.append(row)
    attach_overhead(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out_path, "w"), indent=2)
    print(f"wrote {len(rows)} rows -> {out_path}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--config-file", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("--plan", required=True)
    s.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.mode == "worker":
        worker(Config.from_json(Path(args.config_file).read_text()))
    else:
        sweep(args.plan, args.out)


if __name__ == "__main__":
    main()
