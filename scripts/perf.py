"""Perf-microbench runner (docs/perf-micro-design.md).

Two modes:
  worker --config-file F   run ONE cell in this process, print `RESULT_JSON {...}`.
  sweep  --plan P          launch one subprocess per cell (timeout watchdog), collect, attach
                           overhead, write ONE run file (methodology "perf_micro") into
                           --run-dir (default: the inbox), so perf sweeps flow through the same
                           inbox -> archive -> manager path as every other run. `--out F` also
                           writes the rows as plain JSON. One system per process (vllm-lens /
                           vllm-hook / nnsight never share an interpreter; vLLM EngineCore uses
                           spawn).

Each cell subprocess runs under ISB_PY_<system> (for example a wrapper around `apptainer exec` on
the system's backend image), else under the launcher's own interpreter. The auto-registering
plugins (vLLM-Lens, vLLM-Hook) must not share an interpreter with the other systems. Example:
  CUDA_VISIBLE_DEVICES=5 ISB_PY_vllm_lens=/path/to/lens-python python scripts/perf.py sweep \
      --plan plans/read_footprint_qwen.json --name perf-read-qwen
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

def _python_for(system: str) -> str:
    """The interpreter a system's cells run under: ISB_PY_<system>, else this one. The worker puts
    isb on sys.path itself, so that interpreter does not need this repository installed."""
    return os.environ.get(f"ISB_PY_{system}", sys.executable)


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
    py = _python_for(cfg.system)
    if not os.path.exists(py):
        return _error_row(cfg, f"env python not found for {cfg.system!r}: {py} "
                               f"(set ISB_PY_{cfg.system}=/path/to/python)")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        f.write(cfg.to_json())
        cfgfile = f.name
    cmd = [py, str(Path(__file__).resolve()), "worker", "--config-file", cfgfile]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=cfg.timeout_s)
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


def _write_run_file(plan: list[Config], rows: list[dict], run_dir: str, name: str) -> str:
    """One run file per sweep (methodology "perf_micro"): the rows as outputs, the launcher's
    resolved provenance, and the per-system env pythons the cells actually ran under."""
    from isb.runfile import save_run
    from isb.runs import EngineConfig, RunConfig, resolve_provenance, run_coordinates

    prov = resolve_provenance(RunConfig(engine=EngineConfig("vllm")))
    prov["coordinates"] = run_coordinates(
        spec=name, methodology="perf_micro", family="-",
        repo=", ".join(sorted({c.repo for c in plan})), interface="perf",
        cases=[{"label": r["cell"]} for r in rows])
    prov["perf_envs"] = {s: _python_for(s) for s in sorted({c.system for c in plan})}
    return save_run(run_dir, name, {("perf_rows",): rows, ("__meta__",): {}}, prov)


def sweep(plan_path: str, run_dir: str, name: str | None, out_path: str | None) -> None:
    plan = [Config(**c) for c in json.load(open(plan_path))]
    rows = []
    for i, cfg in enumerate(plan, 1):
        print(f"[{i}/{len(plan)}] {cfg.cell_id()}", file=sys.stderr, flush=True)
        row = _run_cell(cfg)
        _teardown_note(row)
        rows.append(row)
    attach_overhead(rows)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(out_path, "w"), indent=2)
        print(f"wrote {len(rows)} rows -> {out_path}", file=sys.stderr)
    path = _write_run_file(plan, rows, run_dir, name or Path(plan_path).stem)
    print(f"wrote perf run file ({len(rows)} rows) -> {path}", file=sys.stderr)


def main() -> None:
    from isb.runfile import INBOX

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--config-file", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("--plan", required=True)
    s.add_argument("--run-dir", default=INBOX, help="run-file directory (default: the inbox)")
    s.add_argument("--name", default=None, help="run name (default: the plan file's stem)")
    s.add_argument("--out", default=None, help="also write the rows as plain JSON to this path")
    args = ap.parse_args()

    if args.mode == "worker":
        worker(Config.from_json(Path(args.config_file).read_text()))
    else:
        sweep(args.plan, args.run_dir, args.name, args.out)


if __name__ == "__main__":
    main()
