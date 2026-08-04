"""The manager's data layer: one Collection per directory, computed once and cached by mtime.

A collection is a directory of run files with one baseline per spec; every other run is scored
against its spec's baseline through the one scoring path (`score_runs`, with `reference=None`
for the baselines themselves). Pages consume a Collection and never touch files or scoring.
"""
from __future__ import annotations

import os
import shutil
import time

from ..runfile import INBOX, load_run, run_path

_PASS = {"SUPPORTED", "SUPPORTED_DEGRADED", "EQUIVALENT", "EQUIVALENT_DEGRADED"}

_LOAD_CACHE: dict = {}      # (abspath, mtime) -> (outputs, provenance)
_RESULTS_CACHE: dict = {}   # directory fingerprint -> scored results (CellResults are light)


def list_runs(dir_path: str) -> list[str]:
    if not os.path.isdir(dir_path):
        return []
    return sorted(f[:-3] for f in os.listdir(dir_path) if f.endswith(".pt"))


def _load_cached(dir_path: str, name: str):
    path = os.path.abspath(run_path(dir_path, name))
    key = (path, os.path.getmtime(path))
    if key not in _LOAD_CACHE:
        _LOAD_CACHE.pop(next((k for k in _LOAD_CACHE if k[0] == path), None), None)
        _LOAD_CACHE[key] = load_run(dir_path, name)
    return _LOAD_CACHE[key]


def load_dir(dir_path: str) -> dict[str, tuple[dict, dict]]:
    """{run_name: (outputs, provenance)} for every run file in the directory."""
    return {name: _load_cached(dir_path, name) for name in list_runs(dir_path)}


def spec_of(prov: dict) -> str:
    return prov["coordinates"]["spec"].split("@")[0]    # rebound data keeps the base spec name


def probe_results(outputs: dict) -> dict[str, dict]:
    """{probe_name: {state, note, latency_s}} for a construct-probe run; {} for method runs."""
    meta = outputs.get(("__meta__",), {})
    return {k[1]: m for k, m in meta.items()
            if isinstance(k, tuple) and len(k) == 2 and k[0] == "probe"}


def baselines_of(dir_path: str, entries: dict) -> dict[str, str]:
    """{spec_name: baseline run}. One baseline per spec: a run can only reference runs of its
    own spec and data, so a directory holding the whole corpus carries one baseline per method;
    the N-runs-vs-one-baseline structure holds within each spec. The BASELINE file lists one run
    name per line; without it, a spec whose only transformers run is unambiguous uses that run.
    Probe runs never qualify: they self-check, so they cannot reference anything."""
    marker = os.path.join(dir_path, "BASELINE")
    if os.path.exists(marker):
        named = [ln.strip() for ln in open(marker).read().splitlines() if ln.strip()]
        return {spec_of(entries[n][1]): n for n in named if n in entries}
    per_spec: dict[str, list[str]] = {}
    for n, (out, prov) in entries.items():
        if prov["engine"]["kind"] == "transformers" and not probe_results(out):
            per_spec.setdefault(spec_of(prov), []).append(n)
    return {spec: names[0] for spec, names in per_spec.items() if len(names) == 1}


def comparable(prov: dict, base_prov: dict) -> bool:
    """A run gets verdicts only when it ran the baseline's spec on the baseline's data."""
    c, b = prov["coordinates"], base_prov["coordinates"]
    return c["spec"] == b["spec"] and c["data"] == b["data"]


def dir_results(dir_path: str, entries: dict, baselines: dict[str, str]):
    """{run_name: list[CellResult]} for every run through the one scoring path: candidates vs
    their spec's baseline, baselines with no reference (raw states, no verdicts). Runs whose spec
    is unregistered or whose data differ from the baseline's map to None."""
    from ..specs import SPECS
    from ..sweep.score import score_runs

    fp = (os.path.abspath(dir_path),
          tuple(sorted((n, os.path.getmtime(run_path(dir_path, n))) for n in entries)),
          tuple(sorted(baselines.items())))
    if fp in _RESULTS_CACHE:
        return _RESULTS_CACHE[fp]
    if len(_RESULTS_CACHE) > 8:
        _RESULTS_CACHE.clear()

    out = {}
    for name, (_, prov) in entries.items():
        spec_name = spec_of(prov)
        baseline = baselines.get(spec_name)
        if name == baseline:
            out[name] = score_runs(SPECS[spec_name], dir_path, name, None, quiet=True) \
                if spec_name in SPECS else None
            continue
        if baseline is None or spec_name not in SPECS \
                or not comparable(prov, entries[baseline][1]):
            out[name] = None
            continue
        # dtype control: a same-spec run of the candidate's engine at fp32 disambiguates
        # precision near-ties (SUPPORTED ·fp) from real bugs, from data, no live rerun
        ctl = next((n for n, (_, p) in entries.items()
                    if n not in (name, baseline) and spec_of(p) == spec_name
                    and p["engine"]["kind"] == prov["engine"]["kind"]
                    and (p["engine"].get("params") or {}).get("dtype") == "float32"), None)
        out[name] = score_runs(SPECS[spec_name], dir_path, name, baseline, ctl=ctl, quiet=True)
    _RESULTS_CACHE[fp] = out
    return out


def rollup(states: list[str]) -> str:
    """One verdict over a run's result states: SUPPORTED = everything passes; LIMITED = some
    realization works and some does not; when nothing passes, the worst failure (SILENTLY_WRONG
    over ERROR)."""
    if states and all(s in _PASS for s in states):
        return "SUPPORTED"
    if not any(s in _PASS for s in states):
        return "SILENTLY_WRONG" if any(s in ("SILENTLY_WRONG", "DIVERGENT") for s in states) \
            else "ERROR"
    return "LIMITED"


def backend_label(prov: dict) -> tuple[str, str]:
    """(display name, setup line), derived from the run's provenance, never hand-written per
    directory: the setup column must describe the runs it summarizes."""
    e, d = prov["engine"], prov["deployment"]
    if e["kind"] == "transformers":
        return "HF-nnsight", "nnsight on vanilla transformers, eager, in-process"
    mode = e.get("mode") or "async"
    place = "served" if d.get("kind") == "serve" else "in-process"
    return "vLLM-nnsight", f"nnsight on the vLLM {mode} engine, {place}"


def executed(prov: dict, path: str) -> str:
    """The run's own timestamp; file mtime for run files written before it was recorded."""
    return prov.get("executed") or \
        time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(path)))


def _delegation_target(src: str) -> str | None:
    """The helper name when the cell (or its trace-body closure) is a single `return helper(...)`
    delegation; the method's actual operations live in that helper."""
    import ast
    import textwrap

    try:
        outer = ast.parse(textwrap.dedent(src)).body[0]
    except SyntaxError:
        return None
    if not isinstance(outer, ast.FunctionDef):
        return None
    inner = next((n for n in ast.walk(outer)
                  if isinstance(n, ast.FunctionDef) and n is not outer), None)
    body = [s for s in (inner or outer).body
            if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    if len(body) == 1 and isinstance(body[0], ast.Return) \
            and isinstance(body[0].value, ast.Call) \
            and isinstance(body[0].value.func, ast.Name):
        return body[0].value.func.id
    return None


def trace_source(methodology: str, family: str, interface: str) -> str | None:
    """The source holding the method's trace operations. Starts at the registered cell; when the
    cell just delegates to a module helper (the generic cells share their operations this way),
    follows that one hop, because the card must show the operations, never the delegation.
    None when the cell is defined somewhere source-less (a test stub, a REPL)."""
    import inspect

    from ..methodologies.registry import get_cell

    try:
        fn = get_cell(methodology, family, interface)
        fn = getattr(fn, "__wrapped__", fn)
        src = inspect.getsource(fn)
        target = _delegation_target(src)
        if target:
            helper = getattr(inspect.getmodule(fn), target, None)
            if inspect.isfunction(helper):
                return inspect.getsource(helper)
        return src
    except (KeyError, OSError, TypeError):
        return None


def perf_points(entries: dict, run_names: list[str]) -> list[dict]:
    """One operating point per (run, variant, regime) cell that was timed: median latency and
    overhead over that run's own no-intervention baseline. The verdict baseline's cells are
    points like any other: perf is a property of the execution, and its config (e.g. tp=1) is
    exactly what the other configs compare against here. Throughput and peak memory ride along
    for the tooltip when recorded."""
    pts = []
    for name in sorted(run_names):
        meta = entries[name][0].get(("__meta__",), {})
        for k, m in meta.items():
            if not (isinstance(k, tuple) and len(k) == 2) or m.get("error"):
                continue
            if m.get("median_latency_ms") is None or m.get("overhead_vs_baseline") is None:
                continue
            pts.append({"run": name, "regime": k[0], "label": k[1],
                        "lat": m["median_latency_ms"], "ovh": m["overhead_vs_baseline"],
                        "tp": m.get("throughput"), "mem": m.get("peak_mem_mb")})
    return pts


def archive(run_name: str, dest_dir: str, inbox: str = INBOX) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    dest = run_path(dest_dir, run_name)
    shutil.move(run_path(inbox, run_name), dest)
    return dest


def discard(run_name: str, inbox: str = INBOX) -> None:
    os.remove(run_path(inbox, run_name))


class Collection:
    """Everything the pages need about one directory, derived once per access."""

    def __init__(self, dir_path: str, inbox: str = INBOX):
        self.dir_path = dir_path
        self.inbox = inbox
        self.entries = load_dir(dir_path)
        self.baselines = baselines_of(dir_path, self.entries)
        self.results = dir_results(dir_path, self.entries, self.baselines)

    def by_spec(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for name, (_, prov) in self.entries.items():
            groups.setdefault(spec_of(prov), []).append(name)
        return groups

    def inbox_runs(self) -> list[str]:
        return list_runs(self.inbox)

    def load(self, name: str) -> tuple[dict, dict]:
        """A run from the directory, or the inbox when it only exists there."""
        where = self.dir_path if name in self.entries else self.inbox
        return _load_cached(where, name)
