"""Metadata-only collections of saved reports; trusted legacy import is an explicit operation."""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

from ..runfile import INBOX, load_run

_PASS = {"SUPPORTED", "SUPPORTED_DEGRADED", "EQUIVALENT", "EQUIVALENT_DEGRADED"}

def list_runs(dir_path: str) -> list[str]:
    if not os.path.isdir(dir_path):
        return []
    root = Path(dir_path)
    return sorted({p.stem if p.suffix == ".pt" else p.name for p in root.iterdir()
                   if not p.is_symlink() and not p.name.startswith(".")
                   and (p.is_file() and p.suffix == ".pt"
                        or p.is_dir() and (p / "plan.json").is_file())})


def _load_cached(dir_path: str, name: str):
    # Historical name retained for callers; only a small JSON summary is read, never cached tensors.
    from .records import child, read_json
    path = child(Path(dir_path), name + ".summary.json")
    summary = read_json(path)
    artifact = child(Path(dir_path), name + ".pt").stat()
    if summary.get("artifact_stat") != [artifact.st_size, artifact.st_mtime_ns]:
        raise ValueError("legacy artifact changed after import")
    meta = {tuple(key): value for key, value in summary["metadata"]}
    return {("__meta__",): meta, ("perf_rows",): summary.get("perf_rows", [])}, summary["provenance"]


def load_dir(dir_path: str) -> dict[str, tuple[dict, dict]]:
    """{run_name: (outputs, provenance)} for every run file in the directory."""
    return {name: _load_cached(dir_path, name) for name in list_runs(dir_path)
            if (Path(dir_path) / f"{name}.summary.json").is_file()}


def spec_of(prov: dict) -> str:
    return prov.get("view", {}).get("group") or prov["coordinates"]["spec"]


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
    return all(c.get(k) == b.get(k) for k in
               ("spec", "data", "repo", "family", "methodology", "workloads", "tasks"))


def dir_results(dir_path: str, entries: dict, baselines: dict[str, str]):
    """Read explicit legacy-import snapshots. No scoring occurs during browsing."""
    from .records import cells, read_json
    return {name: cells(saved) if saved is not None else None for name in entries
            for saved in [read_json(Path(dir_path) / f"{name}.summary.json").get("cells")]}


def import_legacy(dir_path: str):
    """Explicit, trusted .pt import. Keep only metadata; never guess precision controls."""
    from ..jobs.contract import write_json
    from ..specs import SPECS
    from ..sweep.score import score_runs

    entries = {}
    for name in list_runs(dir_path):
        if not (Path(dir_path) / f"{name}.pt").is_file():
            continue
        outputs, prov = load_run(dir_path, name)
        entries[name] = ({("__meta__",): outputs.get(("__meta__",), {}),
                          ("perf_rows",): outputs.get(("perf_rows",), [])}, prov)
        del outputs
    baselines = baselines_of(dir_path, entries)
    for name, (out, prov) in entries.items():
        base = baselines.get(spec_of(prov))
        spec = SPECS.get(prov["coordinates"]["spec"].split("@")[0])
        rows = None
        if spec is not None and base and comparable(prov, entries[base][1]):
            scored = score_runs(spec, dir_path, name, None if name == base else base, quiet=True)
            rows = [{"label": c.label, "workload": c.workload, "state": c.state,
                     "error": c.error, "metrics": c.metrics,
                     "median_latency_ms": c.latency_s * 1000 if c.latency_s is not None else None}
                    for c in scored]
        write_json(Path(dir_path) / f"{name}.summary.json", {
            "version": 1, "provenance": prov, "reference": base, "cells": rows,
            "metadata": [[list(k), v] for k, v in out[("__meta__",)].items()],
            "perf_rows": out[("perf_rows",)], "legacy": True,
            "artifact_stat": [(Path(dir_path) / f"{name}.pt").stat().st_size,
                              (Path(dir_path) / f"{name}.pt").stat().st_mtime_ns],
        })


def rollup(states: list[str]) -> str:
    """One verdict over a run's result states: SUPPORTED = everything passes; LIMITED = some
    realization works and some does not; when nothing passes, the worst failure (SILENTLY_WRONG
    over ERROR)."""
    if states and all(s in _PASS for s in states):
        return "SUPPORTED"
    if not any(s in _PASS for s in states):
        for state in ("SILENTLY_WRONG", "DIVERGENT", "JOB_FAILED", "ERROR", "INCOMPATIBLE",
                      "INVALID_REFERENCE", "NO_REFERENCE", "RUNNING", "PENDING", "CANCELLED", "NOT_RUN", "RAN"):
            if state in states:
                return state
        return "UNTESTED"
    return "LIMITED"


def backend_label(prov: dict) -> tuple[str, str]:
    """(display name, setup line), derived from the run's provenance, never hand-written per
    directory: the setup column must describe the runs it summarizes."""
    e, d = prov["engine"], prov["deployment"]
    if prov.get("view"):
        return prov["view"]["backend"], f"{e.get('kind')} · {e.get('mode', '-')} · {d.get('kind', '-')}"
    if e["kind"] == "transformers":
        return "HF-nnsight", "nnsight on vanilla transformers, eager, in-process"
    mode = e.get("mode") or "async"
    place = "served" if d.get("kind") == "serve" else "in-process"
    return f"{e['kind']}-nnsight · {mode} · {place}", f"nnsight on the {e['kind']} {mode} engine, {place}"


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


def _rename_exclusive(source, destination):
    """Linux atomic no-replace rename; refuse cross-filesystem moves rather than risking loss."""
    import ctypes
    import errno
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise OSError(errno.ENOTSUP, "atomic no-replace archive requires renameat2")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))


def archive(run_name: str, dest_dir: str, inbox: str = INBOX) -> str:
    """Move a whole bundle (or legacy artifact + summary), never replace a destination."""
    import fcntl
    from .records import child

    source_root, destination = Path(inbox).resolve(), Path(dest_dir).resolve()
    if run_name not in list_runs(inbox):
        raise FileNotFoundError("unknown inbox run")
    source = child(source_root, run_name)
    if not source.is_dir():
        source = child(source_root, run_name + ".pt")
    if destination == source_root or destination.is_relative_to(source):
        raise ValueError("archive destination must be outside the source run/inbox")
    if (destination / "plan.json").exists():
        raise ValueError("archive into a collection directory, not an existing run")
    destination.mkdir(parents=True, exist_ok=True)
    moves = [(source, destination / source.name)]
    summary = source_root / f"{run_name}.summary.json"
    if source.is_file() and summary.exists():
        if summary.is_symlink():
            raise ValueError("refusing summary symlink")
        moves.append((summary, destination / summary.name))
    # Serialize archive requests from manager processes sharing the destination.
    with (destination / ".manager.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if any(dst.exists() or dst.is_symlink() for _, dst in moves):
            raise FileExistsError("an archived run already has this name")
        completed = []
        try:
            for src, dst in moves:
                _rename_exclusive(src, dst)
                completed.append((src, dst))
        except Exception:
            for src, dst in reversed(completed):
                _rename_exclusive(dst, src)
            raise
    return str(moves[0][1])


def discard(run_name: str, inbox: str = INBOX) -> str:
    """Recoverable discard, preserving all reports/logs/tensors together."""
    from .records import child
    trash = child(Path(inbox).resolve(), ".trash")
    return archive(run_name, str(trash / uuid.uuid4().hex), inbox)


class Collection:
    """Everything the pages need about one directory, derived once per access."""

    def __init__(self, dir_path: str, inbox: str = INBOX):
        self.dir_path = dir_path
        self.inbox = inbox
        self.entries, self.results, self.baselines, self.warnings = {}, {}, {}, []
        self._inbox_collection = None
        self.csrf_token = ""
        from .records import bundle_paths, cells, load_bundle, read_json
        root = Path(dir_path)
        for name in list_runs(dir_path):
            if not (root / f"{name}.pt").is_file():
                continue
            try:
                entry = _load_cached(dir_path, name)
                summary = read_json(root / f"{name}.summary.json")
                result = cells(summary["cells"]) if summary.get("cells") is not None else None
                self.entries[name], self.results[name] = entry, result
                if summary.get("reference") == name:
                    self.baselines[spec_of(self.entries[name][1])] = name
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.warnings.append(f"{name}: legacy summary unavailable; run --import-legacy ({error})")
        if self.entries:
            self.warnings.append("Legacy import snapshots: comparisons lack the new runner's frozen-input identity guarantees.")
        for directory in bundle_paths(root) if dir_path else ():
            relative = directory.relative_to(root).as_posix()
            prefix = directory.name if relative == "." else relative
            try:
                entries, results, bases, warnings = load_bundle(directory, prefix)
                self.entries.update(entries)
                self.results.update(results)
                self.baselines.update(bases)
                self.warnings.extend(warnings)
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.warnings.append(f"{prefix}: invalid run manifest ({error})")

    def by_spec(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for name, (_, prov) in self.entries.items():
            groups.setdefault(spec_of(prov), []).append(name)
        return groups

    def inbox_runs(self) -> list[str]:
        return list_runs(self.inbox)

    def load(self, name: str) -> tuple[dict, dict]:
        """A run from the directory, or the inbox when it only exists there."""
        if name in self.entries:
            return self.entries[name]
        if name.startswith("inbox/"):
            item = name.removeprefix("inbox/")
            if item in self.inbox_collection().entries:
                return self.inbox_collection().entries[item]
        raise FileNotFoundError("unknown run")

    def inbox_collection(self):
        if self._inbox_collection is None:
            self._inbox_collection = Collection(self.inbox, "")
        return self._inbox_collection

    def cells_for(self, name):
        if name in self.results:
            return self.results[name]
        if name.startswith("inbox/"):
            return self.inbox_collection().results.get(name.removeprefix("inbox/"))
        return None

    def backend_names(self):
        from ..jobs.local import discover
        return sorted(set(discover()) | {p.get("view", {}).get("backend") or
                      p["coordinates"]["interface"] for _, p in self.entries.values()})
