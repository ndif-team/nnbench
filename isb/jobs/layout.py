"""The backend directory tree: `backends/<system>/<config>/compose.yml` (backends/README.md).

A backend name is the config's path under `backends/`, e.g. `nnsight-vllm/fp32`. A config may nest
under another config whose implementation it shares (`nnsight-vllm/opt/taps`). A config's code is
the nearest `backend.py` / `cells.py` on its own path, searched from the config directory up to the
system directory; a config never reads a sibling's files. A bare system name means its `default`
config. Shared by the host launcher and the in-container entrypoint, so both resolve one tree.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEGMENT = re.compile(r"[a-z0-9][a-z0-9_.-]*\Z")
DEFAULT = "default"


def _segments(name):
    parts = name.split("/") if isinstance(name, str) else []
    if len(parts) < 2 or not all(SEGMENT.fullmatch(p) and p not in {".", ".."} for p in parts):
        raise ValueError(f"backend name {name!r} must be <system>/<config>[/<config>...]")
    return parts


def discover(root=ROOT):
    """Every config: a directory below a system directory that holds a compose.yml."""
    base = (Path(root) / "backends").resolve()
    names = []
    for compose in base.rglob("compose.yml"):
        relative = compose.parent.relative_to(base)
        if any(p.is_symlink() for p in [compose, *compose.parents] if base in p.parents):
            continue
        try:
            names.append("/".join(_segments(relative.as_posix())))
        except ValueError:
            continue
    return sorted(names)


def canonical(name, root=ROOT):
    """The full config name; a bare system name selects `<system>/default`."""
    if isinstance(name, str) and "/" not in name:
        name = f"{name}/{DEFAULT}"
    available = discover(root)
    if name not in available:
        raise ValueError(f"unknown backend {name!r}; available: {', '.join(available)}")
    return name


def config_dir(name, root=ROOT):
    return Path(root).joinpath("backends", *_segments(name))


def nearest(name, filename, root=ROOT):
    """The nearest `filename` from the config directory up to its system directory, or None."""
    parts = _segments(name)
    for depth in range(len(parts), 0, -1):
        path = Path(root).joinpath("backends", *parts[:depth], filename)
        if path.is_file():
            return path
    return None


def module_name(path, root=ROOT):
    """Import name of a file under backends/ (directory names may contain '-', which
    importlib accepts; code inside imports its parents relatively)."""
    relative = Path(path).relative_to(Path(root)).with_suffix("")
    return ".".join(relative.parts)
