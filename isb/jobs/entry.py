"""In-container entrypoint for the bundled backends: `python3 -m isb.jobs.entry`.

Resolves `ISB_BACKEND` (`<system>/<config>`) to the nearest `backend.py` and `cells.py` on the
config's path (isb/jobs/layout.py), imports them as `backends.<system>...` modules, and runs the
shared worker with that `create_backend`. A third-party backend may use any entrypoint instead.
"""
from __future__ import annotations

import importlib
import os

from . import layout


def load(name, root=layout.ROOT):
    """(create_backend, cells module path or None) for one config."""
    backend = layout.nearest(name, "backend.py", root)
    if backend is None:
        raise ValueError(f"{name}: no backend.py on the config's path")
    cells = layout.nearest(name, "cells.py", root)

    def create_backend(spec):
        # Imported inside the worker's failure path, so a dependency error is recorded as a result.
        module = importlib.import_module(layout.module_name(backend, root))
        if cells is not None:
            importlib.import_module(layout.module_name(cells, root))   # registers the config's cells
        return module.create_backend(spec)

    return create_backend, cells


def main():
    from .worker import main as worker

    create_backend, _ = load(os.environ.get("ISB_BACKEND", ""))
    worker(create_backend)


if __name__ == "__main__":   # vLLM EngineCore uses spawn
    main()
