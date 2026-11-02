"""In-container entrypoint for the bundled backends: `python3 -m isb.jobs.entry`.

Resolves `ISB_BACKEND` (`<system>/<config>`) to its system's `backend.py` and `cells.py`
(isb/jobs/layout.py), imports them as `backends.<system>.*` modules, and runs the shared worker
with that `create_backend`. A third-party backend may use any entrypoint instead.

`ISB_RANKS=N` (N > 1) runs the job as N torchrun ranks, one per reserved GPU, for a library whose
multi-GPU mode needs one process per GPU (transformers tensor parallelism). Every rank runs the
whole job; rank 0 alone publishes it (worker.py). Each rank runs 8 OpenMP threads, where torchrun
would set 1; a single-process job keeps torch's default.
"""
from __future__ import annotations

import importlib
import os

from . import layout


def load(name, root=layout.ROOT):
    """(create_backend, cells module path or None) of one backend: its system's backend.py and,
    when the system registers its own cells, cells.py."""
    directory = layout.system_dir(name, root)
    backend = directory / "backend.py"
    if not backend.is_file():
        raise ValueError(f"{name}: {backend} is missing")
    cells = directory / "cells.py"
    cells = cells if cells.is_file() else None

    def create_backend(spec):
        # Imported inside the worker's failure path, so a dependency error is recorded as a result.
        module = importlib.import_module(layout.module_name(backend, root))
        if cells is not None:
            importlib.import_module(layout.module_name(cells, root))   # registers the system's cells
        return module.create_backend(spec)

    return create_backend, cells


def main():
    ranks = int(os.environ.get("ISB_RANKS", "1"))
    if ranks > 1 and "RANK" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = "8"
        os.execvp("torchrun", ["torchrun", "--standalone", f"--nproc_per_node={ranks}",
                               "-m", "isb.jobs.entry"])
    from .worker import main as worker

    create_backend, _ = load(os.environ.get("ISB_BACKEND", ""))
    worker(create_backend)


if __name__ == "__main__":   # vLLM EngineCore uses spawn
    main()
