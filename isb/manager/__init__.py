"""Run manager (design.md §12.12): render one collection directory; inbox; archive = file move.

Layout: model.py (the directory model: Collection, baselines, the one scoring path, caches),
htmlkit.py (every tag and format rule), figure.py (the operating-point scatter), pages.py
(page composition + ROUTES, the single registry the server, the export, and the tests share).

This module re-exports the public API; the render_* wrappers keep the historical
(dir_path, inbox) signatures over the Collection-based pages.
"""
from __future__ import annotations

from ..runfile import INBOX
from . import pages as _pages
from .figure import perf_svg  # noqa: F401
from .model import (Collection, archive, backend_label, baselines_of, comparable,  # noqa: F401
                    dir_results, discard, executed, list_runs, load_dir, perf_points,
                    probe_results, rollup, trace_source)
from .pages import METHOD_INTROS, ROUTES, dispatch, enumerate_paths, export_html  # noqa: F401


def _col(dir_path: str, inbox: str = INBOX) -> Collection:
    return Collection(dir_path, inbox)


def render_overview(dir_path: str, inbox: str = INBOX) -> str:
    return _pages.overview(_col(dir_path, inbox))


def render_micro(dir_path: str, inbox: str = INBOX) -> str:
    return _pages.micro(_col(dir_path, inbox))


def render_method(dir_path: str, name: str, inbox: str = INBOX) -> str:
    return _pages.method(_col(dir_path, inbox), name)


def render_spec(dir_path: str, spec_name: str, inbox: str = INBOX) -> str:
    return _pages.spec(_col(dir_path, inbox), spec_name)


def render_run(dir_path: str, name: str, inbox: str = INBOX) -> str:
    return _pages.run(_col(dir_path, inbox), name)


def render_runs(dir_path: str, inbox: str = INBOX) -> str:
    return _pages.runs(_col(dir_path, inbox))


def render_inbox(dir_path: str, inbox: str = INBOX) -> str:
    return _pages.inbox(_col(dir_path, inbox))


def render_data_index() -> str:
    return _pages.data_index(None)


def render_data_source(name: str) -> str:
    return _pages.data_source(None, name)


def render_data_item(name: str, idx: int) -> str:
    return _pages.data_item(None, name, idx)


def render_models() -> str:
    return _pages.models(None)


def render_model(family: str) -> str:
    return _pages.model_page(None, family)


def render_backends() -> str:
    return _pages.backends(None)


def render_backend(name: str, dir_path: str | None = None) -> str:
    return _pages.backend(_col(dir_path or ""), name)   # "" = an empty collection


def render_doc(title: str, relpath: str) -> str:
    return _pages.doc(None, title, relpath)
