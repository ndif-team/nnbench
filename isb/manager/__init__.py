"""Run manager: render saved reports; inbox; archive complete runs without overwriting.

Layout: records.py (JSON job adapter), model.py (Collection, explicit run-file import, safe moves),
htmlkit.py (every tag and format rule), figure.py (the operating-point scatter), pages.py
(page composition + ROUTES, the single registry the server, the export, and the tests share).
"""
from __future__ import annotations

from .figure import perf_svg  # noqa: F401
from .model import (Collection, archive, backend_label, discard, list_runs, perf_points,  # noqa: F401
                    probe_results, rollup, trace_source)
from .pages import METHOD_INTROS, ROUTES, dispatch, enumerate_paths, export_html  # noqa: F401
