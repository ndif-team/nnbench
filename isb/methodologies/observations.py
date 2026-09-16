"""Observation-only reporting of parameter choices resolved inside explicit cells."""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy

_active_call = ContextVar("isb_active_call", default=None)


@contextmanager
def capture_choices(bound_params, record):
    """Collect choices for this call and restore any enclosing collector on exit."""
    token = _active_call.set((bound_params, record))
    try:
        yield
    finally:
        _active_call.reset(token)


def record_resolved(**choices):
    """Record JSON-compatible choices at their resolution site; direct cell calls work normally.

    Only reported entries appear in ``resolved_params``. Bound arguments remain unchanged.
    A case's choices must agree across its warmups, trials, and aggregated input units.
    """
    active = _active_call.get()
    if active is None:
        return
    from ..jobs.contract import canonical, pack

    bound, record = active
    resolved = record.setdefault("resolved_params", {})
    for name, value in choices.items():
        if name not in bound:
            raise ValueError(f"resolved parameter {name!r} is absent from the bound call")
        encoded = canonical(pack(value))
        if name in resolved and canonical(pack(resolved[name])) != encoded:
            raise ValueError(f"resolved parameter {name!r} changed across case invocations")
        resolved[name] = deepcopy(value)
