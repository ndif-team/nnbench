"""Spec definitions do not import an inference stack."""
from .spec import BaselineSpec, CellConfig, EffectSpec, Workload

__all__ = [
    "CellConfig",
    "Workload",
    "BaselineSpec",
    "EffectSpec",
    "compute_effect_size",
    "execute_run",
    "score_runs",
]


def __getattr__(name):
    from importlib import import_module

    modules = {"compute_effect_size": "guards", "execute_run": "execute", "score_runs": "score"}
    if name not in modules:
        raise AttributeError(name)
    return getattr(import_module(f"{__name__}.{modules[name]}"), name)
