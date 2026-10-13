"""Workloads (design.md §12.16): one per methodology, naming no model and no system.

`instantiate(workload, model)` (isb/sweep/spec.py) resolves one on a model from isb/models.py; a
resolved case is named `<workload>/<model>` (plus `@<data>` when the data is rebound).
"""
from .ablation import ablation
from .activation_patching import activation_patching
from .attention_pattern import attention_pattern
from .attribution_patching import attribution_patching
from .das import das
from .gen_patching import gen_patching
from .gen_steering import gen_steering
from .jacobian_collect import jacobian_collect
from .jacobian_lens import jacobian_lens
from .logit_lens import logit_lens
from .steering import steering

WORKLOADS = {w.name: w for w in (
    logit_lens, jacobian_lens, steering, gen_steering, activation_patching, gen_patching, ablation,
    attention_pattern, attribution_patching, das, jacobian_collect,
)}


def case(name: str):
    """The resolved CellConfig a case name `<workload>/<model>` denotes with its default data, or
    None for any other name (a rebound `@<data>` case, or a saved run's historical spec name)."""
    from ..models import MODELS
    from ..sweep.spec import NotApplicable, instantiate

    workload, _, model = name.partition("/")
    if workload not in WORKLOADS or model not in MODELS:
        return None
    try:
        return instantiate(WORKLOADS[workload], model)[0]
    except NotApplicable:
        return None


__all__ = ["WORKLOADS", "case"]
