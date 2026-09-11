"""Case requirement descriptions for explicit cell programs.

The executor supplies parameters after signature binding and default application. Hooks describe
branches using those values, so defaults remain in the cell functions. Input roles retain the
case's input contract, including paired feeds whose control branch executes one side only.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import replace
from typing import Any

from ..protocol import PROTOCOLS, InterventionSpec


CASE_DESCRIPTIONS: dict[str, tuple[Callable[..., InterventionSpec | None], InterventionSpec | None]] = {}


def case_description(methodology: str, *, override: bool = False,
                     template: InterventionSpec | None = None):
    """Register ``fn(params, *, template, family) -> InterventionSpec | None``.

    A custom methodology can supply a hook beside its cell implementation. A methodology without
    a hook retains its template and records ``protocol_scope='template'`` in case metadata. A hook
    may return None when it cannot specialize a case. Replacing a hook requires ``override=True``.
    Supplying ``template`` restricts a hook to that descriptor, preserving custom frozen templates.
    """
    def register(fn):
        if methodology in CASE_DESCRIPTIONS and not override:
            raise ValueError(f"case description already registered for {methodology!r}")
        CASE_DESCRIPTIONS[methodology] = (fn, template)
        return fn

    return register


def describe_case(methodology: str, params: Mapping[str, Any], *,
                  template: InterventionSpec | None, family: str) -> dict[str, Any]:
    """Describe a bound call, with explicit scope when only a template is available.

    An absent template stays absent. Hooks receive an independent parameter snapshot so metadata
    extensions cannot mutate the parameters subsequently passed to the cell.
    """
    hook, expected_template = CASE_DESCRIPTIONS.get(methodology, (None, None))
    if (template is None or hook is None
            or expected_template is not None and template != expected_template):
        return {"protocol": template.coordinate() if template is not None else None,
                "protocol_scope": "template"}
    descriptor = hook(deepcopy(dict(params)), template=template, family=family)
    if descriptor is None:
        return {"protocol": template.coordinate(), "protocol_scope": "template"}
    if not isinstance(descriptor, InterventionSpec):
        raise TypeError("case description hooks must return an InterventionSpec or None")
    return {"protocol": descriptor.coordinate(), "protocol_scope": "case"}


def _without_write(template: InterventionSpec, *, components: tuple[str, ...]) -> InterventionSpec:
    return replace(template, operations=tuple(set(template.operations) - {"write"}),
                   capabilities=tuple(set(template.capabilities) - {"paired_forward"}),
                   extensions=tuple(set(template.extensions) - {"decode_step_write"}),
                   mechanisms=(), write_components=(), components=components)


def _without_grad(template: InterventionSpec) -> InterventionSpec:
    return replace(template, operations=tuple(set(template.operations) - {"grad"}),
                   capabilities=tuple(set(template.capabilities) - {"grad"}),
                   extensions=tuple(set(template.extensions) -
                                    {"activation_grad", "jacobian_export"}))


@case_description("steering", template=PROTOCOLS["steering"])
def _steering(params, *, template, family):
    if "alpha" not in params:
        return None
    if params["alpha"] == 0:
        return _without_write(template, components=("block_output", "lm_head"))
    return template


@case_description("gen_steering", template=PROTOCOLS["gen_steering"])
def _gen_steering(params, *, template, family):
    if "alpha" not in params:
        return None
    if params["alpha"] == 0:
        return _without_write(template, components=("lm_head",))
    return template


@case_description("activation_patching", template=PROTOCOLS["activation_patching"])
def _activation_patching(params, *, template, family):
    if "patch" not in params:
        return None
    if not params["patch"]:
        return _without_write(template, components=("block_output", "lm_head"))
    return template


@case_description("gen_patching", template=PROTOCOLS["gen_patching"])
def _gen_patching(params, *, template, family):
    if "patch" not in params:
        return None
    if not params["patch"]:
        return _without_write(template, components=("lm_head",))
    return template


@case_description("ablation", template=PROTOCOLS["ablation"])
def _ablation(params, *, template, family):
    if "target" not in params:
        return None
    target = params["target"]
    if target == "none":
        return _without_write(template, components=("block_output", "lm_head"))
    component = {"attn": "attention_output", "mlp": "mlp_output"}.get(target)
    if component is None:
        # A hybrid mixer needs model/layer binding to identify its concrete write component.
        return replace(template, write_components=None)
    return replace(template, components=(component, "block_output", "lm_head"),
                   write_components=(component,) if template.write_components is not None else None)


@case_description("das", template=PROTOCOLS["das"])
def _das(params, *, template, family):
    if "train" not in params:
        return None
    return template if params["train"] else _without_grad(template)


@case_description("attribution_patching", template=PROTOCOLS["attribution_patching"])
@case_description("jacobian_collect", template=PROTOCOLS["jacobian_collect"])
def _grad(params, *, template, family):
    if "grad" not in params:
        return None
    return template if params["grad"] else _without_grad(template)


@case_description("jacobian_lens", template=PROTOCOLS["jacobian_lens"])
def _jacobian_lens(params, *, template, family):
    if "transport" not in params:
        return None
    if params["transport"] is None:
        return replace(template, extensions=tuple(set(template.extensions) - {"linear_transport"}))
    return template


@case_description("logit_lens", template=PROTOCOLS["logit_lens"])
@case_description("attention_pattern", template=PROTOCOLS["attention_pattern"])
def _constant(params, *, template, family):
    return template
