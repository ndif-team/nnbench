"""Cell-registry tests (design.md §12.1) — no GPU; needs torch only."""
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import isb.methodologies  # noqa: F401,E402  (registers cells)
from isb.methodologies.registry import CELLS, families_for, get_cell  # noqa: E402


def test_cells_registered_by_method_family_backend():
    # logit_lens is family-generic (§12.8): registered once under family="*", resolved for any
    # profiled family with that family's ModelProfile bound at lookup
    assert ("logit_lens", "*", "hf") in CELLS
    assert ("logit_lens", "*", "vllm_async") in CELLS
    assert get_cell("logit_lens", "gpt2", "hf") is not None
    assert get_cell("logit_lens", "gpt2", "nope") is None       # missing backend -> None
    assert "gpt2" in families_for("logit_lens", "hf")


def test_generic_cell_serves_every_profiled_family_and_only_those():
    # one "*" registration serves each family that has a ModelProfile; an unprofiled family stays
    # None (no silent guess about an unknown module tree)
    fams = families_for("logit_lens", "hf")
    assert {"gpt2", "llama", "nemotron"} <= set(fams)
    for fam in ("gpt2", "llama", "nemotron"):
        assert get_cell("logit_lens", fam, "hf") is not None
        assert get_cell("logit_lens", fam, "vllm_async") is not None
    assert get_cell("logit_lens", "no-such-family", "hf") is None


def test_explicit_family_registration_overrides_generic():
    # the §12.8 override hatch: an exact (methodology, family, backend) cell beats the "*" cell
    def special(be, model, prompts, **params):
        return "explicit"
    key = ("logit_lens", "gpt2", "hf")
    prev = CELLS.get(key)
    CELLS[key] = special
    try:
        assert get_cell("logit_lens", "gpt2", "hf") is special
        # other families still resolve through the generic cell
        assert get_cell("logit_lens", "llama", "hf") is not special
    finally:
        if prev is None:
            del CELLS[key]
        else:
            CELLS[key] = prev


def test_cell_accepts_variances():
    fn = get_cell("logit_lens", "gpt2", "hf")
    # the profile-bound wrapper carries the generic cell via functools.wraps, so signature
    # inspection sees the real params (m + prompts + the variance knobs)
    params = inspect.signature(fn).parameters
    assert "prompts" in params
    assert "layers" in params and params["layers"].default == "all"
    assert "unembed" in params


def test_no_resolver_or_predict_imports():
    """The abstraction is gone: these modules must not exist anymore."""
    import importlib

    for gone in ("isb.resolve", "isb.spec"):
        try:
            importlib.import_module(gone)
            raise AssertionError(f"{gone} should have been deleted")
        except ModuleNotFoundError:
            pass


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
