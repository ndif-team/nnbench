"""Cell-registry tests (design.md §12.1) — no GPU; needs torch only."""
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import isb.methodologies  # noqa: F401,E402  (registers cells)
from isb.methodologies.registry import CELLS, families_for, get_cell  # noqa: E402


def test_cells_registered_by_method_family_backend():
    assert ("logit_lens", "gpt2", "hf") in CELLS
    assert ("logit_lens", "gpt2", "vllm_async") in CELLS
    assert get_cell("logit_lens", "gpt2", "hf") is not None
    assert get_cell("logit_lens", "gpt2", "nope") is None       # missing cell -> None
    assert "gpt2" in families_for("logit_lens", "hf")


def test_cell_accepts_variances():
    fn = get_cell("logit_lens", "gpt2", "hf")
    params = inspect.signature(fn).parameters
    # prompts (multiple) + observe-knob (layers) + formulation variant (unembed)
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
