"""ModelProfile tests (isb/profiles.py, design.md §12.8) — no GPU; torch only.

The profile must resolve module paths from the model object at runtime, for ANY naming
convention: a profile row with non-standard names (decoder_blocks / final_rms /
output_projection) must work exactly like the gpt2/llama rows, and the generic-cell
binding in the registry must compose with it. This pins "family is data": adding an
architecture is one ModelProfile row, and nothing anywhere assumes transformer.h.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import isb.methodologies  # noqa: F401,E402  (registers cells)
from isb.methodologies.registry import CELLS, get_cell  # noqa: E402
from isb.profiles import PROFILES, ModelProfile  # noqa: E402


def _weird_model():
    """A model whose tree matches NO known convention; profile resolution must still work."""
    blocks = [SimpleNamespace(output=f"block{i}") for i in range(3)]
    return SimpleNamespace(
        core=SimpleNamespace(
            decoder_blocks=SimpleNamespace(stack=blocks),
            final_rms="the-norm",
        ),
        output_projection="the-head",
    )


WEIRD = ModelProfile(
    "weird", "core.decoder_blocks.stack", "core.final_rms",
    head_path="output_projection", residual_denotation={"hf": "plain", "vllm": "fused"},
)


def test_profile_resolves_nonstandard_paths():
    model = _weird_model()
    assert len(WEIRD.blocks(model)) == 3
    assert WEIRD.blocks(model)[1].output == "block1"
    assert WEIRD.norm(model) == "the-norm"
    assert WEIRD.head(model) == "the-head"


def test_profile_wrong_path_raises_loudly():
    # a stale path must AttributeError, never silently return a default
    model = _weird_model()
    bad = ModelProfile("bad", "core.no_such_attr", "core.final_rms")
    try:
        bad.blocks(model)
        raise AssertionError("expected AttributeError on a wrong module path")
    except AttributeError:
        pass


def test_generic_cell_binds_a_new_profile_row():
    # register a throwaway generic cell + a weird-family profile; the registry must bind them
    def probe(be, model, m, prompts, **params):
        return (m.family, m.blocks(model)[0].output, prompts)

    CELLS[("probe_meth", "*", "hf")] = probe
    PROFILES["weird"] = WEIRD
    try:
        fn = get_cell("probe_meth", "weird", "hf")
        assert fn is not None
        fam, first_block, prompts = fn(None, _weird_model(), ["p"])
        assert (fam, first_block, prompts) == ("weird", "block0", ["p"])
        # vllm variant routing composes with the generic fallback
        CELLS[("probe_meth", "*", "vllm_async")] = probe
        assert get_cell("probe_meth", "weird", "vllm_serve") is not None
    finally:
        del CELLS[("probe_meth", "*", "hf")]
        CELLS.pop(("probe_meth", "*", "vllm_async"), None)
        del PROFILES["weird"]


def test_for_backend_prefixes_every_path_on_vllm_only():
    # a wrapper architecture mounts the same text tree behind a prefix on vLLM (qwen3_5's
    # language_model); the engine view must prefix ALL paths there and be identity on hf
    prof = ModelProfile("wrapped", "core.blocks", "core.final_rms",
                        head_path="proj_out", vllm_prefix="text_side")
    assert prof.for_backend("hf") is prof                        # identity, not a copy
    v = prof.for_backend("vllm_async")
    assert (v.blocks_path, v.norm_path, v.head_path) == (
        "text_side.core.blocks", "text_side.core.final_rms", "text_side.proj_out")
    assert v.for_backend("vllm_serve") is v                      # idempotent (prefix consumed)
    # no prefix declared -> identity on every backend
    assert WEIRD.for_backend("vllm_async") is WEIRD


def test_builtin_rows_denote_the_documented_trees():
    # the three shipped rows carry the module trees the per-family cells used to hardcode
    assert PROFILES["gpt2"].blocks_path == "transformer.h"
    assert PROFILES["gpt2"].default_residual("vllm_async") == "plain"
    assert PROFILES["llama"].norm_path == "model.norm"
    assert PROFILES["llama"].default_residual("vllm_async") == "fused"
    assert PROFILES["llama"].default_residual("hf") == "plain"
    # an engine kind with no declared denotation must fail loudly, not default to plain
    try:
        PROFILES["llama"].default_residual("sglang")
        raise AssertionError("undeclared engine kind must raise")
    except KeyError:
        pass
    assert PROFILES["nemotron"].norm_path == "model.norm_f"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
