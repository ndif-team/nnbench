"""No-GPU unit tests for the Resolver + predict (design.md §11.5, §11.6).

Tests behavior on VARIED, non-standard module trees — the resolver must work for a
family that names blocks `decoder_blocks` and the unembed `output_projection`, with
ZERO code changes (profile-only). This is the structural enforcement of the
"don't hardcode to GPT-2/Llama" invariant.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isb.spec import Access, AppState, Selector, Workload  # noqa: E402
from isb.resolve import (  # noqa: E402
    GPT2,
    HF,
    VLLM_ASYNC,
    Binding,
    FamilyProfile,
    Resolver,
    Unsupported,
    predict,
)


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _fake_gpt2(n=6):
    blocks = [_Obj(attn=_Obj(c_proj=_Obj()), mlp=_Obj(act=_Obj())) for _ in range(n)]
    return _Obj(
        transformer=_Obj(h=blocks, ln_f=_Obj()),
        lm_head=_Obj(),
        config=_Obj(n_layer=n, n_head=12, n_inner=3072),
    )


# A deliberately UNCONVENTIONAL architecture: nothing named like GPT-2/Llama.
WEIRD = FamilyProfile(
    type="causal_lm",
    family="weird",
    paths={
        "block": "trunk.decoder_blocks.{i}",
        "attn": "trunk.decoder_blocks.{i}.mixer",
        "mlp": "trunk.decoder_blocks.{i}.feedforward",
        "final_norm": "trunk.final_ln",
        "unembed": "output_projection",
    },
    output_index={"block": None, "attn": None, "mlp": None},
    dims={"n_heads": "num_heads", "head_dim": "head_size", "n_kv_heads": "num_heads", "ffn": "ffn_dim"},
    caps={"block.output", "mlp.output", "attn.output", "read", "write_replace"},
    n_layers_attr="depth",
)


def _fake_weird(n=4):
    blocks = [_Obj(mixer=_Obj(), feedforward=_Obj()) for _ in range(n)]
    return _Obj(
        trunk=_Obj(decoder_blocks=blocks, final_ln=_Obj()),
        output_projection=_Obj(),
        config=_Obj(depth=n, num_heads=8, head_size=64, ffn_dim=2048),
    )


def test_gpt2_block_output_resolves_all_layers():
    model = _fake_gpt2(6)
    R = Resolver(GPT2, model)
    bindings = R.resolve(Selector("block.output", scope="all"))
    assert len(bindings) == 6
    for i, b in enumerate(bindings):
        assert b.module is model.transformer.h[i]   # identity, not a re-derived path
        assert b.output_index == 0                   # GPT-2 block output IS a tuple
        assert b.access == Access.READ
        assert b.site_id == f"L{i}.block.output"


def test_scope_list_and_negative_index():
    model = _fake_gpt2(6)
    R = Resolver(GPT2, model)
    bindings = R.resolve(Selector("block.output", scope=[0, 2, -1]))
    got = [b.module for b in bindings]
    assert got == [model.transformer.h[0], model.transformer.h[2], model.transformer.h[5]]


def test_resolve_one_singletons():
    model = _fake_gpt2()
    R = Resolver(GPT2, model)
    assert R.resolve_one("final_norm").module is model.transformer.ln_f
    assert R.resolve_one("unembed").module is model.lm_head


def test_weird_family_is_profile_only_no_hardcoding():
    """The SAME resolver code binds a totally non-standard tree."""
    model = _fake_weird(4)
    R = Resolver(WEIRD, model)
    bindings = R.resolve(Selector("block.output", scope="all"))
    assert len(bindings) == 4
    assert bindings[2].module is model.trunk.decoder_blocks[2]
    assert R.resolve_one("unembed").module is model.output_projection
    assert R.resolve_one("final_norm").module is model.trunk.final_ln


def test_head_value_binds_oproj_input_side():
    """Per-head value taps the o_proj/c_proj INPUT, not its output (§11.3)."""
    model = _fake_gpt2()
    R = Resolver(GPT2, model)
    b = R.resolve(Selector("attn.head_value", scope=[0], head=3))[0]
    assert b.module is model.transformer.h[0].attn.c_proj   # the o_proj/c_proj module
    assert b.side == "input"                                  # read its INPUT, not output
    assert b.index == ("head", 3, None)                      # head_dim derived at runtime
    assert b.site_id == "L0.attn.head_value.h3"


def test_ffn_dim_derives_for_gpt2():
    """GPT-2 n_inner defaults to None -> ffn derives 4*hidden from the profile."""
    model = _fake_gpt2()
    model.config.n_inner = None
    model.config.n_embd = 768
    R = Resolver(GPT2, model)
    assert R._dim("ffn") == 4 * 768


def test_predict_supported_and_unsupported():
    wl_lens = Workload(
        id="x", motif="logit_lens",
        selectors=[Selector("block.output", access="read")],
    )
    assert predict(wl_lens, GPT2, HF) == AppState.SUPPORTED
    assert predict(wl_lens, GPT2, VLLM_ASYNC) == AppState.SUPPORTED

    wl_attn = Workload(
        id="y", motif="attention_pattern",
        selectors=[Selector("attn.weights", access="read")],
    )
    # attn.weights can't exist under vLLM flash-attn -> frontier marker
    assert predict(wl_attn, GPT2, VLLM_ASYNC) == AppState.UNSUPPORTED_BY_CONSTRUCTION
    assert predict(wl_attn, GPT2, HF) == AppState.SUPPORTED


def test_unsupported_target_raises():
    """A family lacking a target raises Unsupported on resolve (-> predicted UNSUPPORTED)."""
    model = _fake_weird()
    R = Resolver(WEIRD, model)  # caps lack attn.weights
    try:
        R.resolve(Selector("attn.weights"))
        raise AssertionError("expected Unsupported")
    except Unsupported:
        pass


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
