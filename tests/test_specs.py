"""Spec registry gating (isb/specs/__init__.py) — no GPU.

The 14B Qwen TP/PP-equivalence specs must stay resolvable by exact name (`--spec logit_lens_qwen`)
but must NOT be swept by the documented default `bench.py --spec all` — otherwise an `--spec all`
run a user expects to be gpt2-scale silently loads a 14B model (HF fp32 ~56GB) and OOMs/downloads.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isb.specs import SPECS, default_specs  # noqa: E402

_QWEN = ("logit_lens_qwen", "steering_qwen", "ablation_qwen",
         "activation_patching_qwen", "gen_steering_qwen")


def test_large_qwen_specs_resolvable_by_name_but_excluded_from_all():
    names = default_specs()
    for n in _QWEN:
        assert n in SPECS, f"{n} must stay resolvable by exact --spec name"
        assert n not in names, f"{n} is a 14B parallel-only spec; --spec all must not sweep it"


def test_default_all_still_includes_the_gpt2_corpus():
    names = default_specs()
    for n in ("logit_lens_gpt2", "steering_gpt2", "ablation_gpt2"):
        assert n in names, f"{n} is gpt2-scale and belongs in the default --spec all sweep"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
