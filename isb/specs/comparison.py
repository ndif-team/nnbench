"""Cross-system comparison specs (design.md §12.14).

Task parameters carry intervention semantics only (layer, direction, strength, component,
positions). Steering strength is `alpha` times the stream norm, averaged over the forward's tokens
(`mean_norm`, the default) or per token (`token_norm`); patching writes every position (`all`, the
default) or the last one. Each
system's cell picks that system's documented realization, so the same row is comparable across
nnsight and foreign systems. Qwen2.5-7B-Instruct is a Llama-shaped decoder (family "llama"), 28
layers; the MIB IOI pairs are verified length-matched under the Qwen2.5 tokenizer. The rows were
developed on Qwen2.5-1.5B-Instruct (same layer count and tokenizer; tied embeddings, which 7B does
not have); the Delta run uses 7B because it is in the cluster's model cache.
"""
from ..data import DataRef
from ..sweep.spec import BaselineSpec, CellConfig, EffectSpec, ExecutionRegime

_MODEL = "Qwen/Qwen2.5-7B-Instruct"
_PROMPTS = DataRef("counterfact", 32)
_PAIRS = DataRef("mib/ioi", 16)
_STEER = {"target": " Rome", "alpha": 6.0}

cmp_logit_lens = CellConfig(
    name="cmp_logit_lens",
    methodology="logit_lens", family="llama", repo=_MODEL,
    regimes=[ExecutionRegime("interactive", _PROMPTS)],
    tasks=[({"layers": "all"}, "every layer, last token")],
    baseline=BaselineSpec(params={"layers": [-1]}),
)

cmp_steering = CellConfig(
    name="cmp_steering",
    methodology="steering", family="llama", repo=_MODEL,
    regimes=[ExecutionRegime("interactive", _PROMPTS)],
    tasks=[
        ({**_STEER, "layer": 7}, "add toward ' Rome' at layer 7"),
        ({**_STEER, "layer": 14}, "add toward ' Rome' at layer 14"),
        ({**_STEER, "layer": 7, "scale_by": "token_norm"},
         "add toward ' Rome' at layer 7, scaled by each token's norm"),
        ({**_STEER, "layer": 14, "scale_by": "token_norm"},
         "add toward ' Rome' at layer 14, scaled by each token's norm"),
        # alpha=6 saturates (HF puts p=1.0000 on ' Rome' for every prompt), so those rows cannot
        # tell a correct strength from a wrong one; these weaker rows leave the output graded.
        ({**_STEER, "layer": 14, "alpha": 0.1}, "add toward ' Rome' at layer 14, alpha 0.1"),
        ({**_STEER, "layer": 14, "alpha": 0.5}, "add toward ' Rome' at layer 14, alpha 0.5"),
        ({**_STEER, "layer": 14, "alpha": 0.1, "scale_by": "token_norm"},
         "add toward ' Rome' at layer 14, scaled by each token's norm, alpha 0.1"),
        ({**_STEER, "layer": 14, "alpha": 0.5, "scale_by": "token_norm"},
         "add toward ' Rome' at layer 14, scaled by each token's norm, alpha 0.5"),
    ],
    baseline=BaselineSpec(params={**_STEER, "layer": 14, "alpha": 0.0}),
    effect=EffectSpec(baseline_params={**_STEER, "layer": 14, "alpha": 0.0},
                      perturbed_params={**_STEER, "layer": 14}),
)

cmp_gen_steering = CellConfig(
    name="cmp_gen_steering",
    methodology="gen_steering", family="llama", repo=_MODEL,
    regimes=[ExecutionRegime("generation", DataRef("counterfact", 16), new_tokens=8)],
    tasks=[
        ({**_STEER, "layer": 14}, "add toward ' Rome' at layer 14, every step"),
        ({**_STEER, "layer": 14, "scale_by": "token_norm"},
         "add toward ' Rome' at layer 14, every step, scaled by each token's norm"),
        ({**_STEER, "layer": 14, "alpha": 0.5}, "add toward ' Rome' at layer 14, every step, alpha 0.5"),
        ({**_STEER, "layer": 14, "alpha": 0.5, "scale_by": "token_norm"},
         "add toward ' Rome' at layer 14, every step, scaled by each token's norm, alpha 0.5"),
    ],
    baseline=BaselineSpec(params={**_STEER, "layer": 14, "alpha": 0.0}),
    effect=EffectSpec(baseline_params={**_STEER, "layer": 14, "alpha": 0.0},
                      perturbed_params={**_STEER, "layer": 14}),
)

cmp_activation_patching = CellConfig(
    name="cmp_activation_patching",
    methodology="activation_patching", family="llama", repo=_MODEL,
    regimes=[ExecutionRegime("interactive", _PAIRS, aggregate=True)],
    tasks=[
        ({"layer": 7}, "clean residual into corrupt run at layer 7"),
        ({"layer": 21}, "clean residual into corrupt run at layer 21"),
        ({"layer": 7, "positions": "last"}, "clean last-token residual into corrupt run at layer 7"),
        ({"layer": 21, "positions": "last"}, "clean last-token residual into corrupt run at layer 21"),
    ],
    baseline=BaselineSpec(params={"patch": False}),
    effect=EffectSpec(baseline_params={"patch": False},
                      perturbed_params={"layer": 21, "patch": True}),
)

cmp_ablation = CellConfig(
    name="cmp_ablation",
    methodology="ablation", family="llama", repo=_MODEL,
    regimes=[ExecutionRegime("interactive", _PROMPTS)],
    tasks=[
        ({"layer": 1, "target": "mlp"}, "zero the MLP output at layer 1"),
        ({"layer": 1, "target": "attn"}, "zero the attention output at layer 1"),
    ],
    baseline=BaselineSpec(params={"layer": 1, "target": "none"}),
    effect=EffectSpec(baseline_params={"layer": 1, "target": "none"},
                      perturbed_params={"layer": 1, "target": "mlp"}),
)

COMPARISON_SPECS = (cmp_logit_lens, cmp_steering, cmp_gen_steering, cmp_activation_patching,
                    cmp_ablation)
