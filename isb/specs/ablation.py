"""ablation specs, every model.

Baseline = target='none' (no knockout -> pure forward + readout); effect-size = TV(none, attn) on
the control.
"""
from ..data import DataRef
from ..sweep.spec import BaselineSpec, CellConfig, EffectSpec, ExecutionRegime
from ._models import BF16, GPT2, NEMOTRON_4B, NEMOTRON_30B, NEMOTRON_VLLM, QWEN25_7B, QWEN25_14B

# data is a named, swappable source — see logit_lens.py. Default: clean prompts from the
# MIB circuit-track IOI snapshot (data/mib/README.md) — component knockout over the IOI
# task is the MIB ablation setting. The template bank stays available via --data factual.
PROBE = DataRef("mib/ioi_prompts", 100)
BATCHED = DataRef("mib/ioi_prompts", 16)
# Large models: CounterFact prompts, sized for one multi-GPU run.
LARGE = DataRef("counterfact", 32)

# --- smoke (gpt2) -------------------------------------------------------------------------------

ablation_gpt2 = CellConfig(
    name="ablation_gpt2",
    methodology="ablation", family="gpt2", repo=GPT2,
    regimes=[ExecutionRegime("interactive", PROBE), ExecutionRegime("batched", BATCHED)],
    tasks=[
        ({"layer": 6, "target": "mlp"}, "target=mlp"),
        ({"layer": 6, "target": "attn"}, "target=attn"),
    ],
    baseline=BaselineSpec(params={"layer": 6, "target": "none"}),
    effect=EffectSpec(
        baseline_params={"layer": 6, "target": "none"},
        perturbed_params={"layer": 6, "target": "attn"},
    ),
)

# --- parallel (Qwen2.5-14B, GT2) ----------------------------------------------------------------

ablation_qwen = CellConfig(
    name="ablation_qwen",
    methodology="ablation", family="llama", repo=QWEN25_14B,
    regimes=[ExecutionRegime("interactive", LARGE)],
    tasks=[
        ({"layer": 16, "target": "mlp", "residual": "plain"}, "target=mlp"),
        ({"layer": 16, "target": "attn", "residual": "plain"}, "target=attn"),
    ],
    baseline=BaselineSpec(params={"layer": 16, "target": "none", "residual": "plain"}),
    effect=EffectSpec(
        baseline_params={"layer": 16, "target": "none", "residual": "plain"},
        perturbed_params={"layer": 16, "target": "attn", "residual": "plain"},
    ),
    dtype_control=BF16,
)

# --- nemotron (Nemotron 3 Nano) -----------------------------------------------------------------


def _nemotron(suffix: str, repo: str):
    return CellConfig(
        name=f"ablation_nemotron{suffix}",
        methodology="ablation", family="nemotron", repo=repo,
        regimes=[ExecutionRegime("interactive", LARGE)],
        # target="mixer" zeroes the block's single op -> that layer becomes identity. Pick `layer` to
        # choose WHICH op type to knock out (the pattern says which indices are Mamba/attention/MoE).
        tasks=[
            ({"layer": 16, "target": "mixer"}, "layer=16 mixer"),
            ({"layer": 32, "target": "mixer"}, "layer=32 mixer"),
        ],
        baseline=BaselineSpec(params={"layer": 16, "target": "none"}),
        effect=EffectSpec(
            baseline_params={"layer": 16, "target": "none"},
            perturbed_params={"layer": 16, "target": "mixer"},
        ),
        dtype_control=BF16,
        vllm_kwargs=NEMOTRON_VLLM,
    )


ablation_nemotron = _nemotron("", NEMOTRON_30B)
ablation_nemotron_4b = _nemotron("_4b", NEMOTRON_4B)

# --- comparison (Qwen2.5-7B, cross-system) ------------------------------------------------------

cmp_ablation = CellConfig(
    name="cmp_ablation",
    methodology="ablation", family="llama", repo=QWEN25_7B,
    regimes=[ExecutionRegime("interactive", LARGE)],
    tasks=[
        ({"layer": 1, "target": "mlp"}, "zero the MLP output at layer 1"),
        ({"layer": 1, "target": "attn"}, "zero the attention output at layer 1"),
    ],
    baseline=BaselineSpec(params={"layer": 1, "target": "none"}),
    effect=EffectSpec(baseline_params={"layer": 1, "target": "none"},
                      perturbed_params={"layer": 1, "target": "mlp"}),
)
