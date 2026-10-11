"""steering specs, every model.

Baseline = alpha=0 (no write -> pure forward + readout); effect-size = TV(alpha=0, alpha=6) on the
control (the per-mode non-vacuity guard, now declarative).
"""
from ..data import DataRef
from ..sweep.spec import BaselineSpec, CellConfig, EffectSpec, ExecutionRegime
from ._models import BF16, GPT2, NEMOTRON_4B, NEMOTRON_30B, NEMOTRON_VLLM, QWEN25_7B, QWEN25_14B

# data is a named, swappable source — see logit_lens.py. Default: CounterFact factual-recall
# prompts (data/counterfact/README.md); the snapshot's per-item target_new strings are the
# hook for a later per-item steering-target upgrade (today `target` is one task param).
PROBE = DataRef("counterfact", 100)
BATCHED = DataRef("counterfact", 16)
LARGE = DataRef("counterfact", 32)

# --- smoke (gpt2) -------------------------------------------------------------------------------

_S = {"layer": 8, "target": " Rome", "alpha": 6.0}

steering_gpt2 = CellConfig(
    name="steering_gpt2",
    methodology="steering", family="gpt2", repo=GPT2,
    regimes=[ExecutionRegime("interactive", PROBE), ExecutionRegime("batched", BATCHED)],
    tasks=[
        ({**_S, "mode": "inplace"}, "mode=inplace"),
        ({**_S, "mode": "replace"}, "mode=replace"),
    ],
    baseline=BaselineSpec(params={**_S, "alpha": 0.0, "mode": "replace"}),
    effect=EffectSpec(
        baseline_params={**_S, "alpha": 0.0, "mode": "replace"},
        perturbed_params={**_S, "alpha": 6.0, "mode": "replace"},
    ),
)

# --- parallel (Qwen2.5-14B, GT2) and nemotron (Nemotron 3 Nano) ----------------------------------

_L16 = {"layer": 16, "target": " Rome", "alpha": 6.0}

steering_qwen = CellConfig(
    name="steering_qwen",
    methodology="steering", family="llama", repo=QWEN25_14B,
    regimes=[ExecutionRegime("interactive", LARGE)],
    tasks=[
        ({**_L16, "mode": "inplace"}, "mode=inplace"),
        ({**_L16, "mode": "replace"}, "mode=replace"),
    ],
    baseline=BaselineSpec(params={**_L16, "alpha": 0.0, "mode": "replace"}),
    effect=EffectSpec(
        baseline_params={**_L16, "alpha": 0.0, "mode": "replace"},
        perturbed_params={**_L16, "alpha": 6.0, "mode": "replace"},
    ),
    dtype_control=BF16,
)


def _nemotron(suffix: str, repo: str):
    return CellConfig(
        name=f"steering_nemotron{suffix}",
        methodology="steering", family="nemotron", repo=repo,
        regimes=[ExecutionRegime("interactive", LARGE)],
        tasks=[
            ({**_L16, "mode": "inplace"}, "mode=inplace"),
            ({**_L16, "mode": "replace"}, "mode=replace"),
        ],
        baseline=BaselineSpec(params={**_L16, "alpha": 0.0, "mode": "replace"}),
        effect=EffectSpec(
            baseline_params={**_L16, "alpha": 0.0, "mode": "replace"},
            perturbed_params={**_L16, "alpha": 6.0, "mode": "replace"},
        ),
        dtype_control=BF16,
        vllm_kwargs=NEMOTRON_VLLM,
    )


steering_nemotron = _nemotron("", NEMOTRON_30B)
steering_nemotron_4b = _nemotron("_4b", NEMOTRON_4B)

# --- comparison (Qwen2.5-7B, cross-system) ------------------------------------------------------

_STEER = {"target": " Rome", "alpha": 6.0}

cmp_steering = CellConfig(
    name="cmp_steering",
    methodology="steering", family="llama", repo=QWEN25_7B,
    regimes=[ExecutionRegime("interactive", LARGE)],
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
