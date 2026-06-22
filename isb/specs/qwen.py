"""Qwen specs — large-model (Qwen2.5-14B-Instruct) variants for the TP/PP parallelism-equivalence
runs (`bench.py --pp 2 --tp 2`).

family="llama" reuses the model.model.layers / model.model.norm / model.lm_head cells (Qwen2.5 is a
LlamaForCausalLM-shaped decoder). dtype_control="bfloat16" so the (1,1) control fits on ONE 40GB
A100 — fp32 14B (~56GB) would not. Interactive/generation workloads only (batched is gated on the
vLLM async path regardless of model). These exist to exercise the GT2 oracle — candidate (tp,pp) vs
single-GPU (1,1), same dtype — on a real multi-billion-param model.

Note: under GT2 both sides run the SAME cell, so a cell only needs to RUN and be deterministic;
absolute correctness vs HF (the fused-residual subtlety) is not what's scored here. `expected` lists
only cells that genuinely cannot run on vLLM (guarded lm_head.forward, in-place inference-tensor
write, unbounded iter[:] dropping per-step saves) — the "naturally not supported" set.
"""
from ..sweep.spec import BaselineSpec, CellConfig, EffectSpec, Workload
from ._prompts import CLEAN, CORRUPTED, PROBE

_QWEN = "Qwen/Qwen2.5-14B-Instruct"
_BF16 = "bfloat16"
_S = {"layer": 16, "target": " Rome", "alpha": 6.0}

logit_lens_qwen = CellConfig(
    name="logit_lens_qwen",
    methodology="logit_lens", family="llama", repo=_QWEN,
    workloads=[Workload("interactive", PROBE)],
    # residual="plain" (read stream[0]) on BOTH sides: under PP some layers are LazyRemoteTensors and
    # the "fused" (hidden+residual) read isn't symmetric across the stage boundary, which would make
    # the candidate diverge from the control for reasons unrelated to PP correctness. GT2 only needs
    # the two configs to read the residual the same way; absolute-vs-HF fidelity isn't scored here.
    tasks=[
        ({"unembed": "module", "residual": "plain"}, "unembed=module"),
        ({"unembed": "weight", "residual": "plain"}, "unembed=weight"),
    ],
    baseline=BaselineSpec(params={"unembed": "weight", "layers": [-1], "residual": "plain"}),
    effect=None,
    dtype_control=_BF16,
    expected={("vllm_async", "interactive", "unembed=module"): "ERROR"},  # guarded lm_head.forward
)

steering_qwen = CellConfig(
    name="steering_qwen",
    methodology="steering", family="llama", repo=_QWEN,
    workloads=[Workload("interactive", PROBE)],
    tasks=[
        ({**_S, "mode": "inplace"}, "mode=inplace"),
        ({**_S, "mode": "replace"}, "mode=replace"),
    ],
    baseline=BaselineSpec(params={**_S, "alpha": 0.0, "mode": "replace"}),
    effect=EffectSpec(
        baseline_params={**_S, "alpha": 0.0, "mode": "replace"},
        perturbed_params={**_S, "alpha": 6.0, "mode": "replace"},
    ),
    dtype_control=_BF16,
    expected={("vllm_async", "interactive", "mode=inplace"): "ERROR"},  # in-place write on inference tensor
)

ablation_qwen = CellConfig(
    name="ablation_qwen",
    methodology="ablation", family="llama", repo=_QWEN,
    workloads=[Workload("interactive", PROBE)],
    tasks=[
        ({"layer": 16, "target": "mlp"}, "target=mlp"),
        ({"layer": 16, "target": "attn"}, "target=attn"),
    ],
    baseline=BaselineSpec(params={"layer": 16, "target": "none"}),
    effect=EffectSpec(
        baseline_params={"layer": 16, "target": "none"},
        perturbed_params={"layer": 16, "target": "attn"},
    ),
    dtype_control=_BF16,
    expected={},
)

activation_patching_qwen = CellConfig(
    name="activation_patching_qwen",
    methodology="activation_patching", family="llama", repo=_QWEN,
    workloads=[Workload("interactive", [CLEAN, CORRUPTED], aggregate=False)],
    tasks=[
        ({"layer": 8, "residual": "plain"}, "layer=8"),
        ({"layer": 24, "residual": "plain"}, "layer=24"),
    ],
    baseline=BaselineSpec(params={"patch": False, "residual": "plain"}),
    effect=EffectSpec(
        baseline_params={"patch": False, "residual": "plain"},
        perturbed_params={"layer": 24, "residual": "plain", "patch": True},
    ),
    dtype_control=_BF16,
    expected={},
)

gen_steering_qwen = CellConfig(
    name="gen_steering_qwen",
    methodology="gen_steering", family="llama", repo=_QWEN,
    workloads=[Workload("generation", PROBE, new_tokens=8)],
    tasks=[
        ({**_S, "bound": "bounded"}, "bound=iter[0:N]"),
        ({**_S, "bound": "unbounded"}, "bound=iter[:]"),
    ],
    baseline=BaselineSpec(params={**_S, "alpha": 0.0, "bound": "bounded"}),
    effect=EffectSpec(
        baseline_params={**_S, "alpha": 0.0, "bound": "bounded"},
        perturbed_params={**_S, "bound": "bounded"},
    ),
    dtype_control=_BF16,
    expected={("vllm_async", "generation", "bound=iter[:]"): "ERROR"},  # unbounded iter[:] drops saves
)
