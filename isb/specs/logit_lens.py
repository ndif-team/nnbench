"""logit-lens specs (gpt2 + llama) — collapses smoke.py and smoke_llama.py.

Read methodology -> no effect-size guard. The no-intervention baseline is a single-layer lens
(`layers=[-1]`, portable unembed), i.e. one read instead of the full-stack lens, so overhead-vs-
baseline reflects the per-layer lens cost.
"""
from ..sweep.spec import BaselineSpec, CellConfig, Workload
from ._prompts import BATCHED, ONE

logit_lens_gpt2 = CellConfig(
    name="logit_lens_gpt2",
    methodology="logit_lens", family="gpt2", repo="openai-community/gpt2",
    workloads=[Workload("interactive", ONE), Workload("batched", BATCHED)],
    tasks=[
        ({"unembed": "module"}, "unembed=module"),
        ({"unembed": "weight"}, "unembed=weight"),
    ],
    baseline=BaselineSpec(params={"unembed": "weight", "layers": [-1]}),
    effect=None,
)

logit_lens_llama = CellConfig(
    name="logit_lens_llama",
    methodology="logit_lens", family="llama",
    repo="HuggingFaceTB/SmolLM2-135M-Instruct",   # a LlamaForCausalLM; meta-llama is gated + uncached
    workloads=[Workload("interactive", ONE), Workload("batched", BATCHED)],
    tasks=[
        ({"unembed": "module"}, "unembed=module"),
        ({"unembed": "weight"}, "unembed=weight (backend-aware)"),
        ({"unembed": "weight", "residual": "plain"}, "unembed=weight, residual=plain (naive port)"),
    ],
    baseline=BaselineSpec(params={"unembed": "weight", "layers": [-1]}),
    effect=None,
)
