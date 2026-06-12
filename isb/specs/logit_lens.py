"""logit-lens specs (gpt2 + llama) — collapses smoke.py and smoke_llama.py.

Read methodology -> no effect-size guard. The no-intervention baseline is a single-layer lens
(`layers=[-1]`, portable unembed), i.e. one read instead of the full-stack lens, so overhead-vs-
baseline reflects the per-layer lens cost.
"""
from ..sweep.spec import BaselineSpec, CellConfig, Workload
from ._prompts import BATCHED, PROBE

logit_lens_gpt2 = CellConfig(
    name="logit_lens_gpt2",
    methodology="logit_lens", family="gpt2", repo="openai-community/gpt2",
    workloads=[Workload("interactive", PROBE), Workload("batched", BATCHED)],
    tasks=[
        ({"unembed": "module"}, "unembed=module"),
        ({"unembed": "weight"}, "unembed=weight"),
    ],
    baseline=BaselineSpec(params={"unembed": "weight", "layers": [-1]}),
    effect=None,
    expected={
        # idiomatic unembed calls the guarded ParallelLMHead.forward on vLLM -> ERROR (F-2)
        ("vllm_async", "interactive", "unembed=module"): "ERROR",
        # GPT-2 left-pads without position_ids -> padded rows' absolute positions shift -> HF's own
        # batched output diverges from its per-prompt truth (a documented absolute-position artifact)
        ("hf", "batched", "unembed=module"): "SILENTLY_WRONG",
        ("hf", "batched", "unembed=weight"): "SILENTLY_WRONG",
        # vLLM batched is gated on the main dev checkout (run_batched submits only invoke[0]) -> ERROR.
        # A future flip to SUPPORTED would surface as a surprise = "the async multi-prompt fix landed".
        ("vllm_async", "batched", "unembed=module"): "ERROR",
        ("vllm_async", "batched", "unembed=weight"): "ERROR",
    },
)

logit_lens_llama = CellConfig(
    name="logit_lens_llama",
    methodology="logit_lens", family="llama",
    repo="HuggingFaceTB/SmolLM2-135M-Instruct",   # a LlamaForCausalLM; meta-llama is gated + uncached
    workloads=[Workload("interactive", PROBE), Workload("batched", BATCHED)],
    tasks=[
        ({"unembed": "module"}, "unembed=module"),
        ({"unembed": "weight"}, "unembed=weight (backend-aware)"),
        ({"unembed": "weight", "residual": "plain"}, "unembed=weight, residual=plain (naive port)"),
    ],
    baseline=BaselineSpec(params={"unembed": "weight", "layers": [-1]}),
    effect=None,
    expected={
        ("vllm_async", "interactive", "unembed=module"): "ERROR",                          # F-2
        # naive GPT-2 port reads only stream[0]; vLLM-Llama's residual is hidden+residual (Gap 1.2,
        # F-7) -> drops the accumulated residual -> SILENTLY_WRONG. residual=fused is the working form.
        ("vllm_async", "interactive", "unembed=weight, residual=plain (naive port)"): "SILENTLY_WRONG",
        # llama is RoPE (relative positions) -> batched HF matches its per-prompt truth -> SUPPORTED.
        ("vllm_async", "batched", "unembed=module"): "ERROR",                       # gated (+ guard)
        ("vllm_async", "batched", "unembed=weight (backend-aware)"): "ERROR",       # batched gated
        ("vllm_async", "batched", "unembed=weight, residual=plain (naive port)"): "ERROR",
    },
)
