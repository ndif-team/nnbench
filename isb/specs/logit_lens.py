"""logit-lens specs, every model.

Read methodology -> no effect-size guard. The no-intervention baseline is a single-layer lens
(`layers=[-1]`, portable unembed), i.e. one read instead of the full-stack lens, so overhead-vs-
baseline reflects the per-layer lens cost.
"""
from ..data import DataRef
from ..sweep.spec import BaselineSpec, CellConfig, ExecutionRegime
from ._models import BF16, GPT2, NEMOTRON_4B, NEMOTRON_30B, NEMOTRON_VLLM, QWEN25_7B, QWEN25_14B, SMOLLM2

# Data is a named, swappable source (isb/data.py; override at the CLI with --data). The default:
# CounterFact factual-recall prompts (data/counterfact/README.md) — interactive scores over 100
# units; batched pads a size-16 subset (batched is a regime check, not a volume axis). The
# template bank stays available via --data factual.
PROBE = DataRef("counterfact", 100)
BATCHED = DataRef("counterfact", 16)
# Large models: sized for one multi-GPU run.
LARGE = DataRef("counterfact", 32)

# --- smoke (gpt2, SmolLM2) ----------------------------------------------------------------------

logit_lens_gpt2 = CellConfig(
    name="logit_lens_gpt2",
    methodology="logit_lens", family="gpt2", repo=GPT2,
    regimes=[ExecutionRegime("interactive", PROBE), ExecutionRegime("batched", BATCHED)],
    tasks=[
        ({"unembed": "module"}, "unembed=module"),
        ({"unembed": "weight"}, "unembed=weight"),
    ],
    baseline=BaselineSpec(params={"unembed": "weight", "layers": [-1]}),
    effect=None,
)

logit_lens_llama = CellConfig(
    name="logit_lens_llama",
    methodology="logit_lens", family="llama", repo=SMOLLM2,
    regimes=[ExecutionRegime("interactive", PROBE), ExecutionRegime("batched", BATCHED)],
    tasks=[
        ({"unembed": "module"}, "unembed=module"),
        ({"unembed": "weight"}, "unembed=weight (backend-aware)"),
        ({"unembed": "weight", "residual": "plain"}, "unembed=weight, residual=plain (naive port)"),
    ],
    baseline=BaselineSpec(params={"unembed": "weight", "layers": [-1]}),
    effect=None,
)

# --- parallel (Qwen2.5-14B, GT2) ----------------------------------------------------------------

logit_lens_qwen = CellConfig(
    name="logit_lens_qwen",
    methodology="logit_lens", family="llama", repo=QWEN25_14B,
    regimes=[ExecutionRegime("interactive", LARGE)],
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
    dtype_control=BF16,
)

# --- nemotron (Nemotron 3 Nano, 4B dense and 30B-A3B MoE) ----------------------------------------


def _nemotron(suffix: str, repo: str):
    return CellConfig(
        name=f"logit_lens_nemotron{suffix}",
        methodology="logit_lens", family="nemotron", repo=repo,
        regimes=[ExecutionRegime("interactive", LARGE)],
        tasks=[
            ({"unembed": "weight", "residual": "plain"}, "unembed=weight, residual=plain"),
            ({"unembed": "weight", "residual": "fused"}, "unembed=weight, residual=fused"),
            ({"unembed": "module", "residual": "plain"}, "unembed=module"),
        ],
        baseline=BaselineSpec(params={"unembed": "weight", "layers": [-1], "residual": "plain"}),
        effect=None,
        dtype_control=BF16,
        vllm_kwargs=NEMOTRON_VLLM,
    )


logit_lens_nemotron = _nemotron("", NEMOTRON_30B)
logit_lens_nemotron_4b = _nemotron("_4b", NEMOTRON_4B)

# --- comparison (Qwen2.5-7B, cross-system) ------------------------------------------------------

cmp_logit_lens = CellConfig(
    name="cmp_logit_lens",
    methodology="logit_lens", family="llama", repo=QWEN25_7B,
    regimes=[ExecutionRegime("interactive", LARGE)],
    tasks=[({"layers": "all"}, "every layer, last token")],
    baseline=BaselineSpec(params={"layers": [-1]}),
)
