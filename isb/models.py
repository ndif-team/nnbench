"""Models (design.md §12.16): the architectures and checkpoints workloads run on. Torch-free, so
the host can instantiate workloads without loading an inference stack.

`FAMILIES` is the architecture index (§12.8): where a family's blocks, final norm and unembed head
live in the module tree, the family's residual denotation per engine, and the block components a
workload row can name. `MODELS` is one row per checkpoint: repo, family, layer count, control
precision, load options and fitted assets. `isb/profiles.py` wraps `FAMILIES` with the torch helpers.

Nemotron 3 Nano (design.md §12.7) loads differently per backend (measured 2026-06-19):
  - HF: transformers' BUILT-IN NemotronH, trust_remote_code=False. The repo's remote modeling
    hard-requires mamba-ssm; the built-in falls back to a naive torch path when mamba-ssm/causal-conv1d
    are absent ("fast path is not available ... naive implementation" warning) — no extra deps.
  - vLLM: its NATIVE NemotronH for compute, but its config validation refuses an auto_map repo unless
    trust_remote_code=True. That flag only permits reading the config; the remote modeling is never
    executed, so vLLM also needs no mamba-ssm.

HF and vLLM share the module tree `model.model.layers[i]` (each block ONE op exposed as `.mixer` with
a `.block_type`; additive residual) / `model.model.norm_f` / `model.lm_head` (untied); HF returns the
plain residual tensor, vLLM uses fused-residual RMSNorm. The 4B is a DENSE hybrid
(`hybrid_override_pattern = "M-M-M-MM-M-M*-..."`; M=Mamba-2, -=MLP, *=attention), the cheap member used
to MEASURE the family; the 30B-A3B is MoE (`n_routed_experts=128`, top-6, +1 shared) and needs
`--pp/--tp`.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Family:
    """One architecture's module locations. Paths are dotted attribute paths from the model root;
    they are the same on HF and on nnsight's vLLM wrapper (both expose the HF module tree).

    Scope: BOUNDARY-tier locations only (block list, final norm, head, and the within-block attn/mlp
    submodule names). Internal-tier `.source` op names stay out: they are family x implementation
    specific (e.g. GPT-2 HF-eager's `attention_interface_0`) and belong to the explicit cells that
    measure them (attention_pattern stays per-family for exactly this reason)."""
    family: str
    blocks_path: str            # the decoder block list
    norm_path: str              # the final norm before the unembed
    head_path: str = "lm_head"
    # A block-output's DENOTATION is per (family x engine), not per family alone (design §3.2: the
    # same site name means different things per context): HF blocks carry the full residual stream
    # in [0] ("plain") for every family, while vLLM's implementation of RMSNorm families returns
    # (hidden, residual) whose sum is the stream ("fused"). It cannot live on the backend (backends
    # would each need a family table) and cannot be runtime-detected (HF llama blocks ALSO return
    # tuples, whose [1] is attention metadata, not a residual — shape probing cannot tell them
    # apart), so the family row carries it keyed by engine kind. An engine kind with no entry fails
    # loudly: an undeclared denotation read is exactly the SILENTLY_WRONG shape.
    residual_denotation: dict = field(default_factory=lambda: {"hf": "plain", "vllm": "plain"})
    # within-block submodule names (ablation targets); None = the family has no such split (e.g.
    # nemotron's single per-layer `mixer`, whose ablation cells stay explicit)
    attn_name: str | None = "attn"
    mlp_name: str | None = "mlp"
    # Some architectures mount the SAME text tree under an extra prefix on vLLM: Qwen3.5's
    # multimodal wrapper is `model.layers` on HF but `language_model.model.layers` on the nnsight
    # vLLM envoy (measured 2026-07-22 on Qwen/Qwen3.5-4B: `model.layers` absent, prefixed tree
    # present). The registry binds the engine view via `for_backend`, so cells never see this.
    vllm_prefix: str = ""
    mixer: bool = False         # one op per block (`block.mixer`), ablated as a whole

    @property
    def components(self) -> frozenset:
        """Block components a workload row can require (`Row.requires`)."""
        names = {"attn": self.attn_name, "mlp": self.mlp_name}
        found = {key for key, value in names.items() if value is not None}
        return frozenset(found | ({"mixer"} if self.mixer else set()))


_FUSED_ON_VLLM = {"hf": "plain", "vllm": "fused"}

FAMILIES = {
    "gpt2": Family("gpt2", "transformer.h", "transformer.ln_f"),
    # llama covers the Llama-tree checkpoints bound to it below (SmolLM2, Qwen2.5: the same
    # model.layers/model.norm/lm_head tree)
    "llama": Family("llama", "model.layers", "model.norm",
                    residual_denotation=_FUSED_ON_VLLM, attn_name="self_attn"),
    # NemotronH hybrid (Mamba/attention/MoE blocks): same tree on HF and vLLM, norm is norm_f (§12.7);
    # blocks are single-op (`mixer`), so there is no attn/mlp split (ablation cells stay explicit)
    "nemotron": Family("nemotron", "model.layers", "model.norm_f",
                       residual_denotation=_FUSED_ON_VLLM, attn_name=None, mlp_name=None, mixer=True),
    # Qwen3.5 (hybrid linear-attention, e.g. Qwen3.5-4B): HF auto-routes the text-only repo to
    # Qwen3_5ForCausalLM (llama-shaped tree), while vLLM builds the multimodal wrapper, so the same
    # tree sits behind `language_model.` there (the vllm_prefix). vLLM decoder layers return
    # (hidden, residual) like the other RMSNorm families. Blocks mix `linear_attn` and `self_attn`
    # per layer_types, so there is no uniform attn submodule name.
    "qwen3_5": Family("qwen3_5", "model.layers", "model.norm",
                      residual_denotation=_FUSED_ON_VLLM, attn_name=None,
                      vllm_prefix="language_model"),
}


@dataclass(frozen=True)
class Model:
    """One checkpoint. `n_layers` resolves workload depths and is checked against the loaded model.
    `dtype_control` is the precision of the standalone SILENTLY_WRONG-vs-DEGRADED re-check.
    `assets` are fitted artifacts bound to this checkpoint, referenced from rows by `Asset`."""
    key: str
    repo: str
    family: str
    n_layers: int
    dtype_control: str = "float32"
    hf_kwargs: dict = field(default_factory=dict)
    vllm_kwargs: dict = field(default_factory=dict)
    assets: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(f"model {self.key!r}: unknown family {self.family!r}")
        if self.n_layers < 1:
            raise ValueError(f"model {self.key!r}: n_layers must be positive")


# vLLM-only: its config validation refuses an auto_map repo without trust_remote_code, but it uses its
# NATIVE NemotronH for compute (no remote modeling executed, no mamba-ssm). HF stays on the built-in
# path (hf_kwargs left empty -> trust_remote_code=False).
_NEMOTRON_VLLM = {"trust_remote_code": True}

# The upstream fitted Jacobian lens (anthropics/jacobian-lens; neuronpedia/jacobian-lens@qwen-n1000),
# published for Qwen3.5-4B. It maps source layers 0..30 of the 32-layer stack.
_QWEN35_JLENS = {
    "transport": ("hub:neuronpedia/jacobian-lens@qwen-n1000:"
                  "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"),
    "layers": list(range(31)),
}

MODELS = {m.key: m for m in (
    Model("gpt2", "openai-community/gpt2", "gpt2", 12),
    # a LlamaForCausalLM; meta-llama is gated and uncached
    Model("smollm2-135m", "HuggingFaceTB/SmolLM2-135M-Instruct", "llama", 30),
    # Llama-shaped decoders (family "llama"). The 14B's control runs in bf16 so it fits on ONE 40GB
    # A100 (fp32 14B is ~56GB).
    Model("qwen2.5-7b", "Qwen/Qwen2.5-7B-Instruct", "llama", 28),
    Model("qwen2.5-14b", "Qwen/Qwen2.5-14B-Instruct", "llama", 48, dtype_control="bfloat16"),
    # Qwen3 keeps the Llama tree (q/k norms sit inside self_attn). The 32B is the largest cached
    # checkpoint with full-precision weights (~65 GB in bf16: one 141 GB H200, or an 80 GB card).
    Model("qwen3-32b", "Qwen/Qwen3-32B", "llama", 64, dtype_control="bfloat16"),
    # 4-bit AWQ (GEMM, group 128); the only 72B weights in the cache (the full-precision repo holds
    # config and tokenizer only). Loading it on HF needs an AWQ kernel package.
    Model("qwen2.5-72b-awq", "Qwen/Qwen2.5-72B-Instruct-AWQ", "llama", 80, dtype_control="bfloat16"),
    # Text-only checkpoint of a multimodal-wrapper arch: transformers auto-routes it to the text-only
    # ForCausalLM, but nnsight's LanguageModel refuses the repo by model_type (the guard exempts
    # explicit automodels; see HFBackend).
    Model("qwen3.5-4b", "Qwen/Qwen3.5-4B", "qwen3_5", 32,
          hf_kwargs={"force_text_causal": True}, vllm_kwargs={"max_model_len": 1024},
          assets={"jlens": _QWEN35_JLENS}),
    Model("nemotron3-nano-4b", "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", "nemotron", 42,
          dtype_control="bfloat16", vllm_kwargs=_NEMOTRON_VLLM),
    Model("nemotron3-nano-30b", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "nemotron", 52,
          dtype_control="bfloat16", vllm_kwargs=_NEMOTRON_VLLM),
)}
