"""Jacobian-lens spec — the first spec whose workload is a REAL external dataset.

The interactive workload is the upstream multi-hop lens eval (anthropics/jacobian-lens,
`data/jlens/lens-eval-multihop.json`, 93 items): "Fact: The ocean on the coast of the country
where Carnival is most famously celebrated is the " — prompts whose pre-answer position carries a
latent bridge entity. Each prompt runs as its own trace and the oracle aggregates HF-vs-vLLM
top-1/TV over all of them (the multi-trace form; no more hand-picked octet).

Two tasks split the method's two realizations of the transport map:
  - identity transport: the readout IS the plain logit lens at the upstream readout position.
  - seeded-orthogonal transport: the J-matmul code path with a deterministic full-rank map (the
    same matrix on both backends), i.e. the J-lens program shape without the fitted artifact. A
    fitted J_l checkpoint (the `trained`-tag upgrade) drops in through the same param later.

Read methodology -> no effect-size guard. Baseline = single-layer identity read (`layers=[-1]`),
so overhead-vs-baseline reflects the full-band sweep + transport cost.
"""
from ..data import DataRef
from ..sweep.spec import BaselineSpec, CellConfig, ExecutionRegime

# default binding: the upstream multi-hop eval; any other lens-eval source swaps in via --data
MULTIHOP = DataRef("jlens/multihop")

jacobian_lens_gpt2 = CellConfig(
    name="jacobian_lens_gpt2",
    methodology="jacobian_lens", family="gpt2", repo="openai-community/gpt2",
    regimes=[ExecutionRegime("interactive", MULTIHOP)],
    tasks=[
        ({"unembed": "weight", "transport": None}, "transport=identity (logit-lens readout)"),
        ({"unembed": "weight", "transport": 0}, "transport=seeded-orthogonal (the J-matmul path)"),
    ],
    baseline=BaselineSpec(params={"unembed": "weight", "transport": None, "layers": [-1]}),
    effect=None,
)


# --- The REAL fitted lens. Upstream publishes checkpoints only for Qwen3.5-4B / Qwen3.6-27B
# (neuronpedia/jacobian-lens @ qwen-n1000), so the fitted-J bench runs on Qwen3.5-4B: a hybrid
# linear-attention model (mostly linear_attention layers, periodic full_attention), 32 layers,
# d_model 2560; the lens covers source layers 0..30. family="qwen3_5" (profiles.py): HF tree is
# llama-shaped, vLLM mounts it behind `language_model.` (the profile's vllm_prefix). Not in the
# default `--spec all` corpus (4B-scale; the allowlist keeps `all` at smoke scale).
_QWEN35_LENS = ("hub:neuronpedia/jacobian-lens@qwen-n1000:"
                "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt")
_QWEN35_BAND = list(range(31))          # the lens's source layers (0..30 of the 32-layer stack)

jacobian_lens_qwen35 = CellConfig(
    name="jacobian_lens_qwen35",
    methodology="jacobian_lens", family="qwen3_5", repo="Qwen/Qwen3.5-4B",
    regimes=[ExecutionRegime("interactive", MULTIHOP)],
    tasks=[
        ({"unembed": "weight", "transport": None, "layers": _QWEN35_BAND},
         "transport=identity (logit-lens readout)"),
        ({"unembed": "weight", "transport": _QWEN35_LENS, "layers": _QWEN35_BAND},
         "transport=fitted J (upstream qwen-n1000 lens)"),
    ],
    baseline=BaselineSpec(params={"unembed": "weight", "transport": None, "layers": [-1]}),
    effect=None,
    # keep the run tractable at 4B: the verdict still aggregates over all 93 prompts; only the
    # perf trial count shrinks
    warmup=1, n_trials=3,
    # text-only checkpoint of a multimodal-wrapper arch: transformers auto-routes it to the
    # text-only ForCausalLM, but nnsight's LanguageModel refuses the repo by model_type (an
    # interface quirk worth knowing: the guard exempts explicit automodels; see HFBackend)
    hf_kwargs={"force_text_causal": True},
    vllm_kwargs={"max_model_len": 1024},
)
