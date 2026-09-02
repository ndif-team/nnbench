# References — by category

Each entry: what it is, and **the connection** to this benchmark (source of workloads /
prior art for a component / baseline / related-work positioning / consumer).

---

## A. Intervention frameworks — the substrate ("how you express an intervention")

These are alternative abstractions for accessing/modifying model internals. They **compete and
complement**. Our system-under-test is nnsight; the others inform positioning and design reuse.

- **nnsight** (ndif-team/nnsight) — deferred-execution *trace* abstraction; remote execution via
  NDIF; vLLM integration in active development. **This is our SUT.** Branch under test: `dev`.
- **pyvene** (stanfordnlp/pyvene, NAACL 2024) — *declarative* dict-based interventions.
  `IntervenableConfig` / `RepresentationConfig(layer, component, intervention_type, unit, …)`.
  Intervention-type enum: Vanilla / Addition / Subtraction / Zero / Collect / RotatedSpace(DAS) /
  LoRA. Component vocabulary: `block_input/output`, `mlp_input/output/activation`,
  `attention_input/output`, `query/key/value_output`, `head_attention_value_output`. Addresses
  components across architectures via `type_to_module_mapping` / `type_to_dimension_mapping`
  (GPT-2, Llama, Pythia, Mistral, Mixtral, OPT, BLIP, Mamba, …). **Connection:** (1) direct prior
  art for our *workload spec schema* — its declarative config IS "spec-as-data"; (2) its
  component-name vocabulary + type→module mapping is a battle-tested, cross-architecture solution
  to our **Resolver** problem — adopt/align rather than reinvent; (3) a possible cross-framework
  baseline.
- **TransformerLens** — hook-based, transformers-only, exploratory analysis. Related work.
- **baukit** (David Bau) — low-level hooks/edits. Related work / ancestor patterns.
- **vllm-lens** (UK AISI) — vLLM plugin for probes/steering/oracles, "residual stream only."
  Closest *systems* prior art on the vLLM side; narrower scope than nnsight×vLLM.
- **Goodfire SGLang fork** — production activation harvesting (Kimi K2 Thinking, SAE training).
  The "harvesting at production scale" workload our L2 profile models; a forked-per-workflow point
  solution vs nnsight's general programmability.

## B. Method-faithfulness benchmarks — measure *"is the interpretability RESULT correct?"*

**Different question than us.** Ground truth = a known causal model / known circuit; metric =
faithfulness / interchange-intervention-accuracy. We borrow their *workloads* and *harness shape*,
not their metric.

- **causalab** (goodfire-ai/causalab) — causal-abstraction framework. PR #20 replaced its old
  pyvene/Hydra analysis stack with serializable **intervention-protocol documents**: named data
  roles, positions, sites, reads, writes with a closed `do` algebra, intervened models,
  featurizers, metrics, training, and saves. Documents execute through a generic `Backend` seam;
  a workflow composes independently routed document steps. PR #40 adds greedy generation as a
  position frame with prefill-only writes. **Connection:** nnbench adopts the intersection as
  non-executable `InterventionSpec` metadata and keeps system context, realization, correctness
  status, and performance as its own axes. It does not adopt causalab's document executor: that is
  the construction layer nnbench measures. See `causalab-portability-audit.md`.
- **CausalGym** (arXiv 2402.12560) — benchmarking causal interpretability methods on linguistic
  tasks. Faithfulness benchmark; method-comparison framing.
- **InterpBench** (Gupta et al.) — semi-synthetic transformers with *known* circuits as ground
  truth. **Name-collision warning** — do not name our repo "InterpBench."
- **IOI** (Wang et al. 2022, "Interpretability in the Wild") — canonical circuit task reused as a
  workload everywhere (pyvene, causalab demos).

## C. Method corpora / tutorials — runnable recipes ("what technique")

The seed for our **methodology registry** (design.md §4; seed, not ceiling).

- **nnsight-website tutorials** (`ndif-team/nnsight-website`, branch `docs`, `docs/tutorials/`):
  - `tutorials/`: `probing/{logit_lens, diffusion_lens}`,
    `causal_mediation_analysis/{activation_patching, attribution_patching, DAS,
    causal_mediation_analysis_i/ii, causal_models_intro}`,
    `steering/{LoRA_tutorial, dict_learning}`, `get_started/{walkthrough, start_remote_access,
    chat_templates}`.
  - `mini-papers/`: `marks_geometry_of_truth`, `todd_function_vectors`, `csordas_llm_depth`,
    `feucht_dual_route_induction`, `huang_demystifying_memorization`. ← the **Macro** tier
    (end-to-end paper reproductions).
- **nnsightful** (AdamBelfki3/nnsightful) — higher-level method+viz layer on `StandardizedTransformer`
  (logit lens, activation patching, TS/React charts). Inventory of *what researchers want*, not a
  benchmark. Note: `StandardizedTransformer` is **not** present on nnsight `dev` — so our Resolver
  cannot assume it.
- Broader literature backlog (not in tutorials): path/edge patching, attention knockout, EAP/ACDC
  circuit discovery, tuned lens / future lens / **patchscopes**, SAE family (gated/JumpReLU/top-k/
  transcoders/crosscoders), steering family (CAA/ITI/RepE), sparse probing / CCS, integrated
  gradients.

## D. Systems / serving context — the backends + the thesis

- **OSDI '26 abstracts** (in nnsight repo: `docs/blog/vllm-integration/osdi26_abstract.md`,
  `osdi26_poster_abstract_research.md`) — the strategic framing: NNsight×vLLM as a programmability
  layer for production inference engines; three axes **engine / distribution / optimization**.
  Our **L3 sweep matrix instantiates these axes**, and our **coverage matrix produces the gap map**.
- **vLLM**, **SGLang** — target production engines.
- nnsight `docs/developing/vllm-integration.md` — integration internals.

## E. Surveys / theory

- **Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability** (Geiger et al.,
  JMLR 2025 / arXiv 2301.04709).
- **Causal Abstraction in Model Interpretability: A Compact Survey** (arXiv 2410.20161).

---

## The 3-layer map (why these all feel connected, but aren't the same)

```
 (1) FRAMEWORKS          how you express an intervention
     nnsight · pyvene · TransformerLens · baukit · vllm-lens
        │ built on
        ▼
 (2) METHOD LIBRARIES    what technique (DAS, SAE, logit lens, patching, probing)
     nnsightful · pyvene intervention types · causalab/methods/
        │ used by
        ▼
 (3) BENCHMARKS          split by WHAT IS MEASURED:
     (3a) faithfulness  — "is the RESULT correct?"     causalab · CausalGym · InterpBench
     (3b) systems       — "does the WORKLOAD run fast/correctly across backends?"  ← THIS REPO
```

**(3b) is empty.** There is no MLPerf-for-interpretability-on-production-engines. That is the niche.
