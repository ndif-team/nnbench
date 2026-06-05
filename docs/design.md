# Design — interp-serve-bench (living document)

> Status: evolving. Captures decisions as they're made in the design conversation.
> Last structural update: positioning + taxonomy + resolver decouple + reference synthesis.

## 1. Purpose & positioning

A **systems performance + coverage benchmark** for interpretability workloads on nnsight,
swept across serving backends, model types/architectures, and parallelism/optimization configs.

It answers, for each (workload × model × backend × config) cell:
1. **Does it run?** → coverage matrix (= the OSDI "still in active design" gap map, generated).
2. **Is it correct?** → numerical equivalence vs an HF-eager reference oracle, within tolerance.
3. **Is it fast?** → latency, throughput, peak memory, overhead vs baselines, transfer volume.

We are layer **(3b)** in the field map (see `references.md`): the *systems* benchmark. The field
has frameworks (nnsight/pyvene) and *faithfulness* benchmarks (causalab/CausalGym/InterpBench),
but no systems-performance-and-coverage benchmark for interp workloads on production engines.

Strategic backdrop: the nnsight OSDI '26 abstracts frame NNsight×vLLM along three axes —
**engine** (request lifecycle vs one forward call), **distribution** (TP/PP/EP sharding),
**optimization** (continuous batching, CUDA graphs, prefix caching). Our **L3 sweep matrix
instantiates those axes**, so the benchmark is the empirical backbone of the systems story.

## 2. Organizing spine — granularity tiers (map 1:1 onto existing artifacts)

| Tier | = | Seeded from | Role |
|---|---|---|---|
| **Micro** | L0 primitives | `nnsight/tests/performance/benchmark_interventions.py` | overhead floor |
| **Method** | single motif | nnsight-website `tutorials/` | canonical units |
| **Macro** | end-to-end paper repro | nnsight-website `mini-papers/` | "real research runs on this" |

Adding work = adding rows/registry entries, never a harness redesign.

## 3. L0 — Primitives (the interpretability "ISA")

Atomic trace-level ops. Each scales along independent axes: **breadth** (#sites), **tensor size**
(hidden×seq×batch), **depth** (#tokens), **payload** (bytes saved/transferred), **side-compute**
(aux FLOPs).

| Primitive | nnsight surface | Notes |
|---|---|---|
| READ one site | `.output`/`.input`/`.inputs` | |
| CACHE many | `tracer.cache(modules=…)` | breadth |
| WRITE-replace | `x = new` | |
| WRITE-inplace | `x[:] = …` | distinct semantics |
| CROSS-PROMPT | invoke A → invoke B + `barrier` | patching |
| ITERATIVE | `tracer.iter[...]`, `.all()`, `.next()` | decode steps |
| BACKWARD-attribution | `with t.backward(): …grad` | single bwd |
| TRAIN-INTERVENTION | bwd → params → optimizer loop | DAS / dict-learning / probe-train / LoRA (heavy) |
| AUX-MODULE | run SAE/probe/rotation in trace | frozen *or* trainable |
| EDIT | `model.edit()` | persistent weight/module edit |
| LOGITS / SAMPLING | logits & sampled-token props | backend-specific |
| SOURCE | `.source.<op>` | intermediate ops; fragile/family-specific |
| SAVE / transmit | `.save()` | payload, esp. remote |

## 4. L1 — Motif registry (seed, not ceiling)

A **living registry with a fixed schema**: seeded from the nnsight tutorials, extended from the
literature; adding a motif = one registry entry, never a harness change.

Seeded (from website tutorials): logit lens · tuned lens · diffusion lens · activation patching /
causal tracing · attribution patching · DAS / interchange interventions · steering / ActAdd ·
ablation · SAE extract/steer · dictionary learning (SAE *training*) · linear probing · function
vectors · LoRA / edit · attention / per-head · DLA · activation harvesting.

Backlog (literature, additive later): path/edge patching · attention knockout · EAP/ACDC ·
patchscopes · future lens · SAE family (gated/JumpReLU/top-k/transcoders/crosscoders) · steering
family (CAA/ITI/RepE) · sparse probing / CCS · integrated gradients.

**Borrow:** align motif → primitive recipes with **pyvene's intervention-type enum** (Vanilla /
Addition / Subtraction / Zero / Collect / RotatedSpace=DAS / LoRA) where they map, for shared
vocabulary and cross-framework portability.

## 5. L2 — Workload profiles (how motifs get run; the "dataset" distribution)

- **Interactive probe** — 1 trace, 1 prompt (notebook shape)
- **Batched analysis** — 1 motif × N prompts (patching/probing over a dataset)
- **Generation-time intervention** — steering across a multi-token decode (serving shape)
- **Bulk harvesting** — CACHE-many × large corpus, throughput-bound (SAE data collection)
- **Multi-tenant / concurrent** — many independent traces vs one engine (nnsight-serve / NDIF)

L2 exercises the **engine axis** + concurrency — where vLLM should win and HF should struggle.

## 6. L3 — Sweep matrix (system under test = the OSDI three axes as config)

- **Backend (engine axis)**: HF Transformers · vLLM sync · vLLM async · NDIF remote
- **Parallelism (distribution axis)**: single-GPU · TP=2/4/8 · PP · multi-node
- **Engine config (optimization axis)**: enforce_eager vs CUDA graphs · prefix-cache on/off
- **Model type**: causal-LM · diffusion · vision-language  *(first-class axis — "types")*
- **Architecture / scale**: GPT-2 → Llama/Qwen/Gemma/Mistral (7–9B) → 70B → MoE/frontier;
  deliberately include **non-standard module names** to prove the Resolver isn't hardcoded.

## 7. Harness — the spec → resolver → builder → runner → oracle → reporter pipeline

```
Workload spec  (family-INDEPENDENT — the portable, citable "dataset")
    motif + params: logical selectors, scope, prompts/dataset, gen length, aux
        │
        ▼
Resolver  (per type→family — THE ONLY model-aware code)
    logical selector  →  set of (module_ref, tensor_slice|None, access_kind)
        │
        ▼
Builder   (family-INDEPENDENT — compiles spec → nnsight trace closure)
        │
        ▼
Runner    (instantiate SUT on backend; warmup; timed trials; failure-tolerant — record
           "unsupported" structurally, never crash the sweep)
        │
        ▼
Oracle    (compare vs HF-eager reference / golden values, within tolerance)
        │
        ▼
Reporter  (parquet/json → tables, plots, coverage matrix, regression diff)
```

### 7.1 The Resolver decouple (load-bearing decision)

- **Invariant:** family knowledge lives *only* in the Resolver. Motif builders speak *only* the
  logical vocabulary and never see a concrete path — a builder referencing `transformer.h` is a
  bug *by construction*. This makes "don't hardcode to GPT-2/Llama" structurally impossible to
  violate, not a review-time discipline.
- **Two levels:** `type → family`. *Type* fixes which vocabulary exists (causal-LM:
  layers/attn/mlp/residual/heads/unembed; diffusion: down/mid/up blocks + timestep; VLM:
  vision tower / connector / LM). *Family* fixes the concrete binding.
- **Resolution tiers** (a binding is `(module_ref, slice|None, access_kind)`):
  - block-level (`residual_{pre,post}`, `attn_out`, `mlp_out`) — mechanical from tree + config.
  - sub-block (`head[h]`, `neuron[j]`) — needs `config.{num_attention_heads, head_dim,
    intermediate_size}` + a tensor slice.
  - intermediate-op (`attn_weights` pre-softmax, residual add) — needs `.source`; fragile,
    family-specific.
  - runtime/engine (`logits`, `sampled_token`) — backend-specific, not family-specific.
- **Capability declaration → coverage matrix for free.** Each adapter (and backend) declares which
  logical selectors it can realize; unrealizable = structured `unsupported`. That declaration *is*
  the OSDI gap map.
- **Borrow, don't reinvent:** adopt/align **pyvene's component vocabulary** (`block_output`,
  `mlp_output`, `head_attention_value_output`, …) and its `type_to_module_mapping` /
  `type_to_dimension_mapping` approach. Where nnsight later ships a canonical view, the Resolver
  *wraps* it (so we test nnsight's abstraction) and only adds sub-block/source/type tiers on top.

### 7.2 Harness shape — borrow causalab

causalab already solved "declarative spec + config-group sweep + per-task/per-model decouple +
analysis-DAG with artifact deps" — for *faithfulness*. Adopt its proven shape (Hydra config groups
`task/model/analysis/runners`, agent/skill-driven runner) and **swap the metric layer from
faithfulness → systems**. Our config groups ≈ `workload / model / backend / metric`.

## 8. Metrics

- **Performance:** end-to-end latency · throughput (tok/s, traces/s, prompts/s) · peak GPU memory ·
  **overhead vs no-intervention baseline** (per backend) · **overhead vs raw-PyTorch-hooks
  baseline** (HF only — the existing micro-benchmark is the L0 floor) · **transfer volume** (NDIF) ·
  time-to-first-save.
- **Correctness:** (1) coverage (runs?) · (2) cross-backend numerical equivalence (HF-eager oracle
  + tolerance — the "same trace, same answer" claim) · (3) regression vs golden values.

## 9. Cross-cutting concerns

- Baselines: no-intervention generation (per backend) + raw-hooks (HF).
- Determinism: seeds, warmup, repeated trials, cold-vs-warm (model-load excluded by default).
- Tiered run profiles: GPT-2/CPU **smoke tier** for CI vs full **A100 tier**; the full cartesian
  product is huge — log any truncation/sampling, never silently cap coverage.
- Resource teardown: tear down GPU workers / vLLM procs between sweep cells; verify no orphans.

## 10. Open forks / decision log

| # | Decision | Options | Lean | Status |
|---|---|---|---|---|
| F1 | Spec form | data / code / **hybrid** | hybrid (declarative spec compiled by small builders) | proposed |
| F2 | v1 scope | causal-LM observe/steer/patch vs +training +diffusion/VLM | causal-LM, Method+some Macro, full backend sweep; training & diffusion/VLM additive later | **awaiting confirm** |
| F3 | Resolver vocab resolution | (a) block-level / (b) +sub-block / (c) +source | (b) | **awaiting confirm** |
| F4 | Correctness goal | coverage-only vs +numerical equivalence | +equivalence (HF oracle + tolerance) — it's the OSDI claim | proposed |
| F5 | Adopt pyvene vocab + causalab harness shape | yes / build fresh | adopt/align both | proposed |
| F6 | Repo name | provisional `interp-serve-bench` | finalize later (avoid "InterpBench") | open |
