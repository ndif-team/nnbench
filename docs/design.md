# Design — interp-serve-bench (living document)

> Status: evolving. Captures decisions as they're made in the design conversation.
> Last structural update: positioning + taxonomy + resolver decouple + reference synthesis.

## 1. Purpose & positioning

A **systems performance + coverage benchmark** for interpretability workloads on nnsight,
swept across serving backends, model types/architectures, and parallelism/optimization configs.

It answers, for each (workload × model × backend × config) cell:
1. **Is it applicable, and if not, HOW does it fail?** → the **applicability map** (primary
   user-facing deliverable; see §8.1). Multi-valued, not binary — the point is a *guideline* for
   users ("for this access pattern, vLLM+nnsight is/ isn't usable, and here's the failure mode").
2. **Is it correct?** → numerical equivalence vs an HF-eager reference oracle, within tolerance.
3. **Is it fast?** → latency, throughput, peak memory, overhead vs baselines, transfer volume.

**Philosophy (decided):** the corpus is *not* scoped to the runnable-everywhere intersection. It
**deliberately includes non-portable workloads as frontier markers** — the boundary IS the
result. This systematizes scattered tribal knowledge (session-gap, in-place-write failures, PP
tuple-read returning wrong values) into one generated, verified artifact.

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

- **Backend (engine axis)**: **v1 = HF Transformers · vLLM-async** (vLLM-sync + NDIF remote deferred)
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
- **Resolution tiers** (a binding is `(module_ref, slice|None, access_kind)`). Vocabulary spans
  all tiers; v1 build/expectation differs per tier (see F3):
  - **(a) block-level** (`residual_{pre,post}`, `attn_out`, `mlp_out`) — `module.output` directly;
    mechanical from tree + config. *Portable, perf workhorse.*
  - **(b) sub-block** (`head[h]`, `neuron[j]`) — config-driven reshape/slice of a module
    output/input (`attn.output[0].view(B,S,n_heads,head_dim)`; `mlp.act.output[...,j]`); needs
    `config.{num_attention_heads, head_dim, num_key_value_heads (GQA), intermediate_size}`. **No
    `.source`.** *Portable, perf workhorse.* nnsight idiom: `docs/patterns/per-head-attention.md`.
  - **(c) intermediate-op** (`attn_weights`, pre-softmax scores) — needs `.source.<op>` (op name
    differs per family, e.g. `attention_interface_0`) **+ `attn_implementation="eager"`**.
    Attention weights do **not** materialize under SDPA/FlashAttention → **cannot exist on vLLM**.
    v1: implemented **HF-only as frontier markers**. nnsight idiom:
    `docs/patterns/attention-patterns.md`.
  - **runtime/engine** (`logits`, `sampled_token`) — backend-specific, not family-specific.
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

## 8. Metrics & deliverables

### 8.1 Applicability map (PRIMARY deliverable)

For each (workload × backend × config), a multi-valued state — the user guideline:

| State | Meaning | Detected by |
|---|---|---|
| `SUPPORTED` | runs, matches HF reference within tolerance | oracle pass |
| `SUPPORTED_DEGRADED` | correct, but forced a de-opt (disables CUDA graphs / prefix cache) | runs + perf delta |
| `ERROR` | raises a clear, catchable error (user gets a signal) | exception captured |
| `SILENTLY_WRONG` | runs, no error, but fails the oracle — **the dangerous cell** | oracle mismatch |
| `HANG` | deadlocks | timeout |
| `UNSUPPORTED_BY_CONSTRUCTION` | value can't exist (flash-attn attention patterns) | declared + confirmed |

`SILENTLY_WRONG` is **only detectable with the equivalence oracle** → this makes F4 load-bearing,
not optional. Each non-portable workload carries an *expected* state; the runner verifies the
actual state matches (and flags when vLLM returns `SILENTLY_WRONG` where `ERROR` was expected).

### 8.2 Performance

End-to-end latency · throughput (tok/s, traces/s, prompts/s) · peak GPU memory · **overhead vs
no-intervention baseline** (per backend) · **overhead vs raw-PyTorch-hooks baseline** (HF only —
the existing micro-benchmark is the L0 floor) · **transfer volume** (remote, later) ·
time-to-first-save.

### 8.3 Correctness

(1) the applicability state above · (2) cross-backend numerical equivalence (HF-eager oracle +
tolerance — the "same trace, same answer" claim, and the `SILENTLY_WRONG` detector) ·
(3) regression vs golden values.

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
| F2 | v1 scope | — | causal-LM, Method+some Macro; **backends = HF + vLLM-async only**; training & diffusion/VLM & vLLM-sync/NDIF additive later | **DECIDED** |
| F3 | Resolver vocab resolution | (a)/(b)/(c) | **vocabulary spans all of (a/b/c)**; v1 *portable+perf* addressing = (a)+(b); (c) implemented **HF-eager-only as frontier markers** (run on vLLM expecting ERROR/UNSUPPORTED, verified) | **DECIDED** |
| F4 | Correctness goal | coverage-only vs +equivalence | **+equivalence — LOAD-BEARING**: it's the only `SILENTLY_WRONG` detector (§8.1), not just the OSDI claim | **DECIDED** |
| F5 | Adopt pyvene vocab + causalab harness shape | yes / build fresh | adopt/align both | proposed |
| F6 | Repo name | provisional `interp-serve-bench` | finalize later (avoid "InterpBench") | open |
