# Design — interp-serve-bench (living document)

> Status: evolving. Captures decisions as they're made in the design conversation.
> Last structural update: leveled primitive model (§3 rewritten; §§4–6 retitled — levels vs context).

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
**optimization** (continuous batching, CUDA graphs, prefix caching). Our **sweep matrix (§6)
instantiates those axes**, so the benchmark is the empirical backbone of the systems story.

## 2. Organizing spine — granularity tiers (map 1:1 onto existing artifacts)

| Tier | = | Seeded from | Role |
|---|---|---|---|
| **Micro** | Levels 0–2 of the primitive model (§3), measured per context | `nnsight/tests/performance/benchmark_interventions.py` | support/denotation map + overhead floor |
| **Method** | one Level-3 program (§3.5) | nnsight-website `tutorials/` | canonical units |
| **Macro** | end-to-end paper repro | nnsight-website `mini-papers/` | "real research runs on this" |

Adding work = adding rows/registry entries, never a harness redesign.

## 3. The primitive model — leveled (rewritten 2026-06-11)

> **Supersedes the flat "L0 primitives" table.** That table mixed levels — implementation variants
> (WRITE-inplace), addresses (LOGITS), control flow (ITERATIVE), compositions (CROSS-PROMPT) and
> whole programs (TRAIN-INTERVENTION) sat as siblings — which made "what primitives do we have /
> what do workloads need / what does each backend support" unanswerable. The replacement is a
> leveled model, PL-style: a small closed core, an address space, idioms, compositions, programs —
> and an orthogonal execution **context**. Where each old row now lives:
>
> | old flat row | new home |
> |---|---|
> | READ one site | Level 0 READ × Level 1 boundary site |
> | CACHE many | Level 2 fused primitive (READ × breadth + SAVE) |
> | WRITE-replace / WRITE-inplace | Level 0 WRITE; the replace/in-place split is a Level 1.5 realization |
> | CROSS-PROMPT | Level 2 composition (READ in trace A → WRITE in trace B) |
> | ITERATIVE | Level 0 control construct (iteration) |
> | BACKWARD-attribution | Level 0 BACKWARD |
> | TRAIN-INTERVENTION | Level 3 program |
> | AUX-MODULE | Level 0 COMPUTE |
> | EDIT | Level 0 control construct |
> | LOGITS / SAMPLING | Level 1 engine-tier sites |
> | SOURCE | Level 1 internal-tier sites |
> | SAVE / transmit | Level 0 SAVE |

**This model is a vocabulary for declaring footprints and indexing measurements** — metadata and
micro-cells. It is NOT a construction layer: cells stay flat and explicit (§12); nothing generates
intervention code from these declarations. That distinction is the §11 lesson — the Resolver died
because it *constructed* the experiments; the levels only *explain and index* them.

Normative definitions live here; the maintained per-context **status inventory** lives in
`interp-methods-catalog.md` (one copy, so the lists can't diverge again).

### 3.1 Level 0 — core operations (closed set; should essentially never grow)

| op | semantics | nnsight surface |
|---|---|---|
| **READ** | observe the value at a site | `site.output` / `.input` (real tensor inside the trace) |
| **WRITE** | set the value at a site | `site.output = …` / `x[:] = …` (realizations differ, §3.3) |
| **COMPUTE** | run arbitrary torch/Python on live values inside the trace — incl. applying external `nn.Module`s (probes/SAEs) and the model's own modules as functions | plain code in the trace body |
| **SAVE** | move a value across the trace boundary (to the user; over the wire on serve/remote) | `.save()` |
| **BACKWARD** | reverse-mode flow; equivalently READ in gradient space | `with t.backward(): … x.grad` |

**Control / structure constructs** — same level, different kind: they don't touch values, they
shape *when and in what scope* the ops run:

| construct | shapes | nnsight surface |
|---|---|---|
| trace | the base scope: one forward / one generate | `model.trace(...)` |
| invoke | multi-prompt scoping within one trace (the batched regime) | `tracer.invoke(...)` |
| iteration | the decode-time axis: ops per generation step | `tracer.iter[...]` / `.all()` / `.next()` |
| barrier | cross-invoke synchronization (value sharing) | `tracer.barrier(n)` |
| session | multi-trace scoping (variables flow across traces) | `model.session()` |
| edit | persistent model modification outside any trace | `model.edit()` |
| scan | shape-only execution mode | `model.scan(...)` |

### 3.2 Level 1 — the address space (sites)

A *site* is a **name** for a value READ/WRITE can target. Tiers, by depth:

| tier | sites | note |
|---|---|---|
| **engine** | `logits`, sampled tokens | runtime properties; backend-specific, not family-specific |
| **module boundary** | `.output` / `.input` at any tree depth: model root → block → submodule (attn/mlp/norm) → leaf (linear/embedding) | the workhorse |
| **module internal** | `.source.<op>` — intermediate ops inside a forward | op names are family-specific; existence is backend-specific |
| **derived (value-level)** | head *h*, neuron *j*, position *p* | NOT new reads — READ ∘ COMPUTE(view/slice); derived *names* in the address space, not Level-0 ops |
| **gradient space** | `.grad` of any of the above | exists only under BACKWARD |

A site name's **denotation is context-dependent** — the central Level-1 fact. Per context a site
has three properties: **exists?** (attention weights have no denotation under paged attention),
**denotes what?** (vLLM fused-residual blocks: "block output" exists but denotes
`(hidden, residual)` whose SUM is the residual stream — same name, different meaning, F-7),
**writable?**.

### 3.3 Level 1.5 — realizations (idioms)

One abstract op, several concrete spellings; contexts differ in WHICH spelling works. The
"documented working recipe per backend" deliverable = the realization of each abstract op that
works in that context. Cell params like `mode=` / `unembed=` are realization selectors, not
arbitrary knobs.

| abstract op | realizations |
|---|---|
| WRITE | in-place `x[:] = …` vs replacement (new tensor / whole tuple) |
| COMPUTE (unembed) | call the module `lm_head(h)` vs use its weights `F.linear(h, W)` |
| COMPUTE (aux) | bare vs under `torch.no_grad()` |
| cross-prompt transfer | two single-prompt traces vs barrier within one trace |

### 3.4 Level 2 — derived primitives (compositions)

The finite catalog of **op × site-tier × time × realization** combinations, plus compositions
across ops — what the methods-catalog tags (`read`, `write`, `xprompt`, `grad`, `attn-weights`, …)
were groping at. Examples: boundary read; internal read (attention-weights = READ × internal ×
attention); head-sliced write (WRITE × derived site); cross-prompt transplant (READ in trace A →
WRITE in trace B); per-step steering (WRITE × iteration); bulk cache (`tracer.cache` = READ ×
breadth + SAVE, a fused Level-2 primitive); gradient attribution read (BACKWARD + READ × grad
space).

Scaling parameters — **breadth** (#sites), **tensor size** (hidden×seq×batch), **depth**
(#tokens), **payload** (bytes saved/transferred), **side-compute** (aux FLOPs) — are *measures on*
Level-2 entries, never new entries.

### 3.5 Level 3 — methodologies (programs) and footprints

A methodology is a small program over Level-2 primitives plus a readout metric and semantic
intent. Each program has a **footprint**: the set of Level-2 entries (with realizations) it needs.
The catalog's per-method tag rows are footprints. This gives coverage a definition: the method
tier is complete when its programs' footprints jointly cover every Level-2 entry any cataloged
method needs — the gap between "needed by the catalog" and "exercised by a cell" IS the roadmap,
mechanically.

### 3.6 Context — orthogonal to all levels — and the failure-kind taxonomy

`family × backend × engine-config × parallelism × workload-regime` is the environment a program
runs in, not a level. Every level has a *status in a context*; statuses compose upward, and
failures classify into five kinds (which flat per-cell expected entries cannot distinguish):

| failure kind | level | measured example |
|---|---|---|
| operation unsupported | L0 × context | BACKWARD on vLLM (inference mode) — F-11 |
| site absent | L1 × context | attention weights under paged/flash attention — F-10 |
| denotation mismatch | L1 × context | vLLM fused residual: name exists, means something else — F-7 |
| realization unsupported | L1.5 × context | in-place WRITE raises (F-5); `lm_head.forward` guarded (F-2) |
| regime effect | context alone | batched GPT-2 absolute positions: no primitive involved — the model's own semantics change under the regime |

The regime-effect row is why per-cell expected-state overrides exist: it is the class that does
NOT decompose through the levels, and the model makes it an explicit, interesting category instead
of an exception that embarrasses the abstraction.

### 3.7 How the tiers (§2) use the model

- **Micro tier** = measure Levels 0–2 per context: op support, site existence + denotation,
  realization viability. Small (~a dozen rows per backend), and the right surface to
  version-stamp — the primitive-status map is the version-sensitive artifact, not every cell.
- **Method tier** = verify Level-3 programs per context. Expected states become *derivable*
  (footprint ∧ measured support) except the explicitly-marked regime effects; the runner's
  existing surprise mechanism catches wrong derivations.
- When an upstream fix lands, the micro-tier row flips first and every dependent method cell flips
  with it — one cause, reported once.

## 4. Methodology registry — Level-3 programs (seed, not ceiling)

A **living registry with a fixed schema**: seeded from the nnsight tutorials, extended from the
literature; adding a motif = one registry entry, never a harness change.

Seeded (from website tutorials): logit lens · tuned lens · diffusion lens · activation patching /
causal tracing · attribution patching · DAS / interchange interventions · steering / ActAdd ·
ablation · SAE extract/steer · dictionary learning (SAE *training*) · linear probing · function
vectors · LoRA / edit · attention / per-head · DLA · activation harvesting.

Backlog (literature, additive later): path/edge patching · attention knockout · EAP/ACDC ·
patchscopes · future lens · SAE family (gated/JumpReLU/top-k/transcoders/crosscoders) · steering
family (CAA/ITI/RepE) · sparse probing / CCS · integrated gradients.

**Borrow:** align methodology footprints (§3.5) with **pyvene's intervention-type enum** (Vanilla /
Addition / Subtraction / Zero / Collect / RotatedSpace=DAS / LoRA) where they map, for shared
vocabulary and cross-framework portability.

## 5. Context — workload regimes (how methodologies get run; the "dataset" distribution)

- **Interactive probe** — 1 trace, 1 prompt (notebook shape)
- **Batched analysis** — 1 motif × N prompts (patching/probing over a dataset)
- **Generation-time intervention** — steering across a multi-token decode (serving shape)
- **Bulk harvesting** — CACHE-many × large corpus, throughput-bound (SAE data collection)
- **Multi-tenant / concurrent** — many independent traces vs one engine (nnsight-serve / NDIF)

Regimes are context (§3.6), not levels — a regime can change *verdicts* (the batched
absolute-position regime effect), which is why each workload is oracle-checked in its own regime.
The regime axis exercises the **engine axis** + concurrency — where vLLM should win and HF should
struggle.

## 6. Context — sweep matrix (system under test = the OSDI three axes as config)

- **Backend (engine axis)**: **v1 = HF Transformers · vLLM-async** (vLLM-sync + NDIF remote deferred)
- **Parallelism (distribution axis)**: single-GPU · TP=2/4/8 · PP · multi-node
- **Engine config (optimization axis)**: enforce_eager vs CUDA graphs · prefix-cache on/off
- **Model type**: causal-LM · diffusion · vision-language  *(first-class axis — "types")*
- **Architecture / scale**: GPT-2 → Llama/Qwen/Gemma/Mistral (7–9B) → 70B → MoE/frontier;
  deliberately include **non-standard module names** to prove the Resolver isn't hardcoded.

## 7. Harness — the spec → resolver → builder → runner → oracle → reporter pipeline

> **SUPERSEDED by §12** (same supersession as §11 — the Resolver/builder stages were dropped; the
> runner → oracle → reporter tail survives in §12.3). Kept for history.

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
the existing micro-benchmark is the Micro-tier floor) · **transfer volume** (remote, later) ·
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
| F1 | Spec form | data / code / **hybrid** | hybrid: spec=data, builder=code, profile=data (see §11.1) | **DECIDED** |
| F2 | v1 scope | — | causal-LM, Method+some Macro; **backends = HF + vLLM-async only**; training & diffusion/VLM & vLLM-sync/NDIF additive later | **DECIDED** |
| F3 | Resolver vocab resolution | (a)/(b)/(c) | **vocabulary spans all of (a/b/c)**; v1 *portable+perf* addressing = (a)+(b); (c) implemented **HF-eager-only as frontier markers** (run on vLLM expecting ERROR/UNSUPPORTED, verified) | **DECIDED** |
| F4 | Correctness goal | coverage-only vs +equivalence | **+equivalence — LOAD-BEARING**: it's the only `SILENTLY_WRONG` detector (§8.1), not just the OSDI claim | **DECIDED** |
| F5 | Adopt pyvene vocab + causalab harness shape | yes / build fresh | adopt: pyvene names→`FamilyProfile`; causalab Hydra config groups (§11.10) | **DECIDED** |
| F6 | Repo name | provisional `interp-serve-bench` | finalize later (avoid "InterpBench") | open |

---

## 11. Detailed design — Workload spec + Resolver interface

> **SUPERSEDED by §12 (2026-06-06).** The Resolver / FamilyProfile / Binding / predict /
> BackendCtx abstraction below was built and shown to be the wrong tool for a *benchmark*:
> it bakes in nnsight's "one trace runs everywhere" thesis, which is the very thing the
> benchmark must *measure*, not assume — and it leaked at every backend quirk (no_grad,
> lm_head guard, flat buffer, vocab padding) in a single motif. Kept for history; the live
> design is §12. Original text follows.

This is the load-bearing interface. It resolves **F1 (hybrid)** and **F5 (adopt pyvene vocab +
causalab harness)** concretely.

### 11.1 Three artifacts — the data/code split (F1 = hybrid)

| Artifact | Form | Who owns family knowledge | Analogy |
|---|---|---|---|
| **Workload spec** | DATA (YAML → validated dataclass) | none (logical only) | pyvene `IntervenableConfig`; causalab task/analysis YAML |
| **Motif builder** | CODE (registered fn) | none (consumes resolved bindings) | pyvene intervention type; causalab analysis |
| **Family / Backend profile** | DATA (declarative map) | **all of it** | pyvene `type_to_module_mapping` |

Invariant restated structurally: only **profiles** name concrete paths. Specs and builders speak
the logical vocabulary; a builder that types `transformer.h` cannot pass review *because it has no
way to obtain that string* — it only ever receives `Binding`s.

### 11.2 Workload spec schema (the "dataset" unit)

```python
# isb/spec/schema.py  — pydantic for validation; serialized as YAML on disk
class Selector(BaseModel):
    target: TargetKind                       # logical target (see 11.3)
    scope:  Scope = "all"                    # "all" | [int,…] | {start,stop,step} | {fraction}
    head:   int | Literal["all"] | list[int] | None = None     # tier (b)
    neuron: int | Literal["all"] | list[int] | None = None     # tier (b)
    position: int | Literal["last","all"] | list[int] = "all"
    access: AccessKind = "read"              # read|write_replace|write_inplace|cache|grad

class Workload(BaseModel):
    id: str
    motif: str                               # registered builder id
    tier: Literal["micro","method","macro"]
    profile: WorkloadProfile                 # interactive|batched|generation|harvesting|concurrent
    selectors: list[Selector]
    aux: list[AuxSpec] = []                  # unembed | sae | probe  (frozen|trainable)
    inputs: Inputs                           # single|list|dataset; pairs(counterfactual); chat
    generation: Generation = Generation()    # new_tokens (0=forward); per_step
    params: dict = {}                        # motif-specific knobs
    expect: dict[str, AppState] = {}         # per-backend expected state; missing ⇒ predicted (11.6)
```

### 11.3 Logical target vocabulary (causal-LM; pyvene-aligned, F5)

| Logical target | Tier | pyvene name | nnsight access (resolved) |
|---|---|---|---|
| `block.output` / `block.input` | a | `block_output/input` | `block.output[0]` (residual stream) |
| `attn.output` | a | `attention_output` | `attn.output[0]` |
| `mlp.output` | a | `mlp_output` | `mlp.output` |
| `attn.head_value[h]` | b | `head_attention_value_output` | reshape `o_proj/c_proj` **input** → slice head |
| `mlp.neuron[j]` | b | `mlp_activation` | slice `mlp.act.output` / `down_proj.input` |
| `attn.weights` | c | — | `attn.source.<op>.output[1]` (**eager only**) |
| `logits` / `sampled_token` | runtime | — | backend hookable property |

### 11.4 FamilyProfile (DATA) — same spec, different binding

```python
GPT2 = FamilyProfile(type="causal_lm", family="gpt2",
  paths={"block":"transformer.h.{i}", "attn":"transformer.h.{i}.attn",
         "attn_oproj":"transformer.h.{i}.attn.c_proj", "mlp":"transformer.h.{i}.mlp",
         "mlp_act":"transformer.h.{i}.mlp.act",
         "final_norm":"transformer.ln_f", "unembed":"lm_head"},
  output_index={"block":0, "attn":0, "mlp":None},          # tuple-output handling
  dims={"n_heads":"n_head","head_dim":None,"n_kv_heads":"n_head","ffn":"n_inner"},
  caps={"block.output","attn.output","attn.head_value","mlp.output","mlp.neuron",
        "attn.weights","logits"})

LLAMA = FamilyProfile(type="causal_lm", family="llama",
  paths={"block":"model.layers.{i}", "attn":"model.layers.{i}.self_attn",
         "attn_oproj":"model.layers.{i}.self_attn.o_proj", "mlp":"model.layers.{i}.mlp",
         "mlp_down":"model.layers.{i}.mlp.down_proj",
         "final_norm":"model.norm", "unembed":"lm_head"},
  output_index={"block":0, "attn":0, "mlp":None},
  dims={"n_heads":"num_attention_heads","head_dim":"head_dim",
        "n_kv_heads":"num_key_value_heads","ffn":"intermediate_size"},   # GQA-aware
  caps={…})

# Non-standard names MUST be a profile-only change (proves no hardcoding):
WEIRD = FamilyProfile(family="myllm",
  paths={"block":"decoder_blocks.{i}", …, "unembed":"output_projection"}, …)
```

### 11.5 Resolver interface (the ONLY model-aware code; shared, profile-parameterized)

```python
@dataclass
class Binding:
    site_id: str                 # stable label, e.g. "L9.attn.head_value[4]"
    module: Envoy                # resolved nnsight envoy
    output_index: int | None     # tuple index (0 for block/attn; None for mlp)
    reshape: tuple | None        # e.g. (B,S,n_heads,head_dim) for per-head
    index: tuple | None          # slice into reshaped tensor (head/neuron/position)
    access: AccessKind

class Resolver:
    def __init__(self, profile: FamilyProfile, model): ...
    def capabilities(self) -> set[str]: return self.profile.caps
    def resolve(self, sel: Selector) -> list[Binding]:
        if sel.target not in self.profile.caps:
            raise Unsupported(sel.target)                 # → predicted UNSUPPORTED state
        layers = self._scope_to_layers(sel.scope)         # all|list|range|fraction → [i,…]
        return [self._bind(sel, i) for i in layers]       # reshape/slice math from profile.dims
```

The slice/reshape math (per-head, per-neuron, GQA) lives **once** in `_bind`, reading head/ffn
dims by name from `profile.dims`. Adding a family = a new profile entry, never new code.

### 11.6 BackendProfile + applicability prediction (the map, computed a priori)

```python
HF         = BackendProfile("hf", caps=ALL | {"grad","write_inplace","source","attn.weights"})
VLLM_ASYNC = BackendProfile("vllm_async",
    caps={"block.output","attn.output","attn.head_value","mlp.output","mlp.neuron",
          "logits","sampled_token","write_replace","write_inplace"})   # no source/grad/weights

def predict(wl: Workload, fam: FamilyProfile, be: BackendProfile) -> AppState:
    need = motif_requires(wl.motif) | {s.target for s in wl.selectors} \
                                    | {s.access for s in wl.selectors}
    have = fam.caps & be.caps
    return SUPPORTED if need <= have else UNSUPPORTED_BY_CONSTRUCTION
```

`predict` gives the *a priori* cell; the runner then **empirically verifies** it (11.8) and flags
any disagreement (esp. predicted-SUPPORTED but oracle says `SILENTLY_WRONG`, or predicted-ERROR but
actually `SILENTLY_WRONG`).

### 11.7 Motif builder interface (family-independent)

```python
@motif("logit_lens", requires={"cache","aux"})
def logit_lens(wl, R: Resolver, model):
    sites = R.resolve(wl.selectors[0])                    # block.output across scope
    norm  = R.resolve_one("final_norm"); W = R.resolve_one("unembed")
    def program(tracer):
        out = {}
        for s in sites:
            h = read(s)                                   # helper: module.output[idx][…,pos]
            out[s.site_id] = save(apply(W, apply(norm, h)).softmax(-1))   # aux: family-independent
        return out
    return program
```

`read(binding)` / `write(binding, val)` are shared helpers translating a `Binding` into nnsight
access (`.output[idx]`, `.view(reshape)`, slice). Builders never see a path.

### 11.8 Verification flow (per workload × model × backend cell)

```
predicted = predict(wl, fam, be)
if predicted == UNSUPPORTED_BY_CONSTRUCTION and not run_negatives:   record predicted; next
try:      result = run(build(wl, Resolver(fam, model)), timeout)
except CleanError as e:   actual = ERROR(type(e))
except Timeout:           actual = HANG
else:
    ref    = oracle.reference(wl, model)          # HF-eager, matched dtype
    actual = SUPPORTED if equiv(result, ref, tol) else SILENTLY_WRONG
    if forced_deopt(be, wl):  actual = SUPPORTED_DEGRADED
record(cell, predicted, expected=wl.expect.get(be.name, predicted), actual, perf)
```

### 11.9 Worked examples (across tiers, with expected applicability)

```yaml
# logit_lens — tier a, read, forward-only → SUPPORTED on both
id: logit_lens.all_layers
motif: logit_lens
tier: method ; profile: interactive
selectors: [{target: block.output, scope: all, position: last, access: read}]
aux: [{kind: unembed}]
inputs: {kind: single, prompts: ["The Eiffel Tower is in"]}
generation: {new_tokens: 0}
---
# per-head patching (IOI) — tier b, write_replace, cross-prompt → SUPPORTED on both
id: head_patch.ioi
motif: head_patching
tier: method ; profile: batched
selectors: [{target: attn.head_value, scope: [9,10,11], head: all, access: write_replace}]
inputs: {kind: list, pairs: true, prompts: [...clean/corrupted...]}
params: {metric: logit_diff}
---
# attention-pattern read — tier c FRONTIER MARKER
id: attn_pattern.read
motif: attention_pattern
tier: method ; profile: interactive
selectors: [{target: attn.weights, scope: all, access: read}]
inputs: {kind: single, prompts: ["The cat sat on the"]}
expect: {hf: SUPPORTED, vllm_async: UNSUPPORTED_BY_CONSTRUCTION}   # flash-attn: no weights tensor
```

### 11.10 Package + config-group layout (causalab-aligned, F5)

```
isb/
  spec/       # Workload/Selector dataclasses + YAML loader            (data schema)
  resolve/    # FamilyProfile, BackendProfile, Resolver, Binding, predict()
  motifs/     # registered builders (logit_lens, head_patching, …)     (family-independent)
  backends/   # SUT instantiation: hf, vllm_async
  runner/     # sweep dispatch, warmup, timed trials, failure-tolerant exec (11.8)
  oracle/     # HF-eager reference + tolerance equivalence
  report/     # applicability map + perf tables + plots
  configs/
    workload/ model/ backend/ family/ runners/    # Hydra config groups; runner = a sweep preset
```

**F1 → hybrid (resolved):** spec=data, builder=code, profile=data.
**F5 → adopt (resolved):** vocabulary aligned to pyvene component names + `type_to_module_mapping`
(`FamilyProfile`); harness uses causalab's Hydra config-group shape (`workload/model/backend`).

---

## 12. Architecture (LIVE) — fixed per-cell methodologies

Supersedes §11. The benchmark serves **benchmark maintainers**, not arbitrary users, so the
model/family set is **curated and finite** — we never face an unknown architecture and therefore
do not need a general addressing abstraction (no `Resolver`, no `FamilyProfile`, no
`StandardizedTransformer`, no `predict`/caps). A cell is honestly specific, by design.

### 12.1 The matrix

The benchmark is a matrix of **cells**, one per `(methodology, family, backend [, variant])`. The
cell IS the workload AND the unit of failure. A cell is a small, explicit function — readable
top-to-bottom — that writes the real intervention code for that exact combination.

```python
@cell("logit_lens", family="gpt2", backend="hf")
def _(be, model, prompt):
    with be.trace(model, prompt) as t:
        rows = [model.lm_head(model.transformer.ln_f(blk.output[0]))
                for blk in model.transformer.h]
        saved = be.save(be.stack([be.last(r) for r in rows]))
    return be.collect(t, saved)
```

- **Family** is a real row dimension, but handled by *writing the cell*, not by abstraction. The
  Llama cell says `model.model.layers`; the GPT-2 cell says `model.transformer.h`. That is correct
  explicit code, not the hidden-hardcoding the project rule forbids (which is *general* code that
  secretly works for one convention). Cells still read real runtime state.
- **Backend** is the real divergence axis (vLLM ≠ HF). It shows up as separate cells
  (`backend="hf"` vs `backend="vllm_async"`) — explicit, where you can read the `no_grad` /
  weight-matmul / flat-buffer specifics.
- **Variants** (layers touched, overhead, idiomatic-vs-portable unembed) = params or sibling cells.
- **Reuse emerges bottom-up:** when two cells are structurally identical except module names,
  extract a `lens_core(be, blocks, norm, head, ...)` helper they both call with their *own*
  explicit modules. The helper never tries to be universal; the maintainer wires it per cell.
  Do not pre-build it.

### 12.2 Failure model & the per-family control

HF-of-the-same-family is the **control / oracle**. The interesting signal is the **backend-vs-HF
delta within a family**:

| HF (control) | vLLM | meaning |
|---|---|---|
| ✅ | ✅ | portable |
| ✅ | ✗ | **backend bug in vLLM's <family> path** (e.g. vLLM-Llama impl bug) |
| ✗ (HF-other-family ✅) | — | **family-specific architecture quirk** |
| ✗ across families | — | methodology-level issue |

**Consequence:** the oracle reference is per-`(methodology, family)` — the vLLM-Llama cell is
compared against **HF-Llama**, never against GPT-2. `family` is a *grouping key* in the runner,
not an abstraction in the code.

### 12.3 What is kept vs dropped from §11

- **Dropped:** `isb/resolve/` (Resolver, FamilyProfile, Binding, read_value, predict), the heavy
  `Workload`/`Selector` spec, `BackendCtx`.
- **Kept:** the applicability-map output + states (§8.1), the **oracle** (§8.3, now grouped by
  family), the **runner** verify→score→report flow, and the genuinely backend-specific *infra*
  (`be`: open trace, save, collect, last/stack, teardown — HF handle vs vLLM async `output.saves`).

### 12.4 Layout

```
isb/
  states.py             # AppState
  methodologies/
    registry.py         # @cell(methodology, family, backend, variant=...) -> fn ; lookup
    logit_lens.py       # the cells (+ a local helper if/when reuse appears)
  backends/
    hf.py vllm_async.py # `be` infra: trace/save/collect/last/stack/teardown + load
  oracle/equivalence.py # per-family reference comparison (unchanged)
  runner/run.py         # enumerate cells, group by (methodology,family), HF=control, score
  report/applicability.py
scripts/smoke.py        # enumerate the cells to run; print the map
```

### 12.5 Extension points

| Add a… | What you write |
|---|---|
| methodology | new file of `@cell` functions |
| (family,backend) cell | one explicit function for that combination |
| backend | a `be` infra impl (trace/collect/teardown) + its cells per methodology |
| variant | a param or a sibling `@cell` |

### 12.6 The primitive model as index (2026-06-11)

§3's leveled model is the benchmark's *vocabulary*, not an abstraction layer. Concretely:

- Cells stay flat and explicit; the model never constructs them.
- A cell/task may DECLARE its footprint (the Level-2 entries + realizations it needs) as metadata
  next to the registration — pure data, consumed only by reporting/derivation.
- The **micro tier** (§2, §3.7) is one minimal cell per (Level-0/1/1.5 entry × backend) — most
  extractable from existing cell code — producing the measured primitive-status map that
  (a) explains method-level expected states, (b) carries the version stamp (provenance attaches to
  ~a dozen primitive rows, not to every cell), and (c) flips first when an upstream fix lands —
  one cause, reported once, with every dependent method cell flipping alongside it.
  **Built 2026-06-11**: `isb/micro/probes.py` (one probe per row, self-contained denotation
  checks, watchdog HANG detection, vLLM probes ordered safest-first), `scripts/micro.py`;
  measured maps in `results/micro_{hf,vllm_async}.txt`, findings F-12..F-16.
- Per-cell `expected` entries remain for **regime effects** (§3.6) — the non-decomposable residue,
  now an explicit category rather than entries indistinguishable from derivable consequences.
- The maintained Level-0/1 status inventory (measured vs UNTESTED per backend) lives in
  `interp-methods-catalog.md`; the UNTESTED rows there are the coverage queue.
