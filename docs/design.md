# Design — interp-serve-bench (living document)

> Status: evolving. Captures decisions as they're made in the design conversation.
> Last structural update: data-flow / control-flow split + cross-edge movement class (§3.1, §3.4).

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

## 3. The primitive model — leveled (rewritten 2026-06-11; revised 2026-06-12 to the language-level model: boundary-crossing data primitives, control quantifiers, source→destination edges, law-checking)

> **Supersedes the flat "L0 primitives" table.** That table mixed levels — implementation variants
> (WRITE-inplace), addresses (LOGITS), control flow (ITERATIVE), compositions (CROSS-PROMPT) and
> whole programs (TRAIN-INTERVENTION) sat as siblings — which made "what primitives do we have /
> what do workloads need / what does each backend support" unanswerable. The replacement is a
> leveled model, PL-style: a small closed core, an address space, idioms, compositions, programs —
> and an orthogonal execution **context**. Where each old row now lives:
>
> | old flat row | new home |
> |---|---|
> | READ one site | data op READ × Level 1 boundary site |
> | CACHE many | Level 2 fused primitive (READ × breadth + live-out) |
> | WRITE-replace / WRITE-inplace | data op WRITE; the replace/in-place split is a Level 1.5 realization |
> | CROSS-PROMPT | Level 2 cross-edge movement (inter-region communication via the host) |
> | ITERATIVE | control op: the loop (mirrors the decode loop) |
> | BACKWARD-attribution | data op grad (a read on the derivative graph) |
> | TRAIN-INTERVENTION | Level 3 program |
> | AUX-MODULE | meta-compute (demoted from Level 0, §3.1; its measured rows live as realization rows) |
> | EDIT | staging (the named residue — neither data nor control flow) |
> | LOGITS / SAMPLING | Level 1 engine-tier sites |
> | SOURCE | Level 1 internal-tier sites |
> | SAVE / transmit | Level 2 cross-edge movement (live-out of a region) |

**This model is a vocabulary for declaring footprints and indexing measurements** — metadata and
micro-cells. It is NOT a construction layer: cells stay flat and explicit (§12); nothing generates
intervention code from these declarations. That distinction is the §11 lesson — the Resolver died
because it *constructed* the experiments; the levels only *explain and index* them.

Normative definitions live here; the maintained per-context **status inventory** lives in
`interp-methods-catalog.md` (one copy, so the lists can't diverge again).

### 3.1 Level 0 — data primitives and control quantifiers (revised 2026-06-12)

Two orthogonal kinds of operation. **Data primitives** move values across the object/meta
boundary between a model run and the experiment program; **control quantifiers** determine which
runs exist and where in their coordinates the boundary crossings attach. A data-op instance is
located at **(op × site × scope position)** — the site (§3.2) is its SPACE coordinate, the scope
position its TIME/EXTENT coordinate.

**Data primitives — membership criterion (replacing the enumerated framing):** *an operation is
a data primitive iff it crosses the object/meta boundary.* Implementation ground truth: every
data primitive lowers to a Mediator boundary event (nnsight
`src/nnsight/intervention/interleaver.py:349-357` — VALUE / SWAP / SKIP / BARRIER / END /
EXCEPTION); the engine's own event protocol independently confirms the criterion — every user
surface lowers to read(VALUE) / write(SWAP) / skip / sync / end / error.

| op | semantics | nnsight surfaces |
|---|---|---|
| **read**(r, addr) | observe the value at an address of run r (VALUE event) | `.output`, `.input`, `.inputs`, `.source.<op>`, `logits`, `samples`, `tracer.result`; `t.grad` is a read on the derivative graph |
| **write**(r, addr, v) | the **counterfactual run**: r continues as if addr had value v (SWAP event; plus the in-place and skip realizations) | `.output = …`, `.input = …`, `x[:] = …`, `.skip(v)`, `samples = …`, `t.grad = …` |
| **grad**(r, addr; metric) | observe the derivative graph of run r at addr | `with metric.backward(): … x.grad` |

`grad` is kept a primitive (rather than folded into read-on-the-derivative-graph) because
methodologies treat it as one, and because the derivative graph may not exist at all in a
context (gradients are unavailable on vLLM, which runs in inference mode with no autograd) —
exactly the kind of status the inventory must carry.

**What the criterion demotes.** COMPUTE leaves Level 0: the trace body is real Python and the
model's parameters are readable constants, so arbitrary computation — including applying
external `nn.Module`s (probes/SAEs) and the model's own modules as functions — is *meta-level
code*, not a boundary crossing. The measured COMPUTE rows are NOT discarded: they become
realization rows of **meta-compute** in a context (aux compute needs `no_grad` on vLLM because
its activations are inference tensors; module-call unembed is guarded so the unembed must use the
weight matmul). `.save()` also stays out: it does
not act on a value at a site — it moves a value OUT of a region. It is a data **edge** (the
live-out), and lives with the other cross-edge movement in §3.4.

**Control quantifiers — membership criterion:** *a control construct is named iff it quantifies
boundary crossings over runs, steps, addresses, inputs, or run order.* Engine coupling is no
longer the membership test — it is a per-quantifier **property** (some quantifiers are
engine-coupled; some are host-side and free). This admits **sweep** — the host-side loop over
addresses, the control construct most methodologies use most — which the old engine-coupling
criterion wrongly excluded. Plain Python `if`/`for` over real tensors inside a trace remains
ordinary control flow, free since v0.5, and deliberately NOT in the inventory.

| quantifier | over | engine-coupled? | nnsight surface |
|---|---|---|---|
| **run** (region) | one run's existence/extent | yes (request lifecycle; steps=N; mode=fake; `tracer.stop()` truncation) | `model.trace(...)` / `generate` / `scan`; the backward region `with metric.backward():` |
| **step** | the run's unrolled time | yes (the decode loop) | `tracer.iter[0:N]` / `iter[:]` / `.all()` / `tracer.next()` (realizations, §3.3) |
| **sweep** | addresses | **no — host-side, free** | a plain Python loop; breadth (§3.4) is its measure |
| **dataset** | inputs (independent runs) | only in the batched realization | prompt lists; `tracer.invoke(...)` incl. the empty invoke |
| **adaptive** | future runs chosen from past results | no | host `while` / `if` over past results |
| **run DAG** | run ordering with data dependence | yes (multi-request grouping) | `model.session()`; two traces |
| (sync) | fork branches | yes | `tracer.barrier(n)` — a coordination construct whose only observable is its sharing edge (§3.4) |

`generate` and `scan` remain region PARAMETERS, not separate constructs: steps=N
(`max_tokens` / `max_new_tokens`) and mode=fake (`model.scan(...)`). **Staging** (`model.edit()`)
remains the named residue — neither a data primitive nor a quantifier: code defined once,
replayed into every future region; its observable is the replay edge (§3.4).

**Scope position, defined.** A data-op instance's scope position is its path in the engine-side
scope tree — exactly the coordinates the engine knows:

```
session  >  trace  >  invoke  >  step
(multi-      (one        (which      (which decode
 request      request     batch       iteration)
 grouping)    lifecycle)  member)
```

Host-side quantifiers (sweep, dataset-as-sequential-runs, adaptive) do **not** occupy scope
positions; they multiply *instances*, each of which has its own scope position. A footprint
writes scope position as the innermost engine scope the crossing fires in (e.g. `write@step`,
`read@invoke[1]`, `read@trace` default).

**Fork: program or context — resolved.** The dataset quantifier is a *program* element. Whether
it is realized as one batched trace (`invoke`) or N sequential traces is a *realization* (§3.3)
of the dataset quantifier. The *workload regime* (§5) stays context: it describes the input
distribution and concurrency the cell is run under. The dataset-lift law (§3.7) is what connects
the two realizations, and the batched-GPT-2 positions effect is a model-side failure of that law
(§3.6) — not a definitional ambiguity.

Why the engine-coupled quantifiers "feel complex" while the data primitives don't: their
complexity is **inherited from the serving engine**, not invented by nnsight. On HF — one
in-process forward loop — they are all trivial (the micro tier measures 13/13 SUPPORTED). On
vLLM they map onto real scheduler and request-lifecycle machinery, which is exactly where the
measured frontier sits (§3.4).

Orthogonality leak (note it, don't engineer around it): some quantifier parameters alter data
semantics rather than just routing — mode=fake makes read return fake values; steps=N makes a
site's denotation step-indexed.

### 3.2 Level 1 — the address space (sites)

A *site* is a **name** for a coordinate `(module-path × token-position × generation-step
[× batch-member])` of the run's dataflow graph (and its derivative extension). **Membership
criterion:** *a name belongs to the address space iff read/write/grad can target it and its
denotation is checkable per context* (exists? / denotes what? / writable?). Tiers, by depth:

| tier | sites | note |
|---|---|---|
| **engine** | `logits`, `samples` (sampled tokens), `tracer.result`, `generator.output`, `streamer.output` | runtime properties; backend-specific, not family-specific |
| **module boundary** | `.output` / `.input` / `.inputs` (the args+kwargs view) at any tree depth: model root → block → submodule (attn/mlp/norm) → leaf (linear/embedding) | the workhorse |
| **module internal** | `.source.<op>` — intermediate ops inside a forward (incl. nested `.source`) | op names are family-specific; existence is backend-specific |
| **derived (value-level)** | head *h*, neuron *j*, position *p* — **and subspace / direction-valued addresses** (DAS rotations, steering directions): a projection `(site, R)` is a derived NAME, realized as read∘project / project-write; the sweep quantifier ranges over them like any address | derived *names* in the address space, not Level-0 ops |
| **gradient space** | `.grad` of any of the above | exists only under the backward region (grad) |

Derived addresses are **names whose access path is a realization** (§3.3): a head-sliced write
is write × derived address, where the derived address's *realization* is
reshape-slice-reassemble at the boundary site. How a name is reached is its realization — this
dissolves the old site/op blur for derived sites.

A site name's **denotation is context-dependent** — the central Level-1 fact. Per context a site
has three properties: **exists?** (attention weights have no denotation under paged attention),
**denotes what?** (vLLM fused-residual blocks: "block output" exists but denotes
`(hidden, residual)` whose SUM is the residual stream — same name, different meaning, so a plain
read is silently wrong), **writable?**. The inventory (`interp-methods-catalog.md`) is required to carry the per-site
**writable?** property, not just existence/denotation.

### 3.3 Level 1.5 — realizations (idioms) — generalized 2026-06-12

**Definition (generalized):** *a realization is a concrete spelling of ANY language element —
data op, quantifier, edge, or address — such that all spellings have the same language-level
meaning.* Contexts differ in WHICH spelling works; the "documented working recipe per backend"
deliverable = a realization choice per element. Cell params like `mode=` / `unembed=` are
realization selectors, not arbitrary knobs.

**Membership criterion:** two surfaces are realizations of one element iff a language-level
semantics-preserving rewrite connects them (replacement ↔ in-place; bounded ↔ unbounded ↔
`next`; batched invoke ↔ sequential traces; barriered ↔ two-trace transfer; `out[0]` ↔
`out[0]+out[1]` for the same denotation).

| element | realizations |
|---|---|
| write | in-place `x[:] = …` / replacement (new tensor / whole tuple) / skip-with-value `.skip(v)` |
| meta-compute (unembed) | call the module `lm_head(h)` / use its weights `F.linear(h, W)` |
| meta-compute (aux) | bare / under `torch.no_grad()` |
| step quantifier | bounded `iter[0:N]` / unbounded `iter[:]` / `.all()` / `tracer.next()` |
| dataset quantifier | batched invoke (incl. the empty invoke) / sequential traces |
| run↔run transfer edge | barriered / un-barriered cross-invoke (CROSS_INVOKER push/pull) / two traces via the host / session variable |
| residual read (fused-residual families) | plain `out[0]` / fused sum `out[0] + out[1]` |
| read / live-out **value semantics** | alias (reference into the run's storage) / snapshot (clone-at-crossing) — correct value selected by the **engine memory model** (§3.6); under in-place buffer reuse the alias decays silently, so the snapshot realization is required. Runtime-checkable selector (`tensor.is_inference()`), not a backend branch; nnsight applies it automatically (clone-on-save — the vLLM intervention-gap where saved inference tensors must be cloned to survive buffer reuse) |
| read-before-write (user's own downstream write) | alias / clone-first (`before = x.clone().save()`) — distinct from the value-semantics row above: this guards against *your* later write to the same site, not the engine's buffer reuse |
| live-out edge (transport) | in-process / async streaming drain / over-the-wire (serve) |

Note the re-homing: "loop step-selector" and "cross-prompt transfer" were never Level-0 ops —
they are realizations of the **step quantifier** and of the **run↔run transfer edge**
respectively. With realization defined element-generically, the table no longer violates its own
definition.

### 3.4 Level 2 — entries; cross-edge data movement is the frontier class

**Definition.** The finite catalog of tuples **(data-op | edge) × address-tier × scope-position**,
each carrying a **realization coordinate** — what the methods-catalog tags (`read`, `write`,
`xprompt`, `grad`, `attn-weights`, …) were groping at. Examples: boundary read; internal read
(attention-weights = read × internal × attention); head-sliced write (write × derived address);
per-step steering (write × boundary × step); bulk cache (`tracer.cache` = read × breadth +
live-out, a fused Level-2 primitive); gradient attribution read (grad × boundary).

**The membership rule (resolving the 1.5/2 double-classification):** *realizations are
coordinates ON entries, never entries.* An inventory row is an L2 entry; where statuses differ
by spelling, the row splits by its realization coordinate. So bounded-vs-unbounded iteration is
ONE L2 entry (the loop-carried accumulation edge, step → run) with two realization values, of
which one is ERROR on vLLM; barrier-vs-two-trace is ONE entry (the run↔run transfer edge) with
realization values {barriered, two-trace, un-barriered, session-var}. A failure's *kind* (§3.6)
is then read off the coordinate that varies.

**Edge derivation (source → destination).** Every value crossing the boundary has a provenance
`(run, address, step)` and a destination; classifying meta-level dataflow edges by
source → destination scope gives a complete, small set:

| edge class | shape | canonical methodology |
|---|---|---|
| **observation** | read → emit | logit lens, probing data collection |
| **rewiring** | read → compute → write, same run, downstream | path patching; SAE/transcoder splice |
| **transplant** | read in run A → write in run B | activation patching / causal tracing |
| **injection** | meta-constant → write | steering, zero/mean ablation |
| **accumulation** | reads → meta-state → later write or analysis | mean ablation's mean; probe/SAE/DAS training (makes the `trained` tag structural) |
| **derivative** | grad → emit/compute | attribution patching, saliency |

The concrete movement table below is these edges viewed from below — the classic
compiler/parallel-systems notions, and naming them that way is not decoration: it predicts where
things break. **Rewiring and accumulation are the two edge classes with no measuring cell**
(roadmap).

| movement | edge | compiler name |
|---|---|---|
| `.save()` | region → caller (incl. streaming drain and over-the-wire on serve) | live-out |
| iter accumulation (`rows.append` per step) | step → region | loop-carried dependency |
| invoke value sharing (barriered or un-barriered) | fork branch ↔ fork branch | communication at fork/join |
| session variable flow | region → region | live across regions |
| cross-prompt transplant (two traces) | region → region via the host | inter-region communication |
| edit replay (and export/import persistence) | definition → every future region | staging |

**Enumerated membership** (making §3.5's completeness criterion computable): the L2 catalog is
**generated, not curated** — the cross product of the traverse's data ops and edges (§3.8) ×
address tiers (§3.2) × scope positions (§3.1), filtered to combinations any cataloged method's
footprint names. The inventory in `interp-methods-catalog.md` carries this enumeration; coverage
= footprint-needed entries minus probe-or-cell-exercised entries, mechanically computable.

**Measured concentration (micro tier + the construct-gaps diagnosis): cross-edge movement
carries most of the vLLM frontier** — loop-carried saves dropped (unbounded iter), fork/join
sharing broken (barrier), cross-region flow absent (session), staging replay unserializable
(edit) — plus one region mode (scan). Of the micro tier's data-op × site probes, 6/6 pass on
vLLM (bounded iteration, the seventh SUPPORTED probe, is an edge realization, not a site probe).
The non-edge failures are exactly the taxonomy's other kinds (§3.6): grad (op-level — gradients
are unavailable on vLLM's inference mode), attention-weights (site-level — they have no denotation
under paged attention), in-place write and module-call compute (realization-level — in-place
writes raise while whole-tuple replacement works, and the guarded `lm_head` call forces the weight
matmul), fused residual (denotation — the dual `(hidden, residual)` stream whose sum is the true
value). Mechanism for the edge concentration: data ops
execute inside one worker scope; edges must cross nnsight's process/serialization boundaries,
and on a production engine those boundaries are real.

Scaling parameters — **breadth** (#sites — explicitly the sweep quantifier's measure), **tensor
size** (hidden×seq×batch), **depth** (#tokens), **payload** (bytes saved/transferred),
**side-compute** (aux FLOPs) — are *measures on* Level-2 entries, never new entries.

### 3.5 Level 3 — methodologies (programs) and footprints

A methodology is a **meta-program over the language**: boundary crossings + quantifiers + edges,
plus a readout metric and semantic intent. The space is structured as **base programs ×
transformations** — step-lift (steering → generation-time steering), dataset-lift (single-input
→ batched/statistical form), aggregation (ablation → mean ablation), linearization (patching →
attribution patching; a *scientific approximation*, flagged as such, not an equivalence), and
amortization (logit lens → tuned lens; fixed direction → trained probe/SAE). Each registry entry
(§4) names its base program and the transformations applied. Membership criterion for a bench
cell: deterministic (greedy/argmax), oracle-checkable, and its footprint is derivable from its
program text.

**Footprint derivation is syntactic.** A cell's footprint is computed from its program text (or
declared and checked against it — the declaration is the contract, the text is the truth):

1. Every boundary crossing in the text contributes an L2 entry reference
   `(op, address-tier, scope-position, realization)`. E.g. `blk.output = (h + v, out[1])` inside
   `for step in tracer.iter[0:N]` contributes `write × boundary.block × step × replacement`.
2. Every quantifier contributes its name + realization (`step/bounded`, `dataset/batched-invoke`,
   `sweep` with its breadth).
3. Every meta-level dataflow edge contributes its source→destination class (observation /
   rewiring / transplant / injection / accumulation / derivative), plus the live-out for whatever
   is saved.
4. Modes (fake, steps=N) and semantics-relevant CONFIG values are recorded as entry parameters.

**Machine-readable form** — `footprint=[...]` metadata on `@cell` registrations: pure data,
consumed only by reporting/derivation; cells stay flat and explicit (§12/§12.6 untouched).
Footprint vocabulary is the **catalog's tag table** (the single registry; realizations attach
as `/`-suffixes); the site segment may refine to tier-level IDs (`boundary.block.output`,
`internal.attn-weights`) — stable metadata names; cells keep explicit per-family paths:

```python
@cell("gen_steering", family="gpt2", backend="vllm_async",
      footprint=[
        "write/boundary/step/replacement",
        "step/bounded",
        "injection",                 # meta-constant direction → write
        "loop-carried",              # per-step rows.append (a distinct edge from accumulation)
        "live-out",
      ])
```

**Completeness, restated:** the method tier is complete when (a) its programs' footprints
jointly cover every L2 entry any cataloged method needs — the gap between "needed by the
catalog" and "exercised by a cell" IS the roadmap, mechanically — AND (b) every edge type and
every transformation is exercised by at least one cell. (b) is the language-level requirement:
it is what makes the gen-steering cell "the step-lift transformation, verified" rather than just
another method.

### 3.6 Context — orthogonal to all levels — composition (∧), the failure-kind taxonomy, attribution

`family × backend × engine-config × parallelism × workload-regime` is the environment a program
runs in, not a level. **Engine mode (sync/async) is an explicit context coordinate** (barrier and
session statuses differ by it). **The engine memory model is a second explicit context
coordinate**: HF allocates fresh storage per forward (crossed values are de-facto snapshots);
vLLM reuses activation/KV buffers in place (crossed values alias live storage). This one
coordinate selects two realizations and produces a notable failure asymmetry — its **write face**
is the in-place-vs-replacement split (in-place *raises* on vLLM, loud, while whole-tuple
replacement works) and its **read face**
is the alias-vs-snapshot split (an un-cloned read *silently decays* — measured ref-vs-clone
divergence 64.6 / 1013.8, the clone-on-save inference-tensor protection — so the snapshot
realization, §3.3, is required and nnsight applies it automatically). The engine protects writes loudly and reads not at
all, which is why the read face needed an automatic fix. Every level has a *status in a context*;
statuses compose upward — and the composition operator is now defined:

- **Status order (worse = more severe):**
  `SUPPORTED < SUPPORTED_DEGRADED < UNSUPPORTED_BY_CONSTRUCTION < ERROR < HANG < SILENTLY_WRONG`.
- **∧ = max-severity:** the predicted status of a method in a context is the *worst* measured
  status among its footprint's L2 entries in that context
  (`SUPPORTED ∧ SUPPORTED_DEGRADED = SUPPORTED_DEGRADED`).
- **Per-realization:** a footprint names the realization the cell actually uses, so the lookup
  is per-realization — `write/.../replacement` composes against the replacement row (SUPPORTED
  on vLLM), never against "write in general". "Some realization exists" is a *recipe* statement,
  not a composition input.
- **Per-context:** engine mode is a context coordinate, so there is no merging problem — a
  barrier-using method inherits ERROR in `vllm_async` and SILENTLY_WRONG in `vllm_sync`, as two
  cells.
- **UNTESTED inputs:** any footprint entry with no measured row makes the prediction
  `UNDERIVABLE(missing: <entry>)` — never a guess. The UNDERIVABLE set is the probe queue
  generator (§3.7).

Failures classify into seven kinds. The taxonomy is now a **decision procedure** over the L2
coordinates (§3.4's membership rule): vary one coordinate of the failing entry and see what
changes — same entry, one realization red → realization-unsupported; all realizations red →
op/edge-unsupported; name missing → site-absent; name exists but red denotation →
denotation-mismatch.

| failure kind | level (varying coordinate) | measured example |
|---|---|---|
| operation unsupported | L0 × context | grad on vLLM (inference mode, no autograd) |
| site absent | L1 × context | attention weights have no denotation under paged/flash attention |
| denotation mismatch | L1 × context | vLLM fused residual: name exists, means `(hidden, residual)` whose sum is the stream, so a plain read is silently wrong |
| realization unsupported | L1.5 coordinate × context | in-place write raises while whole-tuple replacement works; `lm_head.forward` guarded so unembed uses the weight matmul |
| edge unsupported | L2 edge × context | the vLLM frontier: loop-carried saves dropped under unbounded iteration; fork/join sharing broken by the barrier; cross-region session flow absent; staging replay unserializable |
| mode unsupported | region parameter × context | scan (mode=fake) dies in input prep on vLLM |
| model-side law failure (formerly "regime effect") | context alone — a lifting law fails for the model itself | batched GPT-2 absolute positions: no primitive involved — the dataset-lift law (§3.7) is invalid for absolute-position families |

When a method cell's measured status disagrees with its ∧-prediction, attribute three ways:

- **model-side law failure** — the scientist's lift was invalid for this model
  (absolute positions under dataset-lift). Recorded as a per-cell expected override with
  `attribution: model`; it does NOT impeach the entry statuses or ∧.
- **implementation-side composition failure** — footprint entries interact through shared engine
  state (the upstream saves-clobbering class). Recorded `attribution: implementation`; generates
  a composition cell (the pair becomes a permanent regression row).
- **numerics** — bf16 near-ties; the dtype control decides (the `disambiguate_precision` fp32
  rerun, validated where the activation patch matches HF at fp32 but flips a bf16 near-tie,
  already implements this) → `SUPPORTED_DEGRADED`, `attribution: numerics`.

The model-side-law-failure row is why per-cell expected-state overrides exist: it is the class
that does NOT decompose through the levels, and the model makes it an explicit, interesting
category — with a definition (a named law failing, attributed to the model) instead of a vibe.

### 3.7 How the tiers (§2) use the model — micro checks entries, methods check laws

- **Micro tier = entry-checking:** measure Levels 0–2 per context — data-op support, site
  existence + denotation + writability, realization viability, **edge viability** (the
  cross-edge movements of §3.4 — empirically the rows that carry the frontier). Small (~a dozen
  rows per backend), and the right surface to version-stamp — the primitive-status map is the
  version-sensitive artifact, not every cell. **The UNDERIVABLE set (§3.6) is the probe queue:**
  the footprint entries blocking ∧-derivations are exactly what the micro tier probes next.
- **Method tier = law-checking.** ∧-derivation is only an *a priori* prediction; it silently
  assumes footprint entries compose independently. The transformations' **laws** are that
  assumption made explicit — method cells test laws; micro probes test entries:

  | law | claim | tested by | measured so far |
  |---|---|---|---|
  | step-lift | a per-forward intervention stays valid per step | base cell vs step-lifted cell, SAME backend | no direct test yet (base-vs-lifted comparison queued); the generation-time steering composition result (per-step replacement write inside bounded iteration holds on vLLM exactly) supports it only indirectly — it measured the lifted cell's cross-backend agreement, which is the composition row's confirmation |
  | dataset-lift | batched ≡ sequential per-prompt | batched-invoke cell vs sequential-traces cell | FAILS model-side for absolute-position families (batched GPT-2 positions) |
  | sweep-exchange | sweeping addresses inside one run ≡ one run per address (non-interacting reads) | multi-layer cell vs per-layer cells | untested as a named law (logit-lens cells implicitly assume it) |
  | composition | independent footprint entries compose (∧ is sound) | any method cell vs its ∧-prediction | one confirmation (the generation-time steering composition holds on vLLM exactly); known counterexamples upstream where entries share implementation state (nnsight batched-multitoken saves clobbering; PP iteration hang) |

  A method cell's verdict protocol: `predicted = ∧(footprint, measured map)`; run; compare.
  Agreement confirms both the entry statuses and the law instance ("statuses compose upward",
  confirmed at method tier by the generation-time steering composition result). Disagreement is a
  finding in itself and triggers the
  three-way attribution (§3.6); the runner's existing surprise mechanism catches it.
- When an upstream fix lands, the micro-tier row flips first and every dependent method cell flips
  with it — one cause, reported once.

### 3.8 The complete component list per level (traverse summary, 2026-06-12)

The full 75-row primitive traverse of nnsight's user-facing surface — with `file:line` evidence,
per-row inventory status, and the disposition of every row not yet measured — lives in
`drafts/design-revision-2026-06-12.md` Part 1. The compact normative answer to "what is the
complete component list per level":

| level | complete component list |
|---|---|
| **L0 data primitives** | read · write · grad (closed by the boundary-crossing criterion, §3.1) |
| **L0 control quantifiers** | run (region; params steps=N, mode=fake; truncation `stop()`) · step · sweep · dataset · adaptive · run DAG · (sync/barrier) |
| **residue** | staging (`edit`; replay + export/import persistence) |
| **L1 address tiers** | engine (`logits`, `samples`, `tracer.result`, `generator.output`, `streamer.output`) · boundary (`.output`/`.input`/`.inputs`) · internal (`.source.<op>`, incl. nested) · derived (head/neuron/position + subspace/direction) · gradient (`.grad`) |
| **L1.5 realizations** | per element, the §3.3 table (write ×3 · meta-compute ×2 · step ×3 · dataset ×2 · run↔run transfer ×4 · residual read ×2 · read/live-out value semantics: alias/snapshot · read-before-write clone-first · live-out transport ×3) |
| **L2 entries** | generated: (data ops + edges) × address tiers × scope positions, filtered to footprint-named combinations (§3.4); edges classified observation / rewiring / transplant / injection / accumulation / derivative |
| **L3 methodologies** | the registry (§4), each = base program × transformations (step-lift, dataset-lift, aggregation, linearization, amortization) |
| **meta-level (not in the language)** | COMPUTE (trace body is real Python; measured realization rows: aux compute needs `no_grad` on vLLM, and the guarded `lm_head` forces the weight matmul) · plain `if`/`for` · `save` (= the live-out edge) · harness/address-space machinery (`rename=`, enumeration, device mgmt) |
| **out-of-scope v1** | NDIF plane (remote/session-bundling/non-blocking/`tracer.local`/code shipping — deferred by the v1-scope decision) · deprecated iteration forms · non-greedy sampling (breaks the determinism criterion) · non-LM model classes (the v1-scope decision is causal-LM-only; the §6 model-type axis reserves the slot) |

Maintained per-context statuses for every inventoried component live ONLY in
`interp-methods-catalog.md` (single copy); rows added 2026-06-12 from the traverse are all
**UNTESTED** — no measured status is ever invented.

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
vocabulary and cross-framework portability. Made concrete 2026-06-12: the catalog's tag table
carries the edge-class ↔ pyvene-enum mapping (Collect=observation, Addition/Subtraction=injection,
Vanilla interchange=transplant, RotatedSpace=subspace rewiring, LoRA=staging); subspace addresses
are derived-tier Level-1 citizens (§3.2).

## 5. Context — workload regimes (how methodologies get run; the "dataset" distribution)

- **Interactive probe** — 1 trace, 1 prompt (notebook shape)
- **Batched analysis** — 1 motif × N prompts (patching/probing over a dataset)
- **Generation-time intervention** — steering across a multi-token decode (serving shape)
- **Bulk harvesting** — CACHE-many × large corpus, throughput-bound (SAE data collection)
- **Multi-tenant / concurrent** — many independent traces vs one engine (nnsight-serve / NDIF)

Regimes are context (§3.6), not levels — a regime can change *verdicts* (the batched
absolute-position model-side law failure, formerly "regime effect"), which is why each workload
is oracle-checked in its own regime.
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
  all tiers; v1 build/expectation differs per tier (see the Resolver-vocabulary-resolution decision):
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

`SILENTLY_WRONG` is **only detectable with the equivalence oracle** → this makes the
equivalence-oracle correctness goal load-bearing, not optional. Each non-portable workload carries an *expected* state; the runner verifies the
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

| Decision | Options | Lean | Status |
|---|---|---|---|
| Spec form | data / code / **hybrid** | hybrid: spec=data, builder=code, profile=data (see §11.1) | **DECIDED** |
| v1 scope | — | causal-LM, Method+some Macro; **backends = HF + vLLM-async only**; training & diffusion/VLM & vLLM-sync/NDIF additive later | **DECIDED** |
| Resolver vocab resolution | (a)/(b)/(c) | **vocabulary spans all of (a/b/c)**; v1 *portable+perf* addressing = (a)+(b); (c) implemented **HF-eager-only as frontier markers** (run on vLLM expecting ERROR/UNSUPPORTED, verified) | **DECIDED** |
| Correctness goal | coverage-only vs +equivalence | **+equivalence — LOAD-BEARING**: it's the only `SILENTLY_WRONG` detector (§8.1), not just the OSDI claim | **DECIDED** |
| Adopt pyvene vocab + causalab harness shape | yes / build fresh | adopt: pyvene names→`FamilyProfile`; causalab Hydra config groups (§11.10) | **DECIDED** |
| Repo name | provisional `interp-serve-bench` | finalize later (avoid "InterpBench") | open |

---

## 11. Detailed design — Workload spec + Resolver interface

> **SUPERSEDED by §12 (2026-06-06).** The Resolver / FamilyProfile / Binding / predict /
> BackendCtx abstraction below was built and shown to be the wrong tool for a *benchmark*:
> it bakes in nnsight's "one trace runs everywhere" thesis, which is the very thing the
> benchmark must *measure*, not assume — and it leaked at every backend quirk (no_grad,
> lm_head guard, flat buffer, vocab padding) in a single motif. Kept for history; the live
> design is §12. Original text follows.

This is the load-bearing interface. It resolves the **spec-form decision (hybrid)** and the
**pyvene/causalab-adoption decision** concretely.

### 11.1 Three artifacts — the data/code split (spec form = hybrid)

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

### 11.3 Logical target vocabulary (causal-LM; pyvene-aligned)

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

### 11.10 Package + config-group layout (causalab-aligned)

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

**Spec form → hybrid (resolved):** spec=data, builder=code, profile=data.
**pyvene/causalab adoption → adopt (resolved):** vocabulary aligned to pyvene component names + `type_to_module_mapping`
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
  measured maps in `results/micro_{hf,vllm_async}.txt` — the micro-tier construct findings
  (portable Level-1 sites, and the iteration, barrier, session, and edit/scan construct results).
- Per-cell `expected` entries remain for **model-side law failures** (§3.6; formerly "regime
  effects") — the non-decomposable residue, now an explicit category rather than entries
  indistinguishable from derivable consequences.
- The maintained Level-0/1 status inventory (measured vs UNTESTED per backend) lives in
  `interp-methods-catalog.md`; the UNTESTED rows there are the coverage queue.
