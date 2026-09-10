# Interpretability methods — catalog & nnbench roadmap

A catalog of well-recognized interpretability methods, indexed by the shared CausaLab protocol
vocabulary for **what is intervened where** and by nnbench's separate execution details. The point
is to decide which become nnbench methodologies: a method is a cheap deterministic cell, a new
backend frontier, or a heavyweight needing a trained artifact based on the protocol elements and
realizations it touches.

All methods below are **deterministic** (greedy / argmax, no sampling) → oracle-checkable, which is
the inclusion criterion for an nnbench cell. The sampling-based verifier/oracle direction (BOA /
BEAVER) is parked separately.

## The primitive inventory — leveled (status per context)

The model (normative definitions in design.md §3, revised 2026-06-12 to the language-level
account): **Level 0** — data primitives by the boundary-crossing criterion (**read / write /
grad**) and control **quantifiers** (run / step / sweep / dataset / adaptive / run-DAG + sync;
staging as the named residue; engine coupling is a per-quantifier property, not the membership
test) · **Level 1** address space (sites; per-site exists? / denotes-what? / writable?) ·
**Level 1.5** realizations (of ANY language element) · **Level 2** entries —
(data-op | edge) × address-tier × scope-position with a realization coordinate, incl.
**cross-edge data movement** classified by source→destination (observation / rewiring /
transplant / injection / accumulation / derivative) · **Level 3** methodologies (base programs ×
transformations, with syntactic footprints) · **context** (family × backend × engine-config incl.
engine mode × parallelism × regime), orthogonal to all levels. COMPUTE is meta-level (not a
primitive); its measured rows live as meta-compute realization rows below.

This section is the single maintained **per-context status inventory**. Status values are
bench-measured; the findings themselves are described in `findings.md`. The original Level 0–1
row set is FULLY measured for (GPT-2 × hf/vllm_async) by the **micro tier** — one minimal probe
per row with a self-contained denotation check (`scripts/micro.py`, results in
`results/micro_{hf,vllm_async}.txt`). The 2026-06-12 full primitive traverse
(`design.md` §3.8; `drafts/design-revision-2026-06-12.md` Part 1) added the rows marked
**UNTESTED** below — they are the probe queue, and **no measured status is ever invented for
them**.

**The measured concentration: cross-edge data movement carries most of the vLLM frontier.** Of
the micro tier's data-op × site probes, 6/6 pass on vLLM; the non-edge failures are the
taxonomy's other kinds — gradients unavailable (op-level: no autograd in inference mode),
attention weights absent (site-level: paged attention exposes no probability matrix), in-place
write and module-call compute (realization-level: in-place writes raise, and the guarded
lm_head forces the weight matmul), and the fused residual (denotation-level: the dual residual
stream) — plus one region mode (scan errors cleanly on the vLLM path).

### Data operations × backend

| op | hf | vllm_async / vllm_serve | evidence |
|---|---|---|---|
| read (boundary `.output`/`.input`) | ✓ | ✓ — with per-family denotation caveats (see Level 1) | logit-lens cells; the fused-residual denotation mismatch |
| write (boundary, replacement) | ✓ (both realizations) | ✓ replacement only; in-place raises | the in-place-write restriction; steering/ablation/patching cells |
| write — input side (`module.input = x`) | UNTESTED | UNTESTED | traverse row; the natural form of transcoder-splice and input-side patching |
| write — module skip (`module.skip(replacement)`) | UNTESTED | UNTESTED | traverse row; SKIP is its own Mediator event — not derivable from the SWAP rows; also a perf primitive (elides the module's FLOPs) |
| write — gradient (`t.grad = g` mid-backward) | UNTESTED | UNTESTED | traverse row; vLLM plausibly inherits the no-autograd-on-vLLM result's ERROR but is recorded UNTESTED, never derived-as-measured |
| write — sampler forcing (`model.samples = ids`, vLLM engine site) | n/a (HF site differs) | UNTESTED | traverse row; substrate for constrained/verified decoding |
| grad (formerly BACKWARD) | ✓ (≈3.4× a single forward) | **ERROR** — inference mode, no autograd: the `grad` frontier | the no-autograd-on-vLLM result |
| grad, measured realizations (findings "the gradient class") | ✓ per-pair backward (attribution, 20 labeled pairs); ✓ training loop (DAS: an external rotation parameter receives grad through the frozen model, backward inside the trace); ✓ many-VJP sweep (jacobian collection: batched one-hot cotangents, grads at all 11 source layers per backward) | **ERROR** in all three: `requires_grad_` raises on inference tensors (attribute, vjp_batch); the DAS train step raises "Inference tensors cannot be saved for backward" at the first grad-tracking op. All fail-fast, forward-only: no backward over the async path | the gradient-class finding; train-on-HF / apply-on-vLLM is the working split (DAS seeded apply SUPPORTED_DEGRADED; collected-J transport top1=1.00 at fp32) |
| hooked aux application (`envoy(x, hook=True)` — routes a meta-compute through the boundary, making aux `.input`/`.output` addressable) | UNTESTED | UNTESTED | traverse row; the `ext-module` tag's observability half (SAE latent read requires it) |

(`.save()` is not a data op — it is the live-out EDGE; see cross-edge movement below. COMPUTE is
meta-level code — the trace body is real Python; its measured per-context statuses are the
meta-compute rows of the realizations table — the inference-tensor no_grad requirement and the
guarded lm_head call.)

### Control quantifiers × backend

Quantifiers per design.md §3.1 (membership = quantifies boundary crossings; engine coupling is a
property — sweep and adaptive are host-side and free, so they carry no status rows: sweep is
implicitly exercised by every multi-layer cell, its measure is breadth). The step / sync /
run-DAG / staging constructs are measured through their EDGES — their only observable is data
crossing them — so their rows live in the cross-edge table below.

| quantifier / region parameter | engine structure | hf | vllm_async | evidence |
|---|---|---|---|---|
| run (trace) | one request lifecycle | ✓ | ✓ | every cell |
| run, steps=N (multi-token decode) | the decode loop's extent | ✓ | ✓ — 8-step greedy decode, per-step logits match HF exactly (measured on the steered trajectory — the generation-time steering composition result; the plain decode appears only as its perf baseline) | micro iteration probes; gen_steering — the generation-time steering composition result |
| run, mode=fake (scan) | engine bypass | ✓ — shapes correct, no kernels | **ERROR** — dies building `SamplingParams` on scan's `hook=True` kwarg, before fake mode is entered | micro tier; the edit/scan clean-error result; construct-gaps §5 |
| run, early truncation (`tracer.stop()`) | request teardown mid-forward | UNTESTED | UNTESTED | traverse row; pure perf primitive (elide the rest of the forward) + a correctness question (are pre-stop saves preserved, does the engine survive?) |
| dataset, batched realization (invoke) | the batch | runs; **model-side law failure** (dataset-lift, formerly "regime effect") on absolute-position families — batched GPT-2 diverges from its own per-prompt truth (pending the position_ids fix) | **ERROR** — gated on the dev checkout (async multi-prompt fix pending) | spec `expected` entries; `finding-batched-position-ids.md` |
| dataset, empty-invoke realization (`tracer.invoke()` — full combined batch; the documented out-of-order/trailing-code escape hatch) | the batch | UNTESTED | UNTESTED | traverse row; the recommended fix for two gotchas deserves a measured status |
| step, manual-advance realization (`tracer.next()`) | the decode loop | UNTESTED | UNTESTED | traverse row; third spelling of the step quantifier (bounded/unbounded are measured — the unbounded-iteration saves-drop) |

### Cross-edge data movement × backend — THE FRONTIER

Data flowing across a control-flow edge (compiler names in parentheses — they predict where
things break: ops execute inside one worker scope, edges must cross nnsight's
process/serialization boundaries). Source→destination classes per design.md §3.4: observation /
rewiring / transplant / injection / accumulation / derivative. Per the L2 membership rule,
realizations are coordinates ON entries: bounded/unbounded are two realization rows of ONE
loop-carried entry; barriered / un-barriered / two-trace / session-var are realization rows of
ONE run↔run transfer entry.

| movement (edge) | hf | vllm_async | evidence |
|---|---|---|---|
| `.save()` — region → caller (live-out) | ✓ | ✓ — **conditional on the snapshot realization**: nnsight auto-clones inference-mode tensors on `.save()` (clone-on-save inference-tensor protection); without it the live-out is SILENTLY_WRONG (ref-vs-clone diff 64.6 / 1013.8 — vLLM reuses the buffer). In-process and over-the-wire (serve venue; payload measured) | all cells; serve sweep 22/22 (measured 2026-06-09/10; regenerate via `docker/run_vm.sh`) |
| live-out, async streaming-drain realization — `async for out in tracer.backend()`, saves only on `output.finished`, single-shot generator, request-order ≠ invoke-order | n/a | UNTESTED — it IS the bench's vllm_async transport, but its assumptions have no explicit measured row | traverse row |
| iter accumulation, bounded `iter[0:N]` — step → region (loop-carried) | ✓ | ✓ — 3 steps, step-0 == single-step trace | micro tier; the unbounded-iteration saves-drop |
| iter accumulation, unbounded `iter[:]` / `.all()` | ✓ (stop bound = `default_all` from max_new_tokens) | **ERROR** — ALL saves dropped: the vLLM path never sets a stop bound, the loop overruns and is unwound by Cancelation before the body's final push (the documented idiom is the broken one) | micro tier; the unbounded-iteration saves-drop; construct-gaps §1 |
| barrier value sharing — fork ↔ fork (communication at fork/join) | ✓ — barrier patch == two-trace patch | async: **ERROR** (no saves; stacks with the multi-prompt gate). Sync engine: **SILENTLY_WRONG** — clean exit, saved dict EMPTY (construct-gaps repros) | micro tier; the barrier sync/async split; nnsight `docs/developing/barrier-vllm-not-shared.md` |
| un-barriered cross-invoke flow — fork ↔ fork, automatic push/pull (gated by `CONFIG.APP.CROSS_INVOKER`; the spelling users hit *by accident*) | UNTESTED | UNTESTED — plausibly SILENTLY_WRONG-shaped risk, must be measured not assumed | traverse row |
| session saved flow — region → region (live across regions) | ✓ | async: **ERROR** (no drain point inside a captured session body). Sync engine: **works** (construct-gaps repros) | micro tier; the broken un-saved session flow; construct-gaps §3 |
| session un-saved flow (the session contract) | ✓ (\|Δ\|=0) | **ERROR** on both engines — only saves ship back from the worker; the surfaced UnboundLocalError misleadingly names the downstream variable | micro tier; the broken un-saved session flow; construct-gaps §3 |
| cross-prompt transplant — region → region via the host (inter-region communication) | ✓ | mechanism ✓ at fp32 (top1=1.00 TV=0.0006); default bf16 = **SUPPORTED_DEGRADED** (near-tie top-1 flip — the dtype caveat is load-bearing) | patching cells; the single-forward patch precision near-tie |
| edit replay — definition → every region (staging) | ✓ — edited trace == in-trace ablation; non-inplace edit isolated from the original | **ERROR** — the stored edit mediator fails to pickle into the vLLM worker (`PicklingError: source code unavailable`); the crash is protective — a serialization-only fix would silently drop the edit | micro tier; the edit/scan clean-error result; construct-gaps §4 |
| edit persistence — `export_edits` / `import_edits` (staging × serialization, across processes — the edit/scan clean-error failure class one level up) | UNTESTED | UNTESTED — probe deferred until staging matters on a second backend | traverse row |
| rewiring — same-run read→compute→write downstream (path patching; SAE/transcoder splice) | UNTESTED | UNTESTED | edge class named by four cataloged footprints; roadmap (path-patching / splice cell) |
| accumulation — reads across runs → meta-state → later write/analysis (mean ablation; trained probes/SAEs) | UNTESTED | UNTESTED | edge class; roadmap (mean-ablation cell — the cheapest trained-state proxy) |
| bulk cache (`tracer.cache(...)`) — read × breadth fused with live-out | UNTESTED | UNTESTED | the fused L2 primitive (design §3.4); roadmap top |

Of the six source→destination edge classes, **rewiring** and **accumulation** have no measuring
cell yet — their UNTESTED rows above are the roadmap's next edge cells.

**Engine-mode caveat:** the bench's vLLM backend is the **async** engine; "sync engine" statuses
above are measured via the standalone repros in nnsight `docs/developing/vllm-construct-gaps.md`
("construct-gaps"), verified on vllm 0.19.1 and 0.15.1 — sync is a context (engine-mode) axis the
inventory under-represents until a `vllm_sync` bench backend exists. Where sync and async differ,
the difference is itself the finding (barrier: silent vs loud; session saved-flow: works vs not).

### Level 1 — sites × backend (existence / denotation / writability)

Per design.md §3.2 the inventory carries the per-site **writable?** property; read-status rows
below imply nothing about writability — write statuses live in the data-op and cross-edge tables.

| site tier | hf | vllm_async | evidence |
|---|---|---|---|
| engine (`logits`, sampled tokens) | ✓ — `model.output.logits` == `lm_head.output`; greedy `generator.output` id == logits argmax | ✓ — `model.logits` == portable unembed; `model.samples` == greedy logits argmax | micro tier; the portable-sites result |
| engine — `tracer.result` (the recommended end-of-generation capture, preferred over `generator.output`) | UNTESTED | UNTESTED | traverse row |
| engine — HF per-step token stream (`model.generator.streamer.output`) | UNTESTED | n/a (HF-only site) | traverse row; probe deferred, fold into iteration probes |
| boundary `.output` — block | ✓ | exists, but **denotes differently** on fused-residual families (Llama/Mistral/Qwen2/Gemma): `(hidden, residual)`, true stream = their sum | the fused-residual denotation mismatch |
| boundary `.output` — submodule (attn/mlp) | ✓ | ✓ (ablation write target) | the ablation bf16 near-tie |
| boundary `.input` | ✓ — `h[6].input` == `h[5].output[0]`, exact | ✓ — same check, exact | micro tier; the portable-sites result |
| boundary kwargs sub-site — `args, kwargs = module.inputs` (shares the eproperty key `input` with `.input` — denotation check needed) | UNTESTED | UNTESTED | traverse row |
| internal `.source` — attention weights | ✓ (eager only) | **site absent** — paged/flash attention never materializes the matrix | the attention-weights site-absence |
| internal `.source` — other ops | ✓ — `mlp.source.self_c_fc_0` == `c_fc.output`, exact | ✓ — same check, exact (the vLLM MLP forward is plain Python, so `.source` rewrites it fine) | micro tier; the portable-sites result |
| internal — nested/recursive `.source` (`module.source.<op>.source.<inner>`) | UNTESTED | UNTESTED | traverse row; same op class as `.source`, low priority |
| derived (head *h* / neuron *j*) | ✓ — head view validated by c_proj reconstruction; neuron by `gelu_new(c_fc)` | ✓ — same checks; weight-using reconstruction must run INSIDE the trace (the client-side envoy is the meta model) | micro tier; the portable-sites result |
| derived — subspace / direction-valued addresses (DAS rotations, steering directions; adopted as derived-tier citizens, design.md §3.2) | UNTESTED | UNTESTED — no probe until a DAS-class cell exists | traverse row; subspace-featurizer alignment |
| gradient space (`.grad`) | ✓ | absent (no grad — the no-autograd-on-vLLM result) | the no-autograd-on-vLLM result |

Level-1 notes: name collisions remount under `.nns_output` (address-space quirk, not a row);
`rename=` is adopted as the harness's vary-the-names test tool, not as canonicalization
(design.md §3.5 — footprint metadata uses tier-level site IDs, cells keep explicit per-family
paths).

### Level 1.5 — realizations: the working recipe per backend

Realizations of ANY language element (design.md §3.3); a recipe = the realization choice per
element that works in that context.

| element | hf recipe | vllm recipe |
|---|---|---|
| write | in-place or replacement | replacement ONLY (new tensor / whole tuple) — the in-place-write restriction; skip-with-value realization UNTESTED (data-op table) |
| meta-compute (unembed) | `lm_head(h)` or weight matmul | weight matmul ONLY (`ParallelLMHead.forward` guarded) — the guarded lm_head call |
| meta-compute (aux) | bare | under `torch.no_grad()` — the inference-tensor no_grad requirement |
| step quantifier | bounded or unbounded | bounded `iter[0:N]` ONLY (unbounded drops all loop-carried saves) — the unbounded-iteration saves-drop; `tracer.next()` realization UNTESTED (control table) |
| run↔run transfer | two single-prompt traces (barrier works, measured) | two single-prompt traces ONLY (barrier broken upstream) — the barrier sync/async split; un-barriered realization UNTESTED (cross-edge table) |
| residual read on fused-residual families | `out[0]` | `out[0] + out[1]` — the fused-residual denotation mismatch |
| read / live-out value semantics (engine memory model, design §3.6) | alias is fine (fresh per-forward allocation) | snapshot REQUIRED — alias decays under in-place buffer reuse; nnsight auto-clones inference-mode tensors on `.save()` (clone-on-save inference-tensor protection, `tensor.is_inference()` selector). The read-side dual of the write in-place/replacement split; unhandled = SILENTLY_WRONG |
| read-before-write (user's own downstream write) | clone-first (`before = x.clone().save()`) — distinct from the engine-memory-model row above; denotation check (saved-var-is-a-reference is a SILENTLY_WRONG generator), folded into existing write probes, not a standalone row | same |

### Level 2 — the entry enumeration (the coverage denominator)

Per design.md §3.4 the L2 catalog is **generated, not curated**: (data ops + edges) × address
tiers × scope positions, filtered to combinations any cataloged method's footprint names — and
**the tables above ARE that enumeration**: one row per L2 entry, with realization splits as
sub-rows (bounded/unbounded; barriered/un-barriered/two-trace/session-var) and measured statuses
carrying their evidence (the corresponding finding described in `findings.md`). Everything added
by the 2026-06-12 traverse is UNTESTED. Coverage
= footprint-needed entries minus probe-or-cell-exercised entries, computable from this one copy
(no second status table, so the lists can't diverge). Out-of-scope rows (recorded with reasons
in `drafts/design-revision-2026-06-12.md` Part 1): the NDIF plane (remote trace, session
bundling, non-blocking jobs, `tracer.local()`, code shipping — deferred by the v1-scope decision; a future `ndif`
backend column is reserved, no rows now), deprecated iteration forms, non-greedy sampling
(breaks the determinism criterion), multi-backward, and extension/harness plumbing.

### Protocol index and nnbench execution details

The authoritative methodology templates and parameter classifications live in
[isb/protocol.py](../isb/protocol.py): `PROTOCOLS` holds one `InterventionSpec` per methodology.
Case-description hooks live beside the implementations in
[isb/methodologies/requirements.py](../isb/methodologies/requirements.py).
The method summaries below explain the scientific procedures and their coverage; the Python
definitions own the executable-spec inventory.

Semantic parameters specify the computation: ablation's `target` chooses the component being
ablated, and DAS's `train` selects training or application. Realization parameters specify the
implementation spelling, such as `residual` or bounded iteration. Extensions identify behavior
outside CausaLab v1. The Level-0/1/1.5 vocabulary above supports engine-failure diagnosis.

Saved call records contain effective parameters, semantic/realization values, and the described
requirements. For example, DAS application (`train=0`) is gradient-free while training requires
gradients. `protocol_scope` distinguishes concrete cases from template-only descriptions;
`protocol_coverage` distinguishes described, explicitly opted-out, and legacy-unknown metadata.

Status: ✓ = already an nnbench cell. **frontier** = exercises a primitive where vLLM and HF diverge
(the highest-signal additions).

---

## 1. Reading what a layer represents (observational lenses & probes)

| method | idea | protocol tuple · nnbench details | models / generality | status |
|---|---|---|---|---|
| **Logit lens** | project an intermediate residual through final-norm + unembed → a next-token dist | base · `block_output` + `lm_head` · read · prompt · `full_logits`; unembed/residual realizations | any decoder-only LM; the "fused residual" detail is arch-specific (GPT-2 single tensor vs Llama hidden+residual) | ✓ |
| **Tuned lens** | logit lens with a *trained* affine probe per layer (better-calibrated early layers) | base · `block_output` + `lm_head` · read · prompt; trained affine artifact | needs a tuned-lens checkpoint per model | TODO |
| **Linear probing** | train a linear classifier on activations to test if a concept is linearly decodable | base · `block_output` · read, grad · prompt · `grad`; trained probe artifact | any model; probe is per-model/per-concept (cheap to train) | TODO |
| **Direct logit attribution (DLA)** | decompose the final logit into additive per-component contributions | base · `attention_output` + `mlp_output` + `lm_head` · read · prompt | decoder-only; per-head needs access to head outputs before `W_O` | TODO (read-only, exact) |

## 2. Causal interventions (change something, watch the output)

| method | idea | protocol tuple · nnbench details | models / generality | status |
|---|---|---|---|---|
| **Activation patching / causal tracing** | copy an activation from a clean run into a corrupted run; measure restoration | base + counterfactual · `block_output` + `lm_head` · read, write / `swap` · prompt · `paired_forward`, `full_logits`; residual realization | any transformer | ✓ |
| **Ablation / knockout** | zero- or mean-out a component, measure the damage | base · component selected by case · read, write / `swap` · prompt · `full_logits`; mean variant adds stored aggregate state | any transformer | ✓ (zero); mean = the accumulation roadmap cell |
| **Steering / ActAdd** | add a direction into the residual at run time to push behavior | base · `block_output` + `lm_head` · read, write / `add_scaled` · prompt · `full_logits`; write realization | any decoder-only LM | ✓ |
| **Generation-time steering** | the steering write applied at EVERY decode step of a greedy generation, per-step logits read | base · `block_output` + `lm_head` · read, write / `add_scaled` · prompt + generated · `generate`, `full_logits`; bounded/unbounded and decode-step-write extension | any decoder-only LM; vLLM needs the bounded `iter[0:N]` realization | ✓ — **composition confirmed** (write × bounded-iter SUPPORTED on vLLM, top1=1.00 tv=0.000; unbounded = the unbounded-iteration saves-drop frontier marker; a direct step-lift law test — base vs lifted on one backend — is queued) |
| **Generation-time cross-prompt patching** | the cross-prompt transplant injected at prefill, scored on the generated tokens (the causalab `locate` footprint) | base + counterfactual · `block_output` + `lm_head` · read, write / `swap` · prompt + generated · `paired_forward`, `generate`, `full_logits`; bounded/unbounded realization | any transformer; length-matched pair; vLLM needs bounded `iter[0:N]` | ✓ — **composition confirmed at fp32** (transplant step-lifts correctly); bf16 forks the whole greedy trajectory (top1=0.00 tv=0.711, SUPPORTED_DEGRADED) — precision compounding, NOT a mechanism bug |
| **Attribution patching** | gradient linear-approx of patching for *every* component in one fwd+bwd | base + counterfactual · `block_output` + `lm_head` · read, grad · prompt · `grad`; activation-gradient extension | any differentiable model | ✓ — **frontier confirmed** (`grad`: vLLM ERROR — the no-autograd-on-vLLM result) |
| **Path patching** | patch specific component→component *edges* (not whole activations) | base + counterfactual · `attention_premix` + `block_input` + `attention_output` + `lm_head` · read, write / `swap` · prompt · `paired_forward` | any transformer; more plumbing | TODO (composite; the rewiring edge has no measuring cell) |

## 3. Decomposing representations into features

| method | idea | protocol tuple · nnbench details | models / generality | status |
|---|---|---|---|---|
| **Sparse autoencoders (SAEs)** | sparse overcomplete dict over a layer's activations → monosemantic features | base · `block_output` · read (plus write / `pytorch_fn` for splice) · prompt · `pytorch_fn_local`; trained artifact | needs trained SAEs (available: GPT-2, Gemma-2, Llama, …); latent read needs `hook=True` (UNTESTED) | TODO |
| **Transcoders** | SAE that approximates an MLP's *computation* (read input → write output) | base · `mlp_input` + `mlp_output` · read, write / `pytorch_fn` · prompt · `pytorch_fn_local`; trained artifact and input-write realization | needs trained transcoders | TODO |

## 4. Attention & circuits

| method | idea | protocol tuple · nnbench details | models / generality | status |
|---|---|---|---|---|
| **Attention-pattern read** | read the attention weights (who attends to whom) | base · `attention_probs` · read · prompt | any transformer (HF eager); — | ✓ — **frontier confirmed** (`attention_probs`: site absent on vLLM — the attention-weights site-absence) |
| **Per-head ablation / read** | zero or read an individual attention head's output | base · `attention_output` with head coordinate · read, write / `swap` · prompt; reshape-slice realization | any transformer; needs head-dim reshape | TODO |
| **Induction heads** | identify head pairs implementing "A→B … A→?B" (in-context copying) | base · `attention_probs` + `attention_output` · read (optional write / `swap`) · prompt | emergent in most transformers | TODO (analysis, composite) |
| **Automated circuit discovery (ACDC / EAP)** | iteratively patch/prune edges to find a task's minimal subgraph | base + counterfactual · component graph · read, write / `swap` (plus grad for EAP) · prompt · `paired_forward` (plus `grad`) | any transformer; many runs | TODO (heavyweight) |

## 5. Concept-direction control & full pipelines

| method | idea | protocol tuple · nnbench details | models / generality | status |
|---|---|---|---|---|
| **Representation engineering (RepE)** | derive a concept "reading vector" from contrastive activations, monitor/steer | base + counterfactual · `block_output` · read, write / `add_scaled` · prompt; direction-estimation artifact | any decoder-only LM (vector derived from data, light) | TODO |
| **Circuit Tracer (attribution graphs)** | replace MLPs with cross-layer transcoders → build & intervene on an attribution graph | base · `mlp_input` + `mlp_output` + `lm_head` · read, write, grad / `pytorch_fn` · prompt · `grad`, `pytorch_fn_local`; trained artifacts | only models with trained transcoders (Gemma-2-2B, small Llama, Qwen3-4B) | TODO (heavyweight; full pipeline, not a unit cell) |

---

## Serving-feature context axis — which engine optimization attacks which primitive (predictions)

The backend axis today is single-GPU `hf` vs `vllm_async/serve`. Production serving stacks add a
second axis of **engine optimizations** (`engine-config × parallelism`, design §3.6) that the
inventory above under-represents. This section is that axis. The discipline is unchanged: a feature
does not attack a *method*, it attacks a **primitive component** (a site denotation, an edge, a
realization, a quantifier); a method inherits the break iff its footprint names that component (∧,
§3.6). So the rows are primitive/footprint-level, and every catalogued method maps in through its
cluster.

**Statuses are PREDICTED from (footprint × feature-attack) and are UNTESTED in the bench unless
★-marked.** No measured status is invented (the catalog's rule). ★ = bench- or repro-measured, with
the finding in `findings.md`.

### A. The generator — feature → primitive it attacks → predicted failure

| feature | engine effect | primitive attacked | failure kind | today |
|---|---|---|---|---|
| **TP** tensor-parallel | shards heads / MLP-interm / vocab(`lm_head`); residual all-reduced full-width at the block boundary | L1 denotation of *sharded* sites; reduction order | a sharded-site read/write sees a shard (denotation); all-reduce reorders near-ties | ★ `lm_head` vocab-shard SW (steer/DLA via weights); ★ reduction-order EQUIVALENT_DEGRADED; ★ residual reads EQUIVALENT |
| **PP** pipeline-parallel | layers split across stage processes | L1 site existence (per stage); L2 cross-stage edge | read at layer L only on its owner stage; cross-stage read→write crosses processes | ★ cross-stage write via whole-output replacement; ★ fused-residual `LazyRemoteTensor` summed (logit-lens PP-aware) |
| **EP** expert-parallel (MoE only) | experts sharded; routing is data-dependent | L1 expert-site location; router top-k near-tie | reading a specific expert's activation off-rank; router flips a near-tie | ★ Nemotron param-gather; ★ router near-tie EQUIVALENT_DEGRADED |
| **PD** prefill/decode disagg | prefill and decode on separate engines; KV shipped between | L2 cross-region edge (prefill→decode) | any read-at-prefill → write/read-at-decode spans two engines | UNTESTED (high risk) |
| **KV cache** paged + in-place reuse | KV in paged buffers; activation buffers reused | L1 unnamed KV site; L1.5 read value-semantics | direct KV read/write has no site name; un-cloned read decays | ★ clone-on-save (alias → SW without it) |
| **prefix cache** (often default-on) | a cached prompt prefix is **not recomputed** on a hit | L1 site existence at cached prompt positions | a read/write at a cached position never runs the forward → silently no-op | UNTESTED — predicted SW, the highest-value untested cell |
| **chunked prefill** (V1 default) | a long prompt's prefill is split across scheduler steps | L0 scope position (prefill fragmented); L2 cross-chunk accumulation | "read at prefill" fires per-chunk; multi-position prompt ops fragment | UNTESTED |
| **spec decode** | draft proposes K tokens, target verifies, some rejected | L0 step-quantifier denotation | a per-step decode write/read fires on draft steps that get thrown away | UNTESTED |
| **quantization** fp8/awq/gptq | activations/weights de/quantized | L1 denotation + precision | activation reads degraded; weight-reading ops see quantized weights | UNTESTED |
| **CUDA graphs** (`enforce_eager=False`) | the forward is captured into a replayed graph | L0/L1.5 injected hook vs captured graph | interventions error or are skipped unless eager | partly known (perf runs force `enforce_eager`) |
| (continuous batching) | dynamic batching of requests | model-side lift law (positions) | batched ≢ sequential for absolute-position families | ★ batched GPT-2 positions (model-side law failure) |

### B. Footprint clusters — where every catalogued method lands

Each method inherits its cluster's grid row; composites inherit the worst of their clusters.

- **R · residual read+project** — Logit lens, Tuned lens. `read`(boundary block) `sweep` `live-out` (+`ext-module` for tuned).
- **I · residual inject/write** — Steering/ActAdd, Zero-ablation, RepE-apply. `read` `write/replacement` `injection`.
- **S · internal/derived-site read/write** — DLA (per-head), Per-head ablation/read, Attention-pattern read, Induction heads. `read`/`write` × internal(`.source`/`attn-weights`) / derived(head/neuron).
- **X · cross-run transplant** — Activation patching / causal tracing. `read` `write` `xprompt`.
- **G · per-step generation lift** — Generation-time steering, Generation-time cross-prompt patching. cluster I/X + `step/bounded` `loop-carried`.
- **D · gradient** — Attribution patching, EAP-grad. `grad` `derivative`.
- **E · external-module compute** — SAE read/splice, Transcoder, Linear probe, Circuit Tracer. `read` `ext-module` (`rewiring`/`accumulation`) (+`write`).
- **M · multi-run edge search** — Path patching, ACDC / EAP. `rewiring` `adaptive` (+`grad` for EAP).

### C. Cluster × serving-feature — predicted status

`·` no attacked component (predicted EQUIVALENT) · `SW` silently-wrong · `ERR` loud error · `DEG`
degraded / near-tie · `n/a` op unsupported regardless · ★ measured · `/` idiom-dependent (left =
safe idiom, right = broken idiom). EP applies on MoE families only.

| cluster | TP | PP | EP | PD | KV | PFX | CHP | SPEC | QNT | CG |
|---|---|---|---|---|---|---|---|---|---|---|
| **R** read+project | ★·/SW¹ | ★· | DEG | ·/SW | ★· | SW² | DEG | SW³ | DEG | ERR⁴ |
| **I** inject/write | ★·/★SW¹ | · | DEG | SW | ★· | **SW²** | SW | · | DEG | ERR⁴ |
| **S** internal/derived | SW⁵ | · | SW⁶ | SW | ★·/SW | SW | SW | SW | DEG | ERR⁴ |
| **X** transplant | ·/SW | ★· | DEG | **SW**⁷ | ★· | **SW²** | SW | SW | DEG | ERR⁴ |
| **G** gen-step lift | ·/SW | · | DEG | **SW**⁷ | ★· | SW | ·⁸ | **SW³** | DEG | ERR⁴ |
| **D** gradient | n/a⁹ | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| **E** ext-module | ·/SW⁵ | ·¹⁰ | DEG | SW | ★· | SW | SW | SW | DEG | ERR⁴ |
| **M** multi-run search | ·/SW | · | DEG | SW | ★· | SW | SW | SW | DEG | ERR⁴ |

¹ the residual is full-width (safe), but a manual `lm_head` / weight matmul or `head.weight[id]`
index hits the vocab/row shard → SW (the measured `lm_head` finding); the engine `logits` site is
gathered (safe) — idiom-dependent. ² cached prompt position: the forward never runs there, so the
read is stale/absent and the write is dropped, no error. ³ a per-step or decode-step op fires on
draft steps the verifier rejects → you read/steer a token that is never emitted. ⁴ nnsight
interventions are not in the captured graph → forces eager or errors (why perf uses
`enforce_eager`); read-only may survive, UNTESTED. ⁵ heads / MLP-intermediate are TP-sharded → a
head/neuron read or an SAE applied there sees a shard. ⁶ a specific expert's activation lives on one
rank (the param-gather finding). ⁷ the transplant / per-step edge crosses the prefill→decode engine
split. ⁸ decode is not chunked, so gen-step decode writes are safe; only the prefill-position part
fragments. ⁹ grad is ERROR on vLLM regardless of any optimization. ¹⁰ an external module must live
on the stage that owns layer L (device placement), else it cannot run there.

### The load-bearing predictions

- **prefix cache × {steering, patching, any prompt-position read}** is the densest SILENTLY_WRONG
  region and is entirely unmeasured: a steer injected at a cached prompt token is silently dropped;
  a patch whose clean/corrupted activations are cached never runs. Default-on, no error raised.
- **spec decode × generation-time steering** silently steers rejected draft tokens.
- **PD disaggregation × {transplant, gen-step}** splits one method's read and write across two
  engines.
- **TP / EP × internal-site methods** (per-head, DLA, expert-level SAE) read shards, not whole
  tensors — the measured `lm_head` / param-gather class generalized to every head/expert-level
  method.

### What is measured today

TP (residual EQUIVALENT, `lm_head` shard SW, reduction-order DEG), PP (cross-stage write,
fused-residual), EP (MoE param-gather, router near-tie), KV reuse (clone-on-save), continuous
batching (positions). Everything in columns PD / prefix-cache / chunked-prefill / spec-decode /
quant is PREDICTED and UNTESTED — that gap is the serving-feature roadmap.

---

## nnbench roadmap (prioritized)

**The original Level 0–1 row set is fully measured** (micro tier, refined 2026-06-11: HF 13/13
SUPPORTED; vLLM 7 SUPPORTED / 6 ERROR — the micro-tier construct findings (iteration, barrier,
session, edit/scan)). Attention-pattern read and attribution patching graduated to ✓;
generation-time steering is ✓ DONE (2026-06-12, the generation-time steering composition result —
the composition prediction confirmed exactly; the unbounded form rides along as the frontier
marker / flip-detector for the upstream saves-drop fix). The 2026-06-12 traverse added the UNTESTED rows
above; the queue
below follows the production-engine priority order (design.md revision Part 4d: edges first,
then grad, staging, streaming/serve live-out realizations) — production serving APIs expose
~generate + logprobs and nothing else, so the edges/grad/staging coverage is exactly what an
interp-serving layer must add.

**Micro probes (ordered):**
1. **Multi-layer `tracer.cache`** — the fused bulk-cache primitive (read × breadth + live-out);
   throughput-bound, the harvesting regime; the one Level-2 fused primitive still unprobed.
   (The known upstream PP merge gap is a finding-in-waiting, not a reason to skip.)
2. **Module skip** — SKIP is its own Mediator event; its vLLM status is not derivable from the
   SWAP rows; also a perf primitive.
3. **Early region exit (`tracer.stop()`)** — pre-stop save preservation + engine survival +
   compute saved.
4. **Un-barriered cross-invoke flow** — the accident-prone realization; SILENTLY_WRONG-shaped
   risk.
5. **Input write + kwargs read** — one probe pair; unlocks the transcoder-splice footprint.
6. **Samples write (forced decoding)**.
7. **`tracer.next()` + empty invoke + `tracer.result`** — cheap additions to existing
   iteration/fork probes.
8. **Grad write (HF) + `hook=True` aux application** — unlock gradient-editing and SAE-latent
   footprints.
9. **`trace=False` baseline** — the no-interleave overhead floor per backend (harness baseline,
   not an inventory row).
10. **`vllm_sync` backend** — promotes the construct-gaps sync statuses (barrier SILENTLY_WRONG
    is the motivating cell) from external repros to bench-measured.
- Folded into existing probes: clone-before-modify denotation check; the fused-residual
  denotation-mismatch check already exists. Deferred: edit persistence
  (`export_edits`/`import_edits`), HF streamer per-step token site.

**Method cells (each = one unexercised edge type or transformation, per the §3.5 completeness
criterion):**
1. **Rewiring cell** — path-patching-style same-run read→compute→write downstream (the edge no
   cell exercises); cheapest deterministic form: SAE-free linear splice.
2. **Accumulation cell** — mean ablation (the aggregation transformation; cheapest trained-state
   proxy; makes the `trained` tag structural).
3. **Dataset-lift law cell on a relative-position family** — the positive control for the
   measured GPT-2 absolute-position model-side failure (attribution: model).
4. ~~**Generation-time cross-prompt patching**~~ — ✓ **DONE** (2026-06-15): transplant ×
   step-lift; the composition holds at fp32 (transplant survives the decode loop), bf16 forks the
   trajectory (precision, not a bug). Completes the causalab audit's second flagged prediction
   (locate's footprint) and is the recipe for the Macro-tier locate port.
5. **Per-head ablation** — write × derived address (head-sliced write is still unexercised).
6. **Direct logit attribution** — observation × internal site; exact read-only decomposition
   over the now-measured non-attention `.source` site.
7. **Sweep-exchange law cell** — multi-layer-one-run vs one-run-per-layer logit lens (free data
   from existing cells; names the law).
- Plus: **family axis for the micro tier** — the Level-0/1 map is (GPT-2)-only; rerun on a
  fused-residual family (SmolLM2/Llama) where boundary denotation differs (the fused-residual
  denotation mismatch).

**Tier 2 — need a (cheap/available) trained artifact:**
- Linear probing (train a probe), SAE (use an existing checkpoint, e.g. GPT-2/Gemma), tuned lens.

**Tier 3 — composite / heavyweight (great demos, poor unit cells):**
- Path patching (full), ACDC/EAP, Representation Engineering, Circuit Tracer (transcoders).

**Selection criterion recap:** an ideal new cell is (a) deterministic, (b) needs no trained artifact,
(c) exercises a Level-2 entry (or law/transformation) not yet covered by any cell — ideally one
where vLLM and HF *diverge* — so it adds a real coverage frontier, not just a row. By that test the
next cells to build are **multi-layer `tracer.cache`** and **generation-time cross-prompt
patching** (generation-time steering is done — the generation-time steering composition result).
