# Design revision proposal — full primitive traverse, level clarification, level↔methodology mechanics

> **STATUS: APPLIED (2026-06-12).** Parts 2–5 are folded into `design.md` §3 and
> `interp-methods-catalog.md`; Part 1 (the 75-row traverse with file:line evidence) remains the
> evidence annex referenced by design.md §3.8. Synthesizes: two independent traverses of nnsight's
> implemented primitives (source-level + docs-level, dev branch), the considerations sweep over
> the bench docs, and the external-research sweeps. Normative target: `design.md` §3 (numbering
> kept stable; §3.5/§3.6 are cited externally) and `interp-methods-catalog.md`. The
> language-level framing of `drafts/interp-program-model.md` is the preferred direction
> throughout; implementation detail stays subordinate. **No measured status is invented
> anywhere in this document: every proposed new inventory row is UNTESTED.** The §12
> flat-explicit-cells architecture is untouched — everything here is vocabulary, metadata, and
> roadmap.

---

## Part 1 — Full primitive traverse table

Merged + deduped from the source traverse (`src/nnsight/**`, dev branch) and the docs traverse
(`/disk/u/zikai/nnsight/docs/**`). One row per user-facing primitive.

**Slot vocabulary** (program-model draft): `data-op` (boundary-crossing read/write/grad) ·
`quantifier` (step/sweep/dataset/adaptive/run-DAG — control that determines which runs exist and
where crossings attach) · `edge` (meta-level dataflow classified source→destination) · `address`
(Level-1 site tier) · `realization` (alternate spelling of the same language element) · `mode`
(context/engine parameter) · `staging` (install-into-future-runs residue) · `out-of-model`
(meta-level code or harness/service machinery — not a language element).

**Inventory status:** `MEASURED` (a status row exists in `interp-methods-catalog.md`, cited
F-n) · `LISTED-UNTESTED` (named in the catalog/roadmap but never probed) · `ABSENT` (no row at
all). Every ABSENT row carries a one-line proposed action. Action vocabulary: *new row*
(UNTESTED inventory row), *new probe* (micro-tier probe → measures the new row), *out-of-scope*
(with reason).

### 1.1 Data ops (boundary crossings: read / write / grad)

| # | primitive | surface | evidence | slot | status | action (if ABSENT) |
|---|---|---|---|---|---|---|
| 1 | module output read | `model.<path>.output` | `src/nnsight/intervention/envoy.py:172-179`; getter `interleaver.py:275-315` | data-op (read × boundary) | MEASURED (catalog READ row; F-7 denotation caveat) | — |
| 2 | module output write, replacement | `model.<path>.output = x` | `src/nnsight/intervention/interleaver.py:317-337` (SWAP event) | data-op (write × boundary) | MEASURED (F-5) | — |
| 3 | module output write, in-place | `model.<path>.output[0][:] = 0` | same machinery; `docs/usage/access-and-modify.md:47` | realization of #2 | MEASURED (F-5: raises on vLLM) | — |
| 4 | module input read | `model.<path>.input` | `src/nnsight/intervention/envoy.py:193-213` | data-op (read × boundary) | MEASURED (micro `.input` row; F-12) | — |
| 5 | module input WRITE | `model.<path>.input = x` (setter repacks into `(args,kwargs)`) | `src/nnsight/intervention/envoy.py:193-213` (postprocess) | data-op (write × boundary) | ABSENT | new row + new probe: input-write is the natural form of transcoder-splice and input-side patching |
| 6 | full args/kwargs view | `args, kwargs = model.<path>.inputs` | `src/nnsight/intervention/envoy.py:181-191` | address (boundary, kwargs sub-site) | ABSENT | new row + new probe: kwargs-site read (shares eproperty key `input` with #4 — denotation check needed) |
| 7 | operation-level read/write | `model.<path>.source.<op>.output/.input` | `src/nnsight/intervention/envoy.py:215-238`; `source.py:606-630` | data-op × internal address | MEASURED (non-attn exact both backends; attn-weights site absent on vLLM — F-10/F-12) | — |
| 8 | recursive `.source` | `module.source.<op>.source.<inner>.output` | `src/nnsight/intervention/source.py:632-692`; `docs/usage/source.md:65` | address (internal, nested) | ABSENT | new row + new probe: nested-fn site (e.g. inside `attention_interface`); low priority — same op class as #7 |
| 9 | gradient read | `t.grad` inside `with loss.backward():` | `src/nnsight/intervention/tracing/backwards.py:50-64` | data-op (grad) | MEASURED (F-11: vLLM ERROR) | — |
| 10 | gradient WRITE | `t.grad = g` / `t.grad[:] = 0` mid-backward | `src/nnsight/intervention/tracing/backwards.py:50-64` (setter); `docs/usage/backward-and-grad.md:88` | data-op (write × gradient space) | ABSENT | new row + new probe (HF; vLLM row derivable as ERROR from F-11 but record it UNTESTED, never derived-as-measured) |
| 11 | module skip | `model.<path>.skip(replacement)` | `src/nnsight/intervention/envoy.py:465-485`; SKIP event `interleaver.py:571-582,1257-1297`; `docs/usage/skip.md:13` | data-op (write × boundary, compute-eliding realization) | ABSENT | new row + new probe: ablate-by-bypass — both a WRITE realization and a perf primitive (saves the module's FLOPs) |
| 12 | bulk cache | `tracer.cache(modules=, device=, dtype=, detach=, include_inputs=)` | `src/nnsight/intervention/tracing/tracer.py:545-629`; hooks `hooks.py:476,517` | data-op (fused read × breadth + live-out edge) | LISTED-UNTESTED (catalog roadmap Tier-1 item 2 — "the one Level-2 fused primitive still unprobed") | build the probe (already queued; see Part 5) |
| 13 | ad-hoc module application (COMPUTE) | `y = model.lm_head(model.transformer.ln_f(h))` | `src/nnsight/intervention/envoy.py:240-245` | out-of-model (meta-level compute; parameters are readable constants) | MEASURED as realization rows (F-1 no_grad, F-2 lm_head guard) | — (reclassified, not removed: the measured rows become realization rows of meta-compute, see Part 2) |
| 14 | hooked module application | `envoy(x, hook=True)` | `src/nnsight/intervention/envoy.py:240-245`; `docs/reference/api-quick-reference.md:62` | mode (routes a meta-compute through the boundary, making aux `.input/.output` addressable) | ABSENT | new row + new probe: the `ext-module` tag's observability half (SAE latent read requires it) |

### 1.2 Control quantifiers and region machinery

| # | primitive | surface | evidence | slot | status | action (if ABSENT) |
|---|---|---|---|---|---|---|
| 15 | region (trace) | `with model.trace(input) as tracer:` | `src/nnsight/intervention/envoy.py:249-295`; `tracer.py:339` | quantifier (run scope) | MEASURED | — |
| 16 | region, steps=N (generate) | `with model.generate(input, max_new_tokens=N):` | `src/nnsight/intervention/envoy.py:1002-1023`; `src/nnsight/modeling/language.py:181-203` | quantifier (run scope, unrolled) | MEASURED (micro iteration probes; F-17) | — |
| 17 | region, mode=fake (scan) | `with model.scan(input):` | `src/nnsight/intervention/envoy.py:297-329`; `tracer.py:693-723` | mode | MEASURED (F-16: vLLM ERROR) | — |
| 18 | no-interleave direct call | `model.trace(input, trace=False)` / `model.method(input, trace=False)` | `src/nnsight/intervention/envoy.py:291-293,1010-1014` | mode (bypass) | ABSENT | new probe, *as harness baseline only* — the no-interleave overhead floor per backend; not an inventory row |
| 19 | fork (invoke) | `with tracer.invoke(prompt):` | `src/nnsight/intervention/tracing/tracer.py:513-525`; `invoker.py:14-84` | quantifier (dataset, batched realization) | MEASURED (fork row; vLLM async gated; regime effect) | — |
| 20 | empty invoke | `with tracer.invoke():` (batch_group=None, full combined batch) | `docs/usage/invoke-and-batching.md:18` | realization of #19 (also the out-of-order/trailing-code escape hatch) | ABSENT | new row + new probe: the documented escape hatch deserves a measured status — it's the recommended fix for two gotchas |
| 21 | step selector, bounded | `for step in tracer.iter[0:N]:` (int/slice/list forms) | `src/nnsight/intervention/tracing/tracer.py:533-535`; `iterator.py:55-72,184-291` | quantifier (step) | MEASURED (F-13: SUPPORTED both) | — |
| 22 | step selector, unbounded | `tracer.iter[:]` / `tracer.all()` | `src/nnsight/intervention/tracing/tracer.py:537-538` | realization of #21 | MEASURED (F-13: vLLM drops all saves) | — |
| 23 | manual step advance | `tracer.next(step=1)` | `src/nnsight/intervention/tracing/tracer.py:540-543`; `docs/usage/iter-all-next.md:93` | realization of #21 (straight-line spelling) | ABSENT | new row + new probe: third spelling of the step quantifier; cheap to add to the iteration probe |
| 24 | join/sync (barrier) | `b = tracer.barrier(n); b()` | `src/nnsight/intervention/tracing/tracer.py:631-659,726-745`; BARRIER event `interleaver.py:1212-1247` | quantifier (fork sync; its observable is the sharing edge #34) | MEASURED (F-14) | — |
| 25 | early region exit | `tracer.stop()` | `src/nnsight/intervention/tracing/tracer.py:527-531`; `interleaver.py:1370-1375`; `docs/usage/stop-and-early-exit.md:13` | quantifier (run-scope truncation) | ABSENT | new row + new probe: a pure perf primitive (elide the rest of the forward) AND a correctness question (are pre-stop saves preserved on vLLM, or does the engine die?) |
| 26 | run DAG (session) | `with model.session():` | `src/nnsight/intervention/envoy.py:428-431` | quantifier (run-DAG) | MEASURED (F-15) | — |
| 27 | backward region | `with loss.backward(retain_graph=...):` | `src/nnsight/__init__.py:159-178`; `backwards.py:81-113` | quantifier (run scope over the derivative graph) | MEASURED (F-11) | — |
| 28 | multi-backward | `loss.backward(retain_graph=True)` twice | `docs/usage/backward-and-grad.md:55` | realization of #27 | ABSENT | out-of-scope v1: HF-only autograd bookkeeping, no engine coupling; revisit if a second-order method enters the registry |
| 29 | sweep | host-side `for ℓ in layers:` | no nnsight surface (plain Python) | quantifier (addresses) — **named, per the draft** | implicitly exercised by every multi-layer cell; never named | adopt as named quantifier (Part 2/Part 5); breadth becomes its measure — no probe needed (it is free by construction) |
| 30 | adaptive | host-side `while`/`if` over past results | no nnsight surface | quantifier (future runs) | ABSENT | out-of-scope as a probe (host-side, free); enters via method cells (ACDC-class, Tier 3) |
| 31 | plain `if`/`for` on live tensors | inside any trace | `docs/usage/conditionals-and-loops.md:13` | out-of-model (meta-level control) | declared free, not inventoried (catalog header) | — keep declared-free |
| 32 | vLLM generate alias | `model.generate(input, max_new_tokens=N)` → `max_tokens` rewrite | `src/nnsight/modeling/vllm/vllm.py:479-489` | realization of #16 (portability shim) | MEASURED (gen_steering cells run through it) | — |
| 33 | deprecated iteration forms | `model.iter/.all/.next`, `module.next()`, `with tracer.iter[N]:` | `src/nnsight/intervention/envoy.py:433-463`; `iterator.py:320-327` | realization (deprecated aliases of #21/#23) | ABSENT | out-of-scope: deprecated-but-present; bench measures the canonical spellings only |

### 1.3 Edges (meta-level dataflow, source → destination)

| # | primitive | surface | evidence | slot | status | action (if ABSENT) |
|---|---|---|---|---|---|---|
| 34 | live-out | `x.save()` / `nnsight.save(x)` | `src/nnsight/intervention/tracing/globals.py:24-47`; `src/nnsight/__init__.py:57` | edge (run → meta) | MEASURED (all cells; serve sweep 22/22) | — |
| 35 | loop-carried accumulation | `rows.append(...)` per step | measured through #21/#22 | edge (step → run scope) | MEASURED (F-13) | — |
| 36 | fork↔fork sharing, barriered | value defined in invoke 1, read in invoke 2 after `barrier()` | `docs/usage/barrier.md:15` | edge (fork ↔ fork) | MEASURED (F-14) | — |
| 37 | fork↔fork sharing, un-barriered | same, no barrier, different modules; gated by `CONFIG.APP.CROSS_INVOKER` | `docs/usage/invoke-and-batching.md:80` | realization of #36 (automatic push/pull spelling) | ABSENT | new row + new probe: the automatic flow is the spelling users hit *by accident*; its vLLM status is unknown and plausibly SILENTLY_WRONG-shaped |
| 38 | run → run, saved (session) | `.save()` in trace 1, read in trace 2 | session machinery | edge (run → run) | MEASURED (F-15) | — |
| 39 | run → run, un-saved (session contract) | plain variable across session traces | session machinery | edge (run → run) | MEASURED (F-15: ERROR both vLLM engines) | — |
| 40 | transplant via host | read in trace A → write in trace B (two traces) | patching cells | edge (run → run via meta) | MEASURED (F-8) | — |
| 41 | staging replay | `model.edit()` body → every future region | `src/nnsight/intervention/envoy.py:331-363`; replay `tracer.py:414-418` | staging | MEASURED (F-16) | — |
| 42 | engine result | `tracer.result` (iterate=False eproperty) | `src/nnsight/intervention/tracing/tracer.py:661-663`; `docs/usage/trace.md:91` | address (engine tier) + live-out | ABSENT | new row + new probe: the *recommended* end-of-generation capture (preferred over `generator.output`, `language.py:49-51`) has no measured status |
| 43 | HF generation output site | `model.generator.output.save()` | `src/nnsight/modeling/language.py:44-66` | address (engine tier) | MEASURED (greedy ids == logits argmax) | — |
| 44 | HF per-step token stream | `model.generator.streamer.output` | `src/nnsight/modeling/language.py:53-66` | address (engine tier, step-indexed) | ABSENT | new row, probe deferred: per-step token edge on HF; fold into an existing iteration probe when touched |
| 45 | async streaming drain | `async for out in tracer.backend():` / `await tracer.backend()`; saves only on `output.finished` | `src/nnsight/modeling/vllm/vllm.py:469-476`; `docs/models/vllm.md:190` | edge (run → meta, streaming realization of #34) | ABSENT (it IS the bench's vllm_async transport, but has no explicit row: finished-only collection, single-shot generator, request-order ≠ invoke-order) | new row UNTESTED: make the bench's own transport assumptions measured, not assumed |
| 46 | serve venue | `model.trace(input, serve='http://...')` | `src/nnsight/modeling/vllm/vllm.py:452-468` | edge (live-out over the wire) | MEASURED (serve sweep 22/22; vllm_serve column) | — |
| 47 | NDIF remote region | `model.trace(..., remote=True)` / `remote='local'` / `backend=URL` | `src/nnsight/modeling/mixins/remoteable.py:31-73`; `docs/remote/remote-trace.md:13` | edge (run shipped to a remote engine) | ABSENT | out-of-scope v1 per decision F2 (NDIF deferred); reserve a future backend column `ndif`, do not add rows now |
| 48 | NDIF session bundling | `model.session(remote=True)` | `src/nnsight/modeling/mixins/remoteable.py:76-110` | edge (run-DAG shipped as one job) | ABSENT | out-of-scope v1 (same F2 reason) |
| 49 | non-blocking jobs | `blocking=False`, `job_id`/`job_status`, polling, webhook | `docs/remote/non-blocking-jobs.md:13` | out-of-model (job lifecycle, scheduling) | ABSENT | out-of-scope v1 (same F2 reason; revisit with the multi-tenant regime) |
| 50 | hybrid local step | `with tracer.local():` (server pushes a fn back mid-job) | `src/nnsight/modeling/mixins/remoteable.py:234`; `docs/usage/session.md:83` | edge (bidirectional, remote ↔ meta) | ABSENT | out-of-scope v1 (NDIF) |
| 51 | code shipping | `nnsight.register(module)` / `nnsight.ndif.register` | `src/nnsight/ndif.py:22`; `docs/remote/register-local-modules.md:13` | out-of-model (serialization support for #47) | ABSENT | out-of-scope v1 (NDIF) |
| 52 | NDIF introspection | `nnsight.status()` / `is_model_running()` / `compare()` | `src/nnsight/ndif.py:247,326` | out-of-model (service introspection) | ABSENT | out-of-scope: not a program element |

### 1.4 Addresses (Level-1 sites) and address-space machinery

| # | primitive | surface | evidence | slot | status | action (if ABSENT) |
|---|---|---|---|---|---|---|
| 53 | vLLM logits site | `model.logits` (read or write; iterates per step) | `src/nnsight/modeling/vllm/vllm.py:113-121` | address (engine tier) | MEASURED (== portable unembed; F-12) | — |
| 54 | vLLM samples read | `model.samples` | `src/nnsight/modeling/vllm/vllm.py:123-133` | address (engine tier) | MEASURED (== greedy argmax; F-12) | — |
| 55 | vLLM samples WRITE (forced decoding) | `model.samples = ids` | same eproperty (writable) | data-op (write × engine site) | ABSENT | new row + new probe: forcing the sampler is the substrate for constrained/verified decoding workloads |
| 56 | derived sites (head h / neuron j / position p) | reshape/slice of boundary values | `docs/patterns/per-head-attention.md`; micro probes | address (derived tier) | MEASURED read (F-12); **derived-site WRITE LISTED-UNTESTED** (roadmap item 4, per-head ablation) | build the per-head-ablation cell (Part 5) |
| 57 | subspace / direction-valued addresses | `write(a, R_θ(...))`-style projections (DAS rotations, steering directions) | program-model draft §8 Q4; pyvene RotatedSpace | address (derived tier, proposed new member) | ABSENT | adopt into §3.2 as derived-tier citizens (Part 2/Part 4 decision); no probe until the DAS-class cell exists |
| 58 | site aliasing | `LanguageModel(repo, rename={'.model.layers': '.layers'})` | `src/nnsight/intervention/envoy.py:83,95-98,1086-1135`; `docs/usage/rename-modules.md:13` | out-of-model (address-space metadata; changes which names exist) | ABSENT | out-of-scope as inventory row; **adopt as a harness tool**: rename is how the bench fulfils its vary-the-names testing rule without new model families |
| 59 | address enumeration | `model.get(path)` / `.modules()` / `.named_modules()` | `src/nnsight/intervention/envoy.py:537-603` | out-of-model (harness enumeration) | ABSENT | out-of-scope: harness machinery, not a program element |
| 60 | reserved-name remount | name collision → `.nns_output` | `docs/usage/access-and-modify.md:139` | out-of-model (address-space quirk) | ABSENT | out-of-scope; document in the catalog's Level-1 notes only |

### 1.5 Realizations not already listed above

| # | primitive | surface | evidence | slot | status | action (if ABSENT) |
|---|---|---|---|---|---|---|
| 61 | clone-before-modify | `before = x.clone().save(); x[:] = 0` | `docs/usage/access-and-modify.md:95` | realization (read-before-write idiom) | ABSENT | fold into existing write probes as a denotation check (saved-var-is-a-reference is a SILENTLY_WRONG generator); no standalone row |
| 62 | unembed: module call vs weight matmul | `lm_head(h)` vs `F.linear(h, W)` | smoke cells | realization of meta-compute | MEASURED (F-2) | — |
| 63 | aux compute under no_grad | bare vs `torch.no_grad()` | smoke cells | realization of meta-compute | MEASURED (F-1) | — |
| 64 | fused-residual read | `out[0]` vs `out[0] + out[1]` | `isb/methodologies/logit_lens.py` | realization (denotation repair) | MEASURED (F-7) | — |

### 1.6 Modes, staging, extension plane, non-LM classes

| # | primitive | surface | evidence | slot | status | action (if ABSENT) |
|---|---|---|---|---|---|---|
| 65 | vLLM engine mode | `VLLM(repo, mode='sync'\|'async')` | `src/nnsight/modeling/vllm/vllm.py:70-76,469-476` | mode (context axis) | PARTIALLY MEASURED (async = bench backend; sync statuses via construct-gaps repros only — catalog flags under-representation) | promote `vllm_sync` to a bench backend (Part 5) |
| 66 | vLLM sampling kwargs | `temperature=, top_p=, seed=, logprobs=` root-level + per-invoke override | `docs/models/vllm.md:116,349` | mode | ABSENT | out-of-scope for the oracle (non-greedy breaks determinism — the catalog's inclusion criterion); the per-invoke *heterogeneity* mechanism may matter for the multi-tenant regime later |
| 67 | vLLM deployment knobs | `tensor_parallel_size=, distributed_executor_backend=, gpu_memory_utilization=, dispatch=` | `docs/models/vllm.md:49` | mode (context: §6 sweep axes) | covered by the §6 sweep matrix, not primitive rows | — |
| 68 | CONFIG switches | `CONFIG.APP.CROSS_INVOKER / PYMOUNT / REMOTE_LOGGING; API.HOST` | `docs/reference/api-quick-reference.md:96` | mode (semantics-changing variance params) | ABSENT | record as context/variance parameters in cell metadata (CROSS_INVOKER directly gates row 37); not primitive rows |
| 69 | edit lifecycle | `model.clear_edits()` | `src/nnsight/intervention/envoy.py:365-369` | staging (lifecycle helper) | ABSENT | out-of-scope: bookkeeping, no engine coupling |
| 70 | edit persistence | `model.export_edits(...)` / `import_edits(...)` / constructor `import_edits=` | `src/nnsight/intervention/envoy.py:371-425`; `src/nnsight/modeling/huggingface.py:65,96`; `docs/usage/edit.md:92` | staging (cross-process persistence) | ABSENT | new row UNTESTED, probe deferred: staging × serialization is exactly the F-16 failure class, now across processes; measure when staging matters on a second backend |
| 71 | custom envoys + eproperty | `NNsight(m, envoys={...})`; `@eproperty` + `preprocess/postprocess/transform/provide` | `src/nnsight/intervention/envoy.py:84-113,630-667`; `interleaver.py:71-346`; `docs/usage/extending.md:32,105,237` | out-of-model (extension plane — defines what a site IS) | ABSENT | out-of-scope as rows; note in §3.2 that first-class derived sites (per-head views) are buildable this way — if nnsight ships one, the derived-site rows re-measure against it |
| 72 | Batchable extension | `_prepare_input/_batch` overrides | `docs/usage/invoke-and-batching.md:107` | out-of-model (wrapper extension) | ABSENT | out-of-scope: harness-side plumbing |
| 73 | non-LM model classes | `NNsight(module)` / `VisionLanguageModel` / `DiffusionModel` | `src/nnsight/modeling/{base.py:8,vlm.py:17,diffusion.py:221,475}` | mode (model-type context axis, §6) | ABSENT (bench is causal-LM-only per F2) | out-of-scope v1 per F2; the §6 model-type axis already reserves the slot |
| 74 | event protocol | VALUE/SWAP/SKIP/BARRIER/END/EXCEPTION on the Mediator queue | `src/nnsight/intervention/interleaver.py:349-357,1313-1395` | out-of-model (the lowering target) | n/a | adopt as a §3.1 note: the implementation's own boundary protocol confirms the boundary-crossing criterion — every user surface lowers to read(VALUE)/write(SWAP)/skip/sync/end/error. Ground truth, not a row |
| 75 | device management | `model.to/cuda/cpu/.device/.devices` | `src/nnsight/intervention/envoy.py:487-535` | out-of-model (resource placement) | ABSENT | out-of-scope: context, not program |

**Tally:** 75 rows. MEASURED 32 · LISTED-UNTESTED 2 (cache; derived-site write) · partially
measured 1 (engine mode) · ABSENT 40. Of the 40 ABSENT: **14 get new UNTESTED rows/probes**
(rows 5, 6, 8, 10, 11, 14, 20, 23, 25, 37, 42, 44, 45, 55, plus baseline 18 and check 61 folded
into existing probes), **3 get adopted as vocabulary/harness decisions** (29 sweep, 57 subspace
addresses, 58 rename-as-test-tool), and the rest are explicitly out-of-scope with the reason
recorded above (NDIF plane = decision F2; deprecated forms; extension/harness plumbing;
non-greedy sampling vs the determinism criterion).

Source-traverse negative results worth recording: `Envoy.wait_for` does not exist on this
branch; there are no envoy-level barrier/stop/cache; legacy `nnsight.log/apply/list/dict/
local/cond` are fully removed.

---

## Part 2 — Component clarification per level

Adopts the program-model draft's criteria wherever they are sharper than the current §3 prose.
Each level: definition, membership criterion (a decidable test), complete component list from
the traverse, rationale. This section resolves the inconsistencies the considerations sweep
flagged (COMPUTE's status, the 1.5/2 boundary, the triple-categorized region parameters, the
fork-as-program-vs-context contradiction, the undefined scope position).

### Level 0 — data primitives

**Definition.** The operations that move a value across the object/meta boundary between a model
run and the experiment program.

**Membership criterion (adopted from the draft, replacing the enumerated framing):** *an
operation is a data primitive iff it crosses the object/meta boundary.* Implementation ground
truth: it must lower to a Mediator boundary event (`src/nnsight/intervention/interleaver.py:349-357`).

**Components (closed):**
- `read(r, addr)` — VALUE event. Surfaces: `.output`, `.input`, `.inputs`, `.source.<op>`,
  `logits`, `samples`, `tracer.result`, `t.grad` (a read on the derivative graph).
- `write(r, addr, v)` — SWAP event (plus the in-place and skip realizations). Semantics: the
  counterfactual run. Surfaces: `.output = …`, `.input = …`, `x[:] = …`, `.skip(v)`,
  `samples = …`, `t.grad = …`.
- `grad(r, addr; metric)` — kept a primitive (draft open question 1, **decided**: methodologies
  treat it as one, and operationally the derivative graph may not exist at all — F-11 — which is
  exactly the status the inventory needs to carry; linguistically it remains "read on G′").

**What this demotes.** `COMPUTE` leaves Level 0: it is meta-level code (the trace body is real
Python; parameters are readable constants). The measured COMPUTE rows (F-1, F-2) are **not
discarded** — they become *realization rows of meta-compute in a context* ("aux compute needs
`no_grad` on vLLM"; "module-call unembed is guarded, weight-matmul works"). `save` also stays
out: it is the live-out edge (already the catalog's position). This resolves the
self-undermining "closed set member that is also ambient capability" flagged by the sweep.

**Rationale.** A criterion beats a list: rows 5, 10, 11, 55 of the traverse (input-write,
grad-write, skip, samples-write) are all immediately classifiable — each crosses the boundary —
whereas the old enumerated table needed a new debate per surface.

### Level 0 — control quantifiers

**Definition.** The constructs that determine which runs exist and where in their coordinates
the boundary crossings attach.

**Membership criterion:** *a control construct is named iff it quantifies crossings over runs,
steps, addresses, inputs, or run order.* Engine coupling is no longer the membership test — it
is a **property** (some quantifiers are engine-coupled, some are host-side and free). This
admits sweep, which the old engine-coupling criterion wrongly excluded.

**Components (complete, from the traverse):**

| quantifier | over | engine-coupled? | nnsight surfaces (traverse rows) |
|---|---|---|---|
| **run** (region) | one run's existence/extent | yes (request lifecycle; steps=N; mode) | trace, generate, scan, backward-region, `tracer.stop` truncation (15–18, 25, 27) |
| **step** | the run's unrolled time | yes (decode loop) | `iter` bounded/unbounded, `next` (21–23) |
| **sweep** | addresses | **no — host-side, free** | plain Python loop (29) |
| **dataset** | inputs (independent runs) | only in the batched realization | prompt lists; `invoke` incl. empty invoke (19–20) |
| **adaptive** | future runs from past results | no | host `while`/`if` (30) |
| **run DAG** | run ordering with data dependence | yes (multi-request grouping) | `session`; two traces (26) |
| (sync) | fork branches | yes | `barrier` (24) — a coordination construct whose only observable is its edge |

**Scope position, defined** (closing the sweep's "asserted but never defined" gap): a data-op
instance's scope position is its path in the engine-side scope tree
`session > trace > invoke > step` — exactly the coordinates the engine knows. Host-side
quantifiers (sweep, dataset-as-sequential-runs, adaptive) do **not** occupy scope positions;
they multiply *instances*, each of which has its own scope position. A footprint writes scope
position as the innermost engine scope the crossing fires in (e.g. `write@step`,
`read@invoke[1]`, `read@trace` default).

**Fork: program or context — resolved.** The dataset quantifier is a *program* element. Whether
it is realized as one batched trace (`invoke`) or N sequential traces is a *realization* of the
dataset quantifier. The *workload regime* (§5) stays context: it describes the input
distribution and concurrency the cell is run under. The old §3.1 gloss "fork = the batched
regime" is deleted; the dataset-lift law (Part 3) is what connects the two realizations, and the
batched-GPT-2 effect is a model-side failure of that law, not a definitional ambiguity.

### Level 1 — the address space

**Definition.** Names for coordinates `(module-path × token-position × generation-step
[× batch-member])` of the run's dataflow graph (and its derivative extension). Unchanged in
substance; two amendments:

**Membership criterion:** *a name belongs to the address space iff read/write/grad can target it
and its denotation is checkable per context* (exists? / denotes what? / writable?).

**Components:** engine tier (`logits`, `samples`, `tracer.result`, `generator.output`,
`streamer.output`) · boundary tier (`.output`/`.input`/`.inputs` at any depth) · internal tier
(`.source.<op>`, incl. nested `.source`) · derived tier (head/neuron/position views — **and,
newly adopted, subspace/direction-valued addresses**: a projection `(site, R)` is a derived
NAME, realized as read∘project / project-write; sweep ranges over them like any address — the
Part 4 pyvene decision) · gradient tier (`.grad` of any of the above; exists only under the
backward region).

**Rationale.** The amendments make per-head ablation's decomposition definite (the sweep's
derived-site unclarity): a head-sliced write is **WRITE × derived address**, where the derived
address's *realization* is reshape-slice-reassemble at the boundary site. Derived addresses are
names; how a name is reached is its realization — the site/op blur dissolves once realization
attaches to addresses too (next subsection).

### Level 1.5 — realizations

**Definition (generalized — fixing the self-violation the sweep found):** *a realization is a
concrete spelling of ANY language element — data op, quantifier, edge, or address — such that
all spellings have the same language-level meaning.* Contexts differ in which spelling works;
"the working recipe per backend" = a realization choice per element.

**Membership criterion:** two surfaces are realizations of one element iff a language-level
semantics-preserving rewrite connects them (replacement ↔ in-place; bounded ↔ unbounded ↔
`next`; batched invoke ↔ sequential traces; barriered ↔ two-trace transfer; `out[0]` ↔
`out[0]+out[1]` for the same denotation).

**Components (from the traverse):** write: in-place / replacement / skip-with-value ·
meta-compute: module-call / weight-matmul; bare / no_grad · step quantifier: bounded / unbounded
/ `next` · dataset quantifier: batched invoke (incl. empty invoke) / sequential traces · run→run
transfer: barriered / un-barriered cross-invoke / two-trace via host / session variable ·
residual read: plain / fused-sum · read-before-write: clone-first · live-out: in-process /
streaming drain / over-the-wire.

**Rationale.** "Loop step-selector" and "cross-prompt transfer" were never Level-0 ops — they
are realizations of the step quantifier and of the run→run edge respectively. With realization
defined element-generically, the old table stops violating its own definition.

### Level 2 — entries (and the 1.5/2 membership rule)

**Definition.** The finite catalog of tuples
`(data-op | edge) × address-tier × scope-position`, each carrying a **realization coordinate**.
This — with one rule — is the enumerable substrate §3.5's completeness criterion was missing.

**The disambiguating rule (resolving the double-classification of F-13/F-14):** *realizations
are coordinates ON entries, never entries.* An inventory row is an L2 entry; where statuses
differ by spelling, the row splits by its realization coordinate. So bounded-vs-unbounded is ONE
L2 entry (the loop-carried accumulation edge, step→run) with two realization values, of which
one is ERROR on vLLM; barrier-vs-two-trace is ONE entry (the run↔run transfer edge) with
realization values {barriered, two-trace, un-barriered, session-var}. A failure's *kind* is then
read off the coordinate that varies: same entry, one realization red → realization-unsupported;
all realizations red → edge/op-unsupported; name missing → site-absent; name red-denotation →
denotation-mismatch.

**Edge classification (adopted from the draft, source→destination):** observation (read→emit) ·
rewiring (read→compute→write, same run, downstream) · transplant (read run A → write run B) ·
injection (meta-constant → write) · accumulation (reads → meta-state → later write/analysis;
makes the `trained` tag structural) · derivative (grad→emit). The current §3.4 movement table is
these edges viewed from below; **rewiring and accumulation are the two with no measuring cell**
(Part 5).

**Enumerated membership** (closing the sweep's "uncomputable completeness" gap): the L2 catalog
is generated, not curated — the cross product of the Part 1 traverse's data ops and edges ×
address tiers × scope positions, filtered to combinations any cataloged method's footprint
names. The catalog doc gains this enumeration as a table (Part 5, catalog edit C2); coverage =
footprint-needed entries minus probe-or-cell-exercised entries, now mechanically computable.

Scaling parameters (breadth, tensor size, depth, payload, side-compute) remain measures on
entries — with breadth now explicitly the sweep quantifier's measure.

### Level 3 — methodologies

**Definition (adopted from the draft).** A methodology is a meta-program over the language:
boundary crossings + quantifiers + edges, plus a readout metric and semantic intent. The space
is structured as **base programs × transformations** (step-lift, dataset-lift, aggregation,
linearization, amortization).

**Membership criterion (for a bench cell):** deterministic (greedy/argmax), oracle-checkable,
and its footprint is derivable from its program text (Part 3).

**Components:** the registry (§4) unchanged; each entry now also names its base program and the
transformations applied (e.g. generation-time steering = steering ∘ step-lift; mean ablation =
ablation ∘ aggregation; attribution patching = activation patching ∘ linearization — a
*scientific approximation*, flagged as such; tuned lens = logit lens ∘ amortization).

**Completeness, redefined:** the method tier is complete when (a) every L2 entry any cataloged
footprint names is exercised by at least one cell, AND (b) every edge type and every
transformation is exercised at least once. (b) is the language-level requirement the draft adds;
it is what made gen-steering "the step-lift transformation, verified" rather than just another
method.

### Context

Unchanged: `family × backend × engine-config × parallelism × workload-regime`, orthogonal to all
levels. Two clarifications: engine mode (sync/async) is hereby an explicit context coordinate
(F-14/F-15 statuses differ by it); and "regime effect" is renamed in place to **model-side law
failure** (§3.6 edit, Part 5) — the dataset-lift law failing for absolute-position families is
the measured instance. The failure-kind taxonomy keeps its seven kinds but each now has a
*decision procedure* via the Level-2 rule above, instead of per-failure taste.

---

## Part 3 — Level ↔ methodology interaction (the mechanical account)

The maintainer's unclear part. Four mechanisms, each specified, then two worked examples.

### 3a. Footprint derivation is syntactic

A cell's footprint is computed from its program text (or declared and checked against it — the
declaration is the contract, the text is the truth):

1. Every boundary crossing in the text contributes an L2 entry reference
   `(op, address-tier, scope-position, realization)`. E.g. `blk.output = (h + v, out[1])` inside
   `for step in tracer.iter[0:N]` contributes `write × boundary.block × step × replacement`.
2. Every quantifier in the text contributes its name + realization
   (`step/bounded`, `dataset/batched-invoke`, `sweep` with its breadth).
3. Every meta-level dataflow edge contributes its source→destination class
   (observation / rewiring / transplant / injection / accumulation / derivative), plus the
   live-out for whatever is saved.
4. Modes (fake, steps=N) and semantics-relevant CONFIG values are recorded as entry parameters.

Machine-readable form (the substrate F-17's "derivation" currently lacks — closes the sweep's
"footprints exist only as prose" gap, within §12.6's pure-metadata constraint):

```python
@cell("gen_steering", family="gpt2", backend="vllm_async",
      footprint=[
        "write/boundary.block/step/replacement",
        "quant.step/bounded",
        "edge.injection",                      # meta-constant direction → write
        "edge.accumulation.loop-carried",      # per-step rows.append
        "edge.live-out",
      ])
```

Pure data, consumed only by reporting/derivation; cells stay flat and explicit (§12 untouched).

### 3b. Expected states derive from footprint ∧ measured map — with ∧ defined

The undefined operator, now defined:

- **Status order (worse = more severe):**
  `SUPPORTED < SUPPORTED_DEGRADED < UNSUPPORTED_BY_CONSTRUCTION < ERROR < HANG < SILENTLY_WRONG`.
- **∧ = max-severity (meet in this order):** the predicted status of a method in a context is
  the worst measured status among its footprint's entries in that context.
  `SUPPORTED ∧ SUPPORTED_DEGRADED = SUPPORTED_DEGRADED`.
- **Realization quantification:** a footprint names the realization the cell actually uses, so
  the lookup is per-realization — a footprint naming `write/.../replacement` composes against
  the replacement row (SUPPORTED on vLLM), never against "WRITE in general". "Some realization
  exists" is a *recipe* statement, not a composition input.
- **Per-context splits:** engine mode is a context coordinate, so there is no merging problem —
  a barrier-using method inherits ERROR in `vllm_async` and SILENTLY_WRONG in `vllm_sync`,
  as two cells.
- **UNTESTED inputs:** any footprint entry with no measured row makes the prediction
  `UNDERIVABLE(missing: <entry>)` — never a guess. This is also the probe queue generator: the
  set of entries blocking derivations is exactly what the micro tier should probe next.

F-17 retroactively conforms: footprint {`write/boundary/step/replacement` (F-5 SUPPORTED),
`quant.step/bounded` (F-13 SUPPORTED), live-out (SUPPORTED)} ∧-composes to SUPPORTED;
measured SUPPORTED. The same spec's unbounded sibling composes to ERROR (F-13); measured ERROR.

### 3c. Where the laws enter

∧-derivation is only an *a priori* prediction; it silently assumes footprint elements compose
independently. The transformations' **laws** are exactly that assumption made explicit, so the
method tier's job statement becomes: **method cells test laws; micro probes test entries.**

| law | claim | tested by | measured so far |
|---|---|---|---|
| step-lift | per-forward intervention stays valid per step | base cell vs step-lifted cell | HOLDS on hf + vllm_async (F-17, exact) |
| dataset-lift | batched ≡ sequential per-prompt | batched-invoke cell vs sequential-traces cell | FAILS model-side for absolute-position families (batched GPT-2 positions) |
| sweep-exchange | sweep inside one run ≡ one run per address (non-interacting reads) | multi-layer cell vs per-layer cells | untested as a named law (logit-lens cells implicitly assume it) |
| composition | independent footprint entries compose (∧ is sound) | any method cell vs its ∧-prediction | one confirmation (F-17); known counterexamples upstream where entries share implementation state (nnsight batched-multitoken saves clobbering; PP iteration hang) |

A method cell's verdict protocol: `predicted = ∧(footprint, measured map)`; run; compare.
Agreement confirms both the entry statuses and the law instance. Disagreement is a finding in
itself and triggers the three-way attribution.

### 3d. The residue: verdict attribution for law failures

When measured ≠ predicted, attribute (the draft §5 recast, now operational):

- **model-side law failure** (was: "regime effect") — the scientist's lift was invalid for this
  model (absolute positions under dataset-lift). Recorded as a per-cell expected override with
  `attribution: model`; it does NOT impeach the entry statuses or ∧. This is the §3.6 residue
  class, kept explicit, now with a definition instead of a vibe.
- **implementation-side composition failure** — entries interact through shared engine state
  (the upstream saves-clobbering class). Recorded `attribution: implementation`; generates a
  composition cell (the pair becomes a permanent regression row).
- **numerics** — bf16 near-ties; the dtype control decides (F-8's
  `disambiguate_precision` already implements this) → `SUPPORTED_DEGRADED`, `attribution:
  numerics`.

### 3e. Worked example 1 (read method): logit lens, GPT-2 and Llama-family

**Program** (draft §4): `r = run(x); for ℓ in layers: emit softmax(unembed(norm(read(r, resid[ℓ, last]))))`.

**Syntactic footprint:** `read/boundary.block/trace/⟨residual realization⟩` × sweep(breadth=L) ·
meta-compute (norm + unembed, realization: weight-matmul + no_grad on vLLM) · edge.observation ·
edge.live-out.

**Per-backend ∧-prediction:**
- hf: read SUPPORTED ∧ live-out SUPPORTED → **SUPPORTED**.
- vllm_async, GPT-2, weight-matmul realization: read SUPPORTED ∧ live-out SUPPORTED →
  **SUPPORTED**. With the module-call compute realization: that realization row is ERROR (F-2) →
  **ERROR**.
- vllm_async, Llama-family: the boundary read's denotation coordinate forces the fused-sum
  realization (F-7). Plain-read realization → denotation mismatch → **SILENTLY_WRONG** predicted;
  fused realization → **SUPPORTED**.

**Measured verdicts:** hf SUPPORTED; vLLM weight-matmul SUPPORTED (top1=1.00/TV=0.021); vLLM
module-call ERROR (F-2); vLLM-Llama plain SILENTLY_WRONG (top1=0.13/TV=0.897), fused SUPPORTED
(top1=0.97/TV=0.017) — `findings.md` smoke tables. **Prediction = measurement in all five
cells; no law invoked (single run, observation edge only — sweep is free).**

### 3f. Worked example 2 (write method): steering → generation-time steering

**Base program:** `r = run(x); write(r, resid[ℓ,last], read(...) + α·d); emit read(r, logits)` —
injection edge, write × boundary × trace.

**Base prediction/measurement:** hf SUPPORTED (both write realizations); vllm_async in-place
realization ERROR, replacement SUPPORTED — F-5 measured exactly that split (smoke steering
table). The non-vacuity guard (F-6) is part of the verdict protocol for every write method.

**Transformation:** step-lift — wrap the write in the step quantifier; per-step logits read adds
a loop-carried accumulation edge.

**Lifted footprint:** `write/boundary.block/step/replacement` · `quant.step/⟨bounded|unbounded⟩`
· edge.injection · edge.accumulation.loop-carried · edge.live-out.

**∧-prediction per context:** hf → SUPPORTED either realization. vllm_async bounded →
SUPPORTED; unbounded → ERROR (F-13's loop-carried row).

**Measured verdict (F-17):** bounded SUPPORTED top1=1.00/TV=0.000 (identical greedy trajectory —
the step-lift LAW holds exactly, and ∧ was sound for this pair); unbounded ERROR as predicted
(the realization coordinate carries the failure, the law is untested through it). Attribution
machinery never fires — agreement everywhere.

---

## Part 4 — External alignment (decided recommendations)

**4a. pyvene / causal-abstraction units → adopt subspace addresses; align tags to intervention
types.** pyvene treats intervention units (incl. rotated subspaces, DAS) as first-class; our
address space stops at tensor slices. **Decision:** derived-tier addresses gain
subspace/direction-valued members (Part 2, Level 1) — a projection is a derived NAME with a
realization, not a new op; the sweep quantifier ranges over them. This keeps the language small
(write-at-a-subspace = rewiring with a parameterized projection, exactly the draft's
`write(a, R_θ(...))`) while making DAS/RepE footprints expressible. Also: keep the §4 "borrow"
note but make it concrete — map footprint edge classes to pyvene's enum (Collect=observation,
Addition/Subtraction=injection, Vanilla interchange=transplant, RotatedSpace=subspace rewiring,
LoRA=staging) in the catalog's tag table, for cross-framework legibility.

**4b. TransformerLens hook-point naming → do NOT canonicalize paths; canonicalize footprint
site IDs only.** TL's lesson is double-edged: stable canonical names (`blocks.{i}.hook_resid_post`)
made methods portable, but only by rewriting every model into one reference implementation —
precisely the "one abstraction over all families" move §12 rejected, and it forfeits fidelity to
the engine under test (we benchmark the *production* forward, TL replaces it). **Decision:**
cells keep explicit per-family paths (§12 untouched); the *footprint vocabulary* uses
tier-level site IDs (`boundary.block.output`, `internal.attn-weights`) — TL-style stability
where it costs nothing (metadata), engine fidelity where it matters (code). nnsight's `rename=`
(`src/nnsight/intervention/envoy.py:1086-1135`) is adopted as the harness's vary-the-names test
tool, not as canonicalization.

**4c. MLPerf coverage-unit lessons → closed/open division; per-round version stamps; no single
score.** **Decision:** (i) each methodology gets a *closed* canonical cell (fixed prompts,
params, oracle, tolerance — comparable across backends/versions) and may have *open* variants
(realization experiments, perf tuning) that never enter the applicability map's headline;
(ii) the coverage unit is the L2 entry × context (Part 2's enumerated catalog), version-stamped
the way MLPerf stamps rounds — design.md §3.7 already says the primitive-status map is the
version-sensitive artifact, this makes the unit explicit; (iii) the deliverable stays the
multi-valued map — explicitly no aggregate score, matching the §1 guideline goal.

**4d. Production engines' tiny primitive sets → the frontier tiers are the priority order.**
Production serving APIs expose ~generate + logprobs + (sometimes) per-token callbacks; nothing
exposes write, grad, or cross-request state. Implication, decided as a tier priority: the
portable core (read/write-replacement/bounded-step, measured SUPPORTED) is table stakes — the
benchmark's differentiating coverage is, in order, (1) the edges (measured: the frontier),
(2) grad (measured: ERROR — a whole method class), (3) staging (measured: ERROR), (4) the
streaming/serve live-out realizations (traverse rows 45–46) — because these are exactly what
production engines do NOT give you and what an interp-serving layer must add. New-probe
ordering in Part 5 follows this.

---

## Part 5 — Extension plan (concrete, ordered)

### A. New catalog rows — **ALL status UNTESTED** (no measurement is ever invented)

Add to `interp-methods-catalog.md` inventory, grouped where they'll live:

Data ops / addresses:
1. input WRITE — `module.input = x` (traverse row 5).
2. kwargs-site read — `module.inputs` (row 6).
3. module skip — `module.skip(replacement)`, write realization with compute elision (row 11).
4. gradient WRITE — `t.grad = g` mid-backward (row 10).
5. samples WRITE (forced decoding) — vLLM engine site (row 55).
6. hooked aux application — `envoy(x, hook=True)`, the ext-module observability half (row 14).
7. recursive `.source` — nested internal site (row 8).

Control / edges:
8. early region exit — `tracer.stop()`: pre-stop save preservation + engine survival + compute
   saved (row 25).
9. manual step advance — `tracer.next()` as the third step-quantifier realization (row 23).
10. empty invoke — full-batch fork realization / escape hatch (row 20).
11. un-barriered cross-invoke flow — CROSS_INVOKER push/pull realization of run↔run transfer
    (row 37).
12. `tracer.result` — engine-result site + live-out (row 42).
13. async streaming drain — finished-only saves, single-shot generator (row 45; makes the bench
    transport's own assumptions a measured row).
14. staging persistence — `export_edits`/`import_edits` (row 70; probe deferred).
15. HF streamer per-step token site (row 44; probe deferred, fold into iteration probes).

### B. New micro probes worth building (ordered by Part 4d priority + roadmap continuity)

1. **`tracer.cache`** — already roadmap Tier-1 item 2; the last fused L2 entry. (Known upstream
   PP merge gap is a finding-in-waiting, not a reason to skip.)
2. **skip** (A3) — frontier-shaped: SKIP is its own Mediator event, so its vLLM status is not
   derivable from the SWAP rows.
3. **stop** (A8) — perf primitive + plausible engine-lifecycle frontier on vLLM.
4. **un-barriered cross-invoke flow** (A11) — the accident-prone realization; SILENTLY_WRONG-
   shaped risk.
5. **input WRITE + kwargs read** (A1, A2) — one probe pair; unlocks the transcoder-splice
   footprint.
6. **samples WRITE** (A5).
7. **tracer.next + empty invoke + tracer.result** (A9, A10, A12) — cheap additions to existing
   iteration/fork probes.
8. **grad WRITE** (A4, HF) and **hook=True** (A6) — unlock gradient-editing and SAE-latent
   footprints.
9. **trace=False baseline** (row 18) — harness overhead floor, not an inventory row.
10. **`vllm_sync` backend** (row 65) — promotes the construct-gaps sync statuses from
    "measured via external repros" to bench-measured; the barrier SILENTLY_WRONG row is the
    motivating cell.
11. Probe-side checks folded into existing probes: clone-before-modify denotation check
    (row 61); F-7 fused-residual check already exists.

### C. New method cells (each = one unexercised edge type or transformation, per the Part 2
completeness criterion)

1. **Rewiring cell** — path-patching-style same-run read→compute→write downstream (the edge no
   cell exercises; also draft §7 item 5). Cheapest deterministic form: SAE-free linear splice.
2. **Accumulation cell** — mean ablation (aggregation transformation; cheapest trained-state
   proxy; makes the `trained` tag structural).
3. **Dataset-lift law cell on a relative-position family** — the positive control for the
   measured GPT-2 absolute-position failure (attribution: model).
4. **Generation-time cross-prompt patching** — transplant × step-lift (roadmap item 3; completes
   the causalab audit's second flagged prediction).
5. **Per-head ablation** — WRITE × derived address (roadmap item 4).
6. **DLA** — observation × internal site (roadmap item 5).
7. **Sweep-exchange law cell** — multi-layer-one-run vs one-run-per-layer logit lens (free data
   from existing cells; names the law).

### D. design.md §3 edits (section NUMBERING stable; §3.5/§3.6 externally cited)

| section | change |
|---|---|
| §3.1 | Adopt the boundary-crossing criterion for data primitives (read/write/grad; COMPUTE demoted to meta-level with its measured rows re-homed as meta-compute realizations; save stays an edge). Replace the engine-coupling *membership* test for control ops with the quantifier list (run/step/sweep/dataset/adaptive/run-DAG + sync), engine coupling becoming a per-quantifier property. Define scope position (engine scope tree path; host quantifiers multiply instances). Delete "fork = the batched regime"; fork = the dataset quantifier's batched realization. Add the event-protocol ground-truth note (traverse row 74). |
| §3.2 | Add subspace/direction-valued addresses to the derived tier. State that derived addresses are names whose access path is a realization (resolving the site/op blur); require the per-site writable? property to be carried by the inventory. |
| §3.3 | Generalize realization to all language elements; move "loop step-selector" and "cross-prompt transfer" rows under their elements (step quantifier; run↔run edge). |
| §3.4 | Add the source→destination edge derivation (observation/rewiring/transplant/injection/accumulation/derivative); add rewiring + accumulation as named edges; state the L2 membership rule (realizations are coordinates on entries, never entries) and the generated-enumeration definition of L2. |
| §3.5 | Add the syntactic footprint-derivation procedure and the machine-readable footprint schema (`@cell(..., footprint=[...])` — metadata only, §12 untouched). Completeness restated: cover every footprint-needed L2 entry AND every edge type + transformation once. |
| §3.6 | Define ∧ (max-severity over the ordered status set, per-realization, per-context; UNTESTED ⇒ UNDERIVABLE). Reframe the taxonomy as decision procedure (which coordinate varies). Rename "regime effect" → "model-side law failure" (keep the old term parenthesized for external citations); add the three-way attribution (model / implementation / numerics). |
| §3.7 | Method tier restated as law-checking (step-lift, dataset-lift, sweep-exchange, composition); micro tier = entry-checking; the UNDERIVABLE set is the probe queue. |

**Program-model-draft elements adopted into normative design.md:** the boundary-crossing
criterion (draft §1); the five quantifiers incl. sweep (draft §2); the source→destination edge
set incl. rewiring + accumulation (draft §3); base-programs × transformations and syntactic
footprints (draft §4); laws + the regime-effect recast + three-way attribution (draft §5); the
roadmap cells (draft §7 item 5). Draft open questions decided here: Q1 grad stays a primitive;
Q4 subspace addresses become derived-tier citizens. Q2 (mid-run adaptivity) and Q3 (token
alignment conventions) remain open — Q3 flagged as a real silent-error source the oracle does
not currently see.

### E. interp-methods-catalog.md edits

1. **Tag vocabulary extension** (closes the "footprint vocabulary can't express where methods
   fail" gap): add tags for quantifiers (`step/bounded`, `step/unbounded`, `dataset/batched`,
   `dataset/sequential`, `sweep`, `run-dag`, `sync`), edges (`live-out`, `loop-carried`,
   `transplant`, `injection`, `rewiring`, `accumulation`, `derivative`), staging (`edit-replay`,
   `edit-persist`), and realizations as suffixes (the gen-steering row's improvised
   "write × iteration" becomes `write/boundary/step/replacement + step/bounded`). Add the
   pyvene-enum mapping column (Part 4a).
2. **L2 enumeration table**: the generated entry list (Part 2, Level 2) with per-context status
   cells — measured rows carry their F-n citation; everything else UNTESTED. This is the
   coverage denominator §3.5 needs.
3. **New UNTESTED rows** from list A above, placed in the data-op / control / edge / site
   tables as appropriate.
4. **Method tables**: re-express each method's primitives column in the extended tag vocabulary;
   add base-program × transformation annotations.
5. **Roadmap**: replace the current Tier-1 queue with Part 5 B+C ordering (cache and
   generation-time cross-prompt patching stay at the top, unchanged in substance).

---

## Constraint compliance

- **No invented measurements:** every proposed row in Part 5A is UNTESTED; Part 1 status column
  only repeats catalog/findings citations (F-n) for MEASURED rows.
- **file:line citations** on all traverse claims (Part 1, columns 3).
- **§12 untouched:** cells stay flat and explicit; footprints are registration *metadata*
  consumed by reporting/derivation only (§12.6's existing carve-out).
- **Language-level framing preferred:** Parts 2–3 are written in the program-model draft's
  vocabulary; the implementation map stays subordinate (one ground-truth note, row 74).
