# Interpretability methods — catalog & nnbench roadmap

A catalog of well-recognized interpretability methods, tagged by **what models they apply to** and
**which primitives they need** — their Level-3 *footprint* (design.md §3.5). The point is to decide
which become nnbench methodologies: a method is a cheap deterministic cell, a new backend frontier,
or a heavyweight needing a trained artifact *based on the primitives it touches*.

All methods below are **deterministic** (greedy / argmax, no sampling) → oracle-checkable, which is
the inclusion criterion for an nnbench cell. The sampling-based verifier/oracle direction (BOA /
BEAVER) is parked separately.

## The primitive inventory — leveled (status per context)

The model (normative definitions in design.md §3): **Level 0** core ops + control constructs ·
**Level 1** address space (sites) · **Level 1.5** realizations (idioms) · **Level 2** derived
primitives (op × site × time × realization) · **Level 3** methodologies (programs with footprints) ·
**context** (family × backend × config × regime), orthogonal to all levels.

This section is the single maintained **per-context status inventory**. Status values are
bench-measured (citations `F-n` = `findings.md`). Levels 0–1 are now FULLY measured for
(GPT-2 × hf/vllm_async) by the **micro tier** — one minimal probe per row with a self-contained
denotation check (`scripts/micro.py`, results in `results/micro_{hf,vllm_async}.txt`).

### Level 0 — core ops × backend

| op | hf | vllm_async / vllm_serve | evidence |
|---|---|---|---|
| READ | ✓ | ✓ — with per-family denotation caveats (see Level 1) | logit-lens cells; F-7 |
| WRITE | ✓ (both realizations) | ✓ replacement only; in-place raises | F-5; steering/ablation/patching cells |
| COMPUTE | ✓ | ✓ under `torch.no_grad()` (activations are inference tensors); module-call realization partially guarded | F-1, F-2 |
| SAVE | ✓ | ✓ in-process and over-the-wire (serve venue; payload measured) | all cells; serve sweep 22/22 |
| BACKWARD | ✓ (≈3.4× a single forward) | **ERROR** — inference mode, no autograd: the `grad` frontier | F-11 |

### Level 0 — control constructs × backend

| construct | hf | vllm_async | evidence / note |
|---|---|---|---|
| trace (single forward) | ✓ | ✓ | every cell |
| invoke (batched regime) | runs; **regime effect** on absolute-position families — batched GPT-2 diverges from its own per-prompt truth (pending the position_ids fix) | **ERROR** — gated on the dev checkout (async multi-prompt fix pending) | spec `expected` entries; `finding-batched-position-ids.md` |
| iteration — bounded `iter[0:N]` | ✓ | ✓ — 3 steps, step-0 == single-step trace | micro; F-13 |
| iteration — unbounded `iter[:]` / `.all()` | ✓ (stop bound = `default_all` from max_new_tokens) | **ERROR** — ALL saves dropped: the vLLM path never sets a stop bound, the loop overruns and is unwound by Cancelation before the body's final push (the documented idiom is the broken one) | micro; F-13; construct-gaps §1 |
| barrier | ✓ — barrier patch == two-trace patch | async: **ERROR** (no saves; stacks with the multi-prompt gate). Sync engine: **SILENTLY_WRONG** — clean exit, saved dict EMPTY (construct-gaps repros) | micro; F-14; nnsight `docs/developing/barrier-vllm-not-shared.md` |
| session — saved flow (`.save()` + read after exit) | ✓ | async: **ERROR** (no drain point inside a captured session body). Sync engine: **works** (construct-gaps repros) | micro; F-15; construct-gaps §3 |
| session — un-saved cross-trace flow (the session contract) | ✓ (\|Δ\|=0) | **ERROR** on both engines — only saves ship back from the worker; the surfaced UnboundLocalError misleadingly names the downstream variable | micro; F-15; construct-gaps §3 |
| edit | ✓ — edited trace == in-trace ablation; non-inplace edit isolated from the original | **ERROR** — the stored edit mediator fails to pickle into the vLLM worker (`PicklingError: source code unavailable`); the crash is protective — a serialization-only fix would silently drop the edit | micro; F-16; construct-gaps §4 |
| scan | ✓ — fake-mode shapes correct, no kernels | **ERROR** — dies building `SamplingParams` on scan's `hook=True` kwarg, before fake mode is entered | micro; F-16; construct-gaps §5 |

**Engine-mode caveat:** the bench's vLLM backend is the **async** engine; "sync engine" statuses
above are measured via the standalone repros in nnsight `docs/developing/vllm-construct-gaps.md`
("construct-gaps"), verified on vllm 0.19.1 and 0.15.1 — sync is a context (engine-mode) axis the
inventory under-represents until a `vllm_sync` bench backend exists. Where sync and async differ,
the difference is itself the finding (barrier: silent vs loud; session saved-flow: works vs not).

### Level 1 — sites × backend (existence / denotation)

| site tier | hf | vllm_async | evidence |
|---|---|---|---|
| engine (`logits`, sampled tokens) | ✓ — `model.output.logits` == `lm_head.output`; greedy `generator.output` id == logits argmax | ✓ — `model.logits` == portable unembed; `model.samples` == greedy logits argmax | micro; F-12 |
| boundary `.output` — block | ✓ | exists, but **denotes differently** on fused-residual families (Llama/Mistral/Qwen2/Gemma): `(hidden, residual)`, true stream = their sum | F-7 |
| boundary `.output` — submodule (attn/mlp) | ✓ | ✓ (ablation write target) | F-9 |
| boundary `.input` | ✓ — `h[6].input` == `h[5].output[0]`, exact | ✓ — same check, exact | micro; F-12 |
| internal `.source` — attention weights | ✓ (eager only) | **site absent** — paged/flash attention never materializes the matrix | F-10 |
| internal `.source` — other ops | ✓ — `mlp.source.self_c_fc_0` == `c_fc.output`, exact | ✓ — same check, exact (the vLLM MLP forward is plain Python, so `.source` rewrites it fine) | micro; F-12 |
| derived (head *h* / neuron *j*) | ✓ — head view validated by c_proj reconstruction; neuron by `gelu_new(c_fc)` | ✓ — same checks; weight-using reconstruction must run INSIDE the trace (the client-side envoy is the meta model) | micro; F-12 |
| gradient space (`.grad`) | ✓ | absent (no BACKWARD) | F-11 |

### Level 1.5 — realizations: the working recipe per backend

| abstract op | hf recipe | vllm recipe |
|---|---|---|
| WRITE | in-place or replacement | replacement ONLY (new tensor / whole tuple) — F-5 |
| unembed COMPUTE | `lm_head(h)` or weight matmul | weight matmul ONLY (`ParallelLMHead.forward` guarded) — F-2 |
| aux COMPUTE | bare | under `torch.no_grad()` — F-1 |
| cross-prompt transfer | two single-prompt traces (barrier untested) | two single-prompt traces (barrier broken upstream) |
| residual READ on fused-residual families | `out[0]` | `out[0] + out[1]` — F-7 |

### Tag shorthand (used in the method tables below)

The tags are Level-2 decompositions:

| tag | decomposition |
|---|---|
| `read` | READ × boundary site |
| `write` | WRITE × boundary site (replacement realization unless noted) |
| `xprompt` | cross-prompt transplant: READ (trace A) → WRITE (trace B) |
| `grad` | BACKWARD + READ × gradient space |
| `attn-weights` | READ × internal × attention (HF eager only; site absent on vLLM) |
| `.source` | READ/WRITE × internal site |
| `ext-module` | COMPUTE with an external `nn.Module` (probe / SAE / transcoder) |
| `trained` | needs a pretrained artifact — a dependency, not a primitive |

Status: ✓ = already an nnbench cell. **frontier** = exercises a primitive where vLLM and HF diverge
(the highest-signal additions).

---

## 1. Reading what a layer represents (observational lenses & probes)

| method | idea | nnsight primitives | models / generality | status |
|---|---|---|---|---|
| **Logit lens** | project an intermediate residual through final-norm + unembed → a next-token dist | `read` | any decoder-only LM; the "fused residual" detail is arch-specific (GPT-2 single tensor vs Llama hidden+residual) | ✓ |
| **Tuned lens** | logit lens with a *trained* affine probe per layer (better-calibrated early layers) | `read` `ext-module` `trained` | needs a tuned-lens checkpoint per model | TODO |
| **Linear probing** | train a linear classifier on activations to test if a concept is linearly decodable | `read` `ext-module` `trained` | any model; probe is per-model/per-concept (cheap to train) | TODO |
| **Direct logit attribution (DLA)** | decompose the final logit into additive per-component contributions | `read` (+ `.source` for per-head) | decoder-only; per-head needs access to head outputs before `W_O` | TODO (read-only, exact) |

## 2. Causal interventions (change something, watch the output)

| method | idea | nnsight primitives | models / generality | status |
|---|---|---|---|---|
| **Activation patching / causal tracing** | copy an activation from a clean run into a corrupted run; measure restoration | `read` `write` `xprompt` | any transformer | ✓ |
| **Ablation / knockout** | zero- or mean-out a component, measure the damage | `read` `write` | any transformer | ✓ |
| **Steering / ActAdd** | add a direction into the residual at run time to push behavior | `read` `write` | any decoder-only LM | ✓ |
| **Attribution patching** | gradient linear-approx of patching for *every* component in one fwd+bwd | `read` `grad` | any differentiable model | ✓ — **frontier confirmed** (`grad`: vLLM ERROR, F-11) |
| **Path patching** | patch specific component→component *edges* (not whole activations) | `read` `write` `xprompt` `.source` | any transformer; more plumbing | TODO (composite) |

## 3. Decomposing representations into features

| method | idea | nnsight primitives | models / generality | status |
|---|---|---|---|---|
| **Sparse autoencoders (SAEs)** | sparse overcomplete dict over a layer's activations → monosemantic features | `read` `ext-module` `trained` (+ `write` to steer) | needs trained SAEs (available: GPT-2, Gemma-2, Llama, …) | TODO |
| **Transcoders** | SAE that approximates an MLP's *computation* (read input → write output) | `read` `write` `ext-module` `trained` | needs trained transcoders | TODO |

## 4. Attention & circuits

| method | idea | nnsight primitives | models / generality | status |
|---|---|---|---|---|
| **Attention-pattern read** | read the attention weights (who attends to whom) | `read` `attn-weights` `.source` | any transformer (HF eager); — | ✓ — **frontier confirmed** (`attn-weights`: site absent on vLLM, F-10) |
| **Per-head ablation / read** | zero or read an individual attention head's output | `read` `write` (per-head index) | any transformer; needs head-dim reshape | TODO |
| **Induction heads** | identify head pairs implementing "A→B … A→?B" (in-context copying) | `read` `attn-weights` (+ per-head ablation) | emergent in most transformers | TODO (analysis, composite) |
| **Automated circuit discovery (ACDC / EAP)** | iteratively patch/prune edges to find a task's minimal subgraph | `read` `write` `grad` (EAP) | any transformer; many runs | TODO (heavyweight) |

## 5. Concept-direction control & full pipelines

| method | idea | nnsight primitives | models / generality | status |
|---|---|---|---|---|
| **Representation engineering (RepE)** | derive a concept "reading vector" from contrastive activations, monitor/steer | `read` `write` | any decoder-only LM (vector derived from data, light) | TODO |
| **Circuit Tracer (attribution graphs)** | replace MLPs with cross-layer transcoders → build & intervene on an attribution graph | `read` `write` `grad` `ext-module` `trained` | only models with trained transcoders (Gemma-2-2B, small Llama, Qwen3-4B) | TODO (heavyweight; full pipeline, not a unit cell) |

---

## nnbench roadmap (prioritized)

**Levels 0–1 are fully measured** (micro tier, 2026-06-11: HF 11/11 SUPPORTED; vLLM 6/5
SUPPORTED/ERROR — F-12..F-16). Attention-pattern read and attribution patching graduated to ✓.
The queue is now method-tier breadth over the measured base:

**Tier 1 — cheap deterministic cells, no trained artifact:**
1. **Generation-time steering** — the serving-shaped workload, now VIABLE on both backends via
   the bounded-iteration realization (`iter[0:N]`, measured SUPPORTED — F-13); the unbounded
   form rides along as the frontier marker / flip-detector for the upstream saves-drop fix.
2. **Multi-layer `tracer.cache`** — the fused bulk-cache primitive (READ × breadth + SAVE);
   throughput-bound, the harvesting regime; the one Level-2 fused primitive still unprobed.
3. **Per-head ablation** — turns the measured derived-site READ into a WRITE methodology
   (head-sliced write is still unexercised).
4. **Direct logit attribution** — `read` (+ `.source` for per-head); exact read-only decomposition
   over the now-measured non-attention `.source` site.
5. **Family axis for the micro tier** — the Level-0/1 map is (GPT-2)-only; rerun on a
   fused-residual family (SmolLM2/Llama) where boundary denotation differs (F-7).

**Tier 2 — need a (cheap/available) trained artifact:**
- Linear probing (train a probe), SAE (use an existing checkpoint, e.g. GPT-2/Gemma), tuned lens.

**Tier 3 — composite / heavyweight (great demos, poor unit cells):**
- Path patching, ACDC/EAP, Representation Engineering, Circuit Tracer (transcoders).

**Selection criterion recap:** an ideal new cell is (a) deterministic, (b) needs no trained artifact,
(c) exercises a Level-2 combination not yet covered by any cell — ideally one where vLLM and HF
*diverge* — so it adds a real coverage frontier, not just a row. By that test the next cells to build
are **generation-time steering** and **multi-layer `tracer.cache`**.
