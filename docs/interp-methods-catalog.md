# Interpretability methods — catalog & nnbench roadmap

A catalog of well-recognized interpretability methods, tagged by **what models they apply to** and
**which nnsight primitives they need**. The point is to decide which become nnbench methodologies: a
method is a cheap deterministic cell, a new backend frontier, or a heavyweight needing a trained
artifact *based on the primitives it touches*.

All methods below are **deterministic** (greedy / argmax, no sampling) → oracle-checkable, which is
the inclusion criterion for an nnbench cell. The sampling-based verifier/oracle direction (BOA /
BEAVER) is parked separately.

## Legend — nnsight primitives

| tag | meaning | backend note |
|---|---|---|
| `read` | activation `.output` / `.input` (residual, attn/MLP outputs) | portable |
| `write` | set/patch `.output` (whole-tuple replace, or in-place) | vLLM: replace works, in-place raises (inference tensors) |
| `xprompt` | capture a value in one trace, inject it in another | portable (two single-prompt traces; avoids vLLM cross-invoke) |
| `grad` | `.backward()` / `.grad` | **vLLM frontier** — inference-mode, no autograd → ERROR; HF-only |
| `attn-weights` | read the attention probability matrix | **vLLM frontier** — paged/flash attention never materializes it; HF needs `attn_implementation="eager"` |
| `.source` | intermediate ops inside a module's forward (AST-rewritten) | portable in principle; depends on the backend running that forward in Python |
| `ext-module` | run an auxiliary `nn.Module` on activations (probe / SAE / transcoder) | portable |
| `trained` | needs a pretrained artifact (lens / probe / SAE / transcoder) — not just the base model | — |

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
| **Attribution patching** | gradient linear-approx of patching for *every* component in one fwd+bwd | `read` `grad` | any differentiable model | TODO — **frontier** (`grad`: HF-only) |
| **Path patching** | patch specific component→component *edges* (not whole activations) | `read` `write` `xprompt` `.source` | any transformer; more plumbing | TODO (composite) |

## 3. Decomposing representations into features

| method | idea | nnsight primitives | models / generality | status |
|---|---|---|---|---|
| **Sparse autoencoders (SAEs)** | sparse overcomplete dict over a layer's activations → monosemantic features | `read` `ext-module` `trained` (+ `write` to steer) | needs trained SAEs (available: GPT-2, Gemma-2, Llama, …) | TODO |
| **Transcoders** | SAE that approximates an MLP's *computation* (read input → write output) | `read` `write` `ext-module` `trained` | needs trained transcoders | TODO |

## 4. Attention & circuits

| method | idea | nnsight primitives | models / generality | status |
|---|---|---|---|---|
| **Attention-pattern read** | read the attention weights (who attends to whom) | `read` `attn-weights` `.source` | any transformer (HF eager); — | TODO — **frontier** (`attn-weights`: vLLM) |
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

**Tier 1 — cheap deterministic cells, no trained artifact, each opens a distinct frontier:**
1. **Attention-pattern read** — `attn-weights`/`.source`; vLLM paged-attention frontier (likely ERROR/UNSUPPORTED on vLLM vs HF-eager). Highest coverage signal.
2. **Attribution patching** — `grad`; the clean "this whole class is HF-only" frontier (vLLM is inference-mode).
3. **Direct logit attribution** — `read`(+`.source`); exact read-only decomposition, good portability check.
4. **Per-head ablation** — `read`+`write` with head indexing; tests head-dim reshape across backends.
5. **Multi-layer `tracer.cache`** — exercises the caching path cross-backend (separate from any single method).

**Tier 2 — need a (cheap/available) trained artifact:**
- Linear probing (train a probe), SAE (use an existing checkpoint, e.g. GPT-2/Gemma), tuned lens.

**Tier 3 — composite / heavyweight (great demos, poor unit cells):**
- Path patching, ACDC/EAP, Representation Engineering, Circuit Tracer (transcoders).

**Selection criterion recap:** an ideal new cell is (a) deterministic, (b) needs no trained artifact,
(c) touches a primitive where vLLM and HF *diverge* — so it adds a real coverage frontier, not just a
row. By that test the next cell to build is the **attention-pattern read**.
