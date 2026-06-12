# causalab portability audit — footprint × measured map

> Status: written 2026-06-12. This is the "portability audit" proposed in
> `agents-and-the-primitive-model.md` — a **static join**: each causalab analysis is decomposed
> into its primitive footprint (citing causalab source), and each footprint line is looked up in
> the measured per-context status inventory (`interp-methods-catalog.md`; evidence tags `F-n` =
> `findings.md`). Nothing here was executed; predictions are inferences from measured rows, to be
> validated by Macro-tier runs. Source audited: `goodfire-ai/causalab` @ `bf15b353` (read-only
> clone at `/disk/u/zikai/causalab`).

## 1. As-shipped verdict: 0% on vLLM, by construction

causalab's only model-access layer is `neural/`, which wraps a HuggingFace
`AutoModelForCausalLM` in a pyvene `IntervenableModel` (`neural/activations/intervenable_model.py:15-86`,
`neural/pipeline.py:205`). pyvene plants hooks on in-process HF `nn.Module`s; a vLLM model lives
inside an engine worker and is not hookable from the caller's process. Two corroborating facts:

- `nnsight>=0.5.9` is **declared in `pyproject.toml` but never imported** — zero nnsight (or vllm)
  references anywhere in the code. The execution path is pyvene-on-HF-eager exclusively.
- The pipeline **forces eager attention by default** (`neural/pipeline.py:210-212`,
  `eager_attn: True` → `_attn_implementation = "eager"`) — the suite is built on
  eager-substrate assumptions end to end.

So the as-shipped number is trivial. The non-trivial question the rest of this audit answers:
**re-expressed in nnsight primitives, what fraction of the suite runs on a production engine,
and with which failure kind for the rest?**

## 2. One structural fact that shapes every footprint

Every causalab intervention is applied **during multi-token generation**, not on a single
forward: interchange scoring and steering both go through `pipeline.intervenable_generate(...)`
(`neural/activations/interchange_mode.py:161`, `methods/steer/steer.py:278`;
`max_new_tokens` defaults to 3, `neural/pipeline.py:157`). pyvene's intervenable-generate runs
the source forward, collects, then re-applies the intervention at each decode step of the base
generation.

In nnsight terms every interventional footprint therefore composes with the **iteration
construct**. On vLLM the bounded realization `iter[0:N]` is measured SUPPORTED and the unbounded
`iter[:]` drops all saves (F-13) — so the bounded form is the mandatory idiom for every
re-expressed analysis below.

## 3. Footprints (ops × sites × idioms, with citations)

| analysis | footprint (Level 0/1/1.5 terms) | causalab evidence |
|---|---|---|
| baseline | batched generation + engine-site READ (logits); no internals | `analyses/baseline/main.py:192-225` |
| locate (`method: interchange`) | READ × boundary residual → cross-prompt WRITE (replacement) × generation, scanned over (layer × position) | `analyses/locate/run_interchange.py:66`, `configs/analysis/locate.yaml` (`component: residual_stream`) |
| locate (`method: dbm_binary`) | the above + BACKWARD (mask training) | `configs/analysis/locate.yaml` (`dbm:`), `methods/trained_subspace/train.py:417,536-538` |
| subspace (`method: pca`) | READ+SAVE (collect) × boundary site; PCA offline | `methods/pca.py`, collect intervention `neural/featurizer.py:554` |
| subspace (`method: das/dbm/boundless`) | BACKWARD through a feature-space interchange (AdamW over rotation/mask params) | `methods/trained_subspace/train.py:417` (optimizer), `:536-538` (`loss.backward()`) |
| activation_manifold | post-hoc geometry on cached features; loads a weights-free "lite" pipeline by default | `analyses/activation_manifold/main.py`, `io/pipelines.py` |
| output_manifold | multi-token generation + per-step engine-site READ (logits) | `analyses/output_manifold/main.py` |
| path_steering | COMPUTE (featurizer chain: rotation ∘ standardize ∘ manifold) + WRITE (steering add in feature space, write-back via inverse) × generation | `analyses/path_steering/main.py:169` (chain), `neural/featurizer.py:582` (steering intervention) |
| pullback | BACKWARD through forward passes (LBFGS/Adam trajectory optimization) | `methods/pullback/optimization.py:127,140` |
| attention_pattern | READ × internal attention-weights site (`output_attentions=True`, eager) | `methods/attention_pattern_analysis.py:136` |

Featurizer-space interventions (rotated interchange, masked interchange, steering add) decompose
as COMPUTE (the chain is plain torch) ∘ WRITE (replacement write-back); on vLLM the COMPUTE runs
under `torch.no_grad()` and the WRITE must be the replacement realization (F-1, F-5).

## 4. The join — predicted applicability per backend

HF column: everything is ✓ (causalab's native substrate; our HF control measured every needed
row SUPPORTED). The vLLM column is the audit's content. "Working idiom" = the rewrite that
rescues the analysis; failure kinds per design.md §3.6.

| analysis | vLLM (re-expressed) | failure kind / working idiom | measured rows it rests on |
|---|---|---|---|
| baseline | ✓ predicted | engine sites measured (`model.logits`, `model.samples`); batched regime currently per-prompt only (async multi-prompt gated) | F-12; invoke row |
| locate (interchange) | ✓ predicted — **composition unmeasured** | two-trace idiom for the cross-prompt transplant (barrier broken: F-14); replacement WRITE; bounded `iter[0:N]`; fused-residual read on Llama-family | F-5, F-7, F-13, F-14; patching cells |
| locate (dbm_binary) | ✗ | **operation-unsupported**: no autograd in inference mode (F-11); no working idiom — fall back to `method: interchange` | F-11 |
| subspace (pca) | ✓ predicted | collect = READ+SAVE, measured; PCA is offline | logit-lens cells; micro READ rows |
| subspace (das/dbm/boundless) | ✗ | **operation-unsupported** (F-11); HF-only — train on HF, *apply* the trained rotation on vLLM (apply is COMPUTE∘WRITE, supported) | F-11, F-1, F-5 |
| activation_manifold | ✓ (substrate-independent) | touches no engine; optional decoding eval inherits the generation rows | — |
| output_manifold | ✓ predicted | per-step logits via bounded iteration; the unbounded idiom would silently drop every step | F-12, F-13 |
| path_steering | ✓ predicted — **composition unmeasured** | replacement steering write + featurizer COMPUTE under no_grad + bounded iteration — this is exactly the roadmap's "generation-time steering" cell | F-1, F-5, F-13 |
| pullback | ✗ | **operation-unsupported** (F-11); no working idiom (optimization *is* the method) | F-11 |
| attention_pattern | ✗ | **site-absent**: paged/flash attention never materializes the matrix (F-10); no working idiom | F-10 |

**Headline:** of the eight analyses, **6 are predicted portable** to a production engine once
re-expressed in the working idioms — baseline, locate, subspace, activation_manifold,
output_manifold, path_steering (counting locate and subspace by their gradient-free methods) —
and **2 are blocked outright**: pullback by operation-unsupported (no autograd) and
attention_pattern by site-absent. A third blocker cuts *across* the portable set: every
gradient-trained method variant (DAS, DBM, boundless, locate's `dbm_binary`) is
operation-unsupported, leaving those analyses their gradient-free methods only. All blockers are
*loud* — clean errors, not traps. The traps are in the portable six (next section).

The two "composition unmeasured" flags are the same gap: no bench cell yet composes a WRITE (or
cross-prompt transplant) with the bounded-iteration construct. That composition is causalab's
*default* execution mode (§2), which independently confirms the roadmap's top priority
(generation-time steering) and adds a second composite right behind it (generation-time
cross-prompt patching = locate's footprint).

## 5. The silently-wrong census — why this audit matters

causalab's primary tested model is **Llama-3.1-8B** (README; ≥24 GB VRAM) — a fused-residual
family. On vLLM, every `component: residual_stream` read in a naive port returns only `out[0]`
of `(hidden, residual)` — **half the stream, no error** (F-7: top-1 agreement 0.13 on the same
mistake in our logit-lens cell). That poisons locate's entire (layer × position) heatmap, every
subspace fit on those activations, and every manifold built downstream — and causalab's own
pipeline has no numerical oracle that could notice: scores stay plausible, the faithfulness
conclusion is just wrong. The other documented trap it would inherit: the unbounded-iteration
idiom silently losing all per-step saves on the generation analyses (F-13).

This is the division of labor stated in `agents-and-the-primitive-model.md`: causalab verifies
the science *assuming the substrate*; the substrate assumption is precisely what this map
measures.

## 6. Vocabulary coverage — what causalab forces on the leveled model

The audit doubles as the empirical test of design.md §3's vocabulary (open question 1 of the
agents notes). Verdict: the Level-0/1/1.5 terms covered every footprint line above without
strain, with three findings at the edges:

1. **Cross-model patching is outside the vocabulary.** locate accepts a `source_pipeline` —
   activations collected from a *different model* and patched into the primary
   (`run_interchange.py:69-73`). Our context axis is a single (family × backend × config);
   a footprint spanning two models has no representation. New context-axis entry needed if we
   ever cover it (deliberately out of scope for now: it is not oracle-checkable against a single
   HF control).
2. **Feature-space realization deserves a named idiom row.** "Intervene in f(x)-space, write
   back via f⁻¹" (rotation/PCA/manifold chains) decomposes cleanly as COMPUTE∘WRITE, but it is
   a *recurring* realization with its own failure surface (the chain must run inside the trace,
   under no_grad, on the worker — the meta-model gotcha from F-12 applies to its weights). Worth
   a Level-1.5 row rather than re-deriving per methodology.
3. **Generation-time intervention is a composition, not a primitive — and it is unmeasured.**
   The vocabulary expresses it (WRITE × iteration), the inventory has both rows measured
   separately, but no cell measures the composition. §3.6's "statuses compose upward" claim is
   exactly what the two flagged predictions in §4 will test.

No new Level-0 op was needed — the closed-core claim survived contact with a real external suite.

## 7. What to do with this

> **Addendum (2026-06-12, same day):** the first flagged composition is now MEASURED. The
> generation-time steering cell (`isb/methodologies/gen_steering.py`, finding F-17) runs the
> replacement write inside `iter[0:N]` at every decode step: **SUPPORTED on vLLM, exactly**
> (top1=1.00, tv=0.000 vs the HF control over 8 prompts × 8 steps, at default bf16); the
> unbounded realization errors as predicted (F-13). The path_steering row in §4 is no longer a
> prediction — and it is the first method-tier confirmation of §3.6's composes-upward claim.
> The locate row's composition (cross-prompt transplant × iteration) remains the open one.

- **Macro-tier candidates, in order:** path_steering (= the generation-time steering cell the
  roadmap already ranks first; now doubly motivated), then locate-via-interchange (generation-time
  cross-prompt patching). Together they validate the two unmeasured compositions and turn the §4
  predictions into measurements.
- **The agent-conditioning story is now concrete:** a causalab-style planner that consulted this
  map would (a) route DAS/DBM/pullback to HF automatically, (b) skip attention_pattern on vLLM
  with a clean explanation, (c) rewrite residual reads on Llama-family, (d) bound every
  generation loop. Items (c) and (d) are the two it would otherwise get *silently* wrong.
- **Bookkeeping:** `references.md`'s causalab entry corrected (nnsight declared-but-unused);
  the two composite cells, once built, graduate this doc's predictions into catalog rows.
