# Findings (living)

Each entry is a row in the applicability map and/or a note for the nnsight team.

> **Source of truth — these are mostly VERIFICATIONS, not discoveries.** The architectural
> HF↔vLLM intervention gaps are already documented in
> `nnsight/src/nnsight/modeling/vllm/intervention-gaps/{REPORT.md,VLLM_GUIDE.md}` (13 numbered
> gaps; PR ndif-team/nnsight#662 further cleaned the docs up). This benchmark's job is to
> *systematically re-verify* those gaps across a methodology×family×backend matrix with a
> **numerical-equivalence oracle**, quantify their severity, and flag where the documented
> patterns themselves are **stale** on the current `dev` branch. Where a finding restates a
> documented gap, it cites the gap number. The non-redundant contributions are: (1) the
> oracle-quantified severity per cell, (2) the **dtype-control** protocol that separates a
> precision degradation from a real correctness bug (F-8), (3) the **effect-size guard** for
> write methodologies (F-6), and (4) the doc-staleness discrepancies in §"Doc discrepancies".

## Smoke tier — logit lens, GPT-2, HF vs vLLM-async (2026-06)

Result (`results/smoke_gpt2.txt`):

| workload | hf | vllm_async | note |
|---|---|---|---|
| `logit_lens` (idiomatic: `lm_head(ln_f(h))`) | SUPPORTED | **ERROR** | vLLM `ParallelLMHead.forward()` guards: *"LMHead's weights should be used in the sampler."* |
| `logit_lens.weight` (unembed via weight matmul) | SUPPORTED | **SUPPORTED** | top1=1.00 / TV=0.021 across 12 layers (maxabs=0.89 is kernel/dtype drift, diagnostic only) |

### F-1 — vLLM intermediates are inference-mode tensors
Applying a grad-enabled sub-module (e.g. `ln_f`) to a vLLM activation raises
*"Inference tensors cannot be saved for backward."* Fix that keeps one motif for both
backends: wrap forward-only aux compute in `torch.no_grad()` (harmless on HF, required
on vLLM). → `isb/motifs/logit_lens.py`.

### F-2 — `lm_head` cannot be called directly on vLLM
vLLM's `ParallelLMHead.forward()` deliberately raises; its weights are consumed by the
sampler. So the *idiomatic* logit lens (`model.lm_head(...)`) is a genuine frontier
marker on vLLM. The portable form does `F.linear(normed, lm_head.weight)`, which works.

### F-3 — a worker intervention error can kill the EngineCore
When the idiomatic workload raised in the worker, a *subsequent* workload on the same
async engine got `EngineDeadError`. nnsight's deferred-exception mechanism did **not**
contain this error class. Consequence for the harness: **isolate vLLM cells** (fresh
engine per workload) so an engine-killer can't poison later cells. → runner uses
per-cell `run_cell` for the sweep.

### F-4 (harness, not nnsight) — vLLM pads the vocab
`ParallelLMHead` pads vocab (50257→50304). The oracle must align the last dim to the
real vocab before comparing, or padding positions skew argmax/max-abs and a correct
result is mislabeled `SILENTLY_WRONG`. This near-miss is why the equivalence oracle +
honest verification is load-bearing (§8.1). → `isb/oracle/equivalence.py`.

## Smoke tier — steering (ActAdd), GPT-2, HF vs vLLM-async (2026-06)

First *write* methodology — adds `α·‖resid‖·unit(W[" Rome"])` into block 8's residual,
reads the final-layer portable-unembed next-token distribution. `α=0` is the unsteered
baseline; the effect-size guard measured **HF unsteered vs steered top1=0.00 / TV=0.994**,
so the write genuinely moves the control and the verdicts below are not vacuous. Result
(`results/smoke_steering.txt`):

| workload | hf | vllm_async | note |
|---|---|---|---|
| `steering` (`mode=inplace`, `hidden[:] += vec`) | SUPPORTED | **ERROR** | vLLM: *"Inplace update to inference tensor outside InferenceMode is not allowed."* |
| `steering` (`mode=replace`, whole-tuple new tensor) | SUPPORTED | **SUPPORTED** | top1=1.00 / TV=0.000 / maxabs=0.32 — steered logits match HF exactly |

### F-5 — vLLM in-place residual writes raise on `dev` → the documented in-place steer form is STALE
*(Verifies + corrects intervention-gaps Gap 1.1 / VLLM_GUIDE "Steering".)* The in-place steering
form raises on vLLM because the residual is an inference-mode tensor:
*"Inplace update to inference tensor outside InferenceMode is not allowed."* The whole-tuple
**replacement** form builds a new tensor and is applied faithfully (oracle TV=0.000 vs HF).

This is a **doc-staleness flag, not a new gap**: `VLLM_GUIDE.md` ("Modifying Activations —
Steering", line ~200) still prescribes the in-place form `model.model.layers[L].output[0][-1,:] +=
v`, and Gap 1.1 ("in-place mutation corrupts `.save()`") was marked FIXED via clone-on-save — but
on this `dev` branch / vLLM 0.15.1 the in-place *write* still raises (clone-on-save protects the
saved copy, not the live inference tensor you write to). My prior memory already noted that doc's
"wrong in-place form"; PR #662 began cleaning it. **Confirmed guidance:** on vLLM steer/patch by
*replacement*, never in-place. → `isb/methodologies/steering.py` (smoke `results/smoke_steering.txt`).

### F-6 (methodology) — a write cell's verdict needs an effect-size guard
A backend that silently no-ops a write would score `SUPPORTED` against a control whose own
output barely moved — a false pass. So a write methodology must first prove the write moves
the control (here TV=0.994 unsteered-vs-steered) before any `SUPPORTED`/`SILENTLY_WRONG`
label is trustworthy. The self-calibrating `α·‖resid‖` strength makes this robust across
layers/models without a hard-coded magnitude. → `scripts/smoke_steering.py` `_effect_size`.

## Smoke tier — logit lens, LLAMA architecture, HF vs vLLM-async (2026-06)

Second family (`HuggingFaceTB/SmolLM2-135M-Instruct`, a `LlamaForCausalLM`; meta-llama is
gated + tokenizer not in local cache). The per-family control is HF-llama, never GPT-2.
Result (`results/smoke_llama.txt`):

| workload | hf | vllm_async | note |
|---|---|---|---|
| `logit_lens` (idiomatic `lm_head(...)`) | SUPPORTED | **ERROR** | `ParallelLMHead.forward` guarded — same frontier as GPT-2 (F-2) |
| `logit_lens.weight` + `residual=fused` (backend-aware) | SUPPORTED | **SUPPORTED** | top1=0.97 / TV=0.017 — matches HF |
| `logit_lens.weight` + `residual=plain` (naive GPT-2 port) | SUPPORTED | **SILENTLY_WRONG** | top1=0.13 / TV=0.897 / maxabs=78.55 |

### F-7 — quantifying documented Gap 1.2 (dual residual stream) as a `SILENTLY_WRONG` logit-lens cell ⭐
**This is documented gap 1.2, not a new finding — the contribution is the oracle severity number.**
The intervention-gaps REPORT (Gap 1.2) and VLLM_GUIDE ("Logit Lens") already state: vLLM keeps a
**dual residual stream**, decoder layers return `(hidden_states, residual)`, and *"logit lens,
steering vectors, activation patching all need to account for the dual-stream format"* — the
prescribed form is `hs = layers[i].output[0] + layers[i].output[1]`. (Confirmed in source:
`vllm/model_executor/models/llama.py` layer returns `(hidden, residual)` `:332`; vLLM computes the
sum for its own aux hidden states `:425`.)

What the benchmark adds: the **oracle quantifies the severity** a user incurs if they ignore Gap 1.2
and naively port the single-tensor GPT-2 idiom (`output[0]` only). On vLLM-llama (SmolLM2-135M) that
naive form is **`SILENTLY_WRONG`: top1=0.13, TV=0.897, maxabs=78.55** — no error, no crash; a
coverage-only check would mislabel it `SUPPORTED`. The documented fix (`residual="fused"` =
`out[0]+out[1]`) restores **top1=0.97, TV=0.017** (same quality as GPT-2). The gap is general to any
fused-residual-RMSNorm vLLM model (Llama/Mistral/Qwen2/Gemma). → `isb/methodologies/logit_lens.py`
`_resid` (smoke `results/smoke_llama.txt`).

## Smoke tier — activation patching (causal tracing), GPT-2, HF vs vLLM-async (2026-06)

Cross-prompt write: capture block-L residual from a CLEAN run ("...France...") and transplant it
into a CORRUPTED run ("...Russia...", a length-matched minimal pair) via **two single-prompt
traces** (`be.patch`), then read the corrupted run's next-token logits. The two-trace form is the
**documented** vLLM patching recipe — VLLM_GUIDE "Activation Patching" says *"barriers and
cross-invoke dependencies are not supported... use two separate traces for dependencies"* (the
canonical single-trace form uses `tracer.barrier(2)`, which is not shared across invokes on vLLM;
see `docs/developing/barrier-vllm-not-shared.md`). `be.patch` just packages that recipe. Result
(`results/smoke_patching.txt`, layers 3 & 9 identical):

| backend | state | note |
|---|---|---|
| hf | SUPPORTED | per-family control; non-vacuity guard TV(unpatched, patched)=0.753 |
| vllm_async (default bf16) | **SUPPORTED_DEGRADED** | top1=0.00 TV=0.083 vs HF, but vLLM-**fp32** vs HF = top1=1.00 TV=0.0006 |

### F-8 — a dtype-control protocol that separates precision degradation from a real bug ⭐
*(The "numbers differ across backends" caveat is documented — REPORT "Key differences", item 5:
"vLLM and HF use different kernels... Compare intervention effects, not absolute values." The
contribution here is a concrete protocol to act on it.)* The two-trace patch mechanism **ports
faithfully to vLLM**: forced to `dtype="float32"` it matches HF to TV=0.0006 / top1=1.00. But at
vLLM's **default bf16** the patched top-1 flips vs HF-fp32 (top1=0.00) while the distribution stays
close (TV=0.083) — each backend transplants a residual at its own precision and the prediction is a
near-tie. Honest label: **SUPPORTED_DEGRADED**, not `SILENTLY_WRONG`.

**The protocol (the actual contribution):** a strict cross-backend oracle conflates precision with
correctness — the bf16 run fails the same top1<0.9 gate that caught the real Gap-1.2 bug (F-7). The
**dtype control** disambiguates: re-run the failing backend at the control's precision; matches →
`SUPPORTED_DEGRADED` (precision); persists → `SILENTLY_WRONG` (bug). Contrast: F-7 is FAR (TV=0.897,
dtype-invariant) = real; F-8 is CLOSE (TV=0.083, vanishes at fp32) = precision. This operationalizes
the REPORT's "compare effects not absolutes" caveat into a verdict rule. → `be.patch`,
`scripts/smoke_patching.py` (dtype-control), vLLM `dtype` knob.

## Doc discrepancies — VLLM_GUIDE examples that are STALE on `dev` / vLLM 0.15.1

Empirically run against the current `dev` branch; candidates for the PR #662 doc cleanup. Each is a
case where a documented example would *not run as written*. (Verify before filing — version-sensitive.)

| Doc location | What it shows | What actually happens on `dev` |
|---|---|---|
| `VLLM_GUIDE.md` "Logit Lens" (~line 263) | `logits = model.lm_head(normed)` inside the trace | **Raises** `RuntimeError: LMHead's weights should be used in the sampler.` — `ParallelLMHead.forward` is guarded. Use `F.linear(normed, lm_head.weight)` for a per-layer lens, or `model.logits.output` for the final-token logits. |
| `VLLM_GUIDE.md` "Steering" (~line 200) | `model.model.layers[L].output[0][-1,:] += v` (in-place) | **Raises** `Inplace update to inference tensor outside InferenceMode`. Use whole-tuple replacement (F-5). |

Both reduce to: on `dev`/0.15.1, vLLM activations are inference tensors and `ParallelLMHead.forward`
is guarded, so the GUIDE's in-place-write and `lm_head(...)`-call examples need updating to the
replacement / `model.logits` forms. Tested with GPT-2 and SmolLM2-135M (a `LlamaForCausalLM`).
