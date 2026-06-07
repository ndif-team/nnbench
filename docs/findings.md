# Findings (living)

Real observations the benchmark has surfaced. Each is a candidate row in the
applicability map and/or a note for the nnsight team.

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

### F-5 — vLLM residual writes: in-place RAISES, replacement WORKS (no silent no-op)
The idiomatic in-place steering form (`blk.output[0][:] += vec`) raises on vLLM because
the residual is an inference-mode tensor — a **loud, actionable ERROR** (the message even
prescribes the fix: clone first), not the dangerous silent no-op `SILENTLY_WRONG` cell we
were hunting. The whole-tuple replacement form (`blk.output = (resid + vec, *rest)`) builds
a *new* tensor and is applied faithfully (TV=0.000 vs HF). **Guidance:** on vLLM, do
activation steering/patching by replacement, never in-place. This is the write analogue of
the logit-lens `module`-vs-`weight` split (F-2): idiomatic = frontier, portable = supported.
→ `isb/methodologies/steering.py`.

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

### F-7 — vLLM-Llama logit lens is SILENTLY_WRONG unless you reconstruct the fused residual ⭐
**The first `SILENTLY_WRONG` cell — the one only the numerical oracle can catch.** The exact
portable logit-lens that is correct on GPT-2 (read `block.output[0]`, final-norm, weight-matmul;
F-2 form) returns numerically wrong logits on vLLM-Llama: **top1 agreement 0.13, TV 0.897**, no
error, no crash. A coverage-only (crash-or-not) benchmark would mislabel this `SUPPORTED`.

Root cause (read from `vllm/model_executor/models/llama.py`): vLLM's Llama uses **fused-residual
RMSNorm**. Each `LlamaDecoderLayer.forward(...)` returns `(hidden_states, residual)` (`:332`), and
the actual residual stream at the layer boundary is their **sum** — vLLM itself computes
`hidden_states + residual` for its aux hidden states (`:425`). The naive readout takes only
`output[0]` (= `hidden_states`), dropping the accumulated `residual`, so it normalizes half the
state. HF's Llama adds the residual inside the layer, so `output[0]` already carries the full
stream → HF is correct; the divergence is purely vLLM's representation.

Fix = the backend-aware `residual="fused"` form (`out[0] + out[1]`), which restores TV=0.017
(same quality as GPT-2). This is the residual-stream analogue of the `unembed` split (F-2):
**idiomatic ports → frontier; backend-aware form → supported.** Guidance for the nnsight team
and users: *on vLLM, reading a fused-residual model's mid-stack residual requires summing
`(hidden, residual)`; the single-tensor idiom is silently wrong.* This generalizes to any vLLM
model whose decoder layers use fused-residual RMSNorm (Llama, Mistral, Qwen2, Gemma, …) — a broad
frontier the GPT-2-only smoke could never have surfaced. → `isb/methodologies/logit_lens.py`
`_resid`.

## Smoke tier — activation patching (causal tracing), GPT-2, HF vs vLLM-async (2026-06)

Cross-prompt write: capture block-L residual from a CLEAN run ("...France...") and transplant it
into a CORRUPTED run ("...Russia...", a length-matched minimal pair) via **two single-prompt
traces** (`be.patch`), then read the corrupted run's next-token logits. The two-trace form needs no
multi-invoke/barrier, which is why it runs on vLLM at all. Result (`results/smoke_patching.txt`,
layers 3 & 9 identical):

| backend | state | note |
|---|---|---|
| hf | SUPPORTED | per-family control; non-vacuity guard TV(unpatched, patched)=0.753 |
| vllm_async (default bf16) | **SUPPORTED_DEGRADED** | top1=0.00 TV=0.083 vs HF, but vLLM-**fp32** vs HF = top1=1.00 TV=0.0006 |

### F-8 — vLLM activation patching is correct, but default bf16 flips a near-tie top-1 ⭐
The two-trace patch mechanism **ports faithfully to vLLM**: forced to `dtype="float32"` it matches
HF to TV=0.0006 / top1=1.00 (essentially bit-identical). But at vLLM's **default bf16** the patched
next-token's top-1 flips vs HF-fp32 (top1=0.00) while the distribution stays close (TV=0.083) —
because each backend transplants a residual computed at its *own* precision and the final prediction
is a near-tie. The honest label is **SUPPORTED_DEGRADED**, not `SILENTLY_WRONG`.

**Methodology lesson (load-bearing):** a strict cross-backend oracle *conflates precision-
degradation with correctness-bugs* — the bf16 run fails the same gate (top1<0.9) that caught the
real F-7 bug. The disambiguator is a **dtype control**: re-run the failing backend at the control's
precision; if it then matches, the divergence was precision (`SUPPORTED_DEGRADED`); if it persists,
it is a genuine `SILENTLY_WRONG`. Without this control, a benchmark would cry "silent bug" at every
bf16 near-tie and lose the signal. Contrast the two ⭐ findings: F-7 is FAR (TV=0.897, persists at
any dtype) = real bug; F-8 is CLOSE (TV=0.083, vanishes at fp32) = precision. → `be.patch` (two
single-prompt traces), `scripts/smoke_patching.py` (dtype-control disambiguation), vLLM `dtype` knob.
