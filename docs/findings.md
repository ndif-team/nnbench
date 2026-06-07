# Findings (living)

Each section is a smoke run of one methodology; the table is the applicability-map row(s)
with the oracle's measured top1/TV per cell. Where a cell's behavior reflects a known
HF↔vLLM representational difference, the note cites the relevant intervention-gaps gap number
for context (`nnsight/src/nnsight/modeling/vllm/intervention-gaps/`).

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

### F-5 — vLLM residual writes: in-place raises, replacement works
In-place (`output[0][:] += v`) raises on vLLM (`Inplace update to inference tensor outside
InferenceMode`); whole-tuple replacement is applied faithfully (TV=0.000 vs HF). On vLLM, steer/patch
by replacement. (vLLM activations are inference tensors; cf. intervention-gaps Gap 1.1.)
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

### F-7 — vLLM-Llama logit lens needs the dual residual stream
On vLLM, decoder layers expose a dual residual stream `(hidden, residual)` whose sum is the residual
stream (intervention-gaps Gap 1.2). The single-tensor form (`output[0]` only) is `SILENTLY_WRONG`:
top1=0.13, TV=0.897 — no error. Combining the streams (`residual="fused"` = `out[0]+out[1]`) gives
top1=0.97, TV=0.017, matching HF. Applies to any fused-residual-RMSNorm vLLM model
(Llama/Mistral/Qwen2/Gemma). → `isb/methodologies/logit_lens.py` `_resid`.

## Smoke tier — activation patching (causal tracing), GPT-2, HF vs vLLM-async (2026-06)

Cross-prompt write: capture block-L residual from a CLEAN run ("...France...") and transplant it
into a CORRUPTED run ("...Russia...", a length-matched minimal pair) via two single-prompt traces
(`be.patch`), then read the corrupted run's next-token logits. Two separate traces avoid the
cross-invoke barrier (not shared across invokes on vLLM). Result (`results/smoke_patching.txt`,
layers 3 & 9 identical):

| backend | state | note |
|---|---|---|
| hf | SUPPORTED | per-family control; non-vacuity guard TV(unpatched, patched)=0.753 |
| vllm_async (default bf16) | **SUPPORTED_DEGRADED** | top1=0.00 TV=0.083 vs HF; at fp32, top1=1.00 TV=0.0006 |

### F-8 — activation patching: a dtype control separates precision from a bug
The two-trace patch matches HF at fp32 (top1=1.00, TV=0.0006); at vLLM's default bf16 the patched
top-1 flips (top1=0.00) with TV=0.083 — a near-tie precision effect, so `SUPPORTED_DEGRADED`, not
`SILENTLY_WRONG`. The strict gate alone can't tell these apart, so the smoke re-runs the failing
backend at the control's dtype: matches → `SUPPORTED_DEGRADED`, persists → `SILENTLY_WRONG`.
→ `scripts/smoke_patching.py` (dtype control), vLLM `dtype` knob. Shared helper:
`isb/runner/disambiguate_precision`.

## Smoke tier — ablation (zero-knockout), GPT-2, HF vs vLLM-async (2026-06)

Zero a submodule's output at block 6 (whole-tuple replacement) and read the next-token distribution.
Effect-size guard (HF un-ablated vs ablated): `mlp` top1=1.00/TV=0.115 (WEAK — knocking out one mid
MLP barely moves the top-1, so this cell mostly measures baseline backend precision), `attn`
top1=0.00/TV=0.100 (flips the top-1 — a substantive ablation). Result (`results/smoke_ablation.txt`):

| workload | hf | vllm_async | note |
|---|---|---|---|
| ablation `target=mlp` | SUPPORTED | **SUPPORTED_DEGRADED** | bf16 top1=1.00/TV=0.081; fp32 matches HF |
| ablation `target=attn` | SUPPORTED | **SUPPORTED_DEGRADED** | bf16 top1=1.00/TV=0.061; fp32 matches HF |

### F-9 — ablation ports to vLLM; default bf16 is a near-tie precision divergence
The replacement-form knockout is applied faithfully on vLLM (matches HF at fp32, top-1 agrees). At
default bf16 the ablated distribution diverges from HF-fp32 by TV≈0.06–0.08 — the same precision
near-tie as patching (F-8), resolved to `SUPPORTED_DEGRADED` by the dtype control. In-place zeroing
would raise on vLLM (F-5); the cell uses replacement. → `isb/methodologies/ablation.py`.
