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
*"Inference tensors cannot be saved for backward."* Fix that keeps one methodology for both
backends: wrap forward-only aux compute in `torch.no_grad()` (harmless on HF, required
on vLLM). → `isb/methodologies/logit_lens.py`.

### F-2 — `lm_head` cannot be called directly on vLLM
vLLM's `ParallelLMHead.forward()` deliberately raises; its weights are consumed by the
sampler. So the *idiomatic* logit lens (`model.lm_head(...)`) is a genuine frontier
marker on vLLM. The portable form does `F.linear(normed, lm_head.weight)`, which works.

### F-3 — intervention errors are isolated; the engine survives (corrects an earlier claim)
A worker intervention error (e.g. the guarded `lm_head` call) surfaces as a clean per-cell
`ERROR` via nnsight's deferred-exception mechanism and does **not** kill the EngineCore. With
one engine amortized across all cells, an `lm_head`-guard ERROR is immediately followed by a
`SUPPORTED` cell on the *same* engine (verified by the bench run). An earlier version of this
note claimed the opposite, generalizing from a single observed `EngineDeadError`; that was a
misattribution — ordinary intervention errors are contained, so the benchmark amortizes one
model load across all cells rather than reloading per cell. (Separately: driving one async
engine with repeated `asyncio.run` calls *does* kill it by closing the loop its background task
runs on — the benchmark avoids that with a persistent event loop. → `isb/backends/vllm_async.py`.)

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

## Smoke tier — attention-pattern read, GPT-2, HF vs vLLM-async (2026-06)

Read the post-softmax attention probability matrix for the last query token across all heads of every
layer, via `.source` (a block's `.output` is the value-weighted result, not the probabilities). HF
eager exposes the weights as `attn.source.attention_interface_0.output[1]` (`[B, heads, q, k]`); the
cell emits `log(A)` for the last-query row so the oracle's softmax recovers the true attention
distribution. Result (`isb/specs/attention_pattern.py`):

| workload | hf | vllm_async | note |
|---|---|---|---|
| attention-pattern `layers=all` | SUPPORTED | **ERROR** | `AttributeError: 'SourceEnvoy' has no attribute 'attention_interface_0'` |

### F-10 — reading attention weights is HF-only; vLLM's paged attention exposes no probability matrix
This is the `attn-weights` frontier. HF eager computes `attn_output, attn_weights = attention_interface(...)`,
so `.source.attention_interface_0` returns the matrix. vLLM runs a different forward (paged/flash
attention) that computes attention implicitly and never materializes the probability matrix, and has
no `attention_interface` op — so the `.source` read raises (`AttributeError`, surfaced as a clean
per-cell `ERROR`). This is an architectural limit of the serving backend, not a missing nnsight
feature: there is no probability matrix to read on the vLLM path. → `isb/methodologies/attention_pattern.py`.

## Smoke tier — attribution patching, GPT-2, HF vs vLLM-async (2026-06)

Gradient-based linear approximation of activation patching: one clean forward, one corrupt
forward+backward, attribution per layer = `((resid_clean - resid_corrupt) * grad_corrupt).sum()`,
metric = `logit[" Paris"] - logit[" Moscow"]` on the corrupt run (length-matched France/Russia pair,
portable unembed). New backend primitive `be.attribute` (clean trace + corrupt trace with
`metric.sum().backward()`). Result (`isb/specs/attribution_patching.py`):

| workload | hf | vllm_async | note |
|---|---|---|---|
| attribution `residual=plain` | SUPPORTED (overhead 3.4× fwd) | **ERROR** | `Inference tensors cannot be saved for backward … created in inference mode` |

### F-11 — gradient-based attribution is HF-only; vLLM is inference-mode (no autograd)
This is the `grad` frontier — a whole class of methods (attribution patching, edge attribution
patching, any gradient saliency) is HF-only. HF runs the forward with autograd, so
`requires_grad_` + `with metric.sum().backward(): act.grad` works and the per-layer attribution is
produced (overhead ≈3.4× a single forward: clean fwd + corrupt fwd + backward). vLLM creates
activations under `torch.inference_mode()`, so any attempt to track them for backward raises
(`Inference tensors cannot be saved for backward`), surfaced as a clean per-cell `ERROR`. The cell's
`grad=False` baseline (forward-only metric) runs on both backends — it's specifically the backward
that vLLM cannot do. → `isb/methodologies/attribution_patching.py`, `isb/backends/{hf,vllm_async}.py`
(`be.attribute`).

## Micro tier — Level 0/1 primitive probes, GPT-2, HF vs vLLM-async (2026-06-11)

One minimal probe per previously-UNTESTED inventory row (design.md §3.7; `isb/micro/`,
`scripts/micro.py`), each with a self-contained denotation check. Full maps in
`results/micro_{hf,vllm_async}.txt`. **Refined 2026-06-11** after the per-construct root-cause
diagnosis (nnsight `docs/developing/vllm-construct-gaps.md`; all construct failures verified
identical on vllm 0.19.1 and 0.15.1): iteration split by realization (bounded/unbounded) and
session split by flow (saved/un-saved). **HF: 13/13 SUPPORTED** (all checks exact or ≤1e-3
rel-dev). **vLLM-async: 7 SUPPORTED / 6 ERROR**:

| probe | hf | vllm_async |
|---|---|---|
| boundary `.input` | SUPPORTED | SUPPORTED (exact) |
| engine `logits` | SUPPORTED | SUPPORTED (== portable unembed) |
| engine `samples` | SUPPORTED | SUPPORTED (== greedy argmax) |
| derived head | SUPPORTED | SUPPORTED (recon in-trace) |
| derived neuron | SUPPORTED | SUPPORTED (rel-dev 3e-3, bf16) |
| `.source` non-attention | SUPPORTED | SUPPORTED (exact) |
| iteration — bounded `iter[0:3]` | SUPPORTED | SUPPORTED (3 steps; step-0 == single-step trace) |
| iteration — unbounded `iter[:]` | SUPPORTED | **ERROR** (all saves dropped) |
| scan | SUPPORTED | **ERROR** (`hook` kwarg) |
| edit | SUPPORTED | **ERROR** (mediator pickling) |
| barrier | SUPPORTED | **ERROR** async; **SILENTLY_WRONG on the sync engine** (clean exit, saved dict empty — construct-gaps repros; sync is not yet a bench backend) |
| session — saved flow | SUPPORTED | **ERROR** async (no drain point); works on the sync engine (construct-gaps repros) |
| session — un-saved cross-trace flow | SUPPORTED | **ERROR** (both engines) |

### F-12 — Level-1 sites are portable on vLLM; weight-using checks must run in the worker
Boundary `.input`, engine `logits`/`samples`, derived head/neuron views, and non-attention
`.source` all hold on vLLM with exact (or bf16-roundoff) denotation checks — the vLLM MLP forward
is plain Python, so `.source` rewriting works there; the attention-weights case (F-10) stays the
only absent internal site. Gotcha: the client-side envoy is the META model — a reconstruction that
touches `.weight` must run INSIDE the trace (in the worker), else `Cannot copy out of meta tensor`.
→ `isb/micro/probes.py` (`derived_head_vllm`).

### F-13 — UNBOUNDED `tracer.iter[:]` drops all saves on vLLM; bounded slices work
The documented multi-token idiom (`for step in tracer.iter[:]: rows.append(model.logits)` —
nnsight `docs/models/vllm.md`) yields a finished output that carries **no saves at all**, while
the identical trace without `iter` collects fine. The split is a realization (Level 1.5)
distinction: **bounded `iter[0:3]` is SUPPORTED on both engines** (measured); only the unbounded
forms (`iter[:]`, `.all()`) fail, on sync (UnboundLocalError) and async (no `.saves`) alike.
Root cause (diagnosed upstream, nnsight `docs/developing/vllm-construct-gaps.md` §1): the vLLM
path never sets a stop bound, so the loop overruns the last step, blocks, and is unwound by
`Cancelation` before the body's single final `push()` — the only thing that publishes saves.
Blocks the generation-time workload class on vLLM until fixed; the bounded realization is the
working recipe meanwhile. → `isb/micro/probes.py` (`_iter_vllm`).

### F-14 — barrier on vLLM: loud on async, SILENT on the sync engine
The cross-invoke barrier patch (HF: matches the two-trace patch exactly) fails on vLLM with a
per-engine split. **Async**: no saves at all — two documented causes stack (the async
multi-prompt submission gate and the Barrier object not being shared across invokes, nnsight
`docs/developing/barrier-vllm-not-shared.md`) — a clean ERROR. **Sync engine** (measured via the
construct-gaps repros; a plain two-invoke trace works there): the trace **exits cleanly with the
saved dict EMPTY** — silent post-barrier data loss, the SILENTLY_WRONG state, with no error
signal of any kind. The two-trace `be.patch` recipe remains the working cross-prompt form (F-8).
Note: the sync engine is not yet a bench backend — engine mode is a context axis the inventory
currently under-represents.

### F-15 — session on vLLM: only the UN-SAVED cross-trace flow is broken (plus async entirely)
On HF both flows work (saved read-after-exit, and un-saved trace-1 → trace-2 reuse, |Δ|=0).
On vLLM the row splits (nnsight `docs/developing/vllm-construct-gaps.md` §3):
- **saved flow** (`.save()` in a session trace, read after exit): **works on the sync engine**
  (measured); ERROR on async — there is no drain point inside a captured session body (`async
  for` cannot compile there, and tracer handles don't survive to the caller frame).
- **un-saved cross-trace flow** (the session contract): ERROR on BOTH engines — only values in
  `Globals.saves` ship back from the worker, so the un-saved trace-1 variable never materializes
  client-side; trace 2 dies and the surfaced `UnboundLocalError` misleadingly names the
  *downstream* saved variable.

### F-16 — edit and scan error cleanly on the vLLM path
`model.edit()` stores its mediator, but replaying it into a vLLM worker trace fails to serialize
(`PicklingError: ... source code unavailable`). `model.scan()` fails earlier — the scan machinery
passes a `hook=` kwarg the vLLM execution path rejects. Both are per-cell ERRORs (clear signal, no
silent wrongness), matching `docs/models/vllm.md`'s "not validated on the vLLM path" caveat — now
measured rather than presumed.

## Method tier — generation-time steering, GPT-2, HF vs vLLM-async (2026-06-12)

Result (`results/gen_steering_gpt2.txt`), the first **generation-regime** methodology: the
steering write applied at EVERY decode step of an 8-token greedy generation, per-step logits
oracle-checked row-per-step over 8 probe prompts (64 rows):

| realization | hf | vllm_async | note |
|---|---|---|---|
| bounded `iter[0:N]` | SUPPORTED | **SUPPORTED** | top1=1.00 / TV=0.000 at default bf16 (maxabs=1.18 diagnostic only) |
| unbounded `iter[:]` | SUPPORTED | **ERROR** | all per-step saves dropped (F-13) — the frontier marker for the upstream fix |

Effect-size on the control: the per-step steer flips EVERY step's top-1 (top1=0.00, TV=0.999) —
the verdict is maximally non-vacuous. Perf: vLLM 176 ms / 45.4 tok/s vs HF 197 ms / 40.6 tok/s;
the per-step write costs ≈1.05× over the no-intervention generation baseline.

### F-17 — the WRITE × bounded-iteration composition holds on vLLM, exactly
First method-tier cell measuring a COMPOSITION of two separately measured inventory rows:
replacement WRITE (F-5) inside the bounded iteration construct (F-13). The composed statuses
predict SUPPORTED, and the measurement agrees — *exactly*: per-step logits match HF with
top1=1.00 / TV=0.000 across the full greedy trajectory, i.e. the steered decode follows the
identical token path on both backends, with no precision degradation even at vLLM's default
bf16. This is the first method-tier confirmation of the "statuses compose upward" claim
(design.md §3.6) — and it converts the causalab portability audit's "composition unmeasured"
flag on the path_steering footprint into a measured cell
(`docs/causalab-portability-audit.md` §4). The unbounded realization rides along as predicted
ERROR (F-13), so the spec doubles as the flip-detector for the upstream saves fix.
→ `isb/methodologies/gen_steering.py`, `isb/specs/gen_steering.py`.
