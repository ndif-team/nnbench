# Findings (living)

Real observations the benchmark has surfaced. Each is a candidate row in the
applicability map and/or a note for the nnsight team.

## Smoke tier — logit lens, GPT-2, HF vs vLLM-async (2026-06)

Result (`results/smoke_gpt2.txt`):

| workload | hf | vllm_async | note |
|---|---|---|---|
| `logit_lens` (idiomatic: `lm_head(ln_f(h))`) | SUPPORTED | **ERROR** | vLLM `ParallelLMHead.forward()` guards: *"LMHead's weights should be used in the sampler."* |
| `logit_lens.weight` (unembed via weight matmul) | SUPPORTED | **SUPPORTED** | top1=1.00 across 12 layers, maxabs=0.89 (kernel/dtype drift) |

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
