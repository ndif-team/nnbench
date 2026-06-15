# nnbench

A **systems performance + coverage benchmark for interpretability workloads** run through
[nnsight](https://nnsight.net) across serving backends. nnsight's promise is "write one
intervention, run it on any backend" (HuggingFace, vLLM, NDIF). nnbench measures **where that
actually holds** — which interpretability workloads run, error loudly, *silently produce wrong
numbers*, or merely run degraded — and **how fast** — across HF vs vLLM.

This is **not** a faithfulness benchmark (we do not measure whether an interpretability *result* is
scientifically correct — that is what `causalab` / CausalGym / InterpBench do). We measure whether
an interpretability *workload* **runs**, **runs correctly across backends** (numerical equivalence
vs an HF reference), and **runs fast** (latency / throughput / peak memory / overhead).

## What it produces

For every cell `(methodology × family × backend × params × workload)`, the harness emits an
**applicability map** state plus a **performance** measurement.

The states (a crash-or-not check only sees the first two; the rest need the numerical oracle):

| state | meaning |
|---|---|
| `SUPPORTED` | runs and matches the HF reference |
| `ERROR` | raises a clean, catchable error — you get a signal |
| `SILENTLY_WRONG` | runs with no error but the numbers are wrong — **the dangerous cell** |
| `SUPPORTED_DEGRADED` | diverges only due to precision (e.g. vLLM bf16 vs HF fp32), confirmed by re-running at matched precision |
| `NO_REFERENCE` | the per-family HF control itself failed, so the cell can't be judged |

**The oracle** is the load-bearing piece: HF-of-the-same-family is the per-family numerical control,
and each vLLM cell is scored against it by top-1 token agreement + softmax total-variation distance.
That is what distinguishes `SUPPORTED` from `SILENTLY_WRONG`.

**The performance layer** times each supported cell warm (warmup + N trials, CUDA-synced, median±std)
and reports latency, peak GPU memory, overhead vs a no-intervention baseline, and throughput
(prompts/s on the batched workload).

## Current coverage

- **Methodologies:** logit-lens · steering (activation addition) · activation patching (causal
  tracing) · ablation (zero-knockout) · attention-pattern read · attribution patching ·
  generation-time steering.
- **Families:** GPT-2 · Llama (SmolLM2-135M, a `LlamaForCausalLM`, as the Llama-arch stand-in).
- **Backends:** HuggingFace `LanguageModel` (the per-family control) vs vLLM in both engine modes —
  `vllm_async` (continuous-batching server) and `vllm_sync` (in-process) — plus a `vllm_serve` client
  for a remote nnsight-vllm-serve server. Selectable with `--backend(s)` on `scripts/{bench,micro}.py`.
- **Workloads:** `interactive` (single prompt), `batched` (N prompts; throughput + per-prompt
  oracle), and `generation` (greedy multi-token decode; per-step read/intervention). Batching is a
  coverage axis — it is oracle-checked, not timed blind.

Representative finding: the exact portable logit-lens that is correct on GPT-2 is `SILENTLY_WRONG`
on vLLM-Llama (top-1 agreement 0.13 vs HF, no error) because vLLM keeps a dual residual stream and
the single-tensor read drops half of it; reconstructing the stream restores `SUPPORTED`. A
crash-or-not check would mislabel that `SUPPORTED`.

## Running it

Environment: `nnsight-serve-test` (nnsight editable from `/disk/u/zikai/nnsight/src`, the `dev`
branch, + vLLM **0.15.1**). Note the skew: the nnsight `dev` branch targets vLLM **0.19.1** —
running the benchmark in this env is correct, but nnsight-side vLLM behavior must be verified
under `ndif-dev` (vLLM 0.19.1) with `PYTHONPATH=/disk/u/zikai/nnsight/src`, not characterized
from 0.15.1 alone.

GPU runs use the env's python directly (`conda run` mishandles signals on long vLLM runs — false
timeouts, swallowed SIGABRT) and always go under `timeout`:

```bash
PY=/disk/u/zikai/anaconda3/envs/nnsight-serve-test/bin/python

# one methodology (default backends: HF + vLLM-async)
CUDA_VISIBLE_DEVICES=0 timeout 1800 $PY scripts/bench.py --spec steering_gpt2

# all methodologies / families
CUDA_VISIBLE_DEVICES=0 timeout 1800 $PY scripts/bench.py --spec all

# pick backends explicitly: HF only, or the vLLM sync engine instead of async
CUDA_VISIBLE_DEVICES=0 timeout 1800 $PY scripts/bench.py --spec ablation_gpt2 --backends hf
CUDA_VISIBLE_DEVICES=0 timeout 1800 $PY scripts/bench.py --spec steering_gpt2 --backends hf vllm_sync
```

Specs: `logit_lens_gpt2`, `logit_lens_llama`, `steering_gpt2`, `gen_steering_gpt2`,
`activation_patching_gpt2`, `ablation_gpt2`, `attention_pattern_gpt2`, `attribution_patching_gpt2`.
The llama spec needs `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. The Level-0/1 primitive probes run
via `scripts/micro.py --backend {hf,vllm_async,vllm_sync}`.

No-GPU unit tests (cell logic, oracle, timing, driver) run without a model:

```bash
conda run -n nnsight-serve-test python tests/test_sweep.py   # etc.
```

## Layout

```
isb/
  methodologies/   @cell(methodology, family, backend) -> explicit per-cell intervention code
  backends/        `be` infra: HF (control) + vLLM async / sync / serve (run / patch / batched / teardown)
  oracle/          numerical-equivalence comparison (top-1 + total-variation)
  runner/          run_cell, evaluate (per-family control), dtype-control disambiguation
  perf/            time_cell (warmup + N trials, CUDA-synced, median±std, peak mem)
  sweep/           CellConfig/Workload/EffectSpec + the one-pass run_sweep driver
  specs/           one CellConfig per methodology (what bench.py --spec runs)
  report/          applicability map + performance table
scripts/           bench.py (method specs) · micro.py (Level-0/1 primitive probes)
docs/              design.md (living design) · references.md · findings.md (measured results)
```

## Status

Harness built and validated: the one-pass driver (amortized model load, correctness verified in the
same warm/batched regime perf is measured), the numerical oracle, the performance layer, and the 7
methodologies above. Batched runs the per-prompt multi-invoke pattern: it works on HF and on
`vllm_sync` (each prompt is its own request); on `vllm_async` it is gated on an upstream multi-prompt
submission fix. Next: more families, larger models, and real prompt datasets.

See [`docs/design.md`](docs/design.md) for the living design and [`docs/findings.md`](docs/findings.md)
for the measured results.

> Name note: avoids the existing "InterpBench" (Gupta et al., circuits/faithfulness benchmark) — a
> different question than this repo.
