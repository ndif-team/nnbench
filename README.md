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
- **Docker backends:** `nnsight-hf` and `nnsight-vllm`, discovered from independent directories
  under `backends/`. Legacy micro/serve tools also expose other engine modes.
- **Workloads:** `interactive` (single prompt), `batched` (N prompts; throughput + per-prompt
  oracle), and `generation` (greedy multi-token decode; per-step read/intervention). Batching is a
  coverage axis — it is oracle-checked, not timed blind.

Representative finding: the exact portable logit-lens that is correct on GPT-2 is `SILENTLY_WRONG`
on vLLM-Llama (top-1 agreement 0.13 vs HF, no error) because vLLM keeps a dual residual stream and
the single-tensor read drops half of it; reconstructing the stream restores `SUPPORTED`. A
crash-or-not check would mislabel that `SUPPORTED`.

## Running it

Backend names match directories under `backends/`. Each has an independent Dockerfile, Compose
configuration and entrypoint. Build them once, then run named experiments:

```bash
python scripts/bench.py build nnsight-hf nnsight-vllm

# HF reference and vLLM candidate; no implicit control runs
python scripts/bench.py run --spec logit_lens_gpt2 --data factual:2 \
  --backends nnsight-hf nnsight-vllm --reference nnsight-hf --gpu 0

# one methodology
python scripts/bench.py run --spec steering_gpt2 \
  --backends nnsight-hf nnsight-vllm --reference nnsight-hf --gpu 0

# collect the small default corpus on HF
python scripts/bench.py run --spec all --backends nnsight-hf --gpu 0

python scripts/bench.py list backends
python scripts/bench.py score runs/REPLACE_WITH_RUN_ID
```

Docker owns Python and inference dependencies. The host launcher prepares identical inputs for
each backend; a separate CPU scorer reads the saved artifacts (host CPU PyTorch is sufficient).
Every invocation creates a unique directory under `--out` (default `runs/`). Errors are retained
per cell; crashes and incomplete artifacts fail the job. Use `--strict` for cell-verdict exit codes.
See [`backends/README.md`](backends/README.md) for the contract, configuration, and extension guide.

Specs: `logit_lens_gpt2`, `logit_lens_llama`, `steering_gpt2`, `gen_steering_gpt2`,
`activation_patching_gpt2`, `ablation_gpt2`, `attention_pattern_gpt2`, `attribution_patching_gpt2`.
The llama spec needs `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. The Level-0/1 primitive probes run
via `scripts/micro.py --backend {hf,vllm_async,vllm_sync}`.

No-GPU unit tests exercise cell logic, oracle, timing, and command construction without a model:

```bash
python -m pytest tests/ -q
```

## Browsing results

The result site reads saved JSON reports from a run directory or a collection of run directories:

```bash
python scripts/manager.py --dir runs --port 6688
python scripts/manager.py --dir runs --export results.html
```

Open `http://127.0.0.1:6688` for the local site, or share the self-contained HTML export.
Backend names and experiment histories stay separate. Browsing never loads tensor artifacts or
recomputes verdicts. See [the result-site guide](docs/result-site.md) for legacy imports and safe
inbox/archive operations.

## Layout

```
isb/
  methodologies/   @cell(methodology, family, backend) -> explicit per-cell intervention code
  backends/        `be` infra: HF (control) + vLLM async / sync / serve (run / patch / batched / teardown)
  oracle/          numerical-equivalence comparison (top-1 + total-variation)
  runner/          run_cell, evaluate (per-family control), dtype-control disambiguation
  perf/            time_cell (warmup + N trials, CUDA-synced, median±std, peak mem)
  jobs/            frozen experiments, name-based Docker lifecycle, artifact scoring
  sweep/           experiment definitions and shared cell execution; legacy artifact scorer
  specs/           one CellConfig per methodology (what bench.py --spec runs)
  report/          applicability map + performance table
backends/          independent NAME/{compose.yml,Dockerfile,run.py} packages
scripts/           bench.py (name-based CLI) · legacy execute/score · micro/perf tools
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
