# How inference-engine benchmarks issue requests (survey, 2026-10-20)

The survey behind the load patterns of design.md §12.17. It covers how the standard LLM serving
benchmarks send requests, not their datasets; nnbench keeps its own workload data (CounterFact,
MIB) and borrows only the request patterns. What each pattern tests, and the performance
background, is in `docs/request-patterns.md`.

## The four request patterns

| Pattern | Used by | How requests are sent | Reported |
|---|---|---|---|
| Single stream | MLPerf SingleStream | one request at a time; the next after the previous completes | latency percentile |
| Fixed batch, repeated | `vllm bench latency`; MLPerf MultiStream | the same batch every iteration (vLLM defaults: batch 8, 32 input and 128 output tokens with `ignore_eos`, 10 warm-up and 30 timed iterations) | batch latency, p10–p99 |
| Offline | `vllm bench throughput`; SGLang `bench_offline_throughput`; MLPerf Offline | every prompt in one call (`llm.generate(prompts, ...)`), the engine schedules them | requests/s, tokens/s |
| Online serving | `vllm bench serve`; SGLang `bench_serving`; NVIDIA AIPerf (formerly GenAI-Perf); MLPerf Server | requests arrive over time, capped by a concurrency limit; the arrival rate or the limit is swept | throughput against latency (TTFT, TPOT, ITL, end-to-end; p50/p99) |

## Details by tool

**MLPerf Inference LoadGen** (`loadgen/test_settings.h`) defines the scenarios:

- SingleStream issues queries with one sample; the next query is issued once the previous one
  completes. The result is a latency percentile.
- MultiStream issues queries with N samples, again one at a time; the result is a percentile of the
  query latency.
- Server sends single-sample queries with Poisson arrivals at a target QPS; it passes when a
  latency percentile stays under a threshold. The LLM benchmarks set TTFT and TPOT limits (for
  Llama-2-70B, TTFT 2000 ms and TPOT 200 ms).
- Offline sends all N samples in one query; the result is samples per second.

**vLLM** (`vllm bench`, `vllm/benchmarks/`):

- `latency` builds one batch of random-token prompts once and calls `llm.generate` on it each
  iteration (`--batch-size 8 --input-len 32 --output-len 128 --num-iters-warmup 10 --num-iters 30`).
- `throughput` submits every prompt in one `llm.generate` call (or `llm.enqueue` with
  `--prequeue-requests`; `--async-engine` issues one `generate` per prompt and drains them together).
  It reports requests/s, total tokens/s and output tokens/s.
- `serve` drives a running server. `--request-rate` defaults to `inf` (everything at once); a
  finite rate draws Poisson arrivals, or gamma with `--burstiness` (below 1 burstier, above 1 more
  uniform). `--max-concurrency` caps requests in flight; `--ramp-up-strategy` ramps the rate.
  Metrics: TTFT, TPOT ((end-to-end − TTFT) / (output tokens − 1)), ITL, end-to-end latency, with
  mean, median and P99.

**SGLang** (`bench_serving`, `bench_offline_throughput`): the same knobs as `vllm bench serve`
(`--request-rate inf` or Poisson, `--max-concurrency`); `--random-range-ratio` spreads synthetic
input and output lengths around their means. The offline tool runs the engine in-process and
submits everything at once. The guide advises at least five times as many prompts as the
concurrency limit, so a run reaches steady state.

**NVIDIA AIPerf** (GenAI-Perf's successor): concurrency mode or a request rate with constant,
Poisson or gamma arrivals and ramping. The NIM guide sweeps concurrency (1, 2, 5, 10, 50, 100, 250),
one run per value, and plots throughput against TTFT; streaming is required for TTFT and ITL.

## What this means for nnbench

- `interactive` is the single-stream pattern and `batched` the fixed-batch one: one call repeated.
  Both give a serving engine the least to schedule. A 16-prompt, one-token call is one prefill on
  either engine, bound by reading the weights (measured: plain vLLM 38 ms, plain HF 32 ms,
  Qwen2.5-7B on an A100).
- The `offline` and closed-loop `concurrent` load patterns (design.md §12.17) add the other two.
  The open-loop Poisson form of online serving is not implemented yet.
- An intervention call is mostly one forward, with no decode, so TTFT/TPOT apply only to the
  generation workloads; the load patterns report units/s and request latency percentiles instead.
- HF has no request scheduler. Under concurrency nnbench drives it, and vLLM's sync engine, through
  one lock, as a serial server would; that is how nnsight on HF runs for its users, not a stronger
  baseline that batches queued requests.

## Sources

- vLLM benchmark CLI: https://docs.vllm.ai/en/latest/benchmarking/cli/
- vLLM `benchmarks/throughput.py`, `benchmarks/latency.py`: https://github.com/vllm-project/vllm/tree/main/vllm/benchmarks
- MLPerf LoadGen scenario definitions: https://github.com/mlcommons/inference/blob/master/loadgen/test_settings.h
- MLPerf Inference benchmarks and latency limits: https://docs.mlcommons.org/inference/
- SGLang bench_serving: https://docs.sglang.io/developer_guide/bench_serving.html
- SGLang benchmark and profiling: https://docs.sglang.io/docs/developer_guide/benchmark_and_profiling
- NVIDIA AIPerf: https://developer.nvidia.com/blog/benchmarking-llm-inference-at-scale-with-aiperf
- NVIDIA NIM AIPerf quickstart: https://docs.nvidia.com/nim/benchmarking/llm/latest/quickstart.html
