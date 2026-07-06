# Performance microbenchmark (design)

A second benchmark, separate from the functionality micro tier and the method tier. It
measures the cost of each primitive intervention op in isolation, as overhead over an
unmodified engine, head-to-head across systems. It adopts the vllm-lens layout (one op,
many systems) and vllm-hook's sweep depth and storage metrics.

Scope boundary: this benchmark answers "what does an op cost, and how does nnsight compare
to purpose-built plugins on the ops they share." It does NOT measure coverage, correctness
maps, or the features only nnsight has. Those stay in the method tier. The one place the
two touch is a per-cell correctness cross-check (below), so a perf number is never reported
for a silently-wrong read.

## What is and is not measured here

Primitive ops only, because only primitives are comparable across systems:

- read: capture residual or hidden state at a layer.
- steer: write-replace at a layer.
- qk: capture Q and K at a layer (a read variant; pattern recompute is offline and excluded
  from engine overhead). vllm-lens has no Q/K path, so it is absent from this row.

Explicitly NOT in the microbench (they are not primitives, not unique, or not runnable):

- cache: not a primitive. Reading many layers is the footprint axis of read; `cache` the
  method is a reuse convenience, measured nowhere here.
- cross-prompt: a higher-level idiom (an edge from one prompt's read to another's write).
  Method tier.
- grad: vLLM runs under inference_mode, so backward is unsupported on every vLLM-based
  system including nnsight. Out of the vLLM microbench entirely.
- compose: data-dependent composition needs arbitrary Python (nnsight only); its cost is the
  composition law, which the method tier owns. Static co-configuration of read plus steer in
  one request is a feature-matrix question, not a perf-micro one.

The distinguishing-feature story is entirely the method tier's job. The microbench sells
parity on the shared primitives, nothing more.

## Systems (columns)

- pure_vllm: no intervention. The overhead denominator.
- native_eagle: vLLM's in-tree `ExampleHiddenStatesConnector`. Read baseline (read only).
- vllm_lens: read, steer (add-only, multi-layer, positional). No Q/K path.
- vllm_hook: read, steer, qk. Driven via `HookLLM(worker_name=...)`, not raw `extra_args`; steer
  targets a single `optimal_layer`, so footprint half/all are not expressible for hook steer.
- nnsight_vllm: read, steer, qk (sync / async / serve, async is the default).
- nnsight_hf and raw PyTorch hooks on HF: the "without a production engine" reference
  (optional, added later).

## Axes (sweep)

- footprint: one, half, all layers.
- token_mode: last_token, all_tokens.
- phase: prefill (new_tokens = 1), decode (generation).
- model, a hidden-dim ladder: ~1.5B, ~7-8B, 70B at TP=4. GPT-2 is the CI smoke size only.
- tensor_parallel_size: 1 vs 4 on a model that fits both (gather-on-access cost).
- prompt_len: 16, 64, 256, 512.
- batch: 1, 4, 16 (continuous batching).
- destination: host vs gpu (read only).

Do not run the full cartesian product. One primary sweep per axis with the rest at defaults,
plus a few 2D slices (footprint x model for read, phase x token_mode, tp x footprint). Log
every truncation; never silently cap coverage.

## Metrics

- throughput (tok/s), latency (gen and total).
- overhead vs pure_vllm, percent. The headline. Computed at aggregation against the baseline
  cell sharing the same workload key (repo, phase, prompt_len, batch, tp, new_tokens).
- peak GPU memory: NVML system-wide footprint delta (before vs after load), because torch's
  counter misses the spawned engine child. CAVEAT: currently reservation-dominated. vLLM pre-reserves
  `gpu_memory_utilization` (~24.5 GB) up front, dwarfing intervention buffers, so the column is
  ~flat across cells. To isolate per-op memory, run with a low `gpu_memory_utilization` or measure
  the KV-cache-available delta instead. Plus transfer volume and artifact size.
- one correctness cross-check per cell: captured value matches the HF-eager reference within
  tolerance, via nnbench's oracle. Optional, off by default for speed; on for a correctness
  pass. A perf number on a silently-wrong read is meaningless, and this is the only fold-touch.

## Fairness controls

1. Pin engine config per row: `enforce_eager` identical across all systems in a row, and
   reported. nnsight and vllm-lens force eager; vllm-hook does not force global eager (its hooks
   skip during CUDA-graph capture), so pin it explicitly for the comparison.
2. Match token mode: nnsight full read vs all_tokens; a last-token slice variant of nnsight
   vs vllm-hook's last_token. Equal bytes moved or the numbers are not comparable.
3. Match regime: prefill and decode reported separately, never averaged.
4. Split the timer: gen_lat (in-forward) vs total_lat (gen + retrieval / D2H / disk / collect).
   nnsight keeps reads on GPU until collect, the plugins copy out; report both or the
   comparison is unfair in one direction.
5. Same model, dtype, max_model_len, gpu_memory_utilization, hardware per row.
6. Warmup 5, reps 10, median and standard deviation (vllm-hook's protocol).
7. One system per process (spawn), per-cell timeout watchdog, teardown plus orphan check
   between cells. Exclude model load by default; report cold vs warm separately.

## Harness

`isb/perf/`:

- `core.py`: `Config`, measurement helpers (timer, NVML peak mem, warmup/reps, summarize),
  and aggregation (`attach_overhead`, baseline matching).
- `systems/<name>.py`: one module per system, each exposing `run(cfg) -> dict`, importing only
  its own deps. Separate modules so vllm-lens, vllm-hook, and nnsight never share a process.
- `systems/_echo.py`: a GPU-free stub used only to self-test the runner.

`scripts/perf.py`: two modes. `worker --config-file F` runs one cell in its own process and
prints `RESULT_JSON {...}`. `sweep --plan P --out O` launches one subprocess per cell with a
timeout, collects results, runs teardown, attaches overhead, writes the results file. Mirrors
the existing `scripts/micro.py` one-backend-per-process pattern.

Environment: one conda env per auto-registering plugin (they cross-fire if co-installed), all
aligned on the SAME engine build (vLLM 0.19.1 / torch 2.10+cu128 / transformers 4.57.6 / py3.12) by
cloning a base env and adding only the plugin with `--no-deps`:

- `nnsight_vllm`: env `nnsight-vllm` + `PYTHONPATH=/disk/u/zikai/nnsight/src` (dev branch).
- `pure_vllm`, `native_eagle`: env `nnsight-vllm` (clean vanilla `LLM`; nnsight does not auto-register).
- `vllm_lens`: env `bench-vllm-lens` (clone + vllm-lens; registers the `activations` plugin).
- `vllm_hook`: env `bench-vllm-hook` (clone + vllm-hook; registers `hook_registry` / `vllm_hook`).

Setup script: `/disk/u/zikai/bench-systems/setup_competitor_envs.sh`. Use the env's python directly
(not `conda run`, which gives false timeouts on vLLM), and always wrap runs in a timeout with an
orphan check after. The earlier `nnsight-serve-test` env (vLLM 0.15.1) no longer exists; vLLM 0.19.1
is the branch-target version, so this base is the right one. TODO before the head-to-head run: the
runner (`scripts/perf.py`) must select the env python per system (it currently launches every cell
subprocess with one python).

## Figures and tables it produces

- read overhead vs footprint (one, half, all), per model size. The core parity curve.
- read overhead vs prompt length, last_token vs all_tokens. Data-volume scaling.
- steer overhead across systems. Near-free everywhere.
- peak memory and artifact size across systems (vllm-hook's storage angle, incl. Eagle's
  drafter overhead).
- prefill vs decode overhead per op (table).
- read overhead at TP=1 vs TP=4 (gather-on-access cost).

## Status

- Framework verified without a GPU: `core.py` (config, measurement, aggregation) and the
  `scripts/perf.py` runner, via `tests/test_perf.py` against the `_echo` stub.
- `pure_vllm.py`: run on GPU (GPT-2 + Qwen2.5-1.5B); baseline works.
- `nnsight_vllm.py`: read verified clean on two models (decode overhead footprint-ordered ~24-33%,
  prefill near-free); steer and qk verified-runs with provisional numbers (a co-tenant job
  contaminated the latencies). The idioms are confirmed against dev-branch nnsight + vLLM 0.19.1:
  replacement-form steer (in-place ERRORs), and qkv-projection read for qk (the attention pattern
  is not readable on vLLM).
- `vllm_lens.py`, `vllm_hook.py`: `extra_args` markers resolved and name-verified offline against the
  installed packages (vllm-lens `SteeringVector` fields; vllm-hook `HookLLM` worker names
  `probe_hidden_states` / `probe_hook_qk` / `steer_hook_act`). Not yet run on GPU.
- `native_eagle.py`: still a scaffold (the `kv_transfer` connector config is unresolved).
- NVML memory wired in (see the metrics caveat: reservation-dominated, not yet isolating per-op).
- Remaining before paper-grade numbers: per-system env-python in the runner, a clean uncontended
  GPU re-run at `n_reps=10`, the memory-metric fix, and the larger model rungs (7-8B, 70B/TP=4).
