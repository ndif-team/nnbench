# interp-serve-bench (provisional name)

A **systems performance + coverage benchmark for interpretability workloads** run through
nnsight across serving backends (HF Transformers, vLLM sync/async, NDIF remote), model
types/architectures, and parallelism/optimization configurations.

This is **not** a faithfulness benchmark (we do not measure whether an interpretability
*result* is scientifically correct — that is what `causalab` / CausalGym / InterpBench do).
We measure whether an interpretability *workload* **runs**, **runs correctly across backends**
(numerical equivalence vs an HF reference), and **runs fast** (latency / throughput / memory /
overhead) as it is swept across the production-serving design space.

Status: **design phase.** This repo currently holds evolving design notes only.

- [`docs/design.md`](docs/design.md) — the living design (taxonomy, levels, harness, sweep, metrics, open forks)
- [`docs/references.md`](docs/references.md) — references organized by category + positioning analysis

Provisional name — easy to rename. Avoids the existing "InterpBench" (Gupta et al., circuits
benchmark) collision.
