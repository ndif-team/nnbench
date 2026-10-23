# Request patterns: background, and what each one tests

nnbench times every workload under one or more request patterns: how calls are issued to the
system, how many at once, and what is measured. This document gives the performance background the
patterns rest on and says, for each, what it isolates, what it cannot show, and how to read its
numbers, and which workloads run under which pattern. The implementation contract is design.md §12.5 and §12.17; how the standard benchmarks
issue requests is surveyed in `docs/inference-benchmark-survey.md`; measured results are in
`docs/findings.md`.

## Background

### Latency, throughput, and the closed loop that links them

*Latency* is how long one request takes, from issue to return. *Throughput* is how many units the
system finishes per second. They are different questions: an interactive user cares about the first,
a dataset sweep about the second, and a system can trade one for the other by batching.

When `C` clients each wait for their request to return before sending the next (a *closed loop*,
with no pause between requests), Little's law ties the two together:

    throughput = C / mean latency

So a closed-loop curve of throughput against concurrency also says how latency grows. If throughput
stays flat as `C` rises, latency grows in proportion to `C`. The requests are queueing, not being
served together. If throughput rises, the system is overlapping requests. Measured: HF steering at
64 clients gives 35 units/s and a 1.8 s median latency (64 / 1.8 ≈ 35). The async vLLM engine gives
110 units/s at a 0.5 s median. An *open loop* instead sends requests at a set arrival rate whatever
the system does. Its latency can grow without bound once the rate exceeds capacity, which is what
MLPerf's Server scenario and `--request-rate` measure.

### Where an LLM forward spends its time

- **Prefill and decode.** Prefill runs the prompt's tokens through the model at once; decode then
  produces one token per forward, reusing the cached keys and values (the KV cache). Most
  interpretability calls are a single prefill with one output token. Steered or patched generation
  adds decode steps.
- **Reading the weights sets a floor.** Every forward reads all the weights from GPU memory. For a
  7B model in bf16 that is about 15 GB. On an A100's roughly 1.5 TB/s that is about 10 ms, however few
  tokens the batch holds. With few tokens the forward is *memory-bound*: arithmetic is cheap next to
  the read. Batching more tokens into the same forward shares one weight read among them, so a
  forward over 16 short prompts costs little more than over one. Measured on Qwen2.5-7B, plain HF goes
  from 26–28 ms for 1 prompt to 32 ms for 16; plain vLLM from 18 ms to 38 ms. Only with many tokens does
  the forward become *compute-bound*.
- **Fixed costs per call.** Kernel launches (unless CUDA graphs replay them), Python in the
  framework, tokenization, scheduling and, for vLLM, a hop to its engine process. With little work per
  forward these costs are a large share of the time.

### What a serving engine adds, and when it matters

vLLM's design targets many concurrent requests, each with many decode steps. *Continuous batching*
puts every in-flight request into each forward, admitting new ones as old ones finish, so the weight
read is shared across requests and decode steps. *Paged attention* stores the KV cache in blocks, so
many requests fit in memory at once. *CUDA graphs* remove launch overhead from small forwards.
*Prefix caching* reuses the KV cache of a repeated prompt prefix.

None of this helps a single one-token request: there is nothing to batch it with and no decode to
amortize. A serving engine wins when there is a queue of requests to schedule (the offline and
concurrent patterns) or a long decode (generation). The nnsight configurations here run vLLM
eager (no CUDA graphs) with prefix caching off, nnsight's documented default and a fairness control.

### What changes when the request is an intervention

An interpretability call is not just a forward:

- **The client builds a trace.** nnsight records the intervention code and, for vLLM, ships it to
  the engine's worker. Locally, nnsight 0.8.1 pickles each invoke separately into its request, along
  with the whole envoy tree (45 KB and about 6 ms per invoke on Qwen2.5-7B). The engine unpickles each
  one. This cost grows with the number of invokes, not with the work on the GPU (findings.md,
  2026-10-20).
- **Hooks run inside the forward.** Reads and writes at module boundaries add Python and kernel work
  per hooked module.
- **Saved values come back.** A logit lens over every layer returns `layers × vocabulary` values per
  prompt: 17 MB per prompt on Qwen2.5-7B in fp32 on the host, against a few bytes for a sampled token.
- **Correctness depends on how prompts are grouped.** A padded HF batch shifts positions on padded
  rows; a vLLM batch does not pad. So each pattern is also scored for correctness, not just timed.

Each pattern also times two denominators under the same pattern: the workload's own no-intervention
*baseline* call (for example steering with alpha 0), giving `overhead_vs_baseline`, and the
*plain system* with no trace at all (`vanilla`, `vanilla_many`), giving `overhead_vs_vanilla`. The
first isolates the intervention; the second the whole interpretability layer. Engine load is never
timed: each job loads once, warms up, then measures.

## The patterns

| Pattern | Regime key | Standard counterpart | One sentence |
|---|---|---|---|
| Single stream | `interactive` | MLPerf SingleStream | one unit per call, one call at a time: the notebook user |
| Fixed batch | `batched` | `vllm bench latency`, MLPerf MultiStream | the same N-unit call repeated: one batched step |
| Generation | `generation` | single-stream decode | one prompt per call, several decode steps, intervention every step |
| Offline | `offline/b<B>`, `offline/all` | MLPerf Offline, `vllm bench throughput` | a sweep over M units, B per call: a dataset job |
| Concurrent (closed loop) | `concurrent/c<C>` | `vllm bench serve --max-concurrency`, AIPerf | C clients each sending single-unit calls: a shared server |
| Open loop | `open/r<R>` | MLPerf Server, `vllm bench serve --request-rate` | single-unit calls arriving at R per second: a server under offered load |

### Single stream (`interactive`)

**Issues** one unit per call (a prompt, or a clean/corrupted pair), the next after the previous
returns. **Timed:** the first unit, 3 warm-up and 7 timed calls, median. **Scored:** every unit of
the regime, each as its own call (top-1 agreement and TV aggregated over them).

**Tests** the fixed cost of one intervention call: trace construction, shipping, hooks, readout,
plus one forward. This is the latency an interactive user sees, and it isolates the per-call
overhead of the interpretability layer most directly.

**Cannot show** anything about batching or scheduling. The GPU is mostly idle (memory-bound, one
short prompt), so engines differ mainly by their fixed costs. HF and vLLM are close here (Qwen2.5-7B
steering: 28.5 and 33.4 ms at nnsight 0.8.1).

### Fixed batch (`batched`)

**Issues** one call over N units (16), repeated. HF runs a left-padded batch in one forward; vLLM
runs one trace with N invokes, one engine request each. **Timed** like single stream. **Scored**
against the reference's per-unit outputs (`batched_perprompt`), so a padding artifact counts against
the batch.

**Tests** how an intervention call scales with prompts per call, and whether batching changes the
numbers (the padded-batch position bug in nnsight before 0.8.1 is a correctness finding of this
pattern). For vLLM it exposes per-invoke costs. 16 invokes take 165 ms against 38 ms plain, almost
all of it client-side pickling and engine-side unpickling.

**Cannot show** throughput over a dataset. One batch is one prefill on either engine, so a serving
engine has nothing to schedule. It is the pattern most favourable to HF's padded batch.

### Generation (`generation`)

**Issues** one prompt per call with N decode steps (8 for steered generation, 5 for generation-time
patching); the cell intervenes and reads at every step. **Timed** like single stream; tokens/s is
reported. **Scored** per step against the reference.

**Tests** the decode path: the per-step hook cost, the bounded iteration idiom (`tracer.iter`), and
whether an intervention persists correctly across steps through the KV cache. vLLM's faster decode
shows here even for one request (steered generation, Qwen2.5-7B: 162 ms vLLM, 209 ms HF).

**Cannot show** concurrency. A single decoding request leaves most of a serving engine's capacity
unused.

### Offline (`offline/b<B>`, `offline/all`)

**Issues** a sweep over M units (256), B units per call, calls back to back. `all` hands every unit
to one call, which for vLLM means one trace with M invokes that the engine schedules together, like
`LLM.generate(all prompts)`. **Timed:** one untimed warm-up pass, then 3 timed passes; reports
units/s over the median pass, the pass time and per-call latency. **Scored:** the first 16 units of
the last pass, against the reference's per-unit outputs (`offline_perprompt`).

**Tests** throughput for a dataset job (harvesting activations, a patching sweep, an evaluation
over a corpus) and how it depends on how the client chunks the data. Sweeping B separates three
effects: per-call fixed cost (dominant at B=1), the gain from sharing forwards (B=16, 64), and
memory limits (`all`, where an all-layer logit lens over 256 prompts runs out of GPU memory on both
engines). It is also the pattern that rewards an engine able to take the whole sweep at once.

**Reading it:** HF gains with B because a chunk is one padded forward (steering: 37 → 473 → 768
units/s for B = 1, 16, 64). nnsight on the sync vLLM engine levels off near 90 units/s, because every
unit is an invoke that pays the per-invoke serialization. So the offline gap measures nnsight's vLLM
client path, not vLLM's batching (plain vLLM takes 16 prompts in 38 ms).

**Cannot show** latency under contention, or the async engine (its multi-unit calls would need a
second, sync engine, which does not fit beside a large model).

### Concurrent, closed loop (`concurrent/c<C>`)

**Issues** M single-unit calls (256) from C client threads (1, 4, 16, 64). Each client sends its
next unit as soon as its last returns. A backend whose interface takes overlapping requests (the
async vLLM engine, which has its own scheduler) gets them together. A backend whose interface is
one blocking call on one model (an HF forward through nnsight; vLLM's offline `LLM` class behind
the sync engine) gets them one at a time: requests queue in front of the call, as they would in
front of a server built on that interface. **Timed:** one warm-up pass of M requests under the
same pattern, then three timed passes; the latency distribution pools their 3M requests. Each
request is timed from issue to return, so time spent queueing counts. Calls are prepared before
the pass. Reports units/s and request latency p50, p90, p99 and mean. **Scored:** the first 16
units of the last pass, against the reference's same regime.

**Tests** a shared server: many independent users (or one user's parallel jobs) sending traces to one
engine, which is the shape of nnsight-serve and NDIF. It shows whether the system overlaps requests
(throughput rises with C, latency grows slowly) or serializes them (throughput flat, latency ∝ C;
see Little's law above). It also tests that concurrent traces stay correct. They share one model and
one interleaver, so cross-talk would show as a correctness failure.

**Reading it:** HF and the sync engine stay flat (steering 35 and 33 units/s at every C). The async
engine rises: steering 33 → 110 units/s from 1 to 64 clients, steered generation 6 → 58 (10× HF).
Compare backends at equal C, and read throughput together with p99. A higher C that buys little
throughput but much latency is past the knee.

**Cannot show** behaviour under a fixed offered load. A closed loop slows its own arrivals when the
system slows, so it never overloads it, and a stall delays the requests that would have arrived
during it, so its tail latency understates what users would see (coordinated omission).

### Open loop (`open/r<R>`)

**Issues** M single-unit calls (256) at arrival times drawn from a Poisson process at R per second
(4, 16, 64), each on a thread of its own as soon as it arrives, whether or not earlier requests have
returned. The arrival times come from a fixed seed, so every pass and every backend sees the same
schedule. A backend whose interface takes overlapping requests gets them together; one whose
interface takes one request at a time has them queued in front of it, as in the closed loop.
**Timed:** one warm-up pass at the same arrivals, then three timed passes; the latency
distribution pools their 3M requests. Each request is timed from its scheduled arrival to its
return, so waiting in a queue, or for the system to accept it, counts. Reports achieved units/s,
request latency p50, p90, p99 and mean, the offered rate, and the most requests in flight at once.
**Scored:** the first 16 units of the last pass, against the reference's same regime.

**Tests** a production server under real traffic, including overload. Below capacity, latency stays
near the single-request time and achieved throughput tracks the offered rate (a little under
it, since a pass also waits for its last request). Above it, requests
pile up: the most in flight grows toward M, latency grows through the pass, and achieved throughput
levels off at capacity. That pile-up is where problems that a closed loop cannot reach appear: saved
values and KV cache held for every request in flight, client-side trace serialization (about 6 ms
per invoke for nnsight 0.8.1 on vLLM) becoming the limit before the GPU, and failures or stalls under
overload. One caveat on that serialization limit: the arrival threads, the issuing loop and
nnsight's per-invoke pickling share one Python process and its GIL, so a late issue (charged to
the system, the right side to err on) can be the client's own contention. The ceiling it sets is a
one-process ceiling that N client processes would not share; a sweep whose achieved rate stops at
the client should be repeated from several processes before it is read as an engine limit.

**Reading it:** read p99 against the offered rate. The highest rate whose p99 stays within a latency
target is the server's usable capacity (MLPerf Server's pass criterion). On a serial backend (HF, the
sync engine) the open loop is a single-server queue, so its latency follows from the single-request
time; the suite runs it only on HF, the reference, and the async engine.

**Cannot show** bursty traffic or abandoned requests: arrivals are Poisson only, and a pass has no
request deadline, so an overloaded system drains its queue after the last arrival.

## Which workloads run under which pattern

A workload's methodology is used in one of two settings. In a research job (circuit discovery,
feature training, evaluation sweeps, lens fitting) the researcher holds the data and writes the loop
that issues the calls: the next call goes out when the previous one returns, nobody waits on one
unit's answer, and the number that matters is how long the whole job takes. In deployment (steered
chat, monitors and probes on served generations) the method runs inside requests that users send at
times the system does not control, and what matters is latency at an offered rate. Each workload
declares the load patterns of its settings (`load_patterns` in `isb/workloads/`), and a suite's load
runs only on the workloads that declare it, so no run times a combination nobody uses.

The declaration follows from who controls the arrival process:

- `offline` when the method is run over a dataset the researcher holds. The researcher's loop is a
  closed loop by construction, so throughput at a chosen chunk size is the whole question.
- `concurrent` and `open` only when a deployment runs the method once per live request. The closed
  loop shows whether the engine overlaps requests; the open loop shows latency at a rate the users
  set, including overload. A research job never produces that arrival process, so pushing a
  patching sweep at a Poisson rate would measure a situation no user is in. The logit lens is the
  read under load: a last-position read on a served request is the runtime monitor or probe, so it
  declares all three.
- none of the three when the job is one sequential computation whose calls depend on each other.
  There is no stream of independent requests to shape; the job runs as single stream, one job per
  call, which is the workload's own `interactive` regime.
- none, for a harness or engine reason rather than a usage one, when a sweep could not be scored:
  the attention pattern's per-prompt outputs have different lengths, so they cannot be stacked
  against a per-unit reference, and vLLM has no attention probabilities to read.

Two axes are involved. The request shape is the workload's own regime: one unit per call
(`interactive`), several units per call (`batched`), or one unit decoded for several steps
(`generation`). The load pattern says how those requests are issued. Offline sweeps a workload with
several-unit calls at any B, and a workload whose cells take one unit per call at B=1, which is how
a researcher loops over a dataset with a one-unit API. Concurrent and open issue the single-unit
request shape, decode included.

| Workload | Setting | Request shape (own regimes) | Load patterns |
|---|---|---|---|
| Logit lens | both | one prompt; a batch of prompts | offline (B = 1, 16, 64, all), concurrent, open |
| Jacobian lens | research job | one prompt | offline (B = 1) |
| Ablation | research job | one prompt; a batch of prompts | offline (B = 1, 16, 64, all) |
| Activation patching | research job | one clean/corrupted pair | offline (B = 1) |
| Generation patching | research job | one pair, decoded 5 steps | offline (B = 1) |
| Attribution patching | research job | one labeled pair (forward and backward) | offline (B = 1) |
| Activation harvesting | research job | a batch of passages | offline (B = 1, 16, 64, all) |
| Jacobian collection | research job | the corpus in one call; one prompt per call under load | offline (B = 1) |
| Steering | both | one prompt; a batch of prompts | offline, concurrent, open |
| Steered generation | both | one prompt, decoded 8 steps | offline (B = 1), concurrent, open |
| Per-step capture | both | one prompt, decoded 8 steps | offline (B = 1), concurrent, open |
| Attention pattern | research job | one prompt; output length follows the prompt | single stream only |
| DAS (train and apply) | research job | one fit: the training loop and its held-out check | single stream only |

The last rows follow how the upstream code runs these methods.

- **DAS** is one sequential job. Each training step needs the rotation the previous step updated,
  and the reference implementations run a sweep as one fit after another: pyvene's IOI tutorial
  (`tutorials/advanced_tutorials/tutorial_ioi_utils.py`, `find_variable_at`) builds one
  `IntervenableModel` per (position, layer) and trains it for 10 epochs at 20 pairs per step before
  the next; causalab (`methods/trained_subspace/train.py`) trains one target after another at 32
  pairs per step and parallelises a sweep across slurm jobs. Fits are not batched into one forward.
  Our cell's call is one fit, timed single stream in the `interactive` regime.
- **Jacobian collection** upstream (`anthropics/jacobian-lens`, `jlens/fitting.py`) runs prompts one
  at a time, 8 cotangent rows per backward, and parallelises by fitting disjoint prompt slices and
  merging the lenses. That is one prompt per call with the mean on the client: the offline sweep at
  B = 1. The batch axis is spent on cotangent rows, so B > 1 per call is not the upstream shape.
- **Attention pattern** output is `[layers, heads, prompt length]` per prompt, so outputs of
  different prompts cannot be stacked into one scored sweep, and vLLM has no attention
  probabilities to read, so every pattern there would record the same error.

Every workload keeps its own single-call regimes (`interactive`, `batched`, `generation`): the
notebook user's view and the per-call overhead. Each is one point of a load pattern timed on one
repeated input: `batched` is an offline call at B = 16, `interactive` and `generation` a closed loop
at C = 1.

The DAS and Jacobian collection cells take the form nnsight documents for a multi-trace job: the
whole fit, or the whole sweep, runs inside one `model.session()` (`be.session`, which on vLLM runs
the job's traces as they are reached, since nnsight documents sessions for transformers and remote
models only). A DAS step trains on all 16 pairs of the train split at once, a full-batch step
(pyvene's IOI sweep steps on minibatches of 20, causalab of 32): one left-padded batch, the rotated
interchange at every row's last position, cross-entropy over the rows. Measured on gpt2 (HF, A100, nnsight dev line) against the
1-pair step: the apply row's output is unchanged bit for bit (seeded rotation, 154 ms against
182 ms per call), the train row goes from 1011 ms to 1385 ms per 24-step fit for 16 times the pairs
per step, and its held-out guard still passes. Jacobian collection's output is unchanged bit for
bit under the session (10.0 s against 11.0 s per 4-prompt sweep, one sample each on a shared
card; cause of the difference not determined).

Known gaps, recorded here rather than built:

- A multi-tenant research server (NDIF) receives whole jobs at times its users choose, an open loop
  whose unit is a job (a fit, a 256-prompt lens sweep), not a prompt. That is not implemented. The
  engine behaviour it would expose, overlapping whole sessions, is nearest to what steering under
  the concurrent pattern shows at request level.
- Only the async vLLM engine declares `concurrent_requests`; every other backend has its requests
  queued one at a time. For the nnsight suites here that is the fact being measured (HF and the
  sync engine take one request at a time by their interface). A system whose interface does take
  overlapping requests (interp-engine's async methods, nnsight-serve, vLLM-Lens under `vllm
  serve`, plain vLLM's AsyncLLM) must declare it and issue through that path, or the row would
  measure the harness's queue in front of a system that has its own scheduler.

## Choosing and comparing

- **Per-call overhead of the interpretability layer:** single stream, with `overhead_vs_vanilla`.
- **Correctness of batching:** fixed batch and offline (scored per unit).
- **A dataset job's speed:** offline, the best B for each backend. HF needs a client-chosen B; a
  serving engine should not.
- **Serving many users:** concurrent, throughput against p99 at each C.
- **A server under real traffic, and overload:** open loop, p99 against the offered rate.
- **Decode-time interventions:** generation, and concurrent on a generation workload.

Comparisons need the same workload, data, pattern settings and hardware. Results on different cards
are not comparable for performance: the same HF cell differs by up to 10% between two A100s. Each
pattern records its settings in the run's coordinates (`load`), and the scorer compares only
matching regimes.
