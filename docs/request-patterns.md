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

**Issues** M single-unit calls (128) from C client threads (1, 4, 16, 64). Each client sends its
next unit as soon as its last returns. A backend that declares it can take concurrent requests (the
async vLLM engine) gets them together. Any other (HF, the sync engine) gets them one at a time
through a lock, which is what a server without a scheduler does with concurrent clients. **Timed:**
one warm-up request per client, then one pass of M requests. Each request is timed from issue to
return, so time spent queueing counts. Calls are prepared before the pass. Reports units/s and
request latency p50, p90, p99 and mean. **Scored:** the first 16 units, against the reference's same
regime.

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

**Issues** M single-unit calls (128) at arrival times drawn from a Poisson process at R per second
(4, 16, 64), each on a thread of its own as soon as it arrives, whether or not earlier requests have
returned. The arrival times come from a fixed seed, so every pass and every backend sees the same
schedule. A backend that can take concurrent requests gets them together; any other queues them
behind one lock, as in the closed loop. **Timed:** one warm-up request, then one pass. Each request
is timed from its scheduled arrival to its return, so waiting in a queue, or for the system to accept
it, counts. Reports achieved units/s, request latency p50, p90, p99 and mean, the offered rate, and
the most requests in flight at once. **Scored:** the first 16 units, against the reference's same
regime.

**Tests** a production server under real traffic, including overload. Below capacity, latency stays
near the single-request time and achieved throughput tracks the offered rate (a little under
it, since a pass also waits for its last request). Above it, requests
pile up: the most in flight grows toward M, latency grows through the pass, and achieved throughput
levels off at capacity. That pile-up is where problems that a closed loop cannot reach appear: saved
values and KV cache held for every request in flight, client-side trace serialization (about 6 ms
per invoke for nnsight 0.8.1 on vLLM) becoming the limit before the GPU, and failures or stalls under
overload.

**Reading it:** read p99 against the offered rate. The highest rate whose p99 stays within a latency
target is the server's usable capacity (MLPerf Server's pass criterion). On a serial backend (HF, the
sync engine) the open loop is a single-server queue, so its latency follows from the single-request
time; the suite runs it only on HF, the reference, and the async engine.

**Cannot show** bursty traffic or abandoned requests: arrivals are Poisson only, and a pass has no
request deadline, so an overloaded system drains its queue after the last arrival.

## Which workloads run under which pattern

A workload's methodology is used in one of two settings. In a dataset study (circuit discovery,
feature training, evaluation sweeps) a researcher prototypes on single calls, then runs the method
over a dataset; the speed that matters is the sweep's throughput. In runtime interpretability
(monitors, probes and steering on live traffic) the method runs inside requests users send, and what
matters is latency and throughput under that traffic. Each workload declares the load patterns of
its settings (`load_patterns` in `isb/workloads/`), and a suite's load runs only on the workloads
that declare it, so no run times a combination nobody uses.

Two axes are involved. The request shape is the workload's own regime: one unit per call
(`interactive`), several units per call (`batched`), or one unit decoded for several steps
(`generation`). The load pattern says how those requests are issued. Offline sweeps a workload with
several-unit calls at any B, and a workload whose cells take one unit per call at B=1, which is how
a researcher loops over a dataset with a one-unit API. Concurrent and open issue the single-unit
request shape, decode included.

| Workload | Setting | Request shape (own regimes) | Load patterns |
|---|---|---|---|
| Logit lens | dataset study | one prompt; a batch of prompts | offline (B = 1, 16, 64, all) |
| Jacobian lens | dataset study | one prompt | offline (B = 1) |
| Ablation | dataset study | one prompt; a batch of prompts | offline (B = 1, 16, 64, all) |
| Activation patching | dataset study | one clean/corrupted pair | offline (B = 1) |
| Generation patching | dataset study | one pair, decoded 5 steps | offline (B = 1) |
| Attribution patching | dataset study | one labeled pair (forward and backward) | offline (B = 1) |
| Activation harvesting | dataset study | a batch of passages | offline (B = 1, 16, 64, all) |
| Steering | both | one prompt; a batch of prompts | offline, concurrent, open |
| Steered generation | both | one prompt, decoded 8 steps | offline (B = 1), concurrent, open |
| Per-step capture | both | one prompt, decoded 8 steps | offline (B = 1), concurrent, open |
| Attention pattern | dataset study | one prompt; output length follows the prompt | none |
| DAS (train and apply) | dataset study | one training call over the dataset | none |
| Jacobian collection | dataset study | one fitting call over the corpus | none |

The last three have no load pattern. DAS trains one rotation over the training units and checks it
on held-out units, and Jacobian collection averages its maps over every prompt: each is one call
whose result depends on all its units together, so splitting it into per-unit requests would change
what it computes. How their cost grows with data is measured by the size of that one call. The
attention pattern is `[layers, heads, prompt length]` per prompt, so outputs of different prompts
cannot be stacked into one scored sweep, and vLLM has no attention probabilities to read, so a
sweep would record only errors there. Dataset-study workloads take no
concurrent or open load: live traffic does not run them. Every workload keeps its own single-call
regimes (`interactive`, `batched`, `generation`): the notebook user's view and the per-call
overhead. Each is one point of a load pattern timed on one repeated input: `batched` is an offline
call at B = 16, `interactive` and `generation` a closed loop at C = 1.

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
