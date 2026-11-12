# Measured results (October 2026)

Latency, throughput and workload support of five interpretability systems (nnsight on HF
transformers and on vLLM, vLLM-Lens, vLLM-Hook, interp-engine and the TransformerLens vLLM bridge),
measured with nnbench between 2026-10-26 and 2026-10-30 on one host, with plain vLLM of each
system's vLLM version as the floor. The measurement tables are generated from the saved run
reports; [Source runs](#source-runs) lists each run and the source commit it ran from. Releases
newer than the ones listed under [Systems](#systems) are not measured here. Dates and times on this
page are UTC.

## Setup

### Systems

| System | Release | Engine | Configs |
|---|---|---|---|
| nnsight on HF transformers | 0.8.1 (`de7fec4`) | transformers 5.12.1, torch 2.11.0, sdpa attention | `nnsight-hf/0-8-1`; on several GPUs, `0-8-1/gpu2` and `0-8-1/gpu4` (layers placed across the GPUs), and `0-8-1/tp2` and `0-8-1/tp4` (tensor parallel, transformers 5.19.0) |
| nnsight on vLLM | 0.8.1 (`de7fec4`) | vLLM 0.27.1 | `nnsight-vllm/0-8-1` (the default: the synchronous `LLM` path, eager), `0-8-1/async` (`AsyncLLM`, eager), `0-8-1/opt` (edits installed once in the engine, eager), `0-8-1/opt/taps` (installed edits with CUDA-graph taps; CUDA graphs on, torch.compile off) |
| vLLM-Lens | 1.2.1 | vLLM 0.19.1 | `vllm-lens/default`, `vllm-lens/opt` (one unembedding pass for the lens, ablation through hooks), `vllm-lens/opt/async`; all eager |
| vLLM-Hook | commit `0e34fdd` (`vllm_hook_plugins` 0.2.0) | vLLM 0.19.1 | `vllm-hook/default` (rpc storage), `vllm-hook/disk` (`disk-st-async` storage), `vllm-hook/disk/async`; all eager |
| interp-engine | 1.12.0 | vLLM 0.28.0 | `interp-engine-vllm/default` (hooked, eager), `interp-engine-vllm/static` (`vllm-static`; CUDA graphs on, torch.compile off) |
| TransformerLens, vLLM bridge | 4.0.0 | vLLM 0.20.2 | `transformer-lens-vllm/default` (vLLM's default mode, one prompt per forward), `transformer-lens-vllm/batched` (`enable_batching`, eager), `transformer-lens-vllm/opt` (readout moved to the GPU, vLLM's default mode) |
| plain vLLM | 0.19.1, 0.20.2, 0.27.1, 0.28.0 | | `vllm-plain/<version>` (vLLM's default mode: torch.compile and CUDA graphs); `<version>/tp2` and `<version>/tp4` on two and four GPUs |

A system's first config is the benchmark's default realization of it, written from the system's
documentation at the release listed ([writing workloads](writing-workloads.md)).
`transformer-lens-vllm/default` also turns on the bridge's opt-in `enable_position_interventions`.
The other configs change the engine mode, the storage, or how the documented calls are composed.
They are faster for some loads and slower for others: `transformer-lens-vllm/batched` is slower
than the default on single requests. `transformer-lens-vllm/opt` goes outside the bridge's public
interface. The bridge rebuilds logits on the CPU with no documented GPU option, and this config
projects on the GPU with weights read through the bridge's vLLM driver, so its rows are headroom
beside the bridge's own rows. Two groups of cells use attributes their systems do not document as
public: the TransformerLens cells read the final-norm and unembedding weights through
`model.bridge._driver`, and the vLLM-Hook cells set `llm._output_layers` and `llm._hs_mode`, the
fields its `load_config` fills from a config file.

### Models and hardware

- Qwen2.5-7B-Instruct (28 layers) on one GPU; Qwen2.5-14B-Instruct (48 layers) and Qwen3-32B
  (64 layers) on two GPUs; Llama-3.3-70B-Instruct (80 layers) on four GPUs.
- One host with eight NVIDIA A100-SXM4-80GB GPUs connected by NVLink. Each job ran in its own
  container with one model load, on the GPUs its config reserves.
  [Limits of this data set](#limits-of-this-data-set) lists what else ran on the host and on those
  GPUs.
- Every engine ran in bfloat16 with prefix caching off. vLLM engines used
  `gpu_memory_utilization` 0.85; the TransformerLens bridge used 0.70 on two GPUs and 0.55 with a
  4,096-token context cap on four.

### Workloads

| Table label | Workload | Row | Timed call |
|---|---|---|---|
| Logit lens | `logit_lens` | the residual stream of every layer at the last token, through the final norm and the unembedding | one CounterFact prompt |
| Steering | `steering` | add a vector toward the token " Rome" at half depth, strength 6 times the stream's mean norm | one CounterFact prompt |
| Ablation | `ablation` | zero the MLP output at half depth | one prompt: CounterFact in the single-request table and the 7B, 14B and 32B columns of the model-scale table; IOI in the offline table, the 70B column and the `transformer-lens-vllm/default` single-request cell |
| Patching | `activation_patching` | copy the clean run's last-token residual at quarter depth into the corrupted run | one IOI pair, clean and corrupted |
| Decode steering | `gen_steering` | the steering write at every decode step, scaled by each token's norm | one CounterFact prompt, 8 greedy decode steps |
| Decode capture | `gen_capture` | copy the half-depth residual to host memory at every decode step | one CounterFact prompt, 8 greedy decode steps |
| Harvest | `activation_harvest` | copy the half-depth residual at every prompt token to host memory | 16 WikiText paragraphs in one call |

Four of the workloads have more rows than the one above (other depths, strengths and positions;
[Workload support](#workload-support-on-qwen25-7b) gives the counts). The tables report one row per
workload, the one above.

### How a number is produced

- **Single request.** Three untimed warm-up calls, then seven timed calls, each between CUDA
  synchronizations. The tables give the median in milliseconds.
- **Offline sweep.** 256 prompts per pass, issued in calls of B prompts. One untimed pass, then
  three timed passes. Throughput is prompts per second of the median pass.
- **Closed loop.** C clients each send one single-prompt request and wait for it before sending
  the next: 128 requests per pass, one warm-up request per client, three timed passes.
- **Open loop.** 128 single-prompt requests per pass arrive as a seeded Poisson process at R
  requests per second, whatever the system is doing. Latency runs from a request's scheduled
  arrival to its return. One warm-up request, three timed passes.
- **Floor.** Plain vLLM of the same version, in its default mode (torch.compile and CUDA graphs),
  serving the same prompts with no intervention, measured in the same run. A ratio or percentage
  in parentheses divides a row by the floor of the row's own run. The `vllm-plain` rows printed in
  a table come from one run each, so a printed ratio can differ from the quotient of the two
  printed numbers, by up to 6%. The name comes from latency; in the throughput tables it denotes the
  same plain-vLLM measurement with no intervention.
- **Validation state.** Each row's output is compared with the output of nnsight on HF for the
  same prompts: `nnsight-hf/0-8-1`, its `gpu2` and `gpu4` configs in the two-GPU and four-GPU
  columns, and `nnsight-hf/sdpa` (nnsight 0.8.0rc1, the same options) for 14 single-request cells
  from earlier runs. Those 14 cells are the cells of `interp-engine-vllm/default` and
  `transformer-lens-vllm/default`, plus the patching, decode-capture and harvest cells of
  `vllm-lens/default`. A row passes at top-1 agreement of at least 0.90 and a mean total-variation
  distance of at most 0.05. A row outside those thresholds is `NUMERICAL_MISMATCH`: its correctness
  is unresolved and its timing stands. A row is left out of the tables only when it failed to
  execute or a contract violation was demonstrated.
- **Record selection.** Each cell comes from one job. Where several runs hold a cell, the newest
  supplies it, with the exceptions listed under
  [Limits of this data set](#limits-of-this-data-set).

Marks in the tables:

| Mark | Meaning |
|---|---|
| † | `NUMERICAL_MISMATCH` against the reference |
| ‡ | `INVALID_REFERENCE`: the reference run's effect check for this workload was too weak, so the comparison was not judged |
| n/s | the system's interface cannot express the row; the reason, from its source, is under [Workload support](#workload-support-on-qwen25-7b) |
| excl. | the row ran and is excluded for a demonstrated contract violation |
| n/m | not measured: the row was not run on this config, or the benchmark had no cell for it at the run's commit |

## Workload support on Qwen2.5-7B

How many rows of each workload ran on each vLLM system. `nnsight-hf/0-8-1` ran every row.

| Workload (rows) | nnsight | vLLM-Lens | vLLM-Hook | interp-engine | TransformerLens |
|---|---|---|---|---|---|
| Logit lens (1) | 1 | 1 | 1 | 1 | 1 |
| Steering (8) | 8 | 8 | 0 | 8 | 4 |
| Ablation (4) | 4 | 4, in `opt` only | 0 | 0 | 4 |
| Patching (4) | 4 | 4 | 0 | 2 | 2 |
| Decode steering (4) | 4 | 4 | 0 | 2 | 0 |
| Decode capture (1) | 1 | 1 | 1, measured under open loop only | 1 | 0 |
| Harvest (1) | 1 | 1 | not measured | 1 | 1 |

Reasons the cells record for the rows that did not run, each from the system's source at the
release measured:

- **vLLM-Hook 0.2.0.** `steer_hook_act` writes only the last row of each forward (the last prompt
  token and each decoded token), so a write at every prompt position cannot be expressed. That
  covers steering, decode steering and patching at every position. A last-token patch needs both
  runs read first, and an engine carries one worker extension, so the read worker and the write
  worker cannot share an engine. The built-in workers have no operation that zeroes a component's
  output (ablation). The benchmark's harvest cell for vLLM-Hook was added on 2026-10-28 (commit
  `1201e16`), after the last harvest run that included vLLM-Hook, and no harvest run with
  vLLM-Hook followed. Its decode-capture cell was added on 2026-10-30 (commit `9b8cefb`) and ran in
  the open-loop runs only.
- **interp-engine 1.12.0.** Every write operation acts along one direction, so zeroing a whole
  component output would take one removal per dimension of the stream; the cell declares the
  ablation rows unsupported. A write carries one vector and a request has one position mask, so a
  different value at every position cannot be expressed (patching at every position; the two
  last-token rows ran). The norm-relative write that runs during generation scales by each token's
  own norm, so the two decode-steering rows whose strength follows the mean norm over a forward's
  tokens cannot be expressed.
- **TransformerLens 4.0.0 bridge.** An intervention carries one value per hook, so a strength that
  follows each token's norm (four steering rows) and a different value at every position (two
  patching rows) cannot be expressed. The vLLM driver runs one forward per call and the bridge has
  no `generate()`, so the decode workloads cannot be expressed.
- **vLLM-Lens 1.2.1.** In `vllm-lens/default` hooks fire at decoder-layer boundaries on the summed
  residual stream, so a component output inside a layer is not addressable. `vllm-lens/opt` builds
  ablation from those hooks: a pre-hook on the next layer for the MLP rows, and hooks on two
  layers for the attention rows.

The benchmark's other nine workloads are not on this page. `attribution_patching`, `das` and
`jacobian_collect` have cells for nnsight only. For `jacobian_lens`, `attention_pattern`,
`gen_patching` and `residual_readout`, cells for other systems were added during the evaluation
(commits `1201e16` and `beace01`, 2026-10-28), and no run of them across the systems followed.
`steering_layers` and `activation_harvest_layers` sweep the number of layers and are not reported
here.

## Single request: Qwen2.5-7B on one GPU

Median latency in ms; in parentheses, the ratio to the floor: plain vLLM of the same version in
its default mode.

| Config | Logit lens | Steering | Ablation | Patching | Decode steering | Decode capture | Harvest |
|---|---:|---:|---:|---:|---:|---:|---:|
| `nnsight-hf/0-8-1` | 47.2 | 18.5 | 18.1 | 37.2 | 129.2 | 129.3 | 98.3 |
| `nnsight-vllm/0-8-1` | 70.6 (4.64×) | 27.3 (1.79×) | 26.4‡ (1.75×) | 51.0 (2.05×) | 118.4 (1.35×) | 107.1† (1.23×) | 216.1 (2.51×) |
| `nnsight-vllm/0-8-1/opt` | 47.8 (3.11×) | 19.8 (1.31×) | 19.1‡ (1.26×) | 44.8 (1.81×) | 108.5 (1.24×) | n/m | n/m |
| `nnsight-vllm/0-8-1/opt/taps` | 50.3 (3.27×) | 19.0 (1.25×) | 18.3‡ (1.21×) | 43.4 (1.75×) | 97.1 (1.11×) | n/m | n/m |
| `vllm-lens/default` | 82.8 (6.07×) | 25.8 (1.85×) | n/s | 40.4 (1.65×) | 162.4 (1.90×) | 90.3† (1.05×) | 102.8 (1.22×) |
| `vllm-lens/opt` | 82.7 (6.06×) | 24.8 (1.75×) | 26.9‡ (1.89×) | 42.5 (1.76×) | 167.4 (1.96×) | n/m | n/m |
| `vllm-hook/default` | 29.9 (2.15×) | n/s | n/s | n/s | n/s | n/m | n/m |
| `vllm-hook/disk` | 27.9 (2.05×) | n/s | n/s | n/s | n/s | n/m | n/m |
| `interp-engine-vllm/default` | 43.1 (2.99×) | 36.8 (2.50×) | n/s | 50.6 (1.99×) | 95.5 (1.11×) | 88.1† (1.03×) | 101.2 (1.16×) |
| `interp-engine-vllm/static` | 39.9 (2.66×) | 36.7 (2.45×) | n/s | 50.3 (1.98×) | excl. | 90.3† (1.04×) | 102.7 (1.18×) |
| `transformer-lens-vllm/default` | 34.4 (2.53×) | 62.6 (4.54×) | 30.3‡ (2.15×) | 46.5 (1.91×) | n/s | n/s | 285.0 (3.31×) |
| `transformer-lens-vllm/batched` | 42.7 (2.95×) | 77.7 (5.40×) | 51.5‡ (3.56×) | n/m | n/s | n/m | n/m |
| `transformer-lens-vllm/opt` | 22.0 (1.52×) | 32.3 (2.25×) | 16.3‡ (1.13×) | 33.4 (1.34×) | n/s | n/m | n/m |
| `vllm-plain/0-19-1` | 13.9 | 14.1 | 14.2 | 24.1 | 85.5 | 84.4 | 85.9 |
| `vllm-plain/0-20-2` | 14.5 | 14.4 | 14.5 | 24.8 | 87.4 | 85.3 | 85.9 |
| `vllm-plain/0-27-1` | 15.2 | 15.2 | 15.1 | 24.8 | 87.7 | 86.9 | 86.2 |
| `vllm-plain/0-28-0` | 15.0 | 15.0 | 15.0 | 25.4 | 86.3 | 86.7 | 87.2 |

- Every ablation row is marked ‡. The workload's effect check zeroes the attention output at half
  depth on the reference and needs a total-variation distance above 0.2 from the unablated output,
  or top-1 agreement below 0.5. It measured 0.09 and 0.94 on the CounterFact prompts (0.18 and
  0.81 on the IOI prompts), so the comparison was not judged.
- Every decode-capture row is marked †, with top-1 agreement between 0.89 and 0.90 against the
  reference on every vLLM system. For a captured residual vector, top-1 is the index of its
  largest coordinate. Cause not determined. One known difference: HF generation applies the
  checkpoint's `repetition_penalty` of 1.05, and the vLLM requests use 1.0.
- `interp-engine-vllm/static` decode steering is excluded: its output agrees with the reference
  at the prefill step and diverges at decode steps 1 to 7, because the static mode of release
  1.12.0 drops the writes of the decode steps
  ([findings](findings.md#interp-engine-vllm-static-drops-decode-step-lens-writes-silently_wrong)).
  Upstream fixed this in 1.13.1 (commit `2f31f76`); that release is not measured here.
- The harvest column is one call over 16 paragraphs, so its floor is plain vLLM serving the same
  16 prompts.
- The patching and harvest cells of `transformer-lens-vllm/default` come from the runs made for
  workload support, which ran at the same time as other runs of the evaluation. The timing jobs for
  those two cells are left out: they ran while another user's training processes held the other
  seven GPUs, and the nnsight-on-HF patching rows timed in the same period read 30% to 43% above the
  same rows in another run. The left-out jobs read 47.4 ms (1.92×) and 289.1 ms (3.37×).

## Offline sweep: Qwen2.5-7B on one GPU

Throughput in prompts per second; in parentheses, the percentage of the floor's throughput at the
same B. B=256 hands the system all 256 prompts in one call. The sweep ran nnsight's three vLLM
configs and, for each other system, the configs in the tables. The default configs of vLLM-Lens,
vLLM-Hook and the TransformerLens bridge were not run in this regime, and
`interp-engine-vllm/default` ran for decode steering only, where `static` is excluded.

**Steering**

| Config | B=1 | B=16 | B=64 | B=256 |
|---|---:|---:|---:|---:|
| `nnsight-hf/0-8-1` | 58.3 | 524.2 | 608.2 | 622.5 |
| `nnsight-vllm/0-8-1` | 42.3 (53%) | 100.6 (17%) | 105.4 (10%) | 98.2 (7%) |
| `nnsight-vllm/0-8-1/opt` | 60.2 (76%) | 332.9 (57%) | 502.2 (46%) | 469.9 (32%) |
| `nnsight-vllm/0-8-1/opt/taps` | 64.3 (81%) | 361.9 (63%) | 487.9 (45%) | 458.2 (31%) |
| `vllm-lens/opt` | 45.0 (53%) | 145.0 (24%) | 183.8 (16%) | 176.7 (11%) |
| `vllm-hook/disk` | n/s | n/s | n/s | n/s |
| `interp-engine-vllm/static` | 29.7 (38%) | 169.8 (30%) | 234.7 (22%) | 241.6 (16%) |
| `transformer-lens-vllm/batched` | 13.6 (16%) | 17.5 (3%) | 17.8 (2%) | 17.6 (1%) |
| `transformer-lens-vllm/opt` | 30.4 (36%) | 31.1 (5%) | 31.2 (3%) | 31.1 (2%) |
| `vllm-plain/0-19-1` | 85.4 | 598.9 | 1,128 | 1,619 |
| `vllm-plain/0-20-2` | 84.4 | 598.5 | 1,125 | 1,608 |
| `vllm-plain/0-27-1` | 79.3 | 570.6 | 1,073 | 1,498 |
| `vllm-plain/0-28-0` | 78.9 | 573.3 | 1,062 | 1,510 |

**Logit lens**

| Config | B=1 | B=16 | B=64 | B=256 |
|---|---:|---:|---:|---:|
| `nnsight-hf/0-8-1` | 27.8 | 58.9 | 64.4 | 66.2 |
| `nnsight-vllm/0-8-1` | 15.5 (20%) | 17.1 (3%) | 18.1 (2%) | 15.3 (1%) |
| `nnsight-vllm/0-8-1/opt` | 24.9 (31%) | 29.2 (5%) | 27.9 (3%) | 31.2 (2%) |
| `nnsight-vllm/0-8-1/opt/taps` | 23.7 (30%) | 28.4 (5%) | 27.5 (3%) | 22.2 (2%) |
| `vllm-lens/opt` | 12.1 (14%) | 25.5 (4%) | 28.2 (3%) | 26.5 (2%) |
| `vllm-hook/disk` | 40.2 (47%) | 50.3 (8%) | 56.5 (5%) | 57.3 (4%) |
| `interp-engine-vllm/static` | 27.1 (34%) | 27.3 (5%) | 26.1 (2%) | 26.4 (2%) |
| `transformer-lens-vllm/batched` | 21.3 (25%) | 28.7 (5%) | 56.6 (5%) | 72.0 (4%) |
| `transformer-lens-vllm/opt` | 42.6 (51%) | 33.1 (5%) | 32.6 (3%) | 33.3 (2%) |
| `vllm-plain/0-19-1` | 84.9 | 604.3 | 1,124 | 1,612 |
| `vllm-plain/0-20-2` | 83.9 | 603.5 | 1,119 | 1,610 |
| `vllm-plain/0-27-1` | 79.7 | 571.8 | 1,088 | 1,469 |
| `vllm-plain/0-28-0` | 78.7 | 573.7 | 1,082 | 1,481 |

**Ablation**

| Config | B=1 | B=16 | B=64 | B=256 |
|---|---:|---:|---:|---:|
| `nnsight-hf/0-8-1` | 58.2 | 394.6 | 486.4 | 514.3 |
| `nnsight-vllm/0-8-1` | 43.5 (55%) | 103.4 (21%) | 107.7 (14%) | 106.0 (11%) |
| `nnsight-vllm/0-8-1/opt` | 61.5 (78%) | 329.5 (65%) | 429.2 (56%) | 385.1 (39%) |
| `nnsight-vllm/0-8-1/opt/taps` | 64.5 (82%) | 341.0 (68%) | 390.6 (52%) | 388.7 (40%) |
| `vllm-lens/opt` | 42.2 (50%) | 159.8 (31%) | 201.5 (26%) | 211.5† (21%) |
| `vllm-hook/disk` | n/s | n/s | n/s | n/s |
| `interp-engine-vllm/static` | n/s | n/s | n/s | n/s |
| `transformer-lens-vllm/batched` | 19.4 (23%) | 31.4 (6%) | 72.1 (9%) | 100.2† (10%) |
| `transformer-lens-vllm/opt` | 58.0† (70%) | 63.7† (12%) | 61.7† (8%) | 61.3† (6%) |
| `vllm-plain/0-19-1` | 84.4 | 521.9 | 782.9 | 1,005 |
| `vllm-plain/0-20-2` | 83.2 | 518.2 | 774.3 | 996.9 |
| `vllm-plain/0-27-1` | 78.7 | 500.9 | 755.2 | 972.5 |
| `vllm-plain/0-28-0` | 78.2 | 499.5 | 733.3 | 942.5 |

**Decode steering** (8 decode steps per prompt)

| Config | B=1 | B=16 | B=64 | B=256 |
|---|---:|---:|---:|---:|
| `nnsight-hf/0-8-1` | 7.8 | 90.6 | 155.1 | 188.3 |
| `nnsight-vllm/0-8-1` | 8.6 (72%) | 37.6 (24%) | 46.2 (10%) | 46.3 (6%) |
| `nnsight-vllm/0-8-1/opt` | 9.6 (81%) | 54.0 (34%) | 71.3 (16%) | 77.0 (9%) |
| `nnsight-vllm/0-8-1/opt/taps` | 10.5 (88%) | 54.1 (34%) | 67.6 (15%) | 67.2 (8%) |
| `vllm-lens/opt` | 6.3 (52%) | 26.4 (16%) | 28.6 (6%) | 21.3 (3%) |
| `interp-engine-vllm/default` | 10.7 (90%) | 60.2 (38%) | 82.4 (19%) | 89.2 (11%) |
| `interp-engine-vllm/static` | excl. | excl. | excl. | excl. |
| `transformer-lens-vllm/batched` | n/s | n/s | n/s | n/s |
| `transformer-lens-vllm/opt` | n/s | n/s | n/s | n/s |
| `vllm-plain/0-19-1` | 12.0 | 160.5 | 451.6 | 814.5 |
| `vllm-plain/0-20-2` | 12.0 | 159.6 | 444.0 | 815.4 |
| `vllm-plain/0-27-1` | 11.9 | 157.8 | 444.6 | 798.3 |
| `vllm-plain/0-28-0` | 11.9 | 157.7 | 439.5 | 803.0 |

The `nnsight-hf/0-8-1` rows come from the runs that measured nnsight, vLLM-Lens and vLLM-Hook. HF
ran again, as the reference, in the later runs that measured interp-engine and TransformerLens, and
those HF timings are not shown. They differ from the rows above. In the newest of them the
logit-lens row reads 75.9, 87.7 and 94.0 prompts per second at B=16, 64 and 256, which is 29% to 42%
above the row shown and within 4% of the two runs made before the one shown; the steering row reads
9% and 14% higher at B=64 and 256; the decode-steering row reads 16%, 9% and 5% lower at B=16, 64
and 256. Cause not determined.

## Online serving: Qwen2.5-7B on one GPU

### Open loop

Median / 99th-percentile request latency in ms at R requests per second. nnsight, vLLM-Lens and
vLLM-Hook ran in their `async` configs (vLLM's `AsyncLLM`); interp-engine ran `static`, and
`default` for decode steering; the TransformerLens bridge ran `opt`. The last column is the
throughput at R=64: 128 requests divided by the duration of the median pass. The 128 arrivals span
2.3 s and a pass ends when the last request returns, so the column cannot exceed 55.7. Plain vLLM
reads 55.4 on the single-pass workloads and 53.4 to 53.5 on the decode workloads, whose requests
take about 0.1 s. Values far below those mean requests queued.

**Steering**

| Config | R=4 | R=16 | R=64 | Throughput at R=64 (requests/s) |
|---|---:|---:|---:|---:|
| `nnsight-hf/0-8-1` | 19.7 / 38.9 | 19.2 / 41.2 | 69.7 / 160.4 | 53.0 |
| `nnsight-vllm/0-8-1/async` | 30.3 / 52.2 | 29.2 / 56.2 | 59.8 / 114.2 | 55.0 |
| `vllm-lens/opt/async` | 26.7 / 59.7 | 27.4 / 64.1 | 100.0 / 151.4 | 54.8 |
| `vllm-hook/disk/async` | n/s | n/s | n/s | n/s |
| `interp-engine-vllm/static` | 42.2 / 86.3 | 49.9 / 90.2 | 117.1 / 153.9 | 54.0 |
| `transformer-lens-vllm/opt` | 38.7 / 101.6 | 38.2 / 102.8 | 1,314 / 2,851 | 26.4 |
| `vllm-plain/0-19-1` | 13.2 / 22.3 | 12.6 / 22.1 | 12.8 / 62.0 | 55.4 |
| `vllm-plain/0-20-2` | 14.1 / 22.2 | 12.9 / 22.4 | 13.3 / 66.7 | 55.4 |
| `vllm-plain/0-27-1` | 14.1 / 20.6 | 13.1 / 22.8 | 15.0 / 27.4 | 55.4 |
| `vllm-plain/0-28-0` | 14.5 / 21.3 | 13.4 / 23.4 | 15.0 / 27.2 | 55.4 |

**Logit lens**

| Config | R=4 | R=16 | R=64 | Throughput at R=64 (requests/s) |
|---|---:|---:|---:|---:|
| `nnsight-hf/0-8-1` | 39.0 / 88.5 | 48.7 / 120.9 | 1,552 / 3,492 | 25.7 |
| `nnsight-vllm/0-8-1/async` | 68.0 / 137.1 | 101.4 / 208.3 | 1,423 / 1,820 | 34.9 |
| `vllm-lens/opt/async` | 101.1 / 231.3 | 97.6 / 407.1 | 290.7 / 2,071 | 51.6 |
| `vllm-hook/disk/async` | 31.0 / 59.0 | 31.0 / 65.4 | 81.1 / 137.3 | 54.8 |
| `interp-engine-vllm/static` | 36.7 / 60.0 | 40.1 / 67.6 | 86.0 / 165.0 | 54.1 |
| `transformer-lens-vllm/opt` | 28.0 / 127.9 | 27.9 / 85.3 | 819.5 / 1,777 | 34.9 |
| `vllm-plain/0-19-1` | 13.3 / 21.9 | 12.7 / 22.2 | 13.1 / 67.3 | 55.4 |
| `vllm-plain/0-20-2` | 14.2 / 22.3 | 12.9 / 22.5 | 13.2 / 63.4 | 55.4 |
| `vllm-plain/0-27-1` | 13.8 / 20.9 | 13.2 / 22.3 | 15.0 / 27.4 | 55.4 |
| `vllm-plain/0-28-0` | 14.6 / 22.6 | 13.4 / 22.8 | 15.0 / 27.4 | 55.4 |

**Decode capture**

| Config | R=4 | R=16 | R=64 | Throughput at R=64 (requests/s) |
|---|---:|---:|---:|---:|
| `nnsight-hf/0-8-1` | 128.5 / 318.5 | 3,637 / 7,135 | 7,124 / 13,946 | 7.8 |
| `nnsight-vllm/0-8-1/async` | 108.7 / 142.8† | 136.5 / 157.4† | 165.6 / 196.1† | 52.9† |
| `vllm-lens/opt/async` | 92.0 / 129.9† | 121.7 / 143.8† | 133.3 / 149.5† | 53.3† |
| `vllm-hook/disk/async` | 90.7 / 118.5† | 114.4 / 144.3† | 127.4 / 136.9† | 53.4† |
| `interp-engine-vllm/static` | 90.2 / 132.4† | 120.5 / 140.2† | 134.0 / 147.6† | 53.2† |
| `transformer-lens-vllm/opt` | n/s | n/s | n/s | n/s |
| `vllm-plain/0-19-1` | 84.6 / 102.2 | 94.7 / 102.0 | 98.8 / 177.8 | 53.5 |
| `vllm-plain/0-20-2` | 84.6 / 102.5 | 94.9 / 102.1 | 98.7 / 180.5 | 53.5 |
| `vllm-plain/0-27-1` | 85.2 / 102.9 | 95.5 / 102.8 | 99.6 / 105.2 | 53.4 |
| `vllm-plain/0-28-0` | 85.0 / 102.3 | 94.8 / 102.7 | 98.2 / 105.0 | 53.4 |

**Decode steering**

| Config | R=4 | R=16 | R=64 | Throughput at R=64 (requests/s) |
|---|---:|---:|---:|---:|
| `nnsight-hf/0-8-1` | 142.7 / 393.3 | 4,800 / 9,194 | 8,051 / 15,891 | 7.0 |
| `nnsight-vllm/0-8-1/async` | 128.8 / 175.0 | 174.4 / 215.3 | 316.1 / 469.1 | 50.4 |
| `vllm-lens/opt/async` | 224.7 / 344.9 | 408.3 / 522.0 | 2,086 / 2,542 | 30.8 |
| `vllm-hook/disk/async` | n/s | n/s | n/s | n/s |
| `interp-engine-vllm/default` | 99.8 / 180.2 | 166.9 / 197.7 | 210.7 / 248.3 | 51.9 |
| `interp-engine-vllm/static` | excl. | excl. | excl. | excl. |
| `transformer-lens-vllm/opt` | n/s | n/s | n/s | n/s |
| `vllm-plain/0-19-1` | 84.5 / 102.2 | 94.7 / 102.1 | 98.5 / 178.5 | 53.5 |
| `vllm-plain/0-20-2` | 85.0 / 102.1 | 95.0 / 102.2 | 98.4 / 178.1 | 53.5 |
| `vllm-plain/0-27-1` | 85.4 / 102.1 | 95.4 / 102.7 | 99.7 / 105.1 | 53.4 |
| `vllm-plain/0-28-0` | 84.9 / 102.3 | 94.7 / 102.6 | 98.3 / 104.6 | 53.4 |

### Closed loop

Throughput in requests per second with C clients. Measured for nnsight only.

| Workload | Config | C=1 | C=4 | C=16 | C=64 |
|---|---|---:|---:|---:|---:|
| Steering | `nnsight-hf/0-8-1` | 57.3 | 57.2 | 57.0 | 56.1 |
| Steering | `nnsight-vllm/0-8-1` | 39.7 | 41.3 | 40.6 | 38.9 |
| Steering | `nnsight-vllm/0-8-1/async` | 39.8 | 74.1 | 116.0 | 114.7 |
| Steering | `vllm-plain/0-27-1` | 79.4 | 192.7 | 577.6 | 1,053 |
| Logit lens | `nnsight-hf/0-8-1` | 26.8 | 25.0 | 25.2 | 23.6 |
| Logit lens | `nnsight-vllm/0-8-1` | 16.0 | 16.4 | 15.7 | 13.8 |
| Logit lens | `nnsight-vllm/0-8-1/async` | 15.9 | 24.6 | 33.2 | 34.6 |
| Logit lens | `vllm-plain/0-27-1` | 79.7 | 190.5 | 456.4 | 997.8 |
| Decode capture | `nnsight-hf/0-8-1` | 7.8 | 7.8 | 7.8 | 7.8 |
| Decode capture | `nnsight-vllm/0-8-1` | 9.5† | 9.6† | 9.6† | 9.6† |
| Decode capture | `nnsight-vllm/0-8-1/async` | 9.4† | 28.2† | 75.4† | 109.9† |
| Decode capture | `vllm-plain/0-27-1` | 11.9 | 43.7 | 157.6 | 425.9 |
| Decode steering | `nnsight-hf/0-8-1` | 7.8 | 7.8 | 7.6 | 7.4 |
| Decode steering | `nnsight-vllm/0-8-1` | 8.8 | 8.7 | 8.7 | 8.5 |
| Decode steering | `nnsight-vllm/0-8-1/async` | 8.6 | 22.3 | 48.1 | 61.4 |
| Decode steering | `vllm-plain/0-27-1` | 11.9 | 43.7 | 146.8 | 408.6 |

`nnsight-hf/0-8-1` and the synchronous `nnsight-vllm/0-8-1` serve one request at a time, so their
throughput does not rise with C. `nnsight-vllm/0-8-1/async` hands every request to the engine's
scheduler.

## Model scale: single request on one, two and four GPUs

Median latency in ms; in parentheses, the ratio to the floor on the same GPUs. The 7B column
repeats the single-request table. A row is one config family, run in the config of each column's
GPU count. The vLLM configs use tensor parallelism (for example `nnsight-vllm/0-8-1/tp2` and
`/tp4`, `vllm-lens/opt/tp2`, `interp-engine-vllm/tp2` for the default config, and
`vllm-plain/0-27-1/tp2`). `nnsight-hf/0-8-1/gpu2` and `/gpu4` place the layers across the
GPUs in one process (`device_map=auto`); `nnsight-hf/0-8-1/tp2` and `/tp4` use transformers tensor
parallelism.

**Steering**

| Config | Qwen2.5-7B, 1 GPU | Qwen2.5-14B, 2 GPUs | Qwen3-32B, 2 GPUs | Llama-3.3-70B, 4 GPUs |
|---|---:|---:|---:|---:|
| `nnsight-hf/0-8-1` (`/gpu2`, `/gpu4`) | 18.5 | 36.6 | 60.9 | 110.6 |
| `nnsight-hf/0-8-1/tp2`, `/tp4` | n/m | 45.9 | 74.5 | 193.7 |
| `nnsight-vllm/0-8-1` | 27.3 (1.79×) | 41.8 (2.37×) | 56.6 (1.94×) | 60.3 (1.67×) |
| `nnsight-vllm/0-8-1/opt` | 19.8 (1.31×) | 29.1 (1.65×) | 38.8 (1.33×) | 43.5 (1.21×) |
| `nnsight-vllm/0-8-1/opt/taps` | 19.0 (1.25×) | 22.9 (1.29×) | 34.2 (1.17×) | 42.0 (1.16×) |
| `vllm-lens/opt` | 24.8 (1.75×) | 34.7 (2.10×) | 50.5 (1.80×) | 56.7 (1.66×) |
| `vllm-hook/disk` | n/s | n/s | n/s | n/s |
| `interp-engine-vllm/default` | 36.8 (2.50×) | 56.0 (3.08×) | 73.9 (2.41×) | 87.8 (2.45×) |
| `interp-engine-vllm/static` | 36.7 (2.45×) | 46.8 (2.57×) | 72.3 (2.35×) | 86.7 (2.42×) |
| `transformer-lens-vllm/opt` | 32.3 (2.25×) | 54.1 (3.07×) | 80.9 (2.78×) | 104.1 (2.94×) |
| `vllm-plain/0-19-1` | 14.1 | 16.5 | 28.1 | 34.1 |
| `vllm-plain/0-20-2` | 14.4 | 17.6 | 29.1 | 35.4 |
| `vllm-plain/0-27-1` | 15.2 | 17.7 | 29.3 | 36.1 |
| `vllm-plain/0-28-0` | 15.0 | 18.2 | 30.7 | 35.8 |

**Logit lens**

| Config | Qwen2.5-7B, 1 GPU | Qwen2.5-14B, 2 GPUs | Qwen3-32B, 2 GPUs | Llama-3.3-70B, 4 GPUs |
|---|---:|---:|---:|---:|
| `nnsight-hf/0-8-1` (`/gpu2`, `/gpu4`) | 47.2 | 92.6 | 144.2 | 236.4 |
| `nnsight-hf/0-8-1/tp2`, `/tp4` | n/m | 92.0 | 145.8 | 164.3 |
| `nnsight-vllm/0-8-1` | 70.6 (4.64×) | 85.5 (4.69×) | 169.2 (5.84×) | 203.6 (5.77×) |
| `nnsight-vllm/0-8-1/opt` | 47.8 (3.11×) | 93.4 (5.13×) | 176.6 (6.10×) | 367.2 (10.40×) |
| `nnsight-vllm/0-8-1/opt/taps` | 50.3 (3.27×) | 96.0 (5.27×) | 161.3 (5.57×) | 351.5 (9.95×) |
| `vllm-lens/opt` | 82.7 (6.06×) | 147.5 (9.11×) | 351.6 (12.61×) | 324.3 (9.58×) |
| `vllm-hook/disk` | 27.9 (2.05×) | 42.7 (2.64×) | 80.2 (2.87×) | 109.1 (3.23×) |
| `interp-engine-vllm/default` | 43.1 (2.99×) | 142.1 (7.60×) | 222.6 (7.45×) | 379.7 (10.55×) |
| `interp-engine-vllm/static` | 39.9 (2.66×) | 119.2 (6.37×) | 200.6 (6.71×) | 334.7 (9.45×) |
| `transformer-lens-vllm/opt` | 22.0 (1.52×) | 77.2 (4.64×) | 138.1 (4.71×) | 361.8 (10.36×) |
| `vllm-plain/0-19-1` | 13.9 | 16.2 | 27.9 | 33.8 |
| `vllm-plain/0-20-2` | 14.5 | 16.7 | 29.3 | 34.9 |
| `vllm-plain/0-27-1` | 15.2 | 18.2 | 29.0 | 35.3 |
| `vllm-plain/0-28-0` | 15.0 | 18.7 | 29.9 | 35.4 |

**Ablation**

| Config | Qwen2.5-7B, 1 GPU | Qwen2.5-14B, 2 GPUs | Qwen3-32B, 2 GPUs | Llama-3.3-70B, 4 GPUs |
|---|---:|---:|---:|---:|
| `nnsight-hf/0-8-1` (`/gpu2`, `/gpu4`) | 18.1 | 35.1 | 60.3 | 128.5 |
| `nnsight-hf/0-8-1/tp2`, `/tp4` | n/m | 47.6‡ | 81.1‡ | 74.4‡ |
| `nnsight-vllm/0-8-1` | 26.4‡ (1.75×) | 38.4‡ (2.11×) | 55.9‡ (1.92×) | 63.0‡ (1.62×) |
| `nnsight-vllm/0-8-1/opt` | 19.1‡ (1.26×) | 29.7‡ (1.64×) | 36.7‡ (1.26×) | 57.4‡ (1.48×) |
| `nnsight-vllm/0-8-1/opt/taps` | 18.3‡ (1.21×) | 21.5‡ (1.18×) | 34.2‡ (1.17×) | 43.9‡ (1.13×) |
| `vllm-lens/opt` | 26.9‡ (1.89×) | 36.0‡ (2.21×) | 52.3‡ (1.87×) | 65.6‡ (1.72×) |
| `vllm-hook/disk` | n/s | n/s | n/s | n/s |
| `interp-engine-vllm/default` | n/s | n/s | n/s | n/s |
| `interp-engine-vllm/static` | n/s | n/s | n/s | n/s |
| `transformer-lens-vllm/opt` | 16.3‡ (1.13×) | 26.9‡ (1.53×) | 39.8‡ (1.36×) | 60.2‡ (1.50×) |
| `vllm-plain/0-19-1` | 14.2 | 16.3 | 27.9 | 38.1 |
| `vllm-plain/0-20-2` | 14.5 | 17.5 | 29.2 | 40.0 |
| `vllm-plain/0-27-1` | 15.1 | 18.2 | 29.2 | 38.8 |
| `vllm-plain/0-28-0` | 15.0 | 18.0 | 29.6 | 39.9 |

**Decode steering**

| Config | Qwen2.5-7B, 1 GPU | Qwen2.5-14B, 2 GPUs | Qwen3-32B, 2 GPUs | Llama-3.3-70B, 4 GPUs |
|---|---:|---:|---:|---:|
| `nnsight-hf/0-8-1` (`/gpu2`, `/gpu4`) | 129.2 | 266.0 | 454.4 | 842.7 |
| `nnsight-hf/0-8-1/tp2`, `/tp4` | n/m | 359.2 | 546.0 | 560.3 |
| `nnsight-vllm/0-8-1` | 118.4 (1.35×) | 201.7 (1.98×) | 287.4 (1.49×) | 305.9 (1.33×) |
| `nnsight-vllm/0-8-1/opt` | 108.5 (1.24×) | 189.6 (1.86×) | 265.5 (1.38×) | 292.0 (1.27×) |
| `nnsight-vllm/0-8-1/opt/taps` | 97.1 (1.11×) | 121.7 (1.19×) | 211.2 (1.10×) | 251.1 (1.09×) |
| `vllm-lens/opt` | 167.4 (1.96×) | 238.8 (2.42×) | 374.2 (1.97×) | 425.6 (1.87×) |
| `vllm-hook/disk` | n/s | n/s | n/s | n/s |
| `interp-engine-vllm/default` | 95.5 (1.11×) | 178.4 (1.75×) | 247.7 (1.28×) | 266.5 (1.16×) |
| `interp-engine-vllm/static` | excl. | excl. | excl. | excl. |
| `transformer-lens-vllm/opt` | n/s | n/s | n/s | n/s |
| `vllm-plain/0-19-1` | 85.5 | 98.7 | 189.9 | 227.0 |
| `vllm-plain/0-20-2` | 87.4 | 100.2 | 191.1 | 229.3 |
| `vllm-plain/0-27-1` | 87.7 | 101.9 | 192.7 | 230.1 |
| `vllm-plain/0-28-0` | 86.3 | 101.9 | 193.6 | 230.7 |

On Llama-3.3-70B, each `nnsight-hf/0-8-1/tp4` job measured in one of two latency ranges for one
prompt's forward: about 68 to 77 ms or 180 to 220 ms. Three runs of the steering cell read 74.3,
180.7 and 193.7 ms; the table shows 193.7 for steering and 74.4 for ablation. Cause not determined.

## Limits of this data set

- **Releases.** These are the releases the benchmark pinned in October 2026. As of 2026-11-11
  newer releases exist for every system (nnsight 0.8.2, vLLM-Lens 1.3.0, vLLM-Hook 1.0.1,
  interp-engine 1.14.3, TransformerLens 4.3.0) and are not measured here. Two of them change rows
  on this page: interp-engine fixed the static decode-step writes in 1.13.1, and vLLM-Hook 1.0
  steers every token by default.
- **Different vLLM versions.** Each system runs on the vLLM version its release supports, so
  absolute numbers of different systems come from different engines. The ratios and percentages
  compare each system with the floor of its own version. The floors of the four versions differ
  from each other by up to 10% in the single-request and offline tables, by up to 15% in the
  two-GPU columns and by up to 17% in the open-loop medians. At R=64 the 99th percentile of
  the 0.19.1 and 0.20.2 floors is 62 to 67 ms on the single-pass workloads against 27 ms for
  0.27.1 and 0.28.0, and 178 to 181 ms on the decode workloads against 105 ms.
- **Shared host.** Other users' jobs ran on other GPUs of the host during part of the runs, and from
  2026-10-27 this project kept its own placeholder processes on the GPUs a job did not use. Jobs
  that the evaluation's per-run audit notes record as sharing a GPU with another user's process were
  rerun or left out, and those jobs of one single-request run that ran while another user's training
  processes held the other seven GPUs were left out (see the notes under the single-request table).
  From 2026-10-29 14:46, jobs started on GPUs where one of those placeholder processes stayed,
  holding 4.5 GiB and making no CUDA call. 72 table cells come from such jobs: the interp-engine and
  TransformerLens rows of the two-GPU and four-GPU columns with the 0.20.2 and 0.28.0 floors, the
  `interp-engine-vllm/default` decode-steering rows of the offline and open-loop tables with their
  floor, and the vLLM-Hook open-loop decode-capture row with its floor. GPU processes on the whole
  host were logged from 2026-10-26 16:24, with a gap on 2026-10-28 from 03:57 to 16:02 that holds
  two of the model-scale runs. For a run outside the log, the record is the memory in use on each
  job's first GPU before the job started.
- **Decoding rule.** On Qwen2.5, HF generation applies the checkpoint's `repetition_penalty` of
  1.05 and the vLLM requests use 1.0, so generation rows compare different decoding rules.
- **Repeats and record selection.** Each cell comes from one job: three warm-up calls and seven
  timed calls, or three timed passes. Where several runs hold a cell, the newest supplies it,
  with four exceptions. The single-request table reads three groups of run directories and takes a
  cell from the first group that holds it: `stage1c-opt7b-*`, then `stage1c-cross-system-*`, then
  `stage1a-capability-*`, the runs made for workload support. The single-request logit-lens and
  steering cells of `nnsight-hf/0-8-1`, `nnsight-vllm/0-8-1/opt`, `0-8-1/opt/taps`,
  `vllm-lens/default`, `vllm-hook/default` and the 0.19.1 and 0.27.1 floors have three runs at one
  commit, and the tables use the run with the median latency. `nnsight-hf` cells come from the runs
  that measured nnsight, vLLM-Lens and vLLM-Hook, as noted under the offline tables. Jobs left out
  under Shared host are skipped. Across the three repeat runs the logit-lens latency reads 44.0,
  50.3 and 53.8 ms on `nnsight-vllm/0-8-1/opt/taps` and 28.7, 29.9 and 38.4 ms on
  `vllm-hook/default`; cause not determined.
- **Not on this page.** Per-phase latency breakdowns, memory, the fp32 controls, the gradient
  workloads, the benchmark's other models (GPT-2, SmolLM2-135M, Qwen2.5-72B, Qwen3-8B,
  Qwen3.5-4B, Nemotron) and the older nnsight releases.

## Source runs

Run directories under the evaluation's run tree, with the commit each ran from. All commits are in
this repository's history; the tag `paper-eval-v1` marks the last of them.

| Run | Commit | Tables |
|---|---|---|
| `stage1a-capability-waveB/20261027T200258Z-b29b700a` | `c3cc4dd` | single request |
| `stage1a-capability-waveB/20261027T200258Z-c087e25b` | `c3cc4dd` | single request |
| `stage1c-cross-system-waveA/20261026T151332Z-5f911c21` | `c3cc4dd` | single request |
| `stage1c-cross-system-waveB/20261027T153310Z-a1b0ffd1` | `c3cc4dd` | single request |
| `stage1c-cross-system-waveB/20261028T002449Z-26b95080` | `c3cc4dd` | single request, model scale |
| `stage1c-opt7b-repeats/20261028T194153Z-88981c5c` | `beace01` | single request, model scale |
| `stage1c-opt7b-repeats/20261028T201014Z-ef95ca72` | `beace01` | single request, model scale |
| `stage1c-opt7b-waveA-ext/20261027T214757Z-b212fab4` | `c3cc4dd` | single request |
| `stage1c-opt7b-waveA-ext/20261027T215627Z-e5b9d913` | `c3cc4dd` | single request |
| `stage1c-opt7b-waveA-ext/20261028T031502Z-4075f2cf` | `c3cc4dd` | single request |
| `stage1c-opt7b-waveA-ext/20261028T031937Z-2eddf9ed` | `c3cc4dd` | single request |
| `stage1c-opt7b-waveA/20261026T015456Z-17aa2baa` | `d0ed4b9` | single request |
| `stage1c-opt7b-waveA/20261028T175404Z-a5c7496f` | `beace01` | single request, model scale |
| `stage1c-opt7b-waveA/20261028T192131Z-2e060e31` | `beace01` | single request, model scale |
| `stage1c-opt7b-waveB-ext/20261027T203946Z-aa0a8614` | `c3cc4dd` | single request |
| `stage1c-opt7b-waveB-ext/20261027T204349Z-784975d3` | `c3cc4dd` | single request |
| `stage1c-opt7b-waveB-ext/20261028T012527Z-c336576d` | `c3cc4dd` | single request |
| `stage1c-opt7b-waveB-ext/20261028T013018Z-29d47a6a` | `c3cc4dd` | single request |
| `stage1c-opt7b-waveB/20261027T195426Z-82eebc23` | `c3cc4dd` | single request |
| `stage1c-opt7b-waveB/20261028T012021Z-b4781136` | `c3cc4dd` | single request |
| `stage1c-opt7b-waveB/20261029T060511Z-f7396062` | `ffc13de` | single request, model scale |
| `stage3a-offline/20261026T155944Z-c271a553` | `c3cc4dd` | offline |
| `stage3a-offline/20261027T124826Z-31add31e` | `c3cc4dd` | offline |
| `stage3a-offline/20261027T204752Z-da0f1b1a` | `c3cc4dd` | offline |
| `stage3a-offline/20261028T203854Z-9c4b1032` | `beace01` | offline |
| `stage3a-offline/20261028T214812Z-927e33db` | `beace01` | offline |
| `stage3a-offline/20261029T064601Z-ec4f64da` | `ffc13de` | offline |
| `stage3a-offline/20261029T075501Z-cd2e2c5c` | `ffc13de` | offline |
| `stage3a-offline/20261029T144601Z-badcf746` | `e72b3ce` | offline |
| `stage3b-concurrent/20261026T175554Z-94df114d` | `c3cc4dd` | closed loop |
| `stage3b-concurrent/20261027T132613Z-3a03cc76` | `c3cc4dd` | closed loop |
| `stage3c-open/20261028T225541Z-1ddd5f1f` | `beace01` | open loop |
| `stage3c-open/20261029T082254Z-7cdc26a2` | `ffc13de` | open loop |
| `stage3c-open/20261029T153032Z-aac41a50` | `e72b3ce` | open loop |
| `stage3c-open/20261030T010835Z-d44ca898` | `6d22b26` | open loop |
| `stage3c-open/20261030T013847Z-1204eebe` | `6d22b26` | open loop |
| `stage4b-tp2-qwen2.5-14b/20261028T101629Z-27ea1500` | `beace01` | model scale |
| `stage4b-tp2-qwen2.5-14b/20261029T033953Z-0cf582a3` | `ffc13de` | model scale |
| `stage4b-tp2-qwen2.5-14b/20261029T165124Z-ee6ad111` | `333db8a` | model scale |
| `stage4b-tp2-qwen2.5-14b/20261029T190249Z-a97ff8bd` | `5bc2cec` | model scale |
| `stage4b-tp2-qwen3-32b/20261028T140109Z-2c903dcb` | `beace01` | model scale |
| `stage4b-tp2-qwen3-32b/20261029T032511Z-de05c8b2` | `ffc13de` | model scale |
| `stage4b-tp2-qwen3-32b/20261029T175001Z-2f53b725` | `333db8a` | model scale |
| `stage4b-tp2-qwen3-32b/20261029T190918Z-f27a8d69` | `5bc2cec` | model scale |
| `stage5a-llama70b-fastest/20261027T064253Z-0152fed8` | `c3cc4dd` | model scale |
| `stage5a-llama70b-fastest/20261027T065912Z-90ff849c` | `c3cc4dd` | model scale |
| `stage5a-llama70b-fastest/20261027T073637Z-5853cb81` | `c3cc4dd` | model scale |
| `stage5a-llama70b-fastest/20261027T145037Z-6704c2a8` | `c3cc4dd` | model scale |
| `stage5a-llama70b-fastest/20261028T161934Z-90a79693` | `beace01` | model scale |
| `stage5a-llama70b-fastest/20261029T191635Z-92f507d3` | `5bc2cec` | model scale |
| `stage5a-llama70b-fastest/20261029T204841Z-fa82c8d6` | `5bc2cec` | model scale |
