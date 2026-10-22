# Handoff: the large-model realization comparison on another machine

How to rerun the 72B comparison of 2026-10-22 (nnsight on HF against nnsight on vLLM, at the
fastest form each engine documents, at equal precision, over two GPUs) on a Docker host, with a
different model if needed. It says what the comparison measures, what it delivers, what code it
needs, how to set the host up, how to run and read it, and what to change for another model.

The Delta run this describes: Qwen2.5-72B-Instruct in bf16 on two H200s (gpue02), job 23110478,
runs `runs/large-opt/20261022T151806Z-460acb26` (lens, steering, ablation, steered generation) and
`runs/large-opt/20261022T163216Z-9037ebc0` (activation patching), 1 h 31 min for both including
six model loads. Results are in `docs/findings.md` ("72B" section of 2026-10-22).

## 1. What is being measured

One question: at a size where the model needs more than one GPU, how does nnsight on vLLM compare
with nnsight on HF, when each engine runs the fastest form nnsight documents for it, at the same
precision, on the same card, in the same job, with the plain engine (no trace) beside it?

- **Workloads** (one prompt per call, the interactive regime; generation is 8 new tokens):
  logit lens over every layer, steering (eight rows: layer depth, alpha, per-token scaling), ablation
  (zero an MLP or attention output), steering at every decode step, activation patching (IOI clean
  and corrupted pairs, two forwards). Data: `counterfact:16` for the first four, `mib/ioi:16` for
  patching. Both datasets are in `data/`.
- **Backends, all bf16, two GPUs each:**

  | Column | Backend config | What it is |
  |---|---|---|
  | HF, device_map (reference) | `nnsight-hf/sdpa/0-8-1/gpu2` | nnsight 0.8.1 `LanguageModel`, transformers sdpa, accelerate `device_map="auto"`: layers split across the GPUs and run one after the other |
  | HF, tensor parallel | `nnsight-hf/sdpa/0-8-1/tp2` | the same under nnsight's transformers TP (`DistributedConfig(tp_size=2)`, transformers 5.19, `torchrun` with one rank per GPU) |
  | vLLM, general cell | `nnsight-vllm/sync/0-8-1/byref/tp2` | nnsight 0.8.1 on vLLM 0.27.1, one trace per call, envoys by reference, `tensor_parallel_size=2` |
  | vLLM, installed edits | `nnsight-vllm/opt/0-8-1/byref/tp2` | the block installed once with `model.edit(name=...)`, each timed call a plain request with `edits=[name]` |
  | vLLM, edits + taps | `nnsight-vllm/opt/0-8-1/taps/byref/tp2` | the same edits on an engine started with `taps` naming the touched sites, so CUDA graphs stay on |
  | plain vLLM | `vllm-plain/0-28-0/tp2` | vLLM 0.28.0 compiled, no nnsight; gives the engine's own forward |

  The plain HF number is the HF backend's own no-trace call (every nnsight backend times one; it is
  the `__vanilla__` auxiliary call in each `result.json`).
- **Correctness** is scored against the HF device_map column (top-1 agreement and total-variation
  distance of next-token distributions over all 16 prompts). Under the validation policy
  (`CLAUDE.md`, design §8.1) a numerical difference is `NUMERICAL_MISMATCH` with correctness
  unresolved and the row stays in the timing table; only failures leave it.

The envoy-by-reference prototype is the vLLM baseline on purpose: stock nnsight 0.8.1 pickles the
whole envoy tree with every request, which is the bulk of its per-request cost on vLLM. The fix is
on its way into nnsight (branch `zikai/envoy-by-reference`); until it is released, the `*/byref`
configs load the prototype from `backends/nnsight-vllm/envoy_byref` through `PYTHONPATH`.

## 2. Deliverables

1. The two run directories under `runs/large-opt/` (or your `--out`): `report.json` per run, and
   per backend `experiments/<id>/<backend>/result.json` with every timed call, the plain-engine
   call and the oracle inputs.
2. A findings section in `docs/findings.md`: the table of medians per workload row and column,
   the ratios that matter (general cell vs HF, taps vs HF, taps vs plain vLLM, HF TP vs HF
   device_map, nnsight's own overhead over plain HF), and the correctness states with the rows that
   do not pass named.
3. A report page in the Delta report format, built from the run directory by
   `scripts/report/optlarge.py` (the 72B page) or `scripts/report/opt7b.py` (the 7B page):
   `python scripts/report/optlarge.py runs/large-opt out.html`. The page holds the hero and
   findings, per-workload bar charts (plain engine, general cell, fastest realization per engine),
   the all-rows table with ratios, the realization descriptions, and the correctness table.

## 3. Code: which checkout has what

The 72B run came from `/work/nvme/bdnh/zwang83/nnbench-l72opt`, a copy of the repo carrying work
that is not yet on `main` (several copies were needed because the source guard fails a running job
when `isb/` or `backends/` change under it). Before anything else, consolidate that copy into a
branch; `diff -rq nnbench nnbench-l72opt -x runs -x .apptainer -x .git` lists the differences.
What it adds over `main`:

- `backends/nnsight-hf/sdpa/0-8-1/{gpu2,tp2}` and `backends/nnsight-hf/sdpa/0-4-6/gpu2`: the
  two-GPU HF configs. `tp2` needs the HF image built with `TRANSFORMERS_VERSION=5.19.0`,
  `CLICK_VERSION=8.4.2`, `REMOVE_PACKAGES=spin` (see its `compose.yml`; the Dockerfile grew those
  build args).
- `backends/nnsight-vllm/opt/0-8-1/{byref,taps/byref}` and their `tp2` variants,
  `backends/nnsight-vllm/sync/0-8-1/byref/tp2`, `backends/vllm-plain/0-28-0/tp2`.
- `backends/nnsight-vllm/opt/{backend.py,cells.py}`: the installed-edit cells made TP-safe (the
  steering direction is read from the checkpoint, not from a vocab-sharded `lm_head.weight`;
  `tensor_parallel_size` passed through).
- `isb/backends/hf.py` (`tp_size` option, `DistributedConfig`), `isb/jobs/entry.py` (re-executes
  under `torchrun` when `ISB_RANKS` > 1), `isb/jobs/worker.py` (ranks above 0 write to a scratch
  directory), `isb/methodologies/readout.py` (`head_direction`: a DTensor head falls back to the
  checkpoint), `isb/sweep/execute.py` (`_cached_revision`: transformers 5.18+ drops
  `config._commit_hash`, so the model-identity check reads the cache snapshot), `isb/models.py`
  (the `qwen2.5-72b` entry), the `large-opt` stage in `slurm/delta-compare.slurm`.
- The report generator is in `scripts/report/` of the main checkout (`build.py`, `assets/`,
  `opt7b.py`; `optlarge.py` for the 72B layout). Not committed yet.

## 4. Host setup with Docker

Requirements (`README.md`, "Running it"): Docker Engine, Compose v2, the NVIDIA Container Toolkit,
and a host Python with CPU PyTorch for scoring. Two GPUs with at least 80 GB each for a 72B model
in bf16 (145 GB of weights; vLLM at `gpu_memory_utilization` 0.85 needs about 136 GB free across
the two cards plus KV cache, so two 80 GB cards are the floor and two 141 GB cards comfortable).
Four cards and `tp4` configs are the alternative; see section 7.

1. **Build the images.** Each config's `compose.yml` names its image and build args; `bench.py
   build` builds what the listed configs need:

   ```bash
   python scripts/bench.py build nnsight-hf/sdpa/0-8-1/gpu2 nnsight-hf/sdpa/0-8-1/tp2 \
       nnsight-vllm/sync/0-8-1/byref/tp2 vllm-plain/0-28-0/tp2
   ```

   That is three images: `isb-nnsight-hf-0-8-1` (nnsight 0.8.1, transformers 5.12), the same with
   transformers 5.19 as `isb-nnsight-hf-0-8-1-tf5-19`, `isb-nnsight-vllm-0-8-1` (vLLM 0.27.1,
   shared by the general, edits and taps configs), and `isb-vllm-plain-0-28-0` (vLLM 0.28.0). The
   nnsight ref is pinned in every compose file (`de7fec4d…`, tag v0.8.1).
2. **Stock the model cache.** The containers read `/models` from the named volume `isb-model-cache`
   (override the name with `ISB_MODEL_CACHE`). Either let the first run download (set `HF_TOKEN` if
   the repo is gated) or populate the volume from an existing HF cache (`hub/models--Qwen--...`
   under it) and run offline with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. The identity check
   compares the checkpoint revision between backends, so every backend must see the same snapshot.
3. **Smoke, smallest first.** Before the large model, one GPU, one workload:

   ```bash
   python scripts/bench.py run --workload steering --model qwen2.5-7b --data counterfact:4 \
       --backends nnsight-hf/sdpa/0-8-1 nnsight-vllm/sync/0-8-1/byref --reference nnsight-hf/sdpa/0-8-1 --gpu 0
   ```

   Then the two-GPU HF TP smoke on the 7B (`slurm/delta-compare.slurm hftp-smoke` has the exact
   command): TP must agree with one GPU to bf16 rounding (top-1 0.75 to 1.0 on the few rows that
   differ, TV below 0.04) and will be slower than one GPU at 7B. That is expected: TP pays per
   layer in communication and only wins when the model does not fit one card.

## 5. Running the comparison

The `large-opt` stage, as two `bench run` calls (one per data set). `--gpu 0` means GPUs 0 and 1
for a `tp2`/`gpu2` config (the launcher exposes `ISB_GPU` and `ISB_GPU_1`; `--gpu` must be an
index, not a UUID). Jobs run one after another, each in its own container; the timeout is per job.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # once the cache is stocked
for spec in "logit_lens steering ablation gen_steering|counterfact:16" "activation_patching|mib/ioi:16"; do
    python scripts/bench.py run --workload ${spec%%|*} --model qwen2.5-72b \
        --data "${spec##*|}" --regimes interactive generation --realizations default \
        --backends nnsight-hf/sdpa/0-8-1/gpu2 nnsight-hf/sdpa/0-8-1/tp2 nnsight-vllm/sync/0-8-1/byref/tp2 \
                   nnsight-vllm/opt/0-8-1/byref/tp2 nnsight-vllm/opt/0-8-1/taps/byref/tp2 vllm-plain/0-28-0/tp2 \
        --reference nnsight-hf/sdpa/0-8-1/gpu2 --gpu 0 --timeout 3600 --out runs/large-opt
done
```

Budget: on two H200s the two calls took 1 h 31 min together, most of it the six loads of 145 GB
(HF loads twice, vLLM four times, each engine start a few minutes). Allow 3 h on slower storage.

Then the deliverables:

```bash
python scripts/bench.py score runs/large-opt/<run id>          # re-score if needed
python scripts/report/optlarge.py runs/large-opt large-opt-report.html
```

## 6. Reading the results

- `report.json` holds one cell per (workload row, backend): `median_latency_ms`, `state`
  (`SUPPORTED`, `NUMERICAL_MISMATCH`, `UNSUPPORTED` for the plain engine, which runs no
  intervention), `validation_state` (`NUMERICAL_CHECK_PASSED`, `UNRESOLVED`, `EXECUTION_FAILED`),
  `metrics.top1_agree`, `metrics.tv`. The reference column's own state is `RAN`.
- The plain-engine numbers are not cells: read `auxiliary_calls` in each backend's `result.json`
  for the entry with key `["__vanilla__", <regime>]`. Patching is two forwards, so compare a
  patching cell with twice the plain forward.
- What the 72B run showed, as the pattern to expect: vLLM's general cell at 0.6 to 0.75 of HF
  device_map on single forwards and generation and 1.2 on the logit lens; taps at 0.45 to 0.65 of
  HF and within 1.2 of plain vLLM except the lens, where taps were slower than the general cell;
  HF tensor parallel 1.2 to 1.8 times slower than device_map on single forwards; nnsight's own
  cost on HF about 1.01 of plain for steering and generation, 2.2 for the lens, 1.2 for patching.
  Correctness: lens and steering pass everywhere; ablation and some generation rows are unresolved
  on every backend (effect too weak to score, graded rows fork at a near-tie); patching passes on
  the general cell and on the installed-edit realizations only at quarter depth for the last
  token, the other three installed-edit rows are mismatches (a finding to classify, not a cell to
  patch: the same cells pass at 7B on one GPU).

## 7. Changing the model

1. Add it to `isb/models.py`: `Model(name, repo, family, n_layers, dtype_control="bfloat16")`.
   The family picks the cells (`llama` covers Qwen2.5, Qwen3, Llama, Mistral; others exist for
   GPT-2, Gemma, BERT...). Layers are addressed by depth fraction, so nothing else changes.
   Use the exact repo id the cache holds; `HF_HUB_OFFLINE=1` refuses a different spelling.
2. Size the GPUs. Weights in bf16 are 2 bytes per parameter; vLLM reserves
   `gpu_memory_utilization` (0.85 in the configs) of each card for weights plus KV cache, HF
   device_map just needs the weights to fit across the cards. If two cards are too small, copy the
   `tp2` configs to `tp4` (`tensor_parallel_size: 4`, `tp_size: 4`, `ISB_RANKS: "4"`,
   `device_ids` up to `ISB_GPU_3`), one config per backend. A model that fits one card does not
   need any of this: use the single-GPU configs and the 7B stage as the template.
3. Keep the pairing fair: every column in bf16, the same attention implementation on HF
   (`sdpa`), the reference the HF column itself at the same precision (fp32 is a correctness
   reference only, not a timing peer), the plain engine in the same job.
4. Taps need vLLM 0.27.1 (breakable CUDA graphs); the edits and general configs would run on
   0.28.0 as well, but keep one vLLM version across the three nnsight columns so the comparison is
   about the realization, not the engine release.
5. Datasets: `counterfact:N` and `mib/ioi:N` take any N up to the file size; 16 is enough for the
   oracle to settle most rows. Larger N multiplies the oracle's forwards, not the timed calls.

## 8. Open question to run next: why the logit lens stays slow on vLLM

The lens is the one workload where vLLM loses at every size (7B: 60 against 48 ms; 72B: 132 against
111 ms on HF device_map, and 154 ms with taps). Its cost is not the forward (26 ms plain on vLLM
TP2) but what the cell does on top: it projects 80 last-token hidden states through the
unembedding (80 × 8192 by 152,064, one 2.5 GB weight read on HF, two half reads plus a gather under
vLLM TP) and saves 80 × 152,064 logits, 24 MB in bf16 per prompt. On HF that tensor stays in the
process; on vLLM it is produced in the EngineCore process and pickled back to the client through
vLLM's IPC on every call. The 7B diagnosis (`slurm/diag_cost.py`, findings 2026-10-20) attributed
the lens gap to the saved values rather than the request payload; at 72B this is inferred, not
measured. Taps make it worse because the graph is broken at each of the 80 read sites and one
forward has no decode steps on which replay pays back.

What to measure, on the same two GPUs as the comparison, no scheduler needed (one process, a few
minutes once the engine is up):

1. **Split the lens into its parts** with `slurm/diag_cost.py` as the template (it times the client
   side and the engine side of an N-invoke trace separately and can profile the engine process):
   the bare forward; a trace that saves nothing; a trace that projects all 80 layers but saves only
   the top-k ids and values per layer (80 × k instead of 80 × 152,064); the full lens. The
   differences give the projection, the intervention machinery and the return trip. Run the same
   four on the HF device_map backend for the pairing.
2. **Vary the shape:** 40 layers and 20 layers (the `layers=` argument of the lens cell, see
   `isb/methodologies/logit_lens.py`), and fp32 against bf16 saves, to confirm the return cost
   scales with bytes returned.
3. **Projection form on vLLM:** `model.logits_processor(model.lm_head, normed)` (the documented
   form) against one batched projection of all layers (`backends/nnsight-vllm/opt/cells.py:52`)
   against a plain matmul with the gathered weight; under TP the first two gather logits per call.
4. If the return trip dominates, the fix is on the nnsight side (return less: top-k, or keep saved
   values on the engine and fetch once) and the lens should be reported with the top-k realization
   as its documented fast form; if the projection dominates, it is the engine's unembedding under
   TP and HF and vLLM should be near even. Record the split in `docs/findings.md` and add the
   top-k realization as a cell variant rather than changing the full-vocabulary cell.

## 9. Known issues and gaps

- An engine death inside vLLM (an out-of-memory forward, for example) leaves the client blocked
  until the job timeout; the harness does not detect it. Keep `--timeout` finite.
- `gpu_memory_utilization` 0.85 on two 80 GB cards leaves little room for a 72B's KV cache; drop
  to 0.95 only with a short `max_model_len`, or use four cards.
- The installed-edit patching mismatches at 72B TP (section 6) are unexplained; the 7B single-GPU
  run passes the same rows. The TP-safe cells read the clean activation in one trace and write it
  in a second; a per-rank divergence is the first thing to check.
- The HF TP column timed slower than device_map at every size tried (7B and 72B). It is included
  because it is nnsight's own multi-GPU path for HF; device_map is the practical HF baseline.
- Naming: "plain HF" in the tables is the HF device_map backend with no trace, a bare forward over
  both GPUs, not a single-GPU run. Every HF number at 72B is over two cards.
- Stock nnsight 0.8.1 (without the byref prototype) was not run at 72B; the 7B stock-vs-byref
  numbers are in `docs/findings.md` (2026-10-21).
- The report generator was reconstructed from a session transcript and lives in `scripts/report/`
  uncommitted; commit it with the branch.
