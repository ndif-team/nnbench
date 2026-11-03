# Handoff: Post-Stage Patch & Targeted Rerun Plan (`eval-v1`)

## 0. Timing & Execution Policy
1. **Do NOT interrupt the currently running Stage 4 / Stage 5 queue.** Let the in-flight queue finish under the locked `d0ed4b9` commit first so no partial work is lost.
2. Once the current queue is idle, apply all patches below in a single commit on `refactor/consolidated-architecture` (so every rerun shares one new `(commit, source_sha256)` pair), run the no-GPU `pytest` suite to verify zero regressions, and then execute the targeted reruns listed in **Section 3**.

---

## 1. Required Code & Harness Patches

### Patch 1 — Plain vLLM `trust_remote_code` (`stage1b-nemotron`: 34 errors)
- **File:** `backends/vllm-plain/backend.py:36`
- **Problem:** `AutoConfig.from_pretrained(repo)` is called without forwarding `trust_remote_code` from `self.engine_options`, failing on Nemotron with `KeyError: '-'` and leaving 21 cells without `overhead_vs_plain_vllm`.
- **Fix:** Forward `trust_remote_code=self.engine_options.get("trust_remote_code", False)` into `AutoConfig.from_pretrained(repo, trust_remote_code=...)`.

### Patch 2 — HF `force_text_causal` / `automodel` (`stage1b-jlens`: 3 errors)
- **File:** `isb/backends/hf.py:31-37`
- **Problem:** Passing `kw["automodel"] = _TextCausal` into `LanguageModel(repo, dispatch=True, ...)` causes `nnsight 0.8.1` to forward `automodel=` into `Qwen3_5ForCausalLM.__init__()`, raising `TypeError: got an unexpected keyword argument 'automodel'`.
- **Fix:** Pre-load the causal LM and tokenizer via `AutoModelForCausalLM.from_pretrained(repo, ...)` and `AutoTokenizer.from_pretrained(repo)` (or pass the instantiated module/callable in the way `nnsight 0.8.1`'s `LanguageModel` expects without leaking `automodel` into `from_pretrained(**kwargs)`), and add a unit test covering `force_text_causal=True`.

### Patch 3 — HF `attention_pattern` Call-Site Indexing (`stage1a-capability-waveA`)
- **File:** `isb/methodologies/attention_pattern.py:19`
- **Problem:** Reads `blocks[i].attn.source.attention_interface_0.output[1]`. In `nnsight 0.8.1`, `visit_AnnAssign` binds `attention_interface_0` to the annotated declaration `attention_interface: Callable = ...` (a function object), whereas the actual call site is `attention_interface_1`.
- **Fix:** Change `attention_interface_0` to `attention_interface_1` on HF (keep the expected failure on vLLM PagedAttention).

### Patch 4 — `logits_processor` Path on Wrapper Architectures (`stage1b-jlens` / Qwen3.5-4B)
- **Files:** `isb/profiles.py:81-103`, `isb/methodologies/jacobian_lens.py:131`, `isb/methodologies/logit_lens.py:54`
- **Problem:** `jacobian_lens_vllm` and `logit_lens_vllm` pass `processor=model.logits_processor` at the root `model`. On vLLM 0.19.1's `Qwen3_5ForConditionalGeneration` (`vllm_prefix="language_model"`), `logits_processor` lives at `model.language_model.logits_processor`.
- **Fix:** Add a `logits_processor(self, model)` accessor to `ModelProfile` in `isb/profiles.py` (prefixed by `vllm_prefix` in `for_backend()`, defaulting to `"logits_processor"`), and use `processor=m.logits_processor(model)` in both `jacobian_lens.py` and `logit_lens.py`.

### Patch 5 — Missing Cell Registration Recorded as `UNSUPPORTED`, Not `ERROR` (22 cells)
- **File:** `isb/sweep/execute.py:33-37`
- **Problem:** `_cell()` raises `LookupError(f"no cell for {methodology}/{family}/{name}")`, which `_failure()` records as `ERROR` instead of `UNSUPPORTED`.
- **Fix:** Raise `Unsupported(f"no cell for {methodology}/{family}/{name}")` in `_cell()` so missing cell registrations record `state="UNSUPPORTED"` with their reason string, matching `isb/states.py`.

### Patch 6 — Plain vLLM Concurrent & Open-Loop Floor (`stage3b-concurrent`, `stage3c-open`, Stage 5)
- **Files:** `backends/vllm-plain/backend.py`, `backends/vllm-plain/configs.yml`
- **Problem:** `PlainVLLMBackend` uses synchronous `vllm.LLM` without `concurrent_requests = True`, so `_concurrency` in `isb/sweep/execute.py` serializes `vllm-plain` through a FIFO ticket gate (`load.serial: true`). Dividing `nnsight-vllm/0-8-1/async` (`serial: false`) by a serialized floor produces bogus `0.55x` overhead ratios at `c64`.
- **Fix:**
  1. Add `concurrent_requests = True` and an in-process `AsyncLLMEngine` / `AsyncLLM` path (or `mode: async` / concurrent session matching `VLLMAsyncBackend.concurrent_session`) to `PlainVLLMBackend` when driving concurrent (`c > 1`) and open-loop regimes, so `vllm-plain` batches concurrent client requests natively just like `nnsight-vllm/0-8-1/async`.
  2. Ensure `chunked_prefill` configuration matches between `nnsight-vllm` and its paired `vllm-plain` floor (or explicitly document/align `enable_chunked_prefill` in `configs.yml`).

### Patch 7 — Dataset Size Guard in `isb/data.py` (`activation_harvest`)
- **File:** `isb/data.py:53, 70, 80, 88`
- **Problem:** `data/wikitext/wikitext.json` only has 100 items, and `_wikitext` silently returns `prompts[:n]` when `n=256` is requested.
- **Fix:** Either expand `data/wikitext/wikitext.json` to $\ge 256$ items (preferred if offline sweep requires 256) OR raise a loud `ValueError` in file loaders when `n is not None and n > len(units)` and set the `activation_harvest` spec's offline unit count to match the dataset size so tables and specs never silently diverge.

### Patch 8 — Report Scripts & Auditor Updates
- **Files:** `scripts/report/opt7b.py`, `scripts/report/optlarge.py`, `scripts/audit_runs.py`
- **Fixes:**
  1. Update `PLAIN` / `PLAIN_E` in `scripts/report/opt7b.py` and `scripts/report/optlarge.py` from `0-28-0` to `0-27-1` (matching `stage1c-opt7b-waveA` and Stage 4B), and update the header prose from NCSA Delta / A100-40GB / `byref` to the current hardware and `nnsight 0.8.1` sync/async/opt realizations.
  2. In `scripts/audit_runs.py`:
     - Keep strict intra-run `(commit, source_sha256)` identity (required by `isb/jobs/score.py:_compatible`).
     - Allow `audit_runs.py` to accept `--allow-commits <c1,c2>` (or whitelist both `d0ed4b9` and the new patch commit across distinct run directories) so untouched clean runs from `d0ed4b9` and patched reruns can be audited together without false `GLOBAL SOURCE DRIFT` failures.

---

## 2. Real Engine Findings (Do NOT Patch Away — Document in Findings)
- **`stage1c-fp32-waveA` (`gen_steering` `INVALID_OUTPUT` on vLLM under `fp32`):**
  - Keep as a genuine finding: under `fp32`, vLLM's sampler writes `-inf` in-place onto EOS logits (`min_tokens` masking) on the exact tensor `nnsight` saved, whereas `bf16` casts/copies first. Record and classify this in the findings report.

---

## 3. Targeted Rerun Plan (After Committing All Patches)

> **Important:** Because `isb/jobs/score.py:_compatible` enforces `cexec["source"] == rexec["source"]` within each experiment directory, **rerun the entire affected run block** (both reference and candidate backends) into `/disk/u/zikai/isb-runs/eval-v1/` after moving the superseded `d0ed4b9` run directories to `/disk/u/zikai/isb-runs/eval-v1-superseded/`.

1. **`stage1b-nemotron`** (fixes the 34 `trust_remote_code` plain-vLLM errors and restores the 21 missing `overhead_vs_plain_vllm` ratios).
2. **`stage1b-jlens`** (fixes the 3 HF `automodel=` errors and the vLLM Qwen3.5-4B `language_model.logits_processor` error).
3. **`stage1a-capability-waveA`** (fixes HF `attention_pattern` `attention_interface_1` and reclassifies the 22 missing-cell `ERROR`s to `UNSUPPORTED`).
4. **`stage1b-qwen3-8b-waveA` (Solo Timing Run):**
   - Previously overlapped with `stage1c-fp32-waveA` under the parallel schedule. Rerun **strictly solo on 1 GPU** so its latencies and `logit_lens` numbers are controlled and eligible for latency tables.
5. **`stage3b-concurrent` & `stage3c-open` (Solo Load Runs):**
   - Rerun with the concurrent `vllm-plain` floor so `overhead_vs_plain_vllm` under concurrency (`c64`) and open-loop Poisson load compares apples-to-apples against concurrent vLLM rather than a FIFO-serialized floor.
   - If `wikitext.json` is expanded to 256 items in Patch 7, also rerun the `activation_harvest` experiment in `stage3a-offline` (or all of `stage3a-offline` if keeping a single run directory).

---

## 4. Verification Gate Before Closing
1. `pytest` no-GPU test suite passes 100%.
2. `python3 scripts/audit_runs.py /disk/u/zikai/isb-runs/eval-v1-pilot /disk/u/zikai/isb-runs/eval-v1` passes with **0 errors** (save for the 4 documented `fp32` in-place EOS logit mutations in `stage1c-fp32-waveA`).
3. Both `scripts/report/opt7b.py` and `scripts/report/optlarge.py` render cleanly from the `eval-v1` run directories without `TypeError`.
