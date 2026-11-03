# Engineering Handoff: Forward-Only Export Workloads

**Target Repository:** `/disk/u/zikai/interp-serve-bench`  
**Goal:** Implement two forward-only export workloads—**Bulk Activation Harvesting** (`activation_harvest`) and **Per-Step Activation Capture** (`gen_capture`)—to populate the Export column of the benchmark on serving engines.

---

## 1. Background & Context

In `interp-serve-bench`, the `export` workload category previously contained only `jacobian_collect`. However, `jacobian_collect` calculates input-output Jacobians via Vector-Jacobian Products (VJPs), which require backward passes and autograd. Because serving engines (e.g., vLLM, SGLang) execute under `torch.inference_mode()`, they fail immediately on autograd tensor assertions.

In real-world interpretability at scale (such as Goodfire's infrastructure for harvesting 3 billion activations overnight to train SAEs on Kimi K2 Thinking, DeepMind's Gemma Scope, and online reasoning-trajectory auditing), activation export is **100% forward-only**. 

The systems friction in real export workloads is:
1. **Device-to-Host (D2H) Bandwidth:** Transferring large volumes of activations ($B \times S \times D$) off GPU VRAM without stalling forward compute.
2. **Continuous Batching & Flattened Layouts:** Prefill tokens from concurrent requests are flattened into 1D sequences (`[sum(seq_len), d_model]`), requiring token and document provenance tracking.
3. **Autoregressive Decode Streaming:** Extracting activations per decode step inside the engine's tight generation loop.

Adding these two forward-only workloads gives serving engines legitimate export benchmarks that measure throughput, batching overhead, and D2H transfer efficiency.

---

## 2. Workload Specifications

### Workload 1: Bulk Activation Harvesting (`activation_harvest`)

* **Type:** `export`
* **Regime:** `batched` (16 prompts from WikiText)
* **Dataset:** `DataRef("wikitext", 16)`
* **Layer/Site:** Half-depth residual stream (`Depth(1/2)`)
* **Behavior:**
  * Performs a forward pass on a batch of prompts.
  * Captures the residual stream after layer $L = \lfloor \text{depth} \times n_{\text{layers}} \rfloor$ for all valid sequence tokens.
  * Streams/transfers the captured tensors to host memory (CPU).
* **Baseline (Denominator):** The exact same batch forward pass with no tensor export / save (`export=False`), isolating the D2H transfer and extraction overhead.
* **Output Tensor Contract:**
  * CPU `torch.FloatTensor` containing the extracted activations.
  * Shape: `[batch, seq_len, d_model]` (or concatenated `[total_tokens, d_model]`).
  * Check: Finite, non-empty, and valid tensor artifact registered in `outputs_sha256`.

### Workload 2: Per-Step Activation Capture (`gen_capture`)

* **Type:** `export`
* **Regime:** `generation` (32 CounterFact prompts, 8 greedy decode steps)
* **Dataset:** `DataRef("counterfact", 32)`
* **Layer/Site:** Half-depth residual stream (`Depth(1/2)`)
* **Behavior:**
  * Greedily decodes 8 new tokens per prompt.
  * At each decode step inside the generation loop, captures layer $L$'s residual stream.
  * Streams/accumulates the per-step activations into a CPU tensor.
* **Baseline (Denominator):** Un-instrumented greedy generation of 8 steps (`export=False`), identical to the vanilla decode loop.
* **Output Tensor Contract:**
  * CPU `torch.FloatTensor` containing the per-step activations across decode steps.
  * Shape: `[steps, d_model]` per prompt (or stacked `[prompts, steps, d_model]`).
  * Check: Exactly 8 steps collected, finite, non-empty.

---

## 3. Implementation Blueprint

### Step 1: Workload Definitions (`isb/workloads/`)

#### 1.1 `isb/workloads/activation_harvest.py`
```python
"""activation_harvest: bulk export of intermediate activations during batched prefill.

Forward-only export methodology. Captures the residual stream at target layer(s)
across all tokens in a batch and streams to CPU host memory.
"""
from ..data import DataRef
from ..sweep.spec import Depth, Regime, Row, Workload

_HALF = Depth(1 / 2)

activation_harvest = Workload(
    name="activation_harvest",
    methodology="activation_harvest",
    rows=(
        Row("half depth residual stream, all valid tokens", {"layer": _HALF, "export": True}),
    ),
    regimes=(Regime("batched", DataRef("wikitext", 16)),),
    baseline={"layer": _HALF, "export": False},
)
```

#### 1.2 `isb/workloads/gen_capture.py`
```python
"""gen_capture: capture internal activations at every step of autoregressive generation.

Forward-only decode export. Greedily decodes N steps and streams each step's
residual stream to host memory.
"""
from ..data import DataRef
from ..sweep.spec import Depth, Regime, Row, Workload

_HALF = Depth(1 / 2)

gen_capture = Workload(
    name="gen_capture",
    methodology="gen_capture",
    rows=(
        Row("half depth residual stream, 8 decode steps", {"layer": _HALF, "export": True}),
    ),
    regimes=(Regime("generation", DataRef("counterfact", 32), new_tokens=8),),
    baseline={"layer": _HALF, "export": False},
)
```

#### 1.3 Register in `isb/workloads/__init__.py`
Import and add `activation_harvest` and `gen_capture` to `ALL_WORKLOADS` (or the workload list).

---

### Step 2: Methodologies (`isb/methodologies/`)

#### 2.1 `isb/methodologies/activation_harvest.py`
Follow the pattern of `isb/methodologies/logit_lens.py`:

```python
from __future__ import annotations
import torch
from ..profiles import _resid
from .registry import cell

def _harvest_proxy(blocks, *, layer, residual="plain"):
    """Read the residual stream at layer. Runs INSIDE trace."""
    with torch.no_grad():
        out = blocks[layer].output
        return _resid(out, residual)

@cell("activation_harvest", family="*", backend="hf")
def harvest_hf(be, model, m, prompts, *, layer=None, export=True, residual="plain"):
    layer = layer if layer is not None else len(m.blocks(model)) // 2
    if not export:
        # Denominator baseline: plain forward pass without save
        def noop():
            return m.head(model).output[:, -1, :]
        return be.run(model, prompts, noop)

    def build():
        return _harvest_proxy(m.blocks(model), layer=layer, residual=residual).save()

    return be.run(model, prompts, build)

@cell("activation_harvest", family="*", backend="vllm_async")
def harvest_vllm(be, model, m, prompts, *, layer=None, export=True, residual=None):
    layer = layer if layer is not None else len(m.blocks(model)) // 2
    residual = residual if residual is not None else m.default_residual(be.name)
    if not export:
        def noop():
            return model.logits[-1:, :]
        return be.run(model, prompts, noop)

    def build():
        return _harvest_proxy(m.blocks(model), layer=layer, residual=residual).save()

    return be.run(model, prompts, build)
```

#### 2.2 `isb/methodologies/gen_capture.py`
Follow the pattern of `isb/methodologies/gen_steering.py`:

```python
from __future__ import annotations
import torch
from ..profiles import _resid
from .registry import cell

@cell("gen_capture", family="*", backend="hf", realizations=[{"bound": "unbounded"}])
def gen_capture_hf(be, model, m, prompts, *, layer=None, export=True,
                   bound="bounded", new_tokens=8, residual="plain"):
    layer = layer if layer is not None else len(m.blocks(model)) // 2

    def step():
        if export:
            out = m.blocks(model)[layer].output
            return _resid(out, residual)
        return m.head(model).output[:, -1, :]

    return be.generate(model, prompts, step, new_tokens=new_tokens,
                       bounded=(bound == "bounded"))

@cell("gen_capture", family="*", backend="vllm_async", realizations=[{"bound": "unbounded"}])
def gen_capture_vllm(be, model, m, prompts, *, layer=None, export=True,
                     bound="bounded", new_tokens=8, residual=None):
    layer = layer if layer is not None else len(m.blocks(model)) // 2
    residual = residual if residual is not None else m.default_residual(be.name)

    def step():
        if export:
            out = m.blocks(model)[layer].output
            return _resid(out, residual)
        return model.logits[-1:, :]

    return be.generate(model, prompts, step, new_tokens=new_tokens,
                       bounded=(bound == "bounded"))
```

#### 2.3 Register in `isb/methodologies/__init__.py`
Add `from . import activation_harvest, gen_capture` to register the cells.

---

### Step 3: Backend Output Extraction Check (`isb/backends/`)

Verify how `be.generate` and `be.run` assemble step rows:
1. In `isb/backends/hf.py`:
   - `generate()` stacks step rows: `torch.cat([r.detach().float().cpu() for r in rows], dim=0)`.
   - If `r` is `[1, d_model]`, `cat(..., dim=0)` produces `[new_tokens, d_model]`. This matches!
2. In `isb/backends/vllm_async.py`:
   - `_extract_step_rows` does `mats = [r.detach().float().cpu() for r in v if r is not None]`, `torch.cat(mats, dim=0)`.
   - Ensure that if `r` has shape `[1, d_model]`, it concatenates cleanly along dimension 0.
3. In `isb/jobs/score.py`:
   - Verify that output tensor finite-checks (`torch.isfinite(actual).all()`) and numel-checks pass without requiring reference comparison (since `--reference ''` is used for support/timing).

---

## 4. Verification & Testing

1. **Unit Tests:**
   * Create `tests/test_activation_harvest.py`:
     - Test that `harvest_hf` runs on CPU/smoke GPT-2.
     - Assert returned tensor has expected shape `[batch, seq_len, d_model]` or `[total_tokens, d_model]`, finite values, and non-zero norm.
   * Create `tests/test_gen_capture.py`:
     - Test that `gen_capture_hf` decodes 4 tokens and yields a tensor of shape `[4, d_model]`.
2. **Execute Smoke Test:**
   ```bash
   python scripts/execute.py --case activation_harvest/gpt2 --backend hf --dry-run
   python scripts/execute.py --case gen_capture/gpt2 --backend hf --dry-run
   ```
3. **Full Sweep Integration:**
   * Verify that `python scripts/bench.py run --suite cross-system --dry-run` includes `activation_harvest` and `gen_capture`.
