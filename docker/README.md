# VM-style nnbench: GPU server + GPU-less client

Runs nnbench's `vllm_serve` backend across a real client/server boundary instead of in-process, so the
benchmark measures the over-the-wire deployment (serialize → HTTP → save-transmission) and the
serve-only behaviors the in-process `vllm_async` backend can't see.

- **server** (GPU): `nnsight.modeling.vllm.serve.cli` — a real vLLM async engine behind FastAPI.
- **client** (no GPU): builds a *meta* `VLLM` model and submits the compiled trace over HTTP via
  `model.trace(serve=…)`. Genuinely GPU-less; it keeps CUDA *libraries* (the GPU vLLM wheel's `_C`
  links `libcudart`) but is given no GPU device or driver.

## What this measures (it's about the WORKING recipe, not the gaps)

Each methodology is a *matched pair*: the vanilla form you'd port naively from HF (a frontier marker —
on vLLM it may ERROR or be SILENTLY_WRONG; those gaps are already documented in nnsight's
`intervention-gaps/`), and the **documented-correct form that works on this backend**. The deliverable
is that the working form **survives the serve transport and still matches HF**. The serve map for the
current corpus (bf16 production server, precision-disambiguated):

| methodology (gpt2 unless noted) | working form (the contribution) | over serve | frontier form (marker) | over serve |
|---|---|---|---|---|
| logit_lens | `unembed=weight` (bypass guarded `lm_head.forward`) | SUPPORTED tv=0.021 | `unembed=module` | ERROR (lm_head guard) |
| logit_lens (llama) | `unembed=weight` + `residual=fused` (`out[0]+out[1]`) | SUPPORTED tv=0.017 | `residual=plain` (naive port) | SILENTLY_WRONG tv=0.90 (dual-residual) |
| steering | `mode=replace` (whole-tuple) | **SUPPORTED tv=0.000** | `mode=inplace` | ERROR (in-place on inference tensor) |
| ablation | whole-tuple replacement | SUPPORTED_DEGRADED (bf16) | — | — |
| activation_patching | two-trace cross-prompt (`be.patch`) | SUPPORTED_DEGRADED (bf16) | — | — |
| attention_pattern | (none — vLLM has no `.source` op) | ERROR | — | — |
| attribution_patching | (none — vLLM runs inference-mode, no autograd) | ERROR | — | — |

The write recipes (steering-replace, ablation-replacement, the cross-prompt two-trace patch — including
the CPU-tensor round-trip) all apply correctly over HTTP. attention/attribution have **no** working
vLLM version; the map records that boundary honestly.

## Precision: SUPPORTED_DEGRADED vs SILENTLY_WRONG

ablation/patching diverge from fp32-HF by a small `tv` at the server's bf16 default — a precision
near-tie, not a bug (they match HF to `tv≈0.001` at fp32). The in-process benchmark separates these by
re-running at fp32; the GPU-less client can't, so it disambiguates against a **cached fp32-vLLM**
output (`dump_control_refs`): if the fp32-vLLM result matches the HF control → SUPPORTED_DEGRADED, else
the divergence is a real mechanism bug → stays SILENTLY_WRONG (e.g. llama `residual=plain`, whose
fp32-vLLM output still drops the residual).

## Why the client needs CPU-platform forcing

A meta `VLLM()` build is **not** GPU-free with the stock GPU wheel: vLLM constructs real attention
layers that probe `torch.cuda.get_device_capability()`, and it only auto-selects its CPU platform for a
`+cpu` build (or macOS) — not merely when no GPU is visible (a GPU-less container with the GPU wheel
otherwise resolves to `UnspecifiedPlatform` and can't infer a device). So
`isb/backends/vllm_serve.py:_force_cpu_platform_when_no_gpu` registers a CPU choice in vLLM's platform
registry that activates **only when NVML reports no GPU** — a no-op on the GPU server. Validate it:

```
docker run --rm -e NVIDIA_VISIBLE_DEVICES=void -e HF_HUB_CACHE=/models -e PYTHONPATH=/nnsight/src \
  -v /disk/u/zikai/nnsight:/nnsight:ro -v /disk/u/zikai/interp-serve-bench:/nnbench:ro \
  -v /disk/u/models:/models:ro --entrypoint python3 \
  nnbench-serve:v0.15.1 /nnbench/docker/validate_client.py     # -> "CpuPlatform" + "META BUILD OK"
```

## Run it

1. **Build the image** (vLLM base + nnsight runtime deps):
   ```
   docker build -t nnbench-serve:v0.15.1 -f docker/Dockerfile docker/
   ```
2. **Generate cached references on a GPU host** — the GPU-less client scores against them:
   ```
   # HF control (correctness reference)
   CUDA_VISIBLE_DEVICES=5 python scripts/bench.py --spec all --backends hf --dump-refs results/refs
   # fp32-vLLM control-dtype output (for serve precision disambiguation)
   CUDA_VISIBLE_DEVICES=5 python scripts/bench.py --spec all --dump-ctl-refs results/refs
   ```
3. **Run the split** — specs are grouped by model; one server per model serves all its specs:
   ```
   GPU=5 docker/run_vm.sh                       # every spec
   GPU=5 docker/run_vm.sh steering_gpt2         # selected
   ```
   Each client prints the applicability map + perf, scoring serve cells against `results/refs`
   (HF control + fp32-vLLM disambiguation). `DTYPE=float32 docker/run_vm.sh …` runs the server at
   matched precision (shows the working forms SUPPORTED rather than DEGRADED).

## Security

`serve.cli --host 0.0.0.0` makes the server execute pickled Python from the network. It is only safe on
the trusted compose bridge — do not expose the published `6677` port to untrusted networks.
