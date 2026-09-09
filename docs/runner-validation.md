# Independent backend runner validation — 2026-09-08

The replacement runner was built from the experiment/artifact contracts. Backend selection is
directory-based under `backends/`; the old shared-Compose split launcher has been removed.
Intervention implementations and trace bodies were not rewritten to improve coverage results.

## Real GPU run

Both independent images built through:

```bash
python scripts/bench.py build nnsight-hf nnsight-vllm
```

Final execution command (reusing the existing populated cache):

```bash
ISB_MODEL_CACHE=nnbench-docker-test_model-cache \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/bench.py run --spec logit_lens_gpt2 --data factual:8 \
  --backends nnsight-hf nnsight-vllm --reference nnsight-hf --gpu 5 \
  --out runs/runner-v2-validation --timeout 600
```

Exit status: **0**. Both jobs completed, wrote validated artifacts, and cleaned their Compose
projects. Output directory:
`runs/runner-v2-validation/20260908T190954Z-e1896bde/`.
The experiment directory is `experiments/3c290739dc5df9d2`; the complete identity checksum is
in `experiment.json`.

All four HF cells ran. The vLLM interactive `unembed=weight` cell was **SUPPORTED**, over eight
factual prompts and twelve GPT-2 layers:

| Metric | Value |
|---|---|
| Top-1 agreement | 0.9791666865 |
| Mean total variation | 0.0178535059 |
| Max absolute logit difference (diagnostic) | 1.0916900635 |

Three vLLM cells correctly remained **ERROR**: both direct LM-head calls are guarded by the
engine, and the batched weight cell returns no saved outputs on this pinned nnsight stack.
`score ... --strict` returns 1 for these coverage findings; default scoring does not mistake
cell errors for container startup failures, nor hide them.

Both workers recorded the same model revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e` and the same tokenizer vocabulary checksum. They consumed
the same frozen input checksum. Actual image IDs were inspected from the job containers:

| Backend | Image ID |
|---|---|
| nnsight-hf | `sha256:86a24284580726b6979a8c5623d34e62f96b58cc03a0cd3d0d6d299f1afbac7a` |
| nnsight-vllm | `sha256:9b774e9a8239ef9364ef49e2c8788c0bf54b6c16624659d3a3735f0b0ebcb4e3` |

This was an A100 functional check on a shared host, not a publishable performance measurement.

## Automated checks

- Full host suite after the result-site integration: **236 passed, 3 opt-in Docker tests skipped**.
- Opt-in real Docker lifecycle suite: **3 passed**. Each test creates an unregistered
  `third-party` backend directory and runs it through discovery and the same lifecycle. Success,
  nonzero exit, and timeout all cleaned up correctly. The fixture is CPU-only and supplies no
  inference benchmark measurements.
- Verified paired/labeled input round trips, dataset-level workloads, input identity/count changes,
  duplicate/empty cells, missing/corrupted artifacts, source changes, unsafe service configuration,
  explicit comparison policies, incompatible model/tokenizer identities, batched references,
  nonfinite outputs, cancellation, and cleanup.
- Focused Ruff, shell syntax, and `git diff --check` passed.
- No running or stopped `isb-*` test/job containers remained after cleanup checks.

Initial failed attempts are retained in the same validation parent directory for inspection. They
exposed missing username/cache defaults for non-root execution in the base images. The backend
Compose files now provide writable PyTorch, Triton, FlashInfer, and vLLM cache locations.

## Scope

Validated inference backends: nnsight-HF and local async nnsight-vLLM. Other methods remain in the
suite; this GPU smoke is not a claim that every method/model/backend combination is supported.
The result site now reads these job bundles and saved reports without loading tensors during
browsing; see [the result-site guide](result-site.md). Multi-GPU scheduling, remote serving, and
migration of the standalone micro/performance tools are outside this change.
See [the backend contract and commands](../backends/README.md).
