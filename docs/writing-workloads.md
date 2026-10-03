# Writing a workload

Follow these steps for every new methodology cell, every new family, and every foreign-system cell.
A cell that skips them can read the wrong tensor and still run cleanly: the vLLM ablation and
activation-patching cells read only the first element of a fused-residual layer's output from
June to September 2026, because no step asked what that element denotes.

## 1. Read the system's documentation at the pinned version

The version that runs is the one pinned in `backends/NAME/Dockerfile`. The images do not ship
documentation, so read it in the system's repository at that commit.

- **nnsight** (`NNSIGHT_REF`): `docs/models/vllm.md`, section "What your block sees on vLLM";
  `docs/patterns/<method>.md`; `docs/gotchas/`.
- **This repository**: the realization table and documented gaps in
  `docs/interp-methods-catalog.md`, and measured results in `docs/findings.md`.
- **A foreign system**: its README and examples at the pinned release.

When this repository's catalog and the pinned documentation disagree, the pinned source decides.
Correct the catalog in the same change.

## 2. Establish what every site denotes, from source when the docs leave doubt

For each value the cell reads or writes, write down what tensor it is. If the documentation does not
settle it, read the code at the pinned version:

- **The engine's model file**, for what each module returns. vLLM 0.19.1's Qwen2 decoder layer
  (`vllm/model_executor/models/qwen2.py`) returns `(mlp_output, residual)`; the residual stream
  after the layer is their sum, because the next layer's norm performs the add.
- **The interpretability system's dispatch**, for what it hands user code and how a returned value
  is written back. vLLM-Lens 1.2.1 post-hooks receive the summed stream, and a returned tensor is
  applied as a delta onto the first element (`_worker_ext.py`, `_apply_hook_delta`).

Record the denotation in the cell's docstring with the file and version it came from.

## 3. Check reads and writes separately

A correct read does not make the write correct. The patching cell read the summed stream and then
wrote it into the first element while keeping the second, adding a residual twice. For every
write, state what the stream is after it.

## 4. Make the recommended form the default

The documented, correct realization is the cell's default. Family-dependent defaults come from the
profile (`m.default_residual(be.name)`), never a literal. A naive port stays only as an explicitly
labeled frontier task. Report automatically resolved choices with `record_resolved`.

## 4a. Stay inside the system's public interface

A score measures the system, not the cell author. A cell may compose the system's public,
documented API the way a user would: several requests, a documented engine mode or storage
option, the system's documented extension point, plain user code for steps the system leaves to the
user (loading a checkpoint weight, a final projection the system does not offer). A cell must not
reach into private attributes or internals, and must not re-implement work the system already
does, to make it cheaper. When a workload has no public form, the cell raises `Unsupported` with the
source reason; that is the finding.

Each system's score is then its fastest realization inside that interface: the general cell, a
documented performance mode (nnsight installed edits and `taps`, interp-engine `vllm-static`,
vLLM-Hook `disk-st-async`), or a better composition of the public API (vLLM-Lens's batched lens
readout and hook-built ablation use its documented `Hook`). Faster forms that step outside it (the
TransformerLens GPU readout, which reads weights through the private `_driver` and replaces the
bridge's own logit rebuild) are reported as headroom: evidence of a gap in the system, never as its
score. A cell that cannot avoid an internal (TransformerLens's logit lens and steering direction
read the unembedding through `_driver.get_param`; the bridge has no public accessor) says so in its
docstring and is labeled in the report.

## 5. Test the denotation, not the plumbing

Add one unit test per read and write denotation. Use a fake `(hidden, residual)` output whose two
elements differ, and assert the stream downstream of the cell. A test that passes the naive
reading explicitly checks nothing about the default.

## 6. Smoke against the reference before recording a finding

Run the new cell against `nnsight-hf` on a few inputs. On a mismatch, repeat steps 1 to 3 before
writing it up. For cross-system rows, a second system on the same engine locates the difference:
bitwise-identical outputs mean the gap belongs to the engine, not to either interpretability layer.

## 7. Changing a default

Find every spec that relies on the old default and pass the old value explicitly where that spec
needs it. The Qwen parallelism specs score vLLM against vLLM; a shared wrong reading cancels
there, so those specs never test fidelity to HF.
