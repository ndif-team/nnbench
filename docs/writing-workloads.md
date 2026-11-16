# Writing a workload

Follow these steps for every new methodology cell, every new family, and every foreign-system cell.
A cell that skips them can read the wrong tensor and still run cleanly: the vLLM ablation and
activation-patching cells read only the first element of a fused-residual layer's output from
June to September 2026, because no step asked what that element denotes.

## 1. Read the system's documentation at the pinned version

The version that runs is the one pinned in `backends/<system>/Dockerfile` and the build args of
the config's `compose.yml`. The images do not ship documentation, so read it in the system's
repository at that commit.

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
  applied as a delta onto the first element (`_worker_ext.py`, `_apply_hook_delta`); 1.3.0 stores
  it in the first element and zeroes the second (`_helpers/_hook_output.py`).

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

## 5a. Hand the backend units, never an engine form

A cell hands each backend primitive the units of one call and nothing about how the engine takes
them (`isb/backends/base.py`, design.md §12.13 "Unit form"). Every cell takes `units: list[Unit]`;
B=1 is a one-element list. `run` and `generate` take a nonempty unit list; `patch`,
`generate_patch`, `attribute`, `train_patch` and `vjp_batch` take one list per side of equal
length. Every backend checks that through `Backend.many()` / `Backend.paired()` and raises
`FormError` on a bare prompt, an empty list or unequal sides, and the test spy inherits the check,
so a cell in the wrong form fails in the no-GPU suite. The backend runs the list in the engine's
form (HF: one left-padded batch; the sync vLLM engine: one invoke per unit inside one trace; the
async engine: one request per unit, gathered), and the closures receive `rows`, the indices of the
units the current trace holds (`build(rows)`; `step(index)` for each decode step), to pick their
targets. `train_patch`'s step returns the loss summed over `rows`, so the gradient is the same in
both forms. A reduction over the sequence must honor the batch: HF's steering `mean_norm` masks the
pad positions through the attention mask in `model.inputs` and keeps one scale per unit
(`[B, 1, 1]`); a mean over pad tokens or across prompts is a batch-mixing bug. Read `model.inputs`
before any block's output: nnsight 0.8 runs the trace body alongside the forward and raises
`OutOfOrderError` for a module's input once the model has run past it (on 2026-10-24 every HF
steering row on Qwen3-8B failed until the mask read moved ahead of the block read). A cell that drops
units (taking `prompts[0]` of a several-prompt list), or passes a list because HF would batch it,
has chosen a form for the engine; on vLLM the recorded error then names the harness's form instead
of the engine's limit. The DAS train step on PR #17 (2026-10-21) did this: 16-prompt lists to
`train_patch`, whose contract then took one pair, so the async vLLM row recorded "Multiple prompts
per invoke are not supported" in place of the inference-mode gradient refusal the row exists to
record.

## 5b. Return the user's result; the check output only on a check call

A timed call holds what a user of the system runs: the request, the intervention and the result in
the system's own form. Nothing the benchmark needs goes in it: no readout for the scorer, no cast or
transform for the comparison, no guard, no setup a user does once (build vectors and directions
before the trace, as `readout.py` does). When the correctness check compares something other than
that result, the cell declares `check=False` and returns the check output only when the executor
passes `check=True` (design.md §12.13 "Timed calls and check calls"); `check` is never a case
parameter. The executor makes each case's check calls untimed, after every timed call of the regime
(the baseline, the plain request and every case). A cell that declares `check` needs its
methodology's timed check, registered beside the cells with `timed_check_of`
(`isb/oracle/checks.py` holds the shared forms: `tokens_match`, `generated_tokens_match`,
`read_matches`, `probabilities_match`): the executor relates the last timed result to the check
output of the same units, a failed timed check makes the row `INVALID_OUTPUT`, and a methodology
without a timed check fails the row with a `LookupError`.
Steering is the example: the timed call returns the greedy next token (`next_token`: the
head's argmax on HF, the engine's own sample `model.samples` on vLLM, nnsight 0.8.2
docs/models/vllm.md) with `keep_dtype=True`, so no cast runs; the check call returns the next-token
logits at the documented read site (`final_logits`). A read returns its value as the system
returns it when timed and, where that is not fp32, in fp32 when checked; a timed generation returns
the tokens the decode produced (`tokens=True`: HF's `generate` ids, vLLM's sample at each step) and
passes a per-step closure only when the case does work at every step. A cell whose two forms would be the same tensor declares no `check`. A system
that cannot produce the check output raises `Unsupported` from its check calls only, and its timed
rows still record.

## 6. Run the new cell before writing anything

Run the new cell on each backend that registers it or falls back to it (`vllm_sync` uses the
`vllm_async` cell), on gpt2 or the smallest registered model of its family, against
`nnsight-hf/default`, before any sentence about what it shows, through `scripts/bench.py run`. On a mismatch, repeat steps 1 to 3 before writing it up. For
cross-system rows, a second system on the same engine locates the difference: bitwise-identical
outputs mean the gap belongs to the engine, not to either interpretability layer.

An error on a backend is reported in this order: what the cell passed, what the primitive takes
(quote `base.py`), what the engine's message asked for. The words "engine limit", "finding" and
"frontier marker" come after the cell is in the documented form on every backend, never before.

## 7. Changing a default

A cell default is the system's documented realization of every workload row, so changing it changes
what every case of that methodology measures on that system. Keep the old spelling measurable by
declaring it as an alternative on the registration (`realizations=`, design.md §12.16), never by
writing it into a workload. The `parallel` suite scores vLLM against vLLM; a shared wrong reading
cancels there, so it never tests fidelity to HF.

## 8. Changing a cell or a backend primitive

Before pushing, run only what the diff changed, before and after the change:

1. List the (cell, backend) pairs whose executed code the diff changes. A cell body registered
   for several backends runs on each of them (the DAS body is registered for HF and vLLM, so a
   change to it runs on both); a change to one backend's primitive runs the cells that call it,
   on that backend only.
2. Run those pairs before the change (the main tree, or the last job's run files), on gpt2 or
   the smallest registered model of the cell's family.
3. Make the change; run the no-GPU suite (it catches a one-unit primitive handed several units,
   nothing more about form).
4. Run the same pairs again and compare each with its run before the change
   (`scripts/row_diff.py BEFORE.pt AFTER.pt`, given the two jobs' `result.pt`, prints each
   row's recorded error text and output difference).
5. A row whose output changed needs the reason in the commit message (a new default, a
   batching change). A row whose recorded error text changed is a cell-form change until the
   commit message says otherwise: the engine's message usually names the form it takes.
6. Push, with the comparison summary in the pull request.
