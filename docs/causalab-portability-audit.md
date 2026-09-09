# CausaLab and nnbench: protocol alignment

## 1. Reference revision

Reviewed against CausaLab main at
[`8696e04bfb06a169defe1bf563d8aeef992f85cd`](https://github.com/goodfire-ai/causalab/tree/8696e04bfb06a169defe1bf563d8aeef992f85cd).
The current `can/protocol-refactor` tip, `f0b714c`, has the same application code;
the differences are two GitHub workflow files. GitHub reported no open CausaLab PRs at this check.
Closed, unmerged proposals such as #80–#82 are outside this baseline.

PR #20 merged into the parent branch on August 31, 2026, and PR #19 then merged the
accumulated protocol work into main. The parent branch's historical name, `nnterp-rebase`,
does not identify its final runtime. The current dependency graph contains neither pyvene
nor nnterp. The nnsight extra pins raw nnsight to `8c480727`.

[isb/causalab_vocabulary.json](../isb/causalab_vocabulary.json) records the source revision,
source-file checksums, canonical vocabulary, and deprecated aliases. Verify it with:

```sh
python scripts/check_causalab_alignment.py /path/to/causalab --check
```

For a future update, check out the intended upstream commit and run the command without
`--check` to print a candidate snapshot. Review the vocabulary and engine-policy changes,
update the snapshot and tests together, and record any relevant pending PR separately.
The checker reads Python declarations through the AST, keeping host validation independent
of CausaLab's model/runtime dependencies.

## 2. The current execution architecture

CausaLab executes serializable intervention documents. Its current structure is:

```text
protocol document (or method + application)
  -> validation, expansion, planning, derived requirements
  -> protocol/engine.py: Engine.execute(ExecutionRequest)
     -> neural/shared/: site semantics, shapes, positions, write math, outputs
     -> neural/engines/pytorch_hooks/: reference hook execution and training
     -> neural/engines/nnsight_tracing/: envoy and source-operation tracing
```

The authoritative contracts are
[schema.py](https://github.com/goodfire-ai/causalab/blob/8696e04bfb06a169defe1bf563d8aeef992f85cd/causalab/protocol/schema.py),
[engine.py](https://github.com/goodfire-ai/causalab/blob/8696e04bfb06a169defe1bf563d8aeef992f85cd/causalab/protocol/engine.py),
and [the intervention protocol](https://github.com/goodfire-ai/causalab/blob/8696e04bfb06a169defe1bf563d8aeef992f85cd/docs/intervention_protocol.md).

The N-stack (#56, #58, #59, #63–#68) renamed Backend to Engine, extracted shared services,
introduced component-aware routing, and implemented nnsight module boundaries, attention,
MoE and DeltaNet interiors, and generated-frame reads. Nnsight loads a `TransformersModel`;
shared site resolution addresses its envoy tree, and engine-specific address tables reach
interior operations through `.source`.

The nnsight engine declares generation and writable attention probabilities, alongside
paired forwards, full logits and local Python write functions. Training (`grad`) and
quantized weights remain reference-engine capabilities. These are CausaLab engine
declarations; nnbench's direct nnsight cells have their own measured outcomes.

Engine requirements combine coarse capabilities with `component:<name>` and
`component:<name>:write`. Shape, stream, layer and write-policy constraints remain additional
checks. Component vocabulary membership alone establishes a valid name, rather than a
measurement of support on a particular engine and model.

## 3. What nnbench shares

nnbench uses CausaLab's component, mechanism, featurizer and capability vocabulary to describe
explicit benchmark cells. `TaskSpec` holds semantic values and realization selectors;
`ExecutionRegime` holds input presentation, generation length and aggregation. Backend
selection, oracle comparisons and timing belong to nnbench.

[isb/protocol.py](../isb/protocol.py) owns methodology templates and case specialization.
Its `components` describe the touched surface; `write_components` identify the write targets.
Concrete cases publish component requirements in CausaLab's spelling, plus nnbench extensions.
Older/custom write descriptors with unspecified targets publish incomplete requirements
(`required_capabilities: null`); their executable parameters remain available.

The canonical replacement for `attention_value` is `attention_premix`, the o-projection input.
`attention_value_states` names the actual value vectors. Old descriptors normalize through
upstream's deprecated-alias mapping so the retired spelling retains its original meaning.

The shared vocabulary is larger than nnbench's implemented methodology inventory. Validating
a descriptor against it does not create a cell or claim backend coverage. The local operation
labels (`read`, `write`, `grad`) summarize nnbench execution. Full CausaLab document validation,
metric lowering, engine routing, training and artifact handling remain CausaLab responsibilities.

## 4. Generation and scientific scope

CausaLab's generated frame addresses a continuation with a decode budget. The nnsight engine
executes one generate trace and walks decode occurrences with `tracer.iter`; document writes
remain prefill-only. nnbench's generation patching also injects during prefill and observes the
continuation. Generation steering writes on decode steps and therefore retains the
`decode_step_write` extension.

The current
[protocol corpus](https://github.com/goodfire-ai/causalab/tree/8696e04bfb06a169defe1bf563d8aeef992f85cd/tests/protocols)
contains documents 01–14, including variable-anchor generation, random-subspace controls, and
multi-position patching. The
[demos](https://github.com/goodfire-ai/causalab/tree/8696e04bfb06a169defe1bf563d8aeef992f85cd/demos)
provide method/application and workflow examples. These are sources for future explicit
benchmark cases. This alignment change does not run that corpus or establish numerical
equivalence between nnbench and CausaLab.

The former Python analysis/method stack and Hydra harness are historical. Active nnbench design
uses the protocol/engine seam. Pyvene remains related work only.

## 5. Ownership, provenance and verification

The frozen experiment and shared worker provenance record metadata coverage. Built-in methods
have descriptors; custom methods supply one or a nonempty `protocol_absence_reason`.
Legacy undescribed experiments receive a legacy reason on restoration.

Protocol coordinates carry the vocabulary reference used to describe them. Methodology
templates are specialized after effective dataset defaults and task values are combined.
Provenance is a detached snapshot; each invocation owns its nested parameter values, with
trial setup outside the measured interval.

Verification has two distinct parts: the source checker establishes the pinned upstream
vocabulary, while no-GPU tests exercise canonical aliases, component read/write requirements,
case specialization, serialization and configuration ownership. Model execution and
cross-engine parity require separate benchmark runs.
