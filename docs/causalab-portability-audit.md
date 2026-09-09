# causalab and nnbench: the structural join

> Status: rewritten 2026-08-26 against causalab's protocol refactor (PR #20, `can/protocol-refactor`)
> and the generation frame (PR #40, `can/generate-frame`). The previous version of this file
> (2026-06-12) decomposed eight `analyses/*` programs at `bf15b353`; those programs, the
> `methods/` Python, the pyvene wrapper in `neural/`, and the Hydra configs were all deleted by #20,
> so that version is history. This version is a design document: it fixes how the two projects
> relate and what each takes from the other. It contains no run results; the applicability
> verdicts for the documents below are produced by the evaluation, which has not been run.
> The first nnbench-side refactor landed 2026-09-02: protocol semantics, execution regimes, and
> realization parameters are now distinct data structures and are stamped into provenance.

## 1. What causalab is now

causalab executes **intervention-protocol documents**: JSON with a fixed section layout
(`model`, `data`, `positions`, `sites`, `featurizers`, `params`, `reads`, `writes`,
`intervened_models`, `metrics`, `train`, `save`), sweep wrappers that expand to points, a
canonical form with digests, and a validation checklist (`docs/intervention_protocol.md`).

The model-access seam is one class, `causalab/protocol/backend.py`:

```python
class Backend(abc.ABC):
    name: str
    capabilities: frozenset[str]   # subset of the closed set below
    is_local: bool
    def execute(self, request: ExecutionRequest) -> RunResult: ...
```

`requires(doc)` derives the capability set a document needs from its content; `choose_backend`
picks the first backend whose set covers it. The closed capability vocabulary:

| capability | required when |
|---|---|
| `grad` | a `train` section is present |
| `paired_forward` | a write's operand is a read on a different `input` than the write's model |
| `full_logits` | a full `lm_head` read is saved, or a `top_k`/`class_probs` metric |
| `generate` | any position carries `generated` (#40) |
| `writable_attention_probs` | a write targets `attention_probs` |
| `pytorch_fn_local` | any `pytorch_fn` write |

One backend exists: `causalab/neural/pytorch_hooks/` (native HF hooks). The spec names nnsight,
Megatron, and SGLang as intended future backends and carries a guessed reference matrix for them
(§8). Above documents sits a workflow layer (steps = documents plus select/plot), where each step
is routed independently; that is where a fit-on-one-backend, apply-on-another composition lives.

#40 adds generation as a **position frame**: a greedy decode of derived depth, writes applied in
the prefill only, reads addressing generated tokens by the ordinary anchor grammar, distributions
materialized only where a save or metric needs them.

## 2. The relation

Both projects have the same unit: **one program on one backend**. nnbench's cell is
(methodology, family, backend); causalab's is (document, backend), with family inside the
document's `model`. Cross-backend composition never enters either unit: in causalab it is a
workflow of single-backend steps, and nnbench has no cross-backend cell.

From that:

- **The semantic part of an nnbench benchmark case is a causalab document** (the subset that
  describes what is intervened where: `data`, `positions`, `sites`, `reads`, `writes`,
  `intervened_models`, `featurizers`). `ExecutionRegime` supplies a separate context coordinate.
  nnbench keeps its own axes on top: context (family, backend, dtype, TP/PP layout, workload
  regime), realization (which spelling of each element a cell uses), status (the oracle's state
  label), and perf.
- **causalab owns its scientific document axes:** `metrics`, `causal_model`,
  `save`, canonical form and digests, `ArtifactIdentity`, the workflow layer, dataset resolution.
- **Explicit nnbench cells implement each backend's intervention.** Protocol descriptions
  specify the computation. CausaLab's `Backend`, `SiteResolver`, mechanisms, and planner own
  its document-to-hook implementation.
- **How a backend realizes a document is the backend's concern.** Two-trace transfer, bounded
  iteration, fused-residual reads, `no_grad` wrapping, snapshot saves: these live inside an
  nnsight `Backend`. nnbench's realization axis records the spellings it measures.
- **The interface between the projects is a published description, one direction.** nnbench
  reports, per (backend, context), which document tuples run and in what state. A causalab
  backend author uses it to guide implementation choices.

## 3. Vocabulary: intersection and differences

Same concept, two spellings (nnbench adopts causalab's):

| concept | nnbench (design.md §3) | causalab (protocol §2) |
|---|---|---|
| observe a value | `read` × site × scope position | `reads`: `(site, pos, model, input)` |
| counterfactual value | `write` × site | `writes` with a `do` mechanism |
| derivative | `grad` op, backward region | `train`; capability `grad` |
| address | Level 1 module-boundary sites | `sites` component vocabulary + `layer/head/expert/stream` |
| subspace or direction address | derived address `(site, R)` | `featurizer` on a write (`subspace`, `gate`) |
| token position | position coordinate of a site | `positions` (`index`, `span`, `variable`, `column`, `all`) |
| decode step | `step` quantifier | `generated` frame |
| address loop | `sweep` quantifier | `{"sweep": ...}` wrappers |
| inputs | `dataset` quantifier | `data`: `base`, optional `counterfactual[j]`; everything per-row is a column |
| run ordering with data dependence | run DAG | `intervened_models` graph (acyclic) |
| cross-run transplant | transplant edge | operand read on another `input` (`paired_forward`) |
| injection | injection edge | `add_scaled`/`swap` with a literal or `params` operand |
| accumulation | accumulation edge | `train` |

Write mechanisms: nnbench has one `write` op; causalab's closed `do` set (`swap`, `add_scaled`,
`lerp`, `affine`, `gaussian`, `renormalize`, `clamp`, `pytorch_fn`) names the function applied
before the write. nnbench cells already compute several of these inline (ablation, steering,
patching). The names are adopted as the granularity at which "which document tuples run on which
backend" is stated. `pytorch_fn` is what every nnbench cell body is; nothing to add there.

nnbench only (stays nnbench-side):

- context: backend, family, dtype, TP/PP layout, workload regime, versions
- realizations (§3.3): in-place vs replacement, bounded vs unbounded iteration, barrier vs
  two-trace, `out[0]` vs fused sum, alias vs snapshot, module call vs weight matmul
- status vocabulary and the oracle; perf
- sites causalab does not name: engine tier (`samples`, `tracer.result`), module-internal
  `.source.<op>`, `.input`/`.inputs` at arbitrary depth
- constructs: `adaptive`, `session`, staging (`edit`), `skip`, sync, decode-step writes

causalab only (stays causalab-side):

- `metrics`, `causal_model`, `save`, digests, `ArtifactIdentity`, workflow, dataset resolution
- sites nnbench has no cell for yet: `attention_value`, `router_logits`, `expert_output`
- `dims`, the featurizer error-term contract, `params`
- capability routing and the `Backend` class

Metrics versus measurements: a causalab metric is a per-example scalar from one read and the
dataset's columns, and it is the experiment's output. An nnbench measurement compares the same
workload's saved outputs across two backends (top-1 agreement, TV distance, then a state label).
The oracle can compare a document's saved reads or its saved metric values; nnbench needs no
metric vocabulary of its own.

## 4. The corpus as nnbench workloads

causalab's golden corpus (`tests/protocols/01..11` at `can/generate-frame`) is the concrete
workload set. One row per document; the columns are the tuple that determines which nnbench
elements a backend must realize. Verdict columns are deliberately absent.

| doc | what it is | components | `do` | positions | capabilities | nnbench elements |
|---|---|---|---|---|---|---|
| 01 harvest | reads at 2 layers × 2 positions, no writes | `block_output` | none | `index`, `variable` | none | read, live-out |
| 02 interchange | swap counterfactual residual into base at one layer | `block_output`, `lm_head` | `swap` | `index` | `paired_forward` | transplant edge, replacement write |
| 03 path patching | sender to receiver with two attention outputs frozen; three intervened models | `attention_value`, `block_input`, `attention_output`, `lm_head` | `swap` ×4 | `index` | `paired_forward` | run DAG of depth 3; sites without an nnbench cell |
| 04 DAS | interchange through a trained Cayley rotation | `block_output`, `lm_head` | `swap` via `subspace` featurizer | `index` | `grad`, `paired_forward` | accumulation edge, derived address |
| 05 DBM | interchange through a trained gate | `block_output`, `lm_head` | `swap` via `gate` featurizer | `index` | `grad`, `paired_forward` | accumulation edge |
| 06 hydra effect | resample-ablate one attention layer, inject downstream contributions; five intervened models | `attention_output`, `block_output`, `lm_head` | `swap` ×5 | `index` | `paired_forward` | run DAG of depth 3, many models |
| 07 locate scan | 02 swept over layers × 2 positions (64 points) | as 02 | `swap` | sweep over `index`/`variable` | `paired_forward` | sweep quantifier; compute sharing across points |
| 08 DAS sweep | 04 swept over k × seed, site from an artifact | as 04 | | `artifact` ref | `grad`, `paired_forward` | as 04 |
| 09 DAS apply | 04 with the rotation loaded from a file, no `train` | `block_output`, `lm_head` | `swap` via loaded featurizer | `index` | `paired_forward` | meta-compute (rotation on the worker) ∘ replacement write |
| 10 task-table IIA | 02 over a task-generated table with a `column` position | as 02 | `swap` | `index`, `column` | `paired_forward` | as 02 |
| 11 probe generate | steer add at the last prompt token, greedy-decode 8, top-1 at the last generated token | `block_output`, `lm_head` | `add_scaled` | `generated` + `index` | `generate` | injection edge, bounded step quantifier, engine-site read per step |

What the corpus exercises that nnbench's current cells do not: `attention_value` and
`block_input` sites (03), intervened-model graphs deeper than two (03, 06), a featurizer loaded
from an artifact (09), `variable`/`column` positions (01, 07, 10). What nnbench's cells exercise
that no document can express: writes at decode steps (`gen_steering`), TP/PP as a context axis,
engine-tier and module-internal sites.

## 5. What changes in nnbench

1. **Semantic index: implemented first slice.** `isb/protocol.py` defines an
   `InterventionSpec` using causalab's data-role/component/read/write/mechanism/position/
   featurizer/capability vocabulary. `TaskSpec` partitions concrete params into `semantics` and
   `realization`; their values enter the explicit cell. A later metadata exporter may emit full
   causalab documents. Explicit cells retain ownership of trace code.
2. **Execution regime: implemented.** The old `Workload` class is now `ExecutionRegime` and owns
   only input units, interactive/batched/generation shape, decode length, and aggregation. This
   corrects the earlier shorthand “workload = document”: semantic program and systems regime are
   orthogonal axes.
3. **Cell indexing: implemented as metadata.** The executable registry remains centered on
   `(methodology, family, backend)`. Its protocol metadata is indexed by component name
   (`block_output`, `attention_output`, `lm_head`, ...) and `do` mechanism; per-family module paths
   stay inside explicit cell bodies.
4. **Catalog: implemented.** `interp-methods-catalog.md` method rows are keyed by protocol tuple
   (component × `do` × position frame × capability) with nnbench realization/extensions alongside,
   replacing footprint tags and the pyvene column as the method index. The primitive inventory
   remains because it diagnoses execution failures. The authoritative template inventory and
   parameter classifications live in `isb/protocol.py`; the catalog links to those definitions.
5. **Data: planned.** Paired feeds can be exposed as `base` + `counterfactual`; per-prompt labels
   (target tokens, positions) become columns. Existing `DataRef` inputs are unchanged in this slice.
6. **Macro tier: planned.** A Macro cell is one corpus document on one backend.

CausaLab retains its scientific metrics, document/artifact identities, workflows, and `Backend`.
nnbench's Docker runner owns its experiment and output checksums.

## 6. What nnbench publishes for causalab

Run provenance now records the protocol coordinates plus each case's separated semantic and
realization coordinates. Per (backend, context), reports can therefore publish the
list of document tuples, each with its state, and for every state
other than SUPPORTED the element and realization coordinate that carries it. A causalab backend
author uses it to set that backend's `capabilities` and to know which realization the
`SiteResolver` and planner must use per site and mechanism. The current integration is a
human-readable description for backend authors.

Methodology metadata is stored as `protocol_template` for case specialization.
Each case's `protocol` reflects its parameters (for example DAS apply versus training). Regime
records include decode length, aggregation, dataset knobs, and effective cases after those knobs
are merged with task parameters. Explicit cells execute the requested computation and supply
observed outcomes.

The frozen Docker description and shared worker provenance record `protocol_coverage`.
Built-in methods have descriptors; custom methods provide a descriptor or an explicit
`protocol_absence_reason`. Undescribed legacy experiments receive a legacy reason on restoration.
Provenance is a detached snapshot, and each cell invocation owns its nested parameter values.

## 7. Open points

- **Position resolution without padding.** #40's decode relies on left padding for a ragged
  batch; vLLM has no padding (continuous batching), so a vLLM `PositionFrame` is per-request.
  Backend-internal, but it is the one place the batched regime question still lives.
- **`pytorch_fn_local` on an in-process vLLM backend.** The closure runs on the worker and is
  source-serialized under TP/PP, so "local" has a different meaning than in the reference backend.
- **Decode-step writes** are outside v1 (writes are prefill-only). `gen_steering` stays an
  nnbench workload with no document form.
- **Cross-model patching** (the retired `source_pipeline`) has no document form either; still
  out of scope for nnbench.
