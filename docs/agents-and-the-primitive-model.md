# Agents and the primitive model — connection notes (exploration, not decided design)

> Status: direction-exploration written 2026-06-11. **Artifact 1 (the portability audit) is done
> — see `causalab-portability-audit.md` (2026-06-12)**; it also answered open questions 1
> (vocabulary covers the suite; three edge findings) and 5 (as-shipped 0% — nnsight is declared
> but never imported; pyvene-on-HF-eager only). Nothing else here is committed design; design.md
> §3 (the leveled primitive model) and §12 (flat per-cell architecture) remain the decided ground.
> References: `references.md` §B (causalab), nnsight `docs/developing/agent-evals.md`,
> `docs/developing/vllm-construct-gaps.md`.

## The observation

The benchmark's decomposition of interp workloads —

1. **primitives** (Level 0: data primitives read/write/grad — the boundary-crossing criterion;
   COMPUTE is meta-level — plus control quantifiers; cross-edge data movement —
   live-out/loop-carried/fork-join/cross-region/staging — is the named frontier class at Level 2)
2. **address & range** (Level 1: site tiers; a site is a NAME whose denotation is
   context-dependent)
3. **idioms** (Level 1.5: the working realization per backend)
4. **workflow** (Level 3: methodologies as programs with footprints)

— built for *benchmarking* reasons, has exactly the shape an **autonomous interp agent** needs
for planning. causalab (goodfire-ai/causalab) is the concrete case: an agent/skill-driven
causal-abstraction framework (`/setup-task`, `/plan-experiment`, `/run-experiment`,
`/interpret-experiment`; eight analyses baseline → locate → subspace → manifolds →
path_steering → pullback; pyvene + nnsight ≥ 0.5.9).

## The mapping (decomposition ↔ agent concepts)

| benchmark level | agent concept |
|---|---|
| data ops + engine-coupled control ops | atomic tools |
| address space (name vs denotation) | tool arguments, with environment-dependent meaning |
| idioms / realizations | per-environment tool VARIANTS |
| methodology footprint | plan / skill signature |
| measured support map (micro tier) | precondition table the planner checks |
| failure-kind taxonomy (§3.6) | error model for replanning |

Plan compilation for an agent: method → footprint → bind addresses through the family's naming →
select idioms for the context → emit explicit code. The failure taxonomy gives the replanner its
case analysis:

- **operation unsupported** → switch method (no gradient attribution on vLLM → fall back to
  activation patching)
- **realization unsupported** → same plan, swap idiom (replacement not in-place; weight matmul
  not `lm_head(...)`)
- **denotation mismatch** → rewrite the address read (fused residual: `out[0]+out[1]`)
- **regime effect** → change the experiment regime, not the code

**Key empirical anchor:** the things an unconditioned agent gets wrong when writing nnsight×vLLM
code are *literally the measured cells*: in-place writes, direct `lm_head` calls, unbounded
`tracer.iter[:]`, the plain residual read on fused-residual families, barrier-based patching.
The SILENTLY_WRONG cells are a census of agent traps; the map is what an agent must condition on
to avoid them.

## causalab specifically

causalab's agent pipeline plans at the **workflow level** (analysis × task × model) and *assumes
the substrate works*. That assumption is exactly the layer this benchmark measures. Faithfulness
vs systems division of labor stays clean: they verify the science, we verify the substrate. The
sharp example: a causalab interchange intervention silently reading half the residual stream on
vLLM (the fused-residual denotation mismatch) would not fail their pipeline — it would corrupt
their faithfulness *conclusion*. Substrate verification is a precondition for trusting agent-run
science on production engines.

### Artifact 1 — causalab portability audit (recommended FIRST STEP)

Decompose each causalab analysis into a footprint; join footprint ∧ the measured primitive map →
predict per-backend applicability of the whole suite *before running anything*:

- DAS / DBM (trainable rotation/mask) → needs BACKWARD → operation-unsupported on vLLM → HF-only
- `attention_pattern` → READ × internal × attention → site absent under paged attention
- interchange interventions → cross-prompt transplant → portable, but ONLY via the two-trace
  idiom (barrier is broken; silent on sync)
- locate / subspace reads → boundary READ → portable with the fused-residual denotation caveat

Deliverable: "what fraction of a real causal-abstraction suite runs on a production engine, and
the failure kind for the rest" — the OSDI systems story quantified through a real external
consumer instead of our own cells. Cheap: no new harness; it is a footprint-tagging exercise plus
a table join.

It is also a **test of the model itself**: §3.6 claims statuses compose upward; running causalab
end-to-end compares predicted-from-footprint against measured-at-workflow. Disagreements are
either new regime effects (good findings) or holes in the footprint vocabulary (good corrections).

### Artifact 2 — causalab as the Macro tier

The eight analyses are realistic Level-3 programs — exactly what the Macro tier (§2) was waiting
for. End-to-end causalab-on-vLLM runs validate the map's predictions at workflow scale.

## Two derived directions

### Skill cards generated from the inventory

Each Level-2 entry + working idiom per context is a skill card with preconditions. The nnsight
CLAUDE.md gotcha cheat-sheet is the hand-written, decaying version of this; the inventory makes
it **generated and measured**, with the micro tier as CI: an upstream fix lands → the inventory
row flips → the card regenerates. The same artifact serves nnsight's agent-evals: every
non-trivial inventory row converts into an eval item ("write a logit lens for vLLM-Llama" — does
the agent fuse the residual?), with the map as grading ground truth.

### Agent-driven cell authorship (the §12-compatible automation)

The one place automation may touch cell construction: an **agent** (not a code-gen layer) reads
footprint + idioms + family naming, writes the explicit cell, and the existing oracle +
expected-state machinery verifies it. Cells stay flat and reviewable — the §11/§12 invariant
holds — but authorship scales with the matrix. causalab's pipeline maps onto this loop almost
skill-for-skill: `/plan-experiment` ≈ footprint planning, `/run-experiment` ≈ our runner,
`/interpret-experiment` ≈ our oracle verdicts.

## The caution (same lesson as the Resolver)

Two ways to "help" an agent: hand it a **universal abstraction** (pyvene surface,
StandardizedTransformer) or hand it a **knowledge base + explicit examples**. The abstraction
route makes the agent inherit the abstraction's silent failures — and on a backend like vLLM,
that is exactly where denotation mismatches hide. The decomposition is valuable to agents
precisely because it is the second kind: it makes tribal knowledge legible without pretending
contexts are uniform. Any agent integration must preserve that property (agents consume the map
and produce explicit cells; no layer that "makes the differences disappear").

## Open questions for the next session

1. Footprint coverage: does the Level-2 vocabulary actually cover causalab's eight analyses, or
   do they force new entries (subspace/rotation interventions, path/edge granularity)? (This is
   the open question about the leveled model anyway — artifact 1 answers it empirically.)
2. Where do skill cards live — bench repo (generated docs) vs nnsight repo (agent-facing docs) —
   and what keeps them in sync with the micro tier?
3. Is the agent-eval generation (inventory row → eval item) a bench deliverable or an nnsight
   `agent-evals` deliverable?
4. For agent-authored cells: what is the review gate — oracle pass + expected-state encode +
   human review, or something stricter for write methodologies (effect-size guard mandatory)?
5. Does causalab's pyvene surface itself port to vLLM at all (pyvene wraps HF modules)? If not,
   the portability audit's headline may be "0% as-shipped, N% re-expressed in nnsight primitives"
   — which is itself the finding.
