# The interpretability program model — methodologies as programs over model dataflow

> **STATUS: DRAFT — LARGELY ADOPTED (2026-06-12).** A from-first-principles account of the
> abstraction layer itself: what *language* the primitives + data flow + control flow form, and
> how methodologies are programs in it. Backend/engine implementation is deliberately ONE short
> section at the end — that dimension is already covered by the inventory
> (`interp-methods-catalog.md`) and is not what this draft is about.
>
> **Adopted into normative `design.md` §3 on 2026-06-12** (via
> `drafts/design-revision-2026-06-12.md`): the boundary-crossing criterion for data primitives
> (§1 → design §3.1); the quantifier set incl. sweep (§2 → design §3.1, which additionally names
> run and sync, with scope position defined); the source→destination edge set incl. rewiring +
> accumulation (§3 → design §3.4);
> base programs × transformations and syntactic footprints (§4 → design §3.5); laws, the
> regime-effect recast as model-side law failure, and the three-way attribution (§5 → design
> §3.6/§3.7); the roadmap cells (§7 item 5 → catalog roadmap). Open questions decided: Q1 —
> grad stays a primitive; Q4 — subspace/direction-valued addresses become derived-tier Level-1
> citizens (design §3.2). **Still open: Q2 (mid-run adaptivity) and Q3 (token alignment
> conventions — flagged as a real silent-error source the oracle does not currently see).**
> Where this draft and design.md diverge from here on, design.md is normative.

## 1. Two levels of computation

Every interpretability experiment involves two computations that must not be conflated:

**The object level** — a model execution (a *run*). For a fixed model and input, the run is a
dataflow graph: nodes are intermediate tensors, organized by coordinates
`(module-path × token-position × generation-step [× batch-member])`. Generation unrolls the
graph in steps, with the sampled token feeding the next step. The model's *parameters* are
constants of this graph. The graph also has a derivative extension (the backward graph) once a
scalar metric is fixed. **Addresses** (design.md §3.2's site tiers) are coordinates into this
graph — nothing more.

**The meta level** — the experiment itself: ordinary computation (Python, math, state) that
*instantiates runs, exchanges values with them, and decides what to do next*. A methodology is a
meta-level program. The scientist's deliverable — a patching score, a lens trajectory, a trained
probe — is the meta program's output.

The entire interface between the two levels is three boundary-crossing primitives:

```
v = read(r, addr)            observe the value at an address of run r
    write(r, addr, v)        substitute: run r continues as if addr had value v
g = grad(r, addr; metric)    observe the derivative graph of run r at addr
```

Everything else is ordinary computation on one side or the other. This gives Level 0 a
*criterion* instead of a list: **a data primitive is exactly an operation that crosses the
object/meta boundary.** `COMPUTE` is not a primitive (it is meta-level code — including applying
the model's own modules as pure functions, since parameters are readable constants); `save` is
not a primitive (it is the meta program's output); `grad` is a read on the derivative graph.

A `write` deserves one definitional sentence: its meaning is the **counterfactual run** — the
run in which that address has that value and everything downstream follows. That is the entire
semantics; how an implementation achieves it is below the language.

## 2. Control flow: where ops fire

Meta-level control flow determines *which runs exist* and *where in their coordinates* the
boundary crossings attach. Four quantifiers and one scheduler cover everything in the catalog:

| construct | quantifies over | example | nnsight surface |
|---|---|---|---|
| **step** | the run's unrolled time | per-step steering | `tracer.iter[...]` |
| **sweep** | addresses | "for every layer ℓ" — the most common control construct in real methods | a Python loop |
| **dataset** | inputs (independent runs) | patching over 100 pairs; mean-ablation statistics | prompt lists / `invoke` |
| **adaptive** | future runs chosen from past results | ACDC pruning, searches, training loops | Python `while`/`if` |
| **run DAG** | sequencing runs with data dependence | run B consumes values from run A | `session` / two traces |

Notable: **sweep was invisible in the current model** (filed as a "scaling measure"), yet it is
the control construct most methodologies use most (every layer sweep, every head sweep). It cost
nothing on any backend (it is a host-side loop), which is exactly why it went unmodeled — the
language should name it anyway, because it is the breadth driver for performance and the
enumeration axis for search methods.

## 3. Data flow: the experiment's edges

Every value crossing the boundary has a provenance `(run, address, step)` and a destination.
Classifying meta-level dataflow edges by source → destination scope gives a complete, small set:

| edge | shape | canonical methodology |
|---|---|---|
| **observation** | read → emit | logit lens, probing data collection |
| **rewiring** | read → compute → write, same run, downstream | path patching; SAE splice (`write(a, SAE(read(a)))`) |
| **transplant** | read in run A → write in run B | activation patching / causal tracing |
| **injection** | meta-constant → write | steering, zero/mean ablation |
| **accumulation** | reads across many runs → meta-state → later write or analysis | mean ablation's mean; probe/SAE/DAS training (state = parameters θ) |
| **derivative** | grad → emit/compute | attribution patching, saliency |

This is the abstract origin of what design.md §3.4 calls "cross-edge data movement": those rows
are these edges, viewed from below. At the language level the edges are ordinary variable use —
trivially well-defined. (That they are *hard to implement* on some contexts is the
implementation map's business, §6 — and the reason the bench measures them.)

## 4. Methodologies as programs — and the transformations that relate them

Writing each catalog method as a program makes the footprint *syntactic* (the multiset of
primitives, edges, quantifiers in the text) and exposes algebraic structure the catalog
currently lists as unrelated rows.

```
logit_lens(x):                                    # observation × sweep
  r = run(x)
  for ℓ in layers:                                # sweep
    emit softmax(unembed(norm(read(r, resid[ℓ, last]))))

activation_patching(x_clean, x_corr, A):          # transplant × sweep
  rc = run(x_clean)
  for a in A:                                     # sweep
    h = read(rc, a)
    r = run(x_corr); write(r, a, h)               # transplant edge
    emit metric(read(r, logits))

attribution_patching(x_clean, x_corr, A):         # derivative replaces a sweep of runs
  rc, r = run(x_clean), run(x_corr)
  for a in A:
    emit (read(rc,a) − read(r,a)) · grad(r, a; metric)

steering(x, d, α):                                # injection × step
  r = run(x, steps=N)
  for t in r.steps:                               # step quantifier
    write(r, resid[ℓ, last], read(r, resid[ℓ, last]) + α·d, step=t)
    emit read(r, logits, step=t)

mean_ablation(D, a):                              # accumulation feeding injection
  μ = mean(read(run(x), a) for x in D)            # dataset quantifier → meta-state
  ...; write(r, a, μ); ...

DAS(D, a; θ):                                     # adaptive × accumulation × derivative
  while not converged:                            # adaptive
    for (x, x') in batch(D):
      r = run(x'); write(r, a, R_θ(read(run(x), a), read(r, a)))   # parameterized rewiring
      θ ← θ − η·∇_θ loss(read(r, logits))         # grad THROUGH the splice into meta-state
```

The methods are related by a handful of **program transformations**:

- **step-lifting**: steering → generation-time steering (wrap in the step quantifier);
- **dataset-lifting**: any single-input program → its batched/statistical form;
- **aggregation**: ablation → mean-ablation (insert an accumulation edge);
- **linearization**: patching → attribution patching (replace a sweep of transplant runs with
  one derivative query) — note this one is a *scientific approximation*, not an equivalence;
- **amortization**: logit lens → tuned lens, one-off direction → trained probe/SAE (replace a
  fixed meta-constant with trained meta-state).

So the methodology space is approximately **a few base programs × a few transformations** — the
catalog's five families are not a coincidence. This also says what a *complete* method tier
means: cover each base edge type and each transformation at least once, rather than enumerating
methods (the gen-steering cell, in this language, is precisely "the step-lifting transformation,
verified").

## 5. Laws: what "composition" and "regime effects" actually are

The transformations come with **laws** — claims that transforming the program does not change
what its observations *mean*:

- the **step-lift law**: an intervention valid for one forward stays valid applied per step;
- the **dataset-lift law**: running inputs together (batched) equals running them separately;
- the **sweep-exchange law**: sweeping addresses inside one run equals separate runs per address
  (when reads don't interact);
- the **composition law** generally: independent footprint elements compose.

These laws hold trivially at the language level. **The bench's method tier is, precisely, an
empirical test of which laws survive each implementation context.** Measured so far: the
step-lift law verified exactly (write × bounded-step, F-17); the dataset-lift law *fails for
absolute-position families* (batched GPT-2 left-padding, the position findings) — i.e. what
design.md §3.6 files as a "regime effect" is, in language terms, **a lifting law failing for the
model itself**, not for the implementation. And composition is known NOT to be free where
footprint elements share implementation state (the batched-multitoken saves clobbering and the
PP iteration hang in the nnsight repo's findings) — those pairs are exactly where composition
cells are worth building.

This recasts the three-way attribution of any red cell: a law can fail because of the **model**
(absolute positions — the scientist's lift was invalid), the **implementation** (shared-state
interaction, dropped transport), or **numerics** (bf16 near-ties; the fp32 control decides).
Three different parties, three different fixes; the oracle machinery already distinguishes them.

## 6. The implementation map (subordinate, one section on purpose)

nnsight is an embedding of this language in Python: `trace` = run scope; `.output`/assignment =
read/write; the backward context = grad; `iter` = the step quantifier; prompt lists / `invoke` =
the dataset quantifier; `session` = the run DAG; plain Python = sweep and adaptivity; `.save()` =
emit; `edit` = a program transformation (install an intervention into all future runs). Each
backend context implements some subset of the primitives, quantifiers, and edges with some
fidelity and cost — **that subset is what `interp-methods-catalog.md` measures**, and nothing
about it belongs in the language. The one structural takeaway worth keeping from the
implementation side: the edges of §3 are where implementations break (they are the parts that
cross real process boundaries), while primitives and sweeps are where they don't — which is why
the inventory's frontier table is the edges table.

## 7. What this changes if adopted (not applied)

1. design.md §3.1 gains the **boundary-crossing criterion** for data primitives (read/write/grad
   cross; compute/save don't — they are meta-side), replacing the current enumerated framing.
2. The **sweep quantifier** becomes a named control construct (currently an unmodeled Python
   loop); breadth becomes its measure.
3. §3.4's edge table gets the source→destination derivation (§3 above) and two missing edge
   types: **rewiring** (same-run read→write downstream — path patching's footprint, currently
   unexercised by any cell) and **accumulation** (meta-state; makes the `trained` tag structural
   instead of a dependency note).
4. §3.6's claims become **laws**: the method tier's job is law-checking per context; regime
   effects reclassify as model-side law failures; composition cells are scoped to
   shared-state pairs.
5. Roadmap: one cell per unexercised edge type and per transformation — concretely, a
   **rewiring cell** (path-patching style), an **accumulation cell** (mean-ablation — also the
   cheapest trained-state proxy), and a **dataset-lift law cell** on a relative-position family
   (the positive control for the measured GPT-2 failure).

## 8. Open questions

1. Is `grad` a third primitive or a read against a second graph? (Operationally it differs —
   the derivative graph may not exist at all — but linguistically "read on G′" is cleaner; the
   draft keeps it a primitive because methodologies treat it as one.)
2. Do adaptive experiments need anything beyond host-side `while` — e.g. *mid-run* adaptivity
   (intervene at layer 20 based on a layer-5 read in the same forward)? Expressible today
   (in-order reads then writes), but it is a rewiring edge with control dependence — worth a
   cell of its own?
3. Where do *tokens* live? Inputs are parameters of `run`, but several methods compute over
   token identity (position selection, span alignment in patching pairs). Probably meta-level
   data, but alignment conventions are a real source of silent scientific error the bench
   doesn't currently see.
4. Is the language's address space the right place for **distributed-unit** methods (subspace /
   direction-valued addresses: DAS rotations, steering directions)? `write(a, R_θ(...))` treats
   them as rewiring at a tensor address; pyvene treats subspaces as first-class units. If
   subspace addresses become Level-1 citizens, the sweep quantifier ranges over them too.
