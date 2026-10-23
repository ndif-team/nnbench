"""Jacobian lens: the logit lens with a per-layer transport map before the final norm.

The interactive data is the upstream multi-hop lens eval (anthropics/jacobian-lens,
`data/jlens/lens-eval-multihop.json`, 93 items): prompts whose pre-answer position carries a latent
bridge entity. Each prompt runs as its own trace and the verdict aggregates over all of them.

Rows split the transport:
  - identity: the readout IS the plain logit lens at the upstream readout position.
  - seeded orthogonal: the J-matmul code path with a deterministic full-rank map (the same matrix on
    every backend), the J-lens program shape without a fitted artifact.
  - fitted: the model's published lens (`Model.assets["jlens"]`, today only Qwen3.5-4B), over the
    source layers it maps.

Read methodology -> no effect-size guard. Baseline = single-layer identity read (`layers=[-1]`), so
overhead-vs-baseline reflects the full-band sweep + transport cost.
"""
from ..data import DataRef
from ..sweep.spec import Asset, Regime, Row, Workload

jacobian_lens = Workload(
    name="jacobian_lens", methodology="jacobian_lens",
    rows=(
        Row("identity transport (logit-lens readout)", {"transport": None}),
        Row("seeded orthogonal transport (the J-matmul path)", {"transport": 0}),
        Row("fitted transport (the model's published lens)",
            {"transport": Asset("jlens", "transport"), "layers": Asset("jlens", "layers")}),
    ),
    regimes=(Regime("interactive", DataRef("jlens/multihop")),),
    baseline={"transport": None, "layers": [-1]},
    # a dataset study: notebook calls, then evaluation-set sweeps
    load_patterns=("offline",),
)
