"""activation-patching spec — collapses smoke_patching.py.

Interactive, multi-trace: the workload is a SET of length-matched clean/corrupted pairs (the IOI
task), each a unit the cell consumes via two single-prompt traces. The driver aggregates the verdict
over all pairs (top-1 fraction + mean TV), so a backend right on one pair but wrong on another is
caught — a benchmark, not a single-pair anecdote. Baseline = patch=False (corrupted run, no
transplant); effect-size = TV(unpatched, patched) aggregated over the pairs on the control.
"""
from ..data import DataRef
from ..sweep.spec import BaselineSpec, CellConfig, EffectSpec, Workload

# data is a named, swappable pair source; each unit is one (clean, corrupted) trace.
# Default: the MIB circuit-track IOI snapshot (data/mib/README.md) — clean prompt +
# s2_io_flip counterfactual, BPE-length-matched. The template bank stays available
# via --data ioi_pairs.
_PAIRS = DataRef("mib/ioi", 40)

activation_patching_gpt2 = CellConfig(
    name="activation_patching_gpt2",
    methodology="activation_patching", family="gpt2", repo="openai-community/gpt2",
    # each unit is a (clean, corrupted) pair; aggregate over the set (the driver runs each pair as its
    # own two-trace patch and stacks the verdict, exactly like per-prompt aggregation for reads).
    workloads=[Workload("interactive", _PAIRS, aggregate=True)],
    tasks=[
        ({"layer": 3, "residual": "plain"}, "layer=3"),
        ({"layer": 9, "residual": "plain"}, "layer=9"),
    ],
    baseline=BaselineSpec(params={"patch": False, "residual": "plain"}),
    effect=EffectSpec(
        baseline_params={"patch": False, "residual": "plain"},
        perturbed_params={"layer": 9, "residual": "plain", "patch": True},
    ),
    # The two-trace cross-prompt patch (whole-tuple replace) is the documented vLLM-correct recipe and
    # is faithful at fp32 (top1=1.00, tv≈0.001); at the bf16 default the patched top-1 flips on a
    # near-tie -> precision degradation, not a bug (the single-forward patch matches HF at fp32 but
    # flips a bf16 near-tie, separated by the dtype control).
)
