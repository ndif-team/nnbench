"""activation-patching specs, every model.

Interactive, multi-trace: the workload is a SET of length-matched clean/corrupted pairs (the IOI
task), each a unit the cell consumes via two single-prompt traces. The driver aggregates the verdict
over all pairs (top-1 fraction + mean TV), so a backend right on one pair but wrong on another is
caught — a benchmark, not a single-pair anecdote. Baseline = patch=False (corrupted run, no
transplant); effect-size = TV(unpatched, patched) aggregated over the pairs on the control.
"""
from ..data import DataRef
from ..sweep.spec import BaselineSpec, CellConfig, EffectSpec, ExecutionRegime
from ._models import BF16, GPT2, QWEN25_7B, QWEN25_14B

# data is a named, swappable pair source; each unit is one (clean, corrupted) trace.
# Default: the MIB circuit-track IOI snapshot (data/mib/README.md) — clean prompt +
# s2_io_flip counterfactual, BPE-length-matched. The template bank stays available
# via --data ioi_pairs.
_PAIRS = DataRef("mib/ioi", 40)
# Large models: all 1000 MIB IOI pairs are verified BPE-length-matched under the Qwen2.5 tokenizer,
# which the patch cell's shape check requires.
LARGE_PAIRS = DataRef("mib/ioi", 16)

# --- smoke (gpt2) -------------------------------------------------------------------------------

activation_patching_gpt2 = CellConfig(
    name="activation_patching_gpt2",
    methodology="activation_patching", family="gpt2", repo=GPT2,
    # each unit is a (clean, corrupted) pair; aggregate over the set (the driver runs each pair as its
    # own two-trace patch and stacks the verdict, exactly like per-prompt aggregation for reads).
    regimes=[ExecutionRegime("interactive", _PAIRS, aggregate=True)],
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

# --- parallel (Qwen2.5-14B, GT2) ----------------------------------------------------------------

activation_patching_qwen = CellConfig(
    name="activation_patching_qwen",
    methodology="activation_patching", family="llama", repo=QWEN25_14B,
    # each unit is a (clean, corrupted) IOI pair; the driver runs each pair as its own two-trace
    # patch and aggregates the equivalence verdict over the set
    regimes=[ExecutionRegime("interactive", LARGE_PAIRS, aggregate=True)],
    tasks=[
        ({"layer": 8, "residual": "plain"}, "layer=8"),
        ({"layer": 24, "residual": "plain"}, "layer=24"),
    ],
    baseline=BaselineSpec(params={"patch": False, "residual": "plain"}),
    effect=EffectSpec(
        baseline_params={"patch": False, "residual": "plain"},
        perturbed_params={"layer": 24, "residual": "plain", "patch": True},
    ),
    dtype_control=BF16,
)

# --- comparison (Qwen2.5-7B, cross-system) ------------------------------------------------------

cmp_activation_patching = CellConfig(
    name="cmp_activation_patching",
    methodology="activation_patching", family="llama", repo=QWEN25_7B,
    regimes=[ExecutionRegime("interactive", LARGE_PAIRS, aggregate=True)],
    tasks=[
        ({"layer": 7}, "clean residual into corrupt run at layer 7"),
        ({"layer": 21}, "clean residual into corrupt run at layer 21"),
        ({"layer": 7, "positions": "last"}, "clean last-token residual into corrupt run at layer 7"),
        ({"layer": 21, "positions": "last"}, "clean last-token residual into corrupt run at layer 21"),
    ],
    baseline=BaselineSpec(params={"patch": False}),
    effect=EffectSpec(baseline_params={"patch": False},
                      perturbed_params={"layer": 21, "patch": True}),
)
