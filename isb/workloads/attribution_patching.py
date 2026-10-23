"""attribution patching: gradient-based attribution of the IOI logit difference per layer.

Interactive, multi-trace: the data is a set of labeled (clean, corrupted, answers) units from the MIB
IOI snapshot; each unit is its own two-trace attribution (`be.attribute`) whose per-item answers feed
the logit-difference metric, and the [n_layers] attribution vectors stack across units for the
verdict. Baseline = `grad=False` (forward-only metric, no backward — the overhead denominator, and the
part that also runs on vLLM); the row does the full forward+backward attribution. vLLM runs
inference-mode (no autograd), so the backward raises: the `grad` frontier is HF-only.
"""
from ..data import DataRef
from ..sweep.spec import Regime, Row, Workload

attribution_patching = Workload(
    name="attribution_patching", methodology="attribution_patching",
    rows=(Row("attribute the logit difference to every layer", {"grad": True}),),
    regimes=(Regime("interactive", DataRef("mib/ioi_labeled", 20)),),
    baseline={"grad": False},
    # a dataset study
    load_patterns=("offline",),
)
