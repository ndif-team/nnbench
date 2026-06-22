"""Applicability states (design.md §8.1, §12.2)."""


class AppState:
    SUPPORTED = "SUPPORTED"
    SUPPORTED_DEGRADED = "SUPPORTED_DEGRADED"
    ERROR = "ERROR"
    SILENTLY_WRONG = "SILENTLY_WRONG"      # ran, no error, but != reference (the dangerous cell)
    HANG = "HANG"
    UNSUPPORTED = "UNSUPPORTED"            # no cell implemented for this combination
    NO_REFERENCE = "NO_REFERENCE"          # ran, but the per-family HF control failed -> can't judge
    # Parallelism-equivalence axis (bench.py --pp/--tp, control != "hf"). A SEPARATE question from
    # correctness: the (tp,pp) candidate is scored against single-GPU vLLM, NOT against HF, so the
    # verdict is whether parallelism PRESERVES the computation — never whether the computation is
    # correct vs HF. These are deliberately distinct from SUPPORTED/SILENTLY_WRONG so a read that is
    # known-wrong vs HF (e.g. the naive plain-residual lens that drops the fused residual) is never
    # stamped SUPPORTED merely because the parallel engine faithfully reproduces it. The orthogonal
    # vs-HF correctness is reported alongside (CellResult.correctness), not collapsed into this state.
    EQUIVALENT = "EQUIVALENT"              # parallel engine reproduces single-GPU (within the oracle gate)
    DIVERGENT = "DIVERGENT"                # parallel engine diverges from single-GPU — a real parallelism break
    # Within-noise band between the two: the softmax DISTRIBUTIONS match (tv <= tv_tol) but a few
    # near-tie argmaxes flipped (top1 < top1_thresh). On a parallel engine that signature is TP
    # reduction-order non-determinism — the all-reduce sums in a different order than single-GPU, and a
    # sensitive step (e.g. a MoE router's top-k expert selection) flips on a borderline token — NOT a
    # mechanism divergence. The same-precision (both bf16) analogue of SUPPORTED_DEGRADED; treated as a
    # non-surprise (it is equivalent up to numerical non-determinism), but surfaced distinctly so the
    # near-tie is visible rather than rounded up to a clean EQUIVALENT.
    EQUIVALENT_DEGRADED = "EQUIVALENT_DEGRADED"
