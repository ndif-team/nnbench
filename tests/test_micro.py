"""Micro-tier registry invariants (no GPU)."""
from isb.micro.probes import PROBES, names_for
from isb.states import AppState

# every inventory row gets exactly one probe per backend; iteration and session are
# split by realization (bounded/unbounded, saved/un-saved flow) — the Level-1.5 splits
# the bug diagnosis surfaced (nnsight docs/developing/vllm-construct-gaps.md)
EXPECTED = {
    "input_boundary", "engine_logits", "engine_samples",
    "derived_head", "derived_neuron", "source_mlp",
    "iteration_bounded", "iteration_unbounded",
    "scan", "edit", "barrier",
    "session_saved", "session_unsaved",
}


def test_both_backends_cover_the_inventory_rows():
    for backend in ("hf", "vllm_async"):
        assert set(names_for(backend)) == EXPECTED, backend


def test_vllm_order_is_safest_first():
    # constructs with hang risk (multi-invoke / session / edit on the async loop)
    # must come AFTER the plain single-trace site probes; the session probes
    # (undrained-trace hang risk) dead last
    order = names_for("vllm_async")
    assert set(order[-2:]) == {"session_saved", "session_unsaved"}
    assert order.index("barrier") > order.index("iteration_unbounded")
    risky = {"barrier", "session_saved", "session_unsaved", "edit", "scan",
             "iteration_bounded", "iteration_unbounded"}
    first_risky = min(order.index(n) for n in risky)
    assert all(order.index(n) < first_risky for n in EXPECTED - risky)


def test_probe_watchdog_marks_hang_and_error():
    from isb.micro.run import _run_one

    def hangs(be, model):
        import time
        time.sleep(5)
        return AppState.SUPPORTED, ""

    def raises(be, model):
        raise RuntimeError("boom\nlast line wins")

    state, note, _ = _run_one(hangs, None, None, timeout_s=0.2)
    assert state == AppState.HANG
    state, note, _ = _run_one(raises, None, None, timeout_s=5)
    assert state == AppState.ERROR and note == "last line wins"
