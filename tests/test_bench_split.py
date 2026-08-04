"""Split-run command construction (isb/sweep/split.py); no GPU, pure argv logic.

Pins the contract the orchestrator hands to execute.py subprocesses: the reference run comes
first (hf when present) and every row writes its own run file into the shared out dir; each
backend can run under its own interpreter (the per-env map); the serve URL reaches only the
serve run; the control-dtype vLLM run is appended only when the reference is hf; PP/TP rows
pin the control dtype and memory knobs on both sides.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isb.sweep.split import (  # noqa: E402
    backend_run_commands,
    ctl_run_name,
    parallel_run_commands,
    run_base,
)


def _opt(argv, flag):
    return argv[argv.index(flag) + 1]


def test_reference_run_first_all_rows_share_the_out_dir():
    rows = backend_run_commands("execute.py", "spec_x", ["hf", "vllm_async"], "runs_d")
    assert [b for b, _, _ in rows] == ["hf", "vllm_async"]
    (b0, n0, argv0), (b1, n1, argv1) = rows
    assert n0 == "spec_x-hf" and _opt(argv0, "--engine") == "transformers"
    assert n1 == "spec_x-vllm" and _opt(argv1, "--engine") == "vllm"
    for argv, name in ((argv0, n0), (argv1, n1)):
        assert _opt(argv, "--out") == "runs_d" and _opt(argv, "--name") == name


def test_hf_is_reference_regardless_of_order_else_first():
    rows = backend_run_commands("e.py", "s", ["vllm_async", "hf", "vllm_sync"], "d")
    assert rows[0][0] == "hf"                                     # hf preferred as reference
    assert {b for b, _, _ in rows[1:]} == {"vllm_async", "vllm_sync"}
    rows = backend_run_commands("e.py", "s", ["vllm_sync", "vllm_async"], "d")
    assert rows[0][0] == "vllm_sync"                              # no hf -> first backend
    assert _opt(rows[0][2], "--engine-mode") == "sync"


def test_python_map_selects_per_backend_interpreter():
    rows = backend_run_commands("e.py", "s", ["hf", "vllm_async"], "d",
                                python_map={"hf": "/envs/tf/bin/python"})
    assert rows[0][2][0] == "/envs/tf/bin/python"                 # mapped backend
    assert rows[1][2][0] == sys.executable                        # default: this interpreter


def test_serve_url_reaches_only_the_serve_run():
    rows = backend_run_commands("e.py", "s", ["hf", "vllm_serve", "vllm_async"], "d",
                                serve="http://srv:6677")
    by = {b: argv for b, _, argv in rows}
    assert "--host" not in by["hf"] and "--host" not in by["vllm_async"]
    assert _opt(by["vllm_serve"], "--host") == "http://srv:6677"
    assert _opt(by["vllm_serve"], "--deployment") == "serve"
    raised = False
    try:
        backend_run_commands("e.py", "s", ["vllm_serve"], "d")    # serve without a URL
    except ValueError:
        raised = True
    assert raised, "vllm_serve without a server URL must raise"


def test_ctl_run_appended_only_when_reference_is_hf():
    rows = backend_run_commands("e.py", "s", ["hf", "vllm_async"], "d", ctl_dtype="float32")
    assert rows[-1][0] == "ctl" and rows[-1][1] == "s-vllm-fp32"
    argv = rows[-1][2]
    assert "dtype=float32" in argv and _opt(argv, "--engine") == "vllm"
    # same-engine comparison (no hf reference): precision disambiguation does not apply
    rows = backend_run_commands("e.py", "s", ["vllm_sync", "vllm_async"], "d", ctl_dtype="float32")
    assert all(b != "ctl" for b, _, _ in rows)
    # hf alone: nothing to disambiguate
    rows = backend_run_commands("e.py", "s", ["hf"], "d", ctl_dtype="float32")
    assert all(b != "ctl" for b, _, _ in rows)


def test_data_binding_reaches_children_and_names_runs():
    rows = backend_run_commands("e.py", "spec", ["hf", "vllm_async"], "d", data="jlens/poetry:16",
                                ctl_dtype="bfloat16")
    assert run_base("spec", "jlens/poetry:16") == "spec@jlens-poetry"
    assert rows[0][1] == "spec@jlens-poetry-hf"
    assert rows[-1][1] == ctl_run_name("spec", "jlens/poetry:16", "bfloat16") == "spec@jlens-poetry-vllm-bf16"
    for _, _, argv in rows:
        assert _opt(argv, "--data") == "jlens/poetry:16"


def test_parallel_rows_pin_dtype_and_memory_on_both_sides():
    rows = parallel_run_commands("e.py", "s", "d", pp=2, tp=2, executor="ray",
                                 gpu_mem=0.4, max_model_len=1024, dtype_control="bfloat16")
    (l0, n0, argv0), (l1, n1, argv1) = rows
    assert n0 == "s-tp1pp1" and n1 == "s-tp2pp2"
    for argv in (argv0, argv1):                       # only the topology may vary
        assert "dtype=bfloat16" in argv
        assert "gpu_memory_utilization=0.4" in argv
        assert "max_model_len=1024" in argv
    assert "tensor_parallel_size=2" in argv1 and "pipeline_parallel_size=2" in argv1
    assert "distributed_executor_backend=ray" in argv1
    assert all("parallel_size=2" not in a for a in argv0)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
