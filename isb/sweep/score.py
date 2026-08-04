"""Score two runs (design.md §12.10): a pure CPU comparison over previously written run files.

No models, no GPUs: this step loads the run files of a candidate and a reference and
produces the applicability map. Because execution and scoring are decoupled, any pair of runs is
comparable after the fact (engine vs engine, topology vs single-GPU, the same engine across two
nnsight commits), results can be re-scored without re-running, and the map header NAMES both
stacks (nnsight commit, engine version, host/GPU) — the provenance whose absence once turned a
branch skew into a two-day hunt.

The comparison AXIS is derived from provenance: different engine kinds -> the correctness axis
(SUPPORTED / SILENTLY_WRONG vs the reference); same engine kind -> the parallelism/config
equivalence axis (EQUIVALENT / DIVERGENT). An optional control-dtype run (just another executed
run, e.g. the same engine at fp32) disambiguates precision near-ties exactly like the live rerun
used to, but from data.
"""
from __future__ import annotations

from ..runfile import load_run
from ..runner.run import CellResult, disambiguate_precision, evaluate
from ..report import print_map


def _load(out_dir: str, run_name: str):
    return load_run(out_dir, run_name)


def _stack_line(name: str, prov: dict) -> str:
    c, h = prov["client"], prov["host"]
    nn = c.get("nnsight", {})
    gpus = h.get("gpus") or []
    gpu = f"{gpus[0]['name']} x{len(gpus)}" if gpus else "no GPU recorded"
    return (f"  {name}: engine={prov['engine']['kind']}({prov['engine'].get('mode')}) "
            f"nnsight={str(nn.get('commit'))[:9]}{'*' if nn.get('dirty') else ''} "
            f"vllm={c.get('vllm')} transformers={c.get('transformers')} | {gpu} @ {h.get('hostname')}")


def score_runs(spec, out_dir: str, candidate: str, reference: str | None,
               ctl: str | None = None, quiet: bool = False) -> list:
    """The one path from a run file to CellResult rows. With a `reference`, each cell is scored
    against it (the oracle axis derived from provenance); with `reference=None` the same rows
    carry the run's raw execution states and perf, no verdicts: this is how a baseline run, which
    has nothing to compare against, flows through the same pipeline as every other run. Prints
    the map unless `quiet` (the manager computes states for its own rendering)."""
    cand_out, cand_prov = _load(out_dir, candidate)
    ref_out, ref_prov = _load(out_dir, reference) if reference else ({}, None)
    ctl_out = _load(out_dir, ctl)[0] if ctl else None
    meta = cand_out.get(("__meta__",), {})

    same_engine = reference is not None and \
        cand_prov["engine"]["kind"] == ref_prov["engine"]["kind"]
    control = "vllm_async" if same_engine else "hf"      # the oracle's axis selector

    if not quiet:
        print("\n=== stacks compared ===")
        print(_stack_line(f"candidate {candidate!r}", cand_prov))
        if reference:
            print(_stack_line(f"reference {reference!r}", ref_prov))
            cg = [g["name"] for g in cand_prov["host"].get("gpus") or []]
            rg = [g["name"] for g in ref_prov["host"].get("gpus") or []]
            if cg and rg and cg[0] != rg[0]:
                print(f"  NOTE: differing hardware ({cg[0]} vs {rg[0]}) — correctness comparable, "
                      f"perf numbers are not")
            axis = "config/topology equivalence (same engine)" if same_engine \
                else "correctness (cross-engine, reference = ground truth)"
            print(f"  axis: {axis}")
        else:
            print("  no reference: raw execution states, no verdicts")

    results = []
    # union of output keys and meta keys: an errored cell has NO output entry (execute stores
    # outputs only on success), and it must still appear in the map as its ERROR row
    keys = ({k for k in cand_out if k[0] not in ("__meta__", "batched_perprompt")}
            | {k for k in meta if isinstance(k, tuple) and len(k) == 2
               and k[0] not in ("probe", "__effect__")})
    for workload_kind, label in sorted(keys):
        m = meta.get((workload_kind, label), {})
        cell = CellResult(spec.methodology, spec.family, candidate, label,
                          "RAN" if not m.get("error") else "ERROR",
                          latency_s=(m.get("median_latency_ms") or 0) / 1000.0 or None,
                          error=m.get("error"), value=cand_out.get((workload_kind, label)),
                          workload=workload_kind)
        ref_key = (("batched_perprompt", label) if workload_kind == "batched"
                   else (workload_kind, label))
        ref_val = ref_out.get(ref_key)
        if reference:
            evaluate([cell], control=control, ref_override=ref_val)
            if ctl_out is not None and control == "hf":
                disambiguate_precision([cell], ref_val,
                                       lambda c, k=(workload_kind, label): ctl_out.get(k))
        else:
            cell.value = None            # evaluate() would have released the tensor
        results.append(cell)
        if not quiet:
            print_map(spec.methodology, spec.family, f"{label} [{workload_kind}]", spec.repo, [cell])
    return results
