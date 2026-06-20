"""Reporter — the applicability map (design.md §8.1, primary deliverable)."""
from __future__ import annotations


def print_map(methodology: str, family: str, label: str, repo: str, cells) -> None:
    print("\n=== Applicability map ===")
    title = f"{methodology}  family={family}"
    if label:
        title += f"  [{label}]"
    print(f"task  : {title}")
    print(f"model : {repo}")
    # A --pp/--tp run reports the PARALLELISM-EQUIVALENCE axis: `state` is EQUIVALENT/DIVERGENT vs
    # single-GPU vLLM, a DIFFERENT question from vs-HF correctness. Make that explicit and show the
    # declared vs-HF correctness as its own dimension, so a known-wrong read reads "EQUIVALENT under
    # tp, SILENTLY_WRONG vs HF" rather than being collapsed onto one SUPPORTED label. The
    # default-success state is then EQUIVALENT (not SUPPORTED), so only a non-default expectation is
    # annotated.
    gt2 = any(getattr(c, "correctness", None) is not None for c in cells)
    default_ok = "EQUIVALENT" if gt2 else "SUPPORTED"
    if gt2:
        print("axis  : parallel-equivalence — candidate (tp,pp) vs single-GPU vLLM; "
              "'vs-HF' = declared correctness, NOT measured here")
    print("-" * 86)
    print(f"{'backend':<14}{'actual':<22}{'vs expected':<24}{'latency':<9}{'metrics / note'}")
    print("-" * 86)
    for c in cells:
        lat = f"{c.latency_s:.2f}s" if c.latency_s is not None else "-"
        if c.metrics:
            note = (
                f"top1={c.metrics.get('top1_agree', 0):.2f} "
                f"tv={c.metrics.get('tv', float('nan')):.3f} "
                f"maxabs={c.metrics.get('max_abs', float('nan')):.2f}"
            )
        else:
            note = c.error or ""
        corr = getattr(c, "correctness", None)
        if corr is not None:                         # the orthogonal vs-HF axis (declared), GT2 runs only
            note = f"{note}   vs-HF: {corr}".strip()
        # the delta column: ✓ when actual matches the declared expectation, ⚠ when it doesn't
        if getattr(c, "surprise", False):
            vs = f"⚠ SURPRISE (exp {c.expected})"
        elif c.expected is not None and c.expected != default_ok:
            vs = f"✓ expected {c.expected}"          # a known frontier/degraded cell, as documented
        else:
            vs = "✓"
        print(f"{c.backend:<14}{c.state:<22}{vs:<24}{lat:<9}{note}")
    print("-" * 86)


def print_perf(methodology: str, family: str, label: str, repo: str, cells) -> None:
    """Performance table (design.md §8.2): latency median±std, peak GPU mem, overhead vs the
    no-intervention baseline, and throughput where the workload generates it. Only cells that were
    timed (perf populated) appear."""
    timed = [c for c in cells if c.perf is not None]
    if not timed:
        return
    print("\n=== Performance ===")
    title = f"{methodology}  family={family}"
    if label:
        title += f"  [{label}]"
    print(f"task  : {title}")
    print(f"model : {repo}")
    print("-" * 86)
    print(f"{'backend':<14}{'latency (ms)':<20}{'peak GB':<9}{'overhead×':<11}{'throughput':<14}")
    print("-" * 86)
    for c in timed:
        p = c.perf
        lat = f"{p.median_latency_ms:.1f} ± {p.std_latency_ms:.1f}"
        gb = f"{p.peak_mem_mb / 1024:.2f}" if p.peak_mem_mb else "-"
        ov = f"{p.overhead_vs_baseline:.2f}" if p.overhead_vs_baseline is not None else "-"
        unit = "tok/s" if c.workload == "generation" else "pr/s"
        tp = f"{p.throughput:.1f} {unit}" if p.throughput is not None else "-"
        eager = "  (eager)" if p.enforce_eager else ""
        print(f"{c.backend:<14}{lat:<20}{gb:<9}{ov:<11}{tp:<14}{eager}")
    print("-" * 86)
