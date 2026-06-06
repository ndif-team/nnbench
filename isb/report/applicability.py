"""Reporter — the applicability map (design.md §8.1, primary deliverable)."""
from __future__ import annotations


def print_map(workload, repo: str, cells) -> None:
    print(f"\n=== Applicability map ===")
    print(f"workload : {workload.id}  (motif={workload.motif}, tier={workload.tier})")
    print(f"model    : {repo}")
    print(f"{'-' * 78}")
    print(f"{'backend':<14}{'predicted':<26}{'actual':<22}{'latency':<9}{'metrics'}")
    print(f"{'-' * 78}")
    for c in cells:
        lat = f"{c.latency_s:.2f}s" if c.latency_s is not None else "-"
        if c.metrics:
            met = (
                f"top1={c.metrics.get('top1_agree', 0):.2f} "
                f"maxabs={c.metrics.get('max_abs', float('nan')):.2f}"
            )
        else:
            met = c.error or ""
        print(f"{c.backend:<14}{c.predicted:<26}{c.state:<22}{lat:<9}{met}")
    print(f"{'-' * 78}")
