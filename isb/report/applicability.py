"""Reporter — the applicability map (design.md §8.1, primary deliverable)."""
from __future__ import annotations


def print_map(workload, repo: str, cells) -> None:
    print("\n=== Applicability map ===")
    print(f"workload : {workload.id}  (motif={workload.motif}, tier={workload.tier})")
    print(f"model    : {repo}")
    print("-" * 86)
    print(f"{'backend':<14}{'predicted':<22}{'actual':<22}{'latency':<9}{'metrics / note'}")
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
        # §11.8: flag when the actual state contradicts a hand-annotated expectation.
        expected = workload.expect.get(c.backend)
        flag = f"  [!= expected {expected}]" if expected and expected != c.state else ""
        print(f"{c.backend:<14}{c.predicted:<22}{c.state:<22}{lat:<9}{note}{flag}")
    print("-" * 86)
