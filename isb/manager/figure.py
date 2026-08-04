"""The operating-point scatter (server-side SVG, no client JS)."""
from __future__ import annotations

import math

from .htmlkit import esc

# validated categorical palette (dataviz reference instance, light surface), fixed assignment
# order by run name; runs beyond the palette reuse the last slot.
_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def _ticks(hi: float, n: int = 4) -> list[float]:
    """A few round tick values covering [0, hi]."""
    if hi <= 0:
        return [0, 1]
    step = 10 ** math.floor(math.log10(hi / n))
    for mult in (1, 2, 5, 10, 20, 50):
        if hi / (step * mult) <= n:
            step *= mult
            break
    return [round(i * step, 10) for i in range(int(hi / step) + 2)]


def perf_svg(pts: list[dict]) -> str:
    """Operating-point scatter: x = median latency (ms), y = overhead over the run's baseline.
    Color = run (fixed assignment by sorted name); the y=1 line marks zero intervention cost."""
    if not pts:
        return ""
    w, h, ml, mb, mt, mr = 640, 300, 60, 44, 14, 16
    xt = _ticks(max(p["lat"] for p in pts) * 1.05)
    yt = _ticks(max(max(p["ovh"] for p in pts), 1.0) * 1.1)
    x_hi, y_hi = xt[-1], yt[-1]

    def sx(v):
        return ml + v / x_hi * (w - ml - mr)

    def sy(v):
        return h - mb - v / y_hi * (h - mb - mt)

    runs = sorted({p["run"] for p in pts})
    color = {r: _SERIES[min(i, len(_SERIES) - 1)] for i, r in enumerate(runs)}

    grid = "".join(f'<line x1="{ml}" y1="{sy(v):.0f}" x2="{w - mr}" y2="{sy(v):.0f}" '
                   f'stroke="#DCE3EA"/>'
                   f'<text x="{ml - 6}" y="{sy(v) + 4:.0f}" text-anchor="end">{v:g}</text>'
                   for v in yt)
    grid += "".join(f'<text x="{sx(v):.0f}" y="{h - mb + 16}" text-anchor="middle">{v:g}</text>'
                    for v in xt)
    one = (f'<line x1="{ml}" y1="{sy(1):.0f}" x2="{w - mr}" y2="{sy(1):.0f}" stroke="#5C6C7D" '
           f'stroke-dasharray="4 3"/>' if y_hi >= 1 else "")
    marks, labels = [], []
    for p in pts:
        cx, cy = sx(p["lat"]), sy(p["ovh"])
        extra = "".join(f" · {k}={v:.1f}" for k, v in (("throughput", p["tp"]),
                                                       ("peak MB", p["mem"])) if v)
        tip = (f'{p["run"]} · {p["label"]} [{p["regime"]}] · {p["lat"]:.1f} ms · '
               f'{p["ovh"]:.2f}x{extra}')
        marks.append(f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="4.5" fill="{color[p["run"]]}" '
                     f'stroke="#fff" stroke-width="2"><title>{esc(tip)}</title></circle>')
        labels.append(f'<text x="{cx + 8:.0f}" y="{cy + 4:.0f}" fill="#1C242E">'
                      f'{esc(p["label"])} [{esc(p["regime"])}]</text>')
    legend = "".join(f'<span class="mono"><svg width="10" height="10"><circle cx="5" cy="5" '
                     f'r="4.5" fill="{color[r]}"/></svg> {esc(r)}</span> '
                     for r in runs)
    return (f'<div class="card"><p class="note">One point per timed cell: median latency vs '
            f'overhead over that run\'s no-intervention baseline (dashed line = 1x). '
            f'Hover a point for throughput and peak memory.</p>'
            f'<p>{legend}</p>'
            f'<svg viewBox="0 0 {w} {h}" width="{w}" role="img" font-family="ui-monospace,monospace" '
            f'font-size="11" fill="#5C6C7D" aria-label="Operating points: latency vs overhead">'
            f'{grid}{one}'
            f'<text x="{(ml + w - mr) // 2}" y="{h - 8}" text-anchor="middle">median latency (ms)</text>'
            f'<text x="8" y="{mt - 2}">overhead x</text>'
            f'{"".join(marks)}{"".join(labels)}</svg></div>')
