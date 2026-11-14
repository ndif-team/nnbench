"""Shared HTML helpers of the report pages (opt7b.py, optlarge.py): head, figures, tables and the
chart mounts."""
import html
import json
from pathlib import Path

HERE = Path(__file__).parent
CSS = (HERE / "assets/report.css").read_text()
JS = (HERE / "assets/report.js").read_text()
FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..800'
         '&family=JetBrains+Mono:wght@400;600&family=Source+Serif+4:opsz,wght@8..60,400..650&display=swap">')

E = html.escape
SERIES = ["var(--series-1)", "var(--series-2)", "var(--series-3)"]


def fnum(v, digits=1, suffix=""):
    if v is None:
        return "–"
    return f"{v:,.{digits}f}{suffix}"


def page_head(title, description):
    return (f"<title>{E(title)}</title>\n"
            f'<meta name="description" content="{E(description)}">\n{FONTS}\n<style>\n{CSS}\n</style>\n')


def figure(fid, title, sub, legend=True, extra_cls=""):
    return (f'<figure class="chart {extra_cls}"><figcaption><span class="t">{E(title)}</span>'
            f'<span class="s">{sub}</span></figcaption>'
            + ('<div class="legend" aria-hidden="true"></div>' if legend else "")
            + f'<div class="plot" id="{fid}"></div><div class="chart-tools"></div></figure>')


def provenance(items):
    return ('<dl class="provenance">'
            + "".join(f"<div><dt>{E(k)}</dt><dd>{v}</dd></div>" for k, v in items) + "</dl>")


def mounts(specs):
    """Chart specs travel as JSON; functions (formatters, accessors) are rebuilt in the page."""
    return ("<script>\n" + JS + "\n</script>\n<script>\nconst SPECS = "
            + json.dumps(specs).replace("</", "<\\/") + ";\n" + MOUNT_JS + "\n</script>\n")


# Rebuilds each JSON spec into the chart module's callable form.
MOUNT_JS = r"""
(function () {
  const fmt = {
    ms: (v) => `${v.toFixed(1)} ms`,
    ms0: (v) => `${Math.round(v)} ms`,
    pct: (v) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(0)}%`,
    x: (v) => `${v.toFixed(2)}×`,
    tv: (v) => v.toFixed(3),
  };
  const tick = { ms: (t) => `${t}`, pct: (t) => `${t}%`, x: (t) => `${t}×`, tv: (t) => `${t}` };
  for (const s of SPECS) {
    if (s.kind === "hbars") {
      Report.mount(s.id, "hbars", {
        categories: s.categories, series: s.series, catHead: s.catHead,
        value: (ci, si) => s.values[ci][si], fmt: fmt[s.fmt], tick: tick[s.tickFmt || s.fmt],
        missing: (ci, si) => (s.missing && s.missing[ci] && s.missing[ci][si]) || "n/a",
        labelWidth: s.labelWidth, refLine: s.refLine, domain: s.domain, aria: s.aria,
      });
    }
  }
})();
"""


def table(head, rows, num_cols=(), cls=""):
    th = "".join(f'<th{" class=num" if i in num_cols else ""}>{h}</th>' for i, h in enumerate(head))
    body = []
    for r in rows:
        if isinstance(r, dict) and r.get("group"):
            body.append(f'<tr class="group"><td colspan="{len(head)}">{r["group"]}</td></tr>')
            continue
        body.append("<tr>" + "".join(
            f'<td{" class=num" if i in num_cols else ""}>{c}</td>' for i, c in enumerate(r)) + "</tr>")
    return (f'<div class="table-wrap {cls}"><table><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')
