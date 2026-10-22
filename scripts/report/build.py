"""Render the two report pages from the extracted run data.

usage: python build.py CONFIG.json OUT_DIR
CONFIG names the inputs (fair-run JSON from extract.py, perf rows, construct states, diagnostics).
Writes OUT_DIR/index.html (the artifact page, no document skeleton) and
OUT_DIR/nnsight-versions.html (a complete document, served next to it).
"""
import html
import json
import sys
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

STATE_CHIP = {
    "SUPPORTED": ("good", "✓", "Supported"),
    "EQUIVALENT": ("good", "✓", "Equivalent"),
    "NUMERICAL_MISMATCH": ("warn", "≈", "Numerical mismatch"),
    "SILENTLY_WRONG": ("crit", "!", "Silently wrong"),
    "UNSUPPORTED": ("none", "–", "Unsupported"),
    "INCOMPATIBLE": ("serious", "?", "Incompatible"),
    "ERROR": ("crit", "×", "Error"),
    "JOB_FAILED": ("crit", "×", "Job failed"),
    "HANG": ("crit", "×", "Hang"),
    "RAN": ("none", "·", "Reference"),
}


def chip(state, label=None):
    cls, icon, text = STATE_CHIP.get(state, ("none", "?", state.title()))
    return (f'<span class="chip {cls}"><span class="dot" aria-hidden="true">{icon}</span>'
            f'{E(label or text)}</span>')


def fnum(v, digits=1, suffix=""):
    if v is None:
        return "–"
    return f"{v:,.{digits}f}{suffix}"


def ratio(v):
    return "–" if v is None else f"{v:.2f}×"


def page_head(title, description):
    return (f"<title>{E(title)}</title>\n"
            f'<meta name="description" content="{E(description)}">\n{FONTS}\n<style>\n{CSS}\n</style>\n')


def topbar(current, sections):
    pages = [("index", "./", "Systems compared"), ("versions", "nnsight-versions.html", "nnsight 0.7 → 0.8")]
    links = "".join(f'<a href="{href}"{" aria-current=\"page\"" if key == current else ""}>{E(text)}</a>'
                    for key, href, text in pages)
    secs = "".join(f'<a href="#{sid}">{E(text)}</a>' for sid, text in sections)
    return (f'<nav class="topbar" aria-label="Report"><div class="topbar-inner">'
            f'<a class="brand" href="./">nnbench · Delta</a><div class="pages">{links}</div>'
            f'<div class="sections">{secs}</div></div></nav>')


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
    } else if (s.kind === "dumbbell") {
      Report.mount(s.id, "dumbbell", {
        categories: s.categories, catHead: s.catHead, fmt: fmt[s.fmt], tick: tick[s.fmt],
        labelWidth: s.labelWidth,
        a: { label: s.a.label, color: s.a.color, value: (i) => s.a.values[i] },
        b: { label: s.b.label, color: s.b.color, value: (i) => s.b.values[i] },
        delta: (i) => s.deltas[i],
      });
    } else if (s.kind === "columns") {
      Report.mount(s.id, "columns", {
        categories: s.categories, values: s.values, emph: s.emph, yMax: s.yMax, yTicks: s.yTicks,
        fmt: fmt[s.fmt], tick: tick[s.fmt], emphLabel: s.emphLabel, restLabel: s.restLabel,
        emphColor: s.emphColor, restColor: s.restColor, catName: s.catName, subName: s.subName,
        valueName: s.valueName, height: s.height,
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


def main():
    cfg = json.load(open(sys.argv[1]))
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    import pages  # the two page bodies live next to this file
    index_body, index_specs = pages.systems_page(cfg)
    versions_body, versions_specs = pages.versions_page(cfg)
    (out / "index.html").write_text(
        page_head("nnbench Delta Report", "Five interpretability systems on vLLM, measured with matched "
                  "engines on one A100, plus nnsight 0.7 against 0.8.")
        + topbar("index", pages.SYSTEM_SECTIONS) + index_body + mounts(index_specs))
    (out / "nnsight-versions.html").write_text(
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
        + page_head("nnsight 0.7 → 0.8", "nnsight 0.7.0 against 0.8.0rc1 on HF and vLLM.")
        + "</head><body>\n" + topbar("versions", pages.VERSION_SECTIONS) + versions_body
        + mounts(versions_specs) + "</body></html>\n")
    print("wrote", out / "index.html", out / "nnsight-versions.html")


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main()
