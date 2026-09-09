"""The manager's HTML kit: every tag, class, and format rule lives here and nowhere else.

Pages compose these combinators and pass plain text; the kit escapes. The only helpers that
accept pre-built HTML are the structural ones (`page`, `header_bar`, `card`, `section`,
`table`), whose inputs are the outputs of other kit functions.

The stylesheet and page skeleton are ported from the reviewed design demo (the "nnbench page
structure" artifact): its palette variables, light and dark themes, header bar, eyebrow/lede
typography, panel cards, and wrapped tables. Divergence from the demo is a bug unless the data
forces it.
"""
from __future__ import annotations

import html

esc = html.escape

_LIGHT = """
  --bg:#F5F7F8; --panel:#FFFFFF; --ink:#1C242E; --mut:#5C6C7D; --line:#DCE3EA;
  --acc:#0B7285; --acc-ink:#0B7285;
  --ok:#1E7F43; --ok-bg:#E4F2E9; --deg:#9A660A; --deg-bg:#F7EDD8;
  --bad:#BE3227; --bad-bg:#F9E7E5; --err:#4C5A68; --err-bg:#E9EDF1;
  --un:#93A1AF; --un-bg:#F0F3F5; --code-bg:#F0F3F5;
  --lim:#077E9E; --lim-bg:#E1EFF3;
"""

_DARK = """
  --bg:#12161B; --panel:#1A2027; --ink:#E2E8EF; --mut:#8CA0B3; --line:#2A333D;
  --acc:#45B8CD; --acc-ink:#5FC6D9;
  --ok:#4FBB82; --ok-bg:#16301E; --deg:#D9A644; --deg-bg:#33290F;
  --bad:#E5685C; --bad-bg:#3A1B18; --err:#9FB0C0; --err-bg:#242D36;
  --un:#5E6D7B; --un-bg:#1E252C; --code-bg:#141A20;
  --lim:#45B8CD; --lim-bg:#15282E;
"""

STYLE = f"""<style>
:root{{{_LIGHT}}}
@media (prefers-color-scheme: dark){{:root{{{_DARK}}}}}
:root[data-theme="light"]{{{_LIGHT}}}
:root[data-theme="dark"]{{{_DARK}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}}
a{{color:var(--acc-ink);text-decoration:none}}
a:hover{{text-decoration:underline}}
a:focus-visible,button:focus-visible{{outline:2px solid var(--acc);outline-offset:2px}}
header{{border-bottom:1px solid var(--line);background:var(--panel)}}
.hwrap{{max-width:920px;margin:0 auto;padding:14px 20px;display:flex;gap:14px;
  align-items:baseline;flex-wrap:wrap}}
.navrow{{padding-top:0;margin-top:-8px;padding-bottom:10px}}
.brand{{font:600 15px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;color:var(--ink)}}
.brand a{{color:inherit}}
.crumb{{font:12px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;color:var(--mut)}}
.crumb a{{color:var(--mut)}}
main{{max-width:920px;margin:0 auto;padding:28px 20px 60px}}
.eyebrow{{font:600 11px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  letter-spacing:.09em;text-transform:uppercase;color:var(--mut);margin:0 0 6px}}
h1{{font-size:24px;line-height:1.25;margin:0 0 8px;text-wrap:balance;
   font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-weight:600}}
h2{{font-size:16px;margin:34px 0 10px;text-wrap:balance}}
p{{max-width:68ch;margin:8px 0}}
.lede{{color:var(--mut);max-width:70ch;margin:0 0 4px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:6px;
  padding:16px 18px;margin:12px 0}}
.card h2{{font:600 14px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;margin:0 0 4px}}
.card pre{{margin:10px 0 0}}
.tblwrap{{overflow-x:auto;background:var(--panel);border:1px solid var(--line);
  border-radius:6px;margin:12px 0}}
table{{border-collapse:collapse;width:100%;font-size:13.5px}}
th{{font:600 11px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;letter-spacing:.07em;
   text-transform:uppercase;color:var(--mut);text-align:left;padding:9px 12px;
   border-bottom:1px solid var(--line)}}
td{{padding:8px 12px;border-bottom:1px solid var(--line);vertical-align:top}}
tr:last-child td{{border-bottom:none}}
.mono{{font:12.5px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-variant-numeric:tabular-nums}}
.chip{{display:inline-block;font:600 10.5px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  letter-spacing:.05em;padding:2px 7px;border-radius:4px;white-space:nowrap}}
.s-SUPPORTED,.s-EQUIVALENT,.s-SUPPORTED_DEGRADED,.s-EQUIVALENT_DEGRADED{{
  color:var(--ok);background:var(--ok-bg)}}
.s-SILENTLY_WRONG,.s-DIVERGENT{{color:var(--bad);background:var(--bad-bg)}}
.s-ERROR,.s-JOB_FAILED,.s-INCOMPATIBLE,.s-INVALID_REFERENCE{{color:var(--err);background:var(--err-bg)}}
.s-RAN,.s-NO_REFERENCE,.s-UNTESTED,.s-NOT_RUN,.s-CANCELLED,.s-PENDING,.s-RUNNING{{color:var(--un);background:var(--un-bg)}}
.s-LIMITED,.s-BASELINE{{color:var(--lim);background:var(--lim-bg)}}
.note{{font-size:12.5px;color:var(--mut);max-width:72ch}}
.berow{{display:flex;gap:8px 22px;flex-wrap:wrap;align-items:center;margin-top:12px;
  border-top:1px solid var(--line);padding-top:10px;
  font:12px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;color:var(--mut)}}
.berow span{{display:inline-flex;gap:7px;align-items:center}}
.kv{{display:grid;grid-template-columns:max-content 1fr;gap:4px 18px;font-size:13.5px;margin:6px 0}}
.kv dt{{color:var(--mut);font:12px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  padding-top:1px}}
.kv dd{{margin:0;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:12.5px;overflow-wrap:anywhere}}
pre{{background:var(--code-bg);border:1px solid var(--line);border-radius:6px;
    padding:12px 14px;overflow-x:auto;
    font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;margin:8px 0}}
code{{font:.92em ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:var(--code-bg);
     padding:1px 4px;border-radius:3px}}
pre code{{background:none;padding:0}}
ul{{max-width:70ch;padding-left:22px}}li{{margin:4px 0}}
form{{display:inline}}
button{{font:12px ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;cursor:pointer}}
</style>"""

# secondary header row: how every page reaches the section indexes (the demo reached them
# through overview links; a persistent row keeps them one click away on deep pages too)
NAV = ("<nav class='hwrap navrow crumb'>"
       "<a href='/'>overview</a> · <a href='/micro'>perf micro</a> · "
       "<a href='/runs'>runs</a> · <a href='/inbox'>inbox</a> · <a href='/data'>data</a> · "
       "<a href='/models'>models</a> · <a href='/backends'>backends</a> · "
       "<a href='/findings'>findings</a> · <a href='/catalog'>catalog</a></nav>")


# ---- text-level combinators (escape their inputs) ----------------------------------------------

def link(href: str, label: str) -> str:
    return f"<a href='{esc(href)}'>{esc(label)}</a>"


def chip(state: str) -> str:
    label = "SUPPORTED ·fp" if state == "SUPPORTED_DEGRADED" else \
            "EQUIVALENT ·fp" if state == "EQUIVALENT_DEGRADED" else state
    return f'<span class="chip s-{esc(state)}">{esc(label)}</span>'


def mono(text) -> str:
    return f"<span class='mono'>{esc(str(text))}</span>"


def note(text) -> str:
    return f"<p class='note'>{esc(str(text))}</p>"


def note_html(inner: str) -> str:
    """A note whose content is already kit-built HTML (links, chips)."""
    return f"<p class='note'>{inner}</p>"


def eyebrow(text: str) -> str:
    return f"<p class='eyebrow'>{esc(text)}</p>"


def lede(text: str) -> str:
    return f"<p class='lede'>{esc(text)}</p>"


def lede_html(inner: str) -> str:
    return f"<p class='lede'>{inner}</p>"


def code_block(text: str) -> str:
    return f"<pre><code>{esc(text)}</code></pre>"


def fmt_ms(ms) -> str:
    """One latency format everywhere: milliseconds, switching to seconds at 10 s so probe
    durations (up to the 180 s watchdog) stay readable."""
    if ms is None:
        return "-"
    return f"{ms / 1000:.1f} s" if ms >= 10_000 else f"{ms:.1f} ms"


# ---- structural combinators (inputs are kit-built HTML) ----------------------------------------

def table(headers: list[str] | None, rows: list[list[str]]) -> str:
    """headers are plain text (None for a headerless table); each row cell is kit-built HTML.
    Wrapped for horizontal scroll (the demo's tblwrap)."""
    head = ("<tr>" + "".join(f"<th>{esc(h)}</th>" for h in headers) + "</tr>") if headers else ""
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<div class='tblwrap'><table>{head}{body}</table></div>"


def kv_table(pairs: list[tuple[str, str]]) -> str:
    """(plain-text key, kit-built HTML value) rows, as the demo's definition-list grid."""
    body = "".join(f"<dt>{esc(k)}</dt><dd>{v}</dd>" for k, v in pairs)
    return f"<dl class='kv'>{body}</dl>"


def card(*blocks: str) -> str:
    return f"<div class='card'>{''.join(blocks)}</div>"


def heading(level: int, inner: str) -> str:
    return f"<h{level}>{inner}</h{level}>"


def chip_row(spans: list[str]) -> str:
    return f"<p class='berow'>{' '.join(f'<span>{s}</span>' for s in spans)}</p>"


def header_bar(crumb: str, dir_label: str | None, inbox_html: str | None) -> str:
    """The demo's header: brand, breadcrumb, the rendered directory, inbox link. `crumb` is
    plain text (id'd so the export router can update it); `inbox_html` is kit-built."""
    right = ""
    if dir_label:
        right += (f"<span class='crumb' style='margin-left:auto' title='the manager renders "
                  f"exactly the run files in this directory'>dir: <b>{esc(dir_label)}</b></span>")
        if inbox_html:
            right += f"<span class='crumb'>{inbox_html}</span>"
    elif inbox_html:
        right += f"<span class='crumb' style='margin-left:auto'>{inbox_html}</span>"
    return ("<header><div class='hwrap'>"
            "<span class='brand'><a href='/'>nnbench</a></span>"
            f"<span class='crumb' id='crumb'>{esc(crumb)}</span>"
            f"{right}</div>{NAV}</header>")


def page(title: str, header_html: str, *blocks: str) -> str:
    return (f"<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>{esc(title)}</title>"
            f"{STYLE}{header_html}<main>{''.join(blocks)}</main>")
