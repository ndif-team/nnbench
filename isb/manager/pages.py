"""The manager's pages: composition only. Data comes from a Collection (model.py), markup from
the kit (htmlkit.py). ROUTES is the single registry the server, the export, and the tests share:
a page that exists is a page that is routed, exported, and crawlable, with no second list to
forget."""
from __future__ import annotations

import json
import math
import re
import textwrap

from . import model
from .figure import perf_svg
from .htmlkit import (STYLE, card, chip, chip_row, code_block, esc, eyebrow, fmt_ms, header_bar,
                      heading, kv_table, lede, lede_html, link, mono, note, note_html, page,
                      table)
from .model import Collection

# UI copy (reviewed in the design demo): one line per methodology, shown on its card.
METHOD_INTROS = {
    "logit_lens": "Read each layer's residual stream through the unembedding: the model's "
                  "intermediate next-token prediction per layer.",
    "steering": "Add a scaled token direction to a mid-layer residual stream, read the shifted "
                "next-token distribution.",
    "ablation": "Zero one module's output (MLP or attention), read the effect on the next-token "
                "distribution.",
    "activation_patching": "Capture an activation from a clean prompt, transplant it into a "
                           "corrupted prompt's forward pass.",
    "attribution_patching": "Approximate patching effects with one backward pass (gradient x "
                            "activation difference).",
    "attention_pattern": "Read a layer's attention weight matrix.",
    "gen_steering": "Apply the steering write at every decode step during generation.",
    "gen_patching": "Transplant an activation at each decode step during generation.",
    "jacobian_lens": "Map a mid-layer state to the final logits through a linear transport "
                     "(identity, orthogonal, or fitted).",
}

_BASELINE_CHIP = chip("BASELINE")

# UI copy (reviewed in the design demo): the idealized user-facing trace per methodology, shown
# on its overview card. The measured cell source stays on the method page; the card shows what a
# user of the method would actually write.
TRACE_SNIPPETS = {
    "logit_lens": """W = model.lm_head.weight
for i in layers:
    out = model.model.layers[i].output
    h = out[0] + out[1]                       # vLLM dual residual
    saves[i] = F.linear(norm(h), W)[0, -1].save()""",
    "steering": """vec = F.normalize(model.lm_head.weight[token_id], dim=-1)
out = model.transformer.h[8].output
h = out[0]
model.transformer.h[8].output = (h + alpha * h.norm(dim=-1, keepdim=True) * vec, *out[1:])
logits = model.output.logits[0, -1].save()""",
    "ablation": """out = model.transformer.h[6].mlp.output
model.transformer.h[6].mlp.output = torch.zeros_like(out)
logits = model.output.logits[0, -1].save()""",
    "activation_patching": """with model.trace(clean):                      # trace 1: capture
    h = model.transformer.h[9].output[0].save()
with model.trace(corrupted):                  # trace 2: transplant
    out = model.transformer.h[9].output
    model.transformer.h[9].output = (h, *out[1:])
    logits = model.output.logits[0, -1].save()""",
    "attribution_patching": """h = model.transformer.h[5].output[0]
grad = h.grad.save()
metric = logit_diff(model.output.logits)
metric.backward()
score = (grad * (h_clean - h_corrupt)).sum()""",
    "attention_pattern": """_, attn_probs = model.transformer.h[3].attn.output    # (batch, heads, q, k)
pattern = attn_probs.save()""",
    "gen_steering": """with tracer.iter[0:N]:
    out = model.transformer.h[8].output
    model.transformer.h[8].output = (steer(out[0]), *out[1:])
tokens = generator.output.save()""",
    "gen_patching": """with tracer.iter[0:N]:
    out = model.transformer.h[9].output
    model.transformer.h[9].output = (h_clean, *out[1:])
tokens = generator.output.save()""",
    "jacobian_lens": """h = model.transformer.h[L].output[0][0, -1]
z = transport @ h                             # identity / orthogonal / fitted map
logits = F.linear(norm(z), model.lm_head.weight).save()""",
}


def _page(col: Collection | None, crumb: str, title: str, *blocks: str) -> str:
    """Every page goes through here: the demo's header (brand, breadcrumb, directory, inbox
    count) over the page body. Repo-derived pages may have no collection."""
    if col is not None:
        header = header_bar(crumb, col.dir_path or None,
                            link("/inbox", f"inbox ({len(col.inbox_runs())})"))
    else:
        header = header_bar(crumb, None, link("/inbox", "inbox"))
    warnings = [note(message) for message in col.warnings] if col is not None else []
    return page(title, header, *warnings, *blocks)


def _stack(prov: dict) -> str:
    c = prov.get("client", {})
    nn = c.get("nnsight") or {}
    e = prov.get("engine", {"kind": "unknown"})
    commit = str(nn.get("commit"))[:9]
    url = str(nn.get("remote", "")).removesuffix(".git")
    if url.startswith("git@github.com:"):
        url = "https://github.com/" + url.removeprefix("git@github.com:")
    if url.startswith("https://github.com/"):
        commit = f'<a href="{esc(url)}/commit/{esc(str(nn.get("commit")))}">{esc(commit)}</a>'
    else:
        commit = esc(commit)
    ver = f"vllm {c.get('vllm')}" if e.get("kind") == "vllm" else f"tf {c.get('transformers')}"
    return (f"<span class='mono'>{esc(e.get('kind', 'unknown'))}({esc(str(e.get('mode')))}) · nnsight {commit}"
            f"{'*' if nn.get('dirty') else ''} · {esc(ver)}</span>")


def _legend() -> str:
    """The how-to-read panel (copy reviewed in the design demo)."""
    result_states = table(None, [
        [chip("SUPPORTED"), note("matches the baseline")],
        [chip("SUPPORTED_DEGRADED"),
         note("matches the baseline at fp32; under bf16 a few near-tie top tokens flip")],
        [chip("SILENTLY_WRONG"), note("runs without any error but produces wrong numbers")],
        [chip("ERROR"), note("fails with an error")],
    ])
    rollups = table(None, [
        [chip("SUPPORTED"), note("every variant and regime passes")],
        [chip("LIMITED"), note("some realization passes; at least one variant or regime "
                               "does not")],
        [chip("ERROR"), note("nothing runs")],
    ])
    return card(
        "<b>How to read the map</b>",
        note("A result state is one measurement: one variant of the method, in one workload "
             "regime, on one run, compared against the baseline run."),
        result_states,
        note("A rollup chip summarizes all of a run's result states (every variant and regime). "
             "Open the method for the states behind it."),
        rollups)


def _trace_body(src: str) -> str:
    """The trace-body closure's source. Every cell passes a named inner function (the repo's
    build/capture/step closure convention) to its backend; that body IS what runs inside
    model.trace, so it is what a methodology card shows. The surrounding cell (registry binding,
    backend dispatch) is implementation and stays on the method page."""
    import ast

    dedented = textwrap.dedent(src)
    try:
        outer = ast.parse(dedented).body[0]
    except SyntaxError:
        return src
    if isinstance(outer, ast.FunctionDef):
        inner = next((n for n in ast.walk(outer)
                      if isinstance(n, ast.FunctionDef) and n is not outer), None)
        if inner is not None:
            seg = ast.get_source_segment(dedented, inner)
            # an inner body that is itself a bare delegation shows nothing; the whole helper
            # (e.g. a two-trace capture/patch pair) is the readable trace then
            if seg and model._delegation_target(seg) is None:
                return seg
    return src


def _strip_comments(src: str) -> str:
    import io
    import tokenize

    cuts: dict[int, int] = {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                cuts[tok.start[0]] = min(cuts.get(tok.start[0], tok.start[1]), tok.start[1])
    except (tokenize.TokenError, IndentationError):
        return src
    lines = src.splitlines()
    return "\n".join((ln[:cuts[i + 1]].rstrip() if i + 1 in cuts else ln)
                     for i, ln in enumerate(lines))


def _abridge(src: str, max_lines: int = 12) -> str:
    """The card's code: the trace body, comments and docstring stripped, capped."""
    src = _strip_comments(_trace_body(src))
    src = re.sub(r"\n\s*(?:\"\"\"|''')[\s\S]*?(?:\"\"\"|''')", "", src, count=1)  # docstring
    body = [ln for ln in textwrap.dedent(src).splitlines() if ln.strip()]
    if len(body) > max_lines:
        body = body[:max_lines] + ["# ... (full cell on the method page)"]
    return "\n".join(body)


def _effect_summary(effect: dict) -> str:
    """A supporting call can fail before effect metrics exist, including in historical files."""
    if not isinstance(effect, dict):
        return "effect metrics unavailable"
    failed_sides = [f"{side}: {effect[side]['error']}" for side in ("baseline", "perturbed")
                    if isinstance(effect.get(side), dict) and effect[side].get("error")]
    errors = ([str(effect["error"])] if effect.get("error") else []) + failed_sides
    if errors:
        return "effect check failed: " + "; ".join(errors)
    metrics = [effect.get(key) for key in ("top1_agree", "tv")]
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
               and math.isfinite(value) for value in metrics):
        return "effect metrics unavailable"
    status = ", strong" if effect.get("strong") is True else ", weak" if effect.get("strong") is False else ""
    return f"effect size: top1={metrics[0]:.2f}, tv={metrics[1]:.3f}{status}"


def _cell_rows(col: Collection, name: str, cells) -> list[list[str]]:
    """The uniform variant x regime rows for one run. The baseline's rows carry a BASELINE mark
    where a verdict would sit: nothing exists to compare the reference against."""
    is_baseline = name in col.baselines.values()
    rows = []
    for c in cells:
        m = c.metrics or {}
        metric = (f"top1={m.get('top1_agree', 0):.2f} tv={m.get('tv', float('nan')):.3f} "
                  f"maxabs={m.get('max_abs', float('nan')):.2f}" if m else (c.error or ""))
        state = _BASELINE_CHIP if is_baseline and c.state == "RAN" else chip(c.state)
        lat = fmt_ms(c.latency_s * 1000 if c.latency_s is not None else None)
        rows.append([esc(c.label), esc(c.workload), state, mono(lat), mono(metric)])
    meta = col.load(name)[0].get(("__meta__",), {})
    rows += [[esc("effect guard"), esc(k[1]), "", mono("-"), mono(_effect_summary(e))]
             for k, e in sorted(meta.items(), key=str)
             if isinstance(k, tuple) and len(k) == 2 and k[0] == "__effect__"]
    return rows


def _probe_rows(probes: dict) -> list[list[str]]:
    return [[esc(p), esc("probe"), chip(m["state"]), mono(fmt_ms(m["latency_s"] * 1000)),
             mono(m.get("note") or "")]
            for p, m in probes.items()]


_RUN_TABLE_HEADERS = ["variant", "regime", "state", "latency", "metrics / note"]


def _by_method(col: Collection) -> dict[str, list[str]]:
    """{methodology: [spec names]} over the directory's method runs (probe runs excluded)."""
    groups: dict[str, list[str]] = {}
    for spec_name, run_names in col.by_spec().items():
        if any(model.probe_results(col.entries[n][0]) for n in run_names):
            continue                           # construct-probe runs render on backend pages
        method = col.entries[run_names[0]][1]["coordinates"]["methodology"]
        if method == "perf_micro":
            continue                           # perf sweeps render on the perf micro page
        groups.setdefault(method, []).append(spec_name)
    return groups


# ---- pages -------------------------------------------------------------------------------------

def overview(col: Collection) -> str:
    setups = {}
    for _, prov in col.entries.values():
        if prov["coordinates"]["methodology"] == "perf_micro":
            continue                           # a perf sweep spans systems; not one backend setup
        label, setup = model.backend_label(prov)
        setups[label] = setup
    backends = table(["backend", "setup of the runs in this directory"],
                     [[esc(label), note(setup)] for label, setup in sorted(setups.items())])

    cards = []
    for method, spec_names in sorted(_by_method(col).items()):
        run_names = [n for s in spec_names for n in col.by_spec()[s]]
        # one card per METHODOLOGY: the pure procedure. Models and data are run coordinates;
        # they live on the method page's matrix, never on the card.
        per_backend: dict[str, list[str]] = {}
        baseline_labels: set[str] = set()
        for name in run_names:
            label, _ = model.backend_label(col.entries[name][1])
            if name in col.baselines.values() and all(c.state == "RAN" for c in col.results.get(name) or []):
                baseline_labels.add(label)
                continue
            cells = col.results.get(name)
            if cells is None:
                continue
            per_backend.setdefault(label, []).extend(c.state for c in cells)
        chips = [f"{esc(label)} {_BASELINE_CHIP}"
                 for label in sorted(baseline_labels - set(per_backend))]
        chips += [f"{esc(label)} {chip(model.rollup(states))}"
                  for label, states in sorted(per_backend.items())]
        if not per_backend:
            chips.append("<span class='note'>no scored runs</span>")
        coords = col.entries[run_names[-1]][1]["coordinates"]
        snippet = TRACE_SNIPPETS.get(method)
        if snippet is None:
            src = model.trace_source(method, coords["family"], coords["interface"])
            snippet = _abridge(src) if src else None
        cards.append(card(
            heading(2, link(f"/method/{method}", method)),
            note(METHOD_INTROS.get(method, "")),
            code_block(snippet) if snippet else "",
            chip_row(chips)))

    baselines_line = (esc(f"{len(col.baselines)} reference jobs, one per experiment "
                          f"(marked on the runs index)") if col.baselines else
                      esc("none (reference selection is recorded by the benchmark)"))
    dir_panel = card(kv_table([
        ("directory", f"{esc(col.dir_path)} · {link('/runs', f'{len(col.entries)} runs')}"),
        ("baselines", baselines_line),
        ("inbox", link("/inbox", f"{len(col.inbox_runs())} new runs")),
    ]))
    return _page(
        col, "", f"nnbench · {col.dir_path}",
        eyebrow("Applicability map"),
        heading(1, "nnbench"),
        lede_html(f"Which interpretability methods run, run correctly, and run fast on each "
                  f"serving backend. These are saved benchmark reports, not comparisons "
                  f"recomputed by the site. Browse a run directory or a collection of runs. "
                  f"The {link('/inbox', 'inbox')} can archive complete runs."),
        dir_panel,
        heading(2, "Backends"),
        backends,
        note_html(f"Per-construct support for each backend is on its backend page. Primitive op "
                  f"cost (read / steer / qk per system) is on the {link('/micro', 'perf micro')} "
                  f"page."),
        _legend(),
        heading(2, "Methodologies"),
        note("One card per method: what it does, the trace of its recommended form (abridged; "
             "current checkout source is on the method page), and an aggregate of all recorded "
             "results per named backend in this directory. This is not a latest-run selector."),
        *cards)


_OP_ORDER = {"read": 0, "steer": 1, "qk": 2}
_FP_ORDER = {"one": 0, "half": 1, "all": 2}
_OP_INTROS = {
    "read": "Capture the hidden state at the footprint's layers.",
    "steer": "Write-replace the hidden state at the footprint's layers.",
    "qk": "Capture Q and K at the footprint's layers.",
}
_SYSTEM_LABELS = {"pure_vllm": "pure vLLM (baseline)", "nnsight_vllm": "nnsight",
                  "vllm_lens": "vllm-lens", "vllm_hook": "vllm-hook",
                  "native_eagle": "native Eagle"}


def _perf_row_cells(r: dict) -> list[str]:
    sys_label = _SYSTEM_LABELS.get(r["system"], r["system"])
    where = [esc(sys_label), mono(r.get("footprint") or "-"), mono(r.get("phase") or "-")]
    if r.get("error"):
        return where + [mono("-"), mono("-"), chip("ERROR"), note(r["error"])]
    lat = fmt_ms(r["median_total_lat_s"] * 1000) if r.get("median_total_lat_s") else "-"
    tps = f"{r['median_tok_per_s']:.0f}" if r.get("median_tok_per_s") else "-"
    ovh = (f"{r['overhead_pct']:+.1f}%" if r.get("overhead_pct") is not None else "-")
    return where + [mono(lat), mono(tps), mono(ovh), ""]


def _op_card(op: str, rows: list[dict]) -> str:
    def key(r):
        return (r["system"] != "pure_vllm", r["system"],
                _FP_ORDER.get(r.get("footprint"), 9), r.get("phase") != "prefill")
    return card(
        heading(2, esc(op)),
        note(_OP_INTROS.get(op, "")),
        table(["system", "footprint", "phase", "total", "tok/s", "overhead", ""],
              [_perf_row_cells(r) for r in sorted(rows, key=key)]))


def micro(col: Collection) -> str:
    sections = []
    for name, (out, prov) in sorted(col.entries.items()):
        if prov["coordinates"]["methodology"] != "perf_micro":
            continue
        rows = out.get(("perf_rows",), [])
        base_rows = [r for r in rows if r.get("op") == "none"]
        sections += [heading(2, link(f"/run/{name}", name)), note_html(_stack(prov))]
        for op in sorted({r["op"] for r in rows if r.get("op") != "none"},
                         key=lambda o: _OP_ORDER.get(o, 9)):
            sections.append(_op_card(op, base_rows + [r for r in rows if r["op"] == op]))
    if not sections:
        sections = [note("No perf-micro runs in this directory; run scripts/perf.py sweep "
                         "and archive its run file here.")]
    return _page(
        col, "perf micro", "perf micro",
        eyebrow("Perf micro"),
        heading(1, "Primitive op cost"),
        lede("The cost of read, steer, and qk as overhead over the unmodified engine, compared "
             "across systems; the baseline run is pure vLLM with no intervention. Overhead is "
             "lost throughput versus the baseline sharing the same workload (model, phase, "
             "batch, lengths)."),
        *sections)


def method(col: Collection, name: str) -> str:
    """One methodology: the full trace, then the model × backend matrix. A matrix cell rolls up
    one model's runs on one backend and links the run details."""
    spec_names = _by_method(col).get(name, [])
    labels = sorted({model.backend_label(col.entries[n][1])[0]
                     for s in spec_names for n in col.by_spec()[s]})
    rows = []
    for spec_name in sorted(spec_names):
        run_names = col.by_spec()[spec_name]
        coords = col.entries[run_names[0]][1]["coordinates"]
        cells_by_label: dict[str, list[str]] = {}
        for rn in run_names:
            cells = col.results.get(rn)
            if cells is None or (rn in col.baselines.values() and all(c.state == "RAN" for c in cells)):
                continue
            cells_by_label.setdefault(
                model.backend_label(col.entries[rn][1])[0], []).extend(c.state for c in cells)
        row = [f"{link(f'/spec/{spec_name}', coords['repo'])} "
               f"<span class='note'>data {esc(', '.join(coords['data']) or '-')} · "
               f"experiment {esc(spec_name)}</span>"]
        row += [chip(model.rollup(cells_by_label[lb])) if lb in cells_by_label
                else "<span class='note'>-</span>" for lb in labels]
        rows.append(row)
    src = None
    if spec_names:
        coords = col.entries[col.by_spec()[spec_names[-1]][-1]][1]["coordinates"]
        src = model.trace_source(name, coords["family"], coords["interface"])
    return _page(
        col, name, name,
        eyebrow("Methodology"),
        heading(1, esc(name)),
        lede(METHOD_INTROS.get(name, "")),
        heading(2, "Procedure") if src else "",
        note("Source from the current checkout, not a historical snapshot of the executed code. "
             "The run details record source identity.") if src else "",
        code_block(src) if src else "",
        heading(2, "Model × backend"),
        _legend(),
        table(["model"] + labels, rows),
        note_html(f"A cell rolls up one model's runs on one backend; open the model for every "
                  f"run's states, metrics, and operating points. {link('/', 'back')}"))


def spec(col: Collection, spec_name: str) -> str:
    baseline = col.baselines.get(spec_name)
    names = sorted(col.by_spec().get(spec_name, []), reverse=True)
    if baseline in names:
        names = [baseline] + [n for n in names if n != baseline]

    sections = []
    for name in names:
        outputs, prov = col.entries[name]
        head = [heading(2, link(f"/run/{name}", name)), note_html(_stack(prov)),
                note(f"comparison: {prov.get('view', {}).get('comparison', 'legacy import')} · "
                     f"reference: {prov.get('view', {}).get('reference', baseline)}")]
        probes = model.probe_results(outputs)
        cells = col.results.get(name)
        if probes:
            sections += head + [table(_RUN_TABLE_HEADERS, _probe_rows(probes))]
        elif cells is None:
            sections += head + [note("no saved verdict (legacy spec/data may differ from the reference)")]
        else:
            sections += head + [table(_RUN_TABLE_HEADERS, _cell_rows(col, name, cells))]

    method = (col.entries[names[0]][1]["coordinates"]["methodology"] if names else spec_name)
    base_line = (f"vs baseline {link(f'/run/{baseline}', baseline)}"
                 if baseline else esc("no baseline"))
    fig = perf_svg(model.perf_points(col.entries, names))
    return _page(
        col, spec_name, spec_name,
        eyebrow("Results"),
        heading(1, esc(spec_name)),
        lede(METHOD_INTROS.get(method, "")),
        note_html(f"{base_line} · {link('/', 'back')}"),
        fig,
        *sections)


def run(col: Collection, name: str) -> str:
    outputs, prov = col.load(name)
    meta = outputs.get(("__meta__",), {})
    facts = kv_table([
        ("stack", _stack(prov)),
        ("coordinates", esc(str(prov.get("coordinates")))),
        ("deployment", esc(str(prov.get("deployment")))),
        ("host", esc(str(prov.get("host", {}).get("hostname")))),
        ("gpus", esc(str([g.get("name") for g in prov.get("host", {}).get("gpus") or []]))),
        ("job", esc(str(prov.get("view", {})))),
    ])
    rows = []
    for k, m in meta.items():
        if not (isinstance(k, tuple) and len(k) == 2):
            continue
        if m.get("error"):
            shown = m["error"]
        elif "state" in m:                     # construct probe: self-checked state + note
            shown = f"{m['state']} · {m.get('note') or ''}"
        elif m.get("median_latency_ms") is not None:
            shown = fmt_ms(m["median_latency_ms"])
        else:
            shown = "-"
        rows.append([mono(str(k)), mono(str(shown))])
    is_baseline = name in col.baselines.values()
    return _page(
        col, f"run / {name}", name,
        eyebrow("Run · baseline" if is_baseline else "Run"),
        heading(1, esc(name)),
        facts,
        heading(2, "Cells"),
        table(_RUN_TABLE_HEADERS, _cell_rows(col, name, col.cells_for(name)))
        if col.cells_for(name) is not None else table(["cell", "latency / error"], rows),
        note_html(link("/", "back")))


def inbox(col: Collection) -> str:
    rows = []
    for name in col.inbox_runs():
        from pathlib import Path
        is_bundle = (Path(col.inbox) / name / "plan.json").is_file()
        title = link(f"/inbox-run/{name}", name) if is_bundle else link(f"/run/inbox/{name}", name)
        description = "complete benchmark run" if is_bundle else "legacy artifact (import summary before viewing)"
        token = f"<input type='hidden' name='csrf' value='{esc(col.csrf_token)}'>"
        actions = (f"<form method='post' action='/archive'>"
                   f"{token}"
                   f"<input type='hidden' name='name' value='{esc(name)}'>"
                   f"<button>archive to {esc(col.dir_path)}</button></form> "
                   f"<form method='post' action='/discard'>"
                   f"{token}"
                   f"<input type='hidden' name='name' value='{esc(name)}'>"
                   f"<button>move to trash</button></form>")
        rows.append([title, note(description), actions])
    return _page(
        col, "inbox", "inbox",
        eyebrow("Inbox"),
        heading(1, "Inbox"),
        lede_html(f"Complete runs can be archived without splitting their artifacts. Discard "
                  f"moves them into the inbox's .trash directory for recovery. {link('/', 'back')}"),
        table(["run", "stack", ""], rows))


def runs(col: Collection) -> str:
    baselines = set(col.baselines.values())
    rows = []
    for name, (_, prov) in sorted(col.entries.items()):
        c = prov["coordinates"]
        mark = f" {_BASELINE_CHIP}" if name in baselines else ""
        rows.append([
            link(f"/run/{name}", name) + mark,
            mono(prov.get("executed") or "unknown"),
            _stack(prov),
            mono(f"{c['spec']} · {', '.join(c['data']) or '-'}"),
        ])
    return _page(
        col, "runs", "runs",
        eyebrow("Runs"),
        heading(1, esc(f"Runs in {col.dir_path}")),
        table(["run", "executed", "stack", "spec · data"], rows))


# ---- repo-derived pages (directory-independent) -------------------------------------------------

def _units_of(name: str):
    """(units, sample_note): file sources load fully; generated sources need a size, so the page
    shows a fixed sample of them."""
    from ..data import SOURCES

    try:
        return SOURCES[name].load(None), ""
    except Exception:
        return SOURCES[name].load(32), " (generated source; showing a sample of 32)"


def data_index(col: Collection) -> str:
    from ..data import SOURCES

    rows = [[link(f"/data/{name}", name), mono(src.unit),
             mono(str(src.knobs) if src.knobs else "-")]
            for name, src in sorted(SOURCES.items())]
    return _page(
        col, "data", "data",
        eyebrow("Data"),
        heading(1, "Data sources"),
        lede("Named, swappable sources; a spec binds one by name and a run records which slice "
             "it used."),
        table(["source", "unit", "knobs"], rows))


def data_source(col: Collection, name: str) -> str:
    from ..data import SOURCES

    units, sample = _units_of(name)
    src = SOURCES[name]
    shown = units[:20]
    rows = [[link(f"/data/{name}/{i}", f"{i:03d}"), esc(str(u)[:100])]
            for i, u in enumerate(shown)]
    return _page(
        col, f"data / {name}", name,
        eyebrow("Data source"),
        heading(1, esc(name)),
        note(f"unit: {src.unit} · knobs: {src.knobs if src.knobs else '-'} · {len(units)} "
             f"items{sample} · showing {len(shown)}"),
        table(["id", "preview"], rows))


def data_item(col: Collection, name: str, idx: int) -> str:
    units, _ = _units_of(name)
    u = units[idx]
    detail = (kv_table(list(zip(("clean", "corrupted"), (esc(str(v)) for v in u))))
              if isinstance(u, tuple) else note(str(u)))
    return _page(
        col, f"data / {name} / {idx:03d}", f"{name}/{idx}",
        eyebrow("Data item"),
        heading(1, esc(f"{name} · {idx:03d}")),
        detail,
        note_html(link(f"/data/{name}", f"back to {name}")))


def models(col: Collection) -> str:
    from ..profiles import PROFILES

    rows = [[link(f"/model/{fam}", fam), mono(p.blocks_path), mono(str(p.residual_denotation))]
            for fam, p in sorted(PROFILES.items())]
    return _page(
        col, "models", "models",
        eyebrow("Models"),
        heading(1, "Model families"),
        lede("Profiles: how the benchmark addresses each family's modules on each engine."),
        table(["family", "blocks", "residual denotation"], rows))


def model_page(col: Collection, family: str) -> str:
    from ..profiles import PROFILES
    from ..specs import SPECS

    p = PROFILES[family]
    facts = kv_table([
        ("blocks", esc(p.blocks_path)), ("final norm", esc(p.norm_path)),
        ("head", esc(p.head_path)),
        ("residual denotation", esc(str(p.residual_denotation))),
        ("attn / mlp names", esc(str((p.attn_name, p.mlp_name)))),
        ("vLLM tree prefix", esc(p.vllm_prefix or "-")),
    ])
    specs = "".join(f"<li>{esc(s.name)} "
                    f"<span class='note'>{esc(s.repo)}</span></li>"
                    for s in SPECS.values() if s.family == family) or "<li>none</li>"
    recorded = "".join(f"<li>{link(f'/spec/{group}', group)}</li>"
                       for group, names in col.by_spec().items()
                       if col.entries[names[0]][1]["coordinates"]["family"] == family) if col else ""
    return _page(
        col, f"model / {family}", family,
        eyebrow("Model family"),
        heading(1, esc(family)),
        facts,
        heading(2, "Specs on this family"),
        f"<ul>{specs}</ul>", heading(2, "Recorded experiments"),
        f"<ul>{recorded or '<li>none</li>'}</ul>")


_BACKEND_SETUPS = {
    "hf": "nnsight on vanilla transformers, eager, in-process",
    "vllm_async": "nnsight on the vLLM async engine, in-process",
    "vllm_sync": "nnsight on the vLLM sync engine, in-process",
    "vllm_serve": "nnsight on the vLLM async engine, served over HTTP",
}


def backends(col: Collection) -> str:
    from ..jobs.local import discover
    packaged = set(discover())
    names = col.backend_names() if col is not None else sorted(packaged)
    rows = [[link(f"/backend/{name}", name), mono(f"backends/{name}/compose.yml" if name in packaged
                                                 else "recorded backend/interface")]
            for name in names]
    return _page(col, "backends", "backends",
                 eyebrow("Backends"), heading(1, "Backends"),
                 table(["backend", "configuration"], rows))


def backend(col: Collection, name: str) -> str:
    from ..backends import IMPLS
    if name not in col.backend_names() and name not in IMPLS:
        raise KeyError(name)
    sections = []
    for rn, (out, prov) in sorted(col.entries.items()):
        if prov.get("view", {}).get("backend") == name:
            sections += [heading(2, link(f"/run/{rn}", rn)), note_html(_stack(prov)),
                         table(_RUN_TABLE_HEADERS, _cell_rows(col, rn, col.results[rn]))]
            continue
        if prov["coordinates"]["interface"] != name:
            continue
        probes = model.probe_results(out)
        if not probes:
            continue
        rows = [[esc(p), chip(m["state"]), note(m.get("note") or "")] for p, m in probes.items()]
        sections += [heading(2, f"Construct support <span class='note'>from "
                                f"{link(f'/run/{rn}', rn)}</span>"),
                     table(["construct", "state", "check / note"], rows)]
    if not sections:
        sections = [note_html("No recorded jobs for this backend in the directory. "
                              + (f"Legacy probes: <code>scripts/micro.py --backend {esc(name)}</code>."
                                 if name in IMPLS else f"Configuration: <code>backends/{esc(name)}/compose.yml</code>."))]
    return _page(
        col, f"backend / {name}", name,
        eyebrow("Backend"),
        heading(1, esc(name)),
        lede_html(f"{esc(_BACKEND_SETUPS.get(name, name))} · documented gaps in the "
                  f"{link('/catalog', 'methods catalog')}"),
        *sections)


# ---- rendered repo docs -------------------------------------------------------------------------

def _inline(s: str) -> str:
    s = esc(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    return s


def _md_html(md: str) -> str:
    """Headings (anchored), fenced code, pipe tables, paragraphs. Everything else passes through
    as text; this renders the repo's own docs, no foreign markdown."""
    out, in_code, in_table = [], False, False
    for line in md.splitlines():
        if line.startswith("```"):
            out.append("</pre>" if in_code else "<pre>")
            in_code = not in_code
            continue
        if in_code:
            out.append(esc(line))
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            tag = "td" if in_table else "th"
            if not in_table:
                out.append("<table>")
                in_table = True
            out.append("<tr>" + "".join(f"<{tag}>{_inline(c)}</{tag}>" for c in cells) + "</tr>")
            continue
        if in_table:
            out.append("</table>")
            in_table = False
        m = re.match(r"(#{1,4}) +(.*)", line)
        if m:
            lvl = min(len(m.group(1)) + 1, 4)   # doc h1 -> page h2: the page has its own h1
            anchor = re.sub(r"[^a-z0-9]+", "-", m.group(2).lower()).strip("-")
            out.append(f"<h{lvl} id='{anchor}'>{_inline(m.group(2))}</h{lvl}>")
            continue
        out.append(f"<p>{_inline(line)}</p>" if line.strip() else "")
    if in_table:
        out.append("</table>")
    if in_code:
        out.append("</pre>")
    return "\n".join(out)


def doc(col: Collection, title: str, relpath: str) -> str:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    return _page(col, title.lower(), title,
                 eyebrow("Docs"), heading(1, esc(title)),
                 _md_html((root / relpath).read_text()))


# ---- the single route registry ------------------------------------------------------------------

def _data_router(col: Collection, rest: str) -> str:
    from ..data import SOURCES

    if rest in SOURCES:                        # source names may contain "/"
        return data_source(col, rest)
    src, idx = rest.rsplit("/", 1)
    return data_item(col, src, int(idx))


def inbox_run(col: Collection, name: str):
    from pathlib import Path
    from .records import child
    if name not in col.inbox_runs():
        raise FileNotFoundError(name)
    directory = child(Path(col.inbox), name)
    if not (directory / "plan.json").is_file():
        raise FileNotFoundError(name)
    incoming = col.inbox_collection()
    rows = [[link(f"/run/inbox/{key}", key),
             chip(model.rollup([c.state for c in incoming.results[key] or []]))]
            for key in incoming.entries if key.startswith(name + "/")]
    return _page(col, f"inbox / {name}", name, heading(1, esc(name)), table(["job", "state"], rows))


ROUTES = [
    (re.compile(r"^/$"), lambda col, m: overview(col)),
    (re.compile(r"^/micro$"), lambda col, m: micro(col)),
    (re.compile(r"^/runs$"), lambda col, m: runs(col)),
    (re.compile(r"^/inbox$"), lambda col, m: inbox(col)),
    (re.compile(r"^/inbox-run/([^/]+)$"), lambda col, m: inbox_run(col, m.group(1))),
    (re.compile(r"^/run/(.+)$"), lambda col, m: run(col, m.group(1))),
    (re.compile(r"^/method/(.+)$"), lambda col, m: method(col, m.group(1))),
    (re.compile(r"^/spec/(.+)$"), lambda col, m: spec(col, m.group(1))),
    (re.compile(r"^/data$"), lambda col, m: data_index(col)),
    (re.compile(r"^/data/(.+)$"), lambda col, m: _data_router(col, m.group(1))),
    (re.compile(r"^/models$"), lambda col, m: models(col)),
    (re.compile(r"^/model/(.+)$"), lambda col, m: model_page(col, m.group(1))),
    (re.compile(r"^/backends$"), lambda col, m: backends(col)),
    (re.compile(r"^/backend/(.+)$"), lambda col, m: backend(col, m.group(1))),
    (re.compile(r"^/findings$"), lambda col, m: doc(col, "Findings", "docs/findings.md")),
    (re.compile(r"^/catalog$"),
     lambda col, m: doc(col, "Methods catalog", "docs/interp-methods-catalog.md")),
]


def dispatch(col: Collection, path: str) -> str | None:
    for pattern, handler in ROUTES:
        m = pattern.match(path)
        if m:
            try:
                return handler(col, m)
            except (KeyError, IndexError, FileNotFoundError, ValueError):
                return None
    return None


def enumerate_paths(col: Collection, items_per_source: int = 20) -> list[str]:
    """Every concrete path the ROUTES serve for this collection: the export and the crawl tests
    walk exactly this list, so a new page is exported the moment it is routed here."""
    from ..data import SOURCES
    from ..profiles import PROFILES

    paths = ["/", "/micro", "/runs", "/inbox", "/models", "/backends", "/data",
             "/findings", "/catalog"]
    paths += [f"/method/{m}" for m in sorted(_by_method(col))]
    paths += [f"/spec/{s}" for s in sorted(col.by_spec())]
    paths += [f"/run/{n}" for n in sorted(col.entries)]
    paths += [f"/run/inbox/{n}" for n in sorted(col.inbox_collection().entries)]
    from pathlib import Path
    paths += [f"/inbox-run/{n}" for n in col.inbox_runs() if (Path(col.inbox) / n / "plan.json").is_file()]
    paths += [f"/model/{f}" for f in sorted(PROFILES)]
    paths += [f"/backend/{b}" for b in col.backend_names()]
    for src in sorted(SOURCES):
        paths.append(f"/data/{src}")
        units, _ = _units_of(src)
        paths += [f"/data/{src}/{i}" for i in range(min(len(units), items_per_source))]
    return paths


def export_html(dir_path: str, inbox: str | None = None, items_per_source: int = 20) -> str:
    """The whole site as ONE self-contained page (hash-routed), for sharing a collection where
    no server runs. The demo's header persists across pages (the router updates its breadcrumb).
    Read-only: archive/discard forms are stripped."""
    from ..runfile import INBOX

    col = Collection(dir_path, inbox if inbox is not None else INBOX)

    def parts(full_page: str) -> list[str]:
        """[breadcrumb, hash-rewritten main content] for one rendered page."""
        m = re.search(r"id='crumb'>(.*?)</span>", full_page)
        body = full_page[full_page.index("<main>") + 6:full_page.rindex("</main>")]
        body = re.sub(r"<form.*?</form>", "", body, flags=re.S)     # read-only export
        body = body.replace("href='/", "href='#/").replace('href="/', 'href="#/')
        return [m.group(1) if m else "", body]

    pages = {p: parts(dispatch(col, p)) for p in enumerate_paths(col, items_per_source)}
    payload = json.dumps(pages).replace("</", "<\\/")
    header = header_bar("", dir_path, link("/inbox", f"inbox ({len(col.inbox_runs())})"))
    header = header.replace("href='/", "href='#/")
    return (f"<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>nnbench · {esc(dir_path)}</title>{STYLE}"
            f"{header}<main id='app'></main><script>const PAGES={payload};\n"
            "const go = () => {\n"
            "  const p = location.hash.replace(/^#/, '') || '/';\n"
            "  const e = PAGES[p];\n"
            "  document.getElementById('crumb').innerHTML = e ? e[0] : '';\n"
            "  document.getElementById('app').innerHTML = e ? e[1] :\n"
            "    '<h1>not exported</h1><p><a href=\"#/\">overview</a></p>';\n"
            "  window.scrollTo(0, 0);\n"
            "};\n"
            "window.addEventListener('hashchange', go); go();\n"
            "</script>")
