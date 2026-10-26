"""nnsight 0.8.1 best realizations on Qwen2.5-7B, HF against vLLM, all bf16: one report page (hero, findings,
per-workload bars, all-rows table, correctness). GPUs, versions and dates are read from the results.

usage: python opt7b.py RUN_DIR OUT.html
RUN_DIR holds the bench.py runs (report.json + experiments/**/result.json).
"""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build import E, SERIES, chip, figure, fnum, mounts, page_head, provenance, table  # noqa: E402

HF = "nnsight-hf/0-8-1"   # also the correctness reference
GEN, OPT, TAPS = "nnsight-vllm/0-8-1", "nnsight-vllm/0-8-1/opt", "nnsight-vllm/0-8-1/opt/taps"
PLAIN, PLAIN_E = "vllm-plain/0-27-1", "vllm-plain/0-27-1/eager"
COLS = [  # (backend, head, sub)
    (HF, "nnsight on HF", "general cell · transformers 5.12.1, sdpa"),
    (GEN, "nnsight on vLLM", "general cell · vLLM 0.27.1, eager"),
    (OPT, "nnsight on vLLM", "installed edits · vLLM 0.27.1, eager"),
    (TAPS, "nnsight on vLLM", "installed edits + taps · vLLM 0.27.1, CUDA graphs"),
]
SPEC_NAMES = [("logit_lens", "Logit lens"), ("steering", "Steering, one forward"), ("ablation", "Ablation"),
              ("gen_steering", "Steering at every decode step (8 tokens)"),
              ("activation_patching", "Activation patching (IOI pairs)")]
HEADLINE = [  # (fig id, spec, label, title)
    ("o-ll", "logit_lens", "every layer, last token", "Logit lens, every layer"),
    ("o-st", "steering", "add toward ' Rome' at half depth, scaled by each token's norm", "Steering at half depth, per-token norm"),
    ("o-ab", "ablation", "zero the MLP output at layer 1", "Zero the MLP output at layer 1"),
    ("o-pa", "activation_patching", "clean last-token residual into corrupt run at quarter depth", "Last-token patching at quarter depth"),
    ("o-gs", "gen_steering", "add toward ' Rome' at half depth, every step, scaled by each token's norm",
     "8-token generation, steered every step (per-token norm)"),
]
VSTATE = {"NUMERICAL_CHECK_PASSED": ("good", "✓", "Passed"), "NUMERICAL_MISMATCH": ("warn", "≈", "Mismatch"),
          "UNRESOLVED": ("none", "·", "Unresolved"), "EXECUTION_FAILED": ("crit", "×", "Failed"),
          "INVALID_REFERENCE": ("serious", "?", "Weak reference"), "RAN": ("none", "·", "Reference")}


def vchip(state):
    cls, icon, text = VSTATE.get(state, ("none", "?", state.title()))
    return f'<span class="chip {cls}"><span class="dot" aria-hidden="true">{icon}</span>{E(text)}</span>'


class Run:
    def __init__(self, run_dir):
        self.cells, self.ids, self.vanilla, self.order = {}, [], {}, []
        self.prov, self.dates, self.gpus = {}, set(), set()   # per backend, and over every job
        for path in sorted(glob.glob(f"{run_dir}/*/report.json")):
            d = json.load(open(path))
            self.ids.append(d["run"])
            for x in d["experiments"]:
                spec = x["spec"].split("/")[0]
                for c in x["cells"]:
                    self.cells[(spec, c["label"], c["backend"])] = c
                    if (spec, c["label"]) not in self.order:
                        self.order.append((spec, c["label"]))
                # each backend's own no-trace request (plain engine) for this experiment
                for res in glob.glob(f"{Path(path).parent}/experiments/{x['experiment_id'][:16]}*/**/result.json",
                                     recursive=True):
                    r = json.load(open(res))
                    p = self.prov[r["backend"]] = r["provenance"]
                    self.dates.add(p["executed"][:10])
                    self.gpus.add(f'{len(p["host"]["gpus"])} × {p["host"]["gpus"][0]["name"].removeprefix("NVIDIA ")}')
                    for a in r.get("auxiliary_calls", []):
                        if a["key"][0] == "__vanilla__" and a["record"].get("median_latency_ms"):
                            self.vanilla[(spec, r["backend"], a["key"][1])] = a["record"]["median_latency_ms"]
        order = {s: i for i, (s, _) in enumerate(SPEC_NAMES)}
        self.order.sort(key=lambda k: order[k[0]])

    def get(self, spec, label, b):
        return self.cells.get((spec, label, b))

    def ms(self, spec, label, b):
        c = self.get(spec, label, b)
        return c.get("median_latency_ms") if c and c.get("performance_eligible", True) else None

    def plain(self, spec, label, b):
        """The engine with nothing attached: plain compiled vLLM for a vLLM cell, plain HF for an HF cell."""
        c = self.get(spec, label, b)
        regime = c["workload"] if c else "interactive"
        v = self.vanilla.get((spec, HF if b.startswith("nnsight-hf") else PLAIN, regime))
        return v * 2 if v and spec == "activation_patching" else v   # patching is two forwards

    def vstate(self, spec, label, b):
        if b == HF:
            return "RAN"   # the reference column is what the others are scored against
        c = self.get(spec, label, b)
        return (c.get("validation_state") or c["state"]) if c else "not run"

    def nnsight(self, b):
        n = self.prov[b]["client"]["nnsight"]
        return f'{n["version"]} ({n["commit"][:7]})'

    def layers(self):
        return self.prov[HF]["experiment"]["spec"]["n_layers"]

    def correctness(self, cols):
        """One sentence counting the scored cells of `cols` by validation outcome."""
        cells = [c for k in self.order for b in cols if (c := self.get(k[0], k[1], b))]
        passed = sum(c.get("validation_state") == "NUMERICAL_CHECK_PASSED" for c in cells)
        weak = sum(c["state"] == "INVALID_REFERENCE" for c in cells)
        off = [c["metrics"] for c in cells if c["state"] == "NUMERICAL_MISMATCH"]
        other = len(cells) - passed - weak - len(off)
        span = lambda key: f'{fnum(min(m[key] for m in off), 2)} to {fnum(max(m[key] for m in off), 2)}'  # noqa: E731
        return (f"Of {len(cells)} cells, {passed} pass the numerical check against the reference, {weak} have a reference "
                f"effect too weak to score and {len(off)} differ from it"
                + (f" (top-1 {span('top1_agree')}, TV {span('tv')})" if off else "")
                + (f"; {other} are in other states (see Correctness)" if other else "")
                + "; under the validation policy weak-reference and differing cells are unresolved and stay in the "
                  "timing table.")


def short(label):
    return (label.replace("add toward ' Rome' at ", "").replace("clean ", "").replace(" into corrupt run", "")
            .replace("scaled by each token's norm", "per-token norm").replace("zero the ", ""))


def headline_spec(R, fid, spec, label):
    regime = R.get(spec, label, HF)["workload"]
    hf_plain, vl_plain = R.vanilla.get((spec, HF, regime)), R.vanilla.get((spec, PLAIN, regime))
    vals = [[hf_plain, R.ms(spec, label, HF), None],
            [vl_plain, R.ms(spec, label, GEN), min(v for v in (R.ms(spec, label, OPT), R.ms(spec, label, TAPS)) if v)]]
    taps, opt = R.ms(spec, label, TAPS), R.ms(spec, label, OPT)
    fastest = "installed edits + taps" if taps and (not opt or taps <= opt) else "installed edits"
    return {"kind": "hbars", "id": fid,
            "categories": [{"label": "HF transformers", "sub": "nnsight 0.8.1 · one general form"},
                           {"label": "vLLM 0.27.1", "sub": f"nnsight 0.8.1 · {fastest}"}],
            "values": vals, "missing": [[None, None, "no faster public form"], [None, None, None]],
            "series": [{"label": "plain engine, nothing attached", "color": "var(--deemph)"},
                       {"label": "general cell", "color": SERIES[1]},
                       {"label": "fastest documented realization", "color": SERIES[0]}],
            "fmt": "ms", "tickFmt": "ms", "catHead": "Engine", "labelWidth": 190}


def full_table(R):
    head = ["Workload", 'plain HF<span class="sub">no trace</span>'] + [f'{E(h)}<span class="sub">{E(s.split(" · ")[0])}</span>' for _, h, s in COLS] \
        + ['plain vLLM<span class="sub">0.27.1, compiled</span>', 'plain vLLM<span class="sub">0.27.1, eager</span>']
    rows, cur = [], None
    for spec, label in R.order:
        if spec != cur:
            rows.append({"group": E(dict(SPEC_NAMES)[spec])})
            cur = spec
        regime = R.get(spec, label, HF)["workload"]
        r = [E(short(label)), f"{fnum(R.vanilla.get((spec, HF, regime)))} ms"]
        for b, _, _ in COLS:
            ms, pl = R.ms(spec, label, b), R.plain(spec, label, b)
            r.append(f'<span class="sub">{E(R.vstate(spec, label, b).lower())}</span>' if ms is None
                     else f"{fnum(ms)} ms" + (f'<span class="sub">{ms / pl:.2f}× plain</span>' if pl else ""))
        r += [f"{fnum(R.vanilla.get((spec, PLAIN, regime)))} ms", f"{fnum(R.vanilla.get((spec, PLAIN_E, regime)))} ms"]
        rows.append(r)
    return table(head, rows, num_cols=set(range(1, len(head))))


def state_table(R):
    head = ["Workload"] + [f'{E(h)}<span class="sub">{E(s.split(" · ")[0])}</span>' for _, h, s in COLS]
    rows, cur = [], None
    for spec, label in R.order:
        if spec != cur:
            rows.append({"group": E(dict(SPEC_NAMES)[spec])})
            cur = spec
        r = [E(short(label))]
        for b, _, _ in COLS:
            c = R.get(spec, label, b)
            m = (c or {}).get("metrics") or {}
            extra = (f'<span class="cell-metrics">top-1 {m["top1_agree"]:.2f} · TV {m["tv"]:.3f}</span>'
                     if m.get("tv") is not None else "")
            r.append(vchip(R.vstate(spec, label, b)) + extra)
        rows.append(r)
    return table(head, rows)


def page(run_dir):
    R = Run(run_dir)
    specs, parts = [], []
    st = "add toward ' Rome' at half depth"
    ll, gs = HEADLINE[0][2], "add toward ' Rome' at half depth, every step"
    pa = HEADLINE[3][2]
    hf = {k: R.ms(k[0], k[1], HF) for k in R.order}
    gen = {k: R.ms(k[0], k[1], GEN) for k in R.order}
    opt = {k: R.ms(k[0], k[1], OPT) for k in R.order}
    taps = {k: R.ms(k[0], k[1], TAPS) for k in R.order}
    r_gen = {k: gen[k] / hf[k] for k in R.order if gen[k] and hf[k]}
    r_taps = {k: taps[k] / hf[k] for k in R.order if taps[k] and hf[k]}
    one = [k for k in R.order if k[0] in ("steering", "ablation", "activation_patching")]
    fwd = [k for k in one if k[0] != "activation_patching"]   # one prompt per call, like the plain request
    vs_plain = lambda col: {k: col[k] / R.plain(k[0], k[1], GEN) for k in R.order if col[k]}  # noqa: E731
    gen_vs_plain, opt_vs_plain, taps_vs_plain = vs_plain(gen), vs_plain(opt), vs_plain(taps)
    rng = lambda d, ks: f"{fnum(min(d[k] for k in ks), 2)}–{fnum(max(d[k] for k in ks), 2)}"  # noqa: E731
    hf_plain1 = R.vanilla[("steering", HF, "interactive")]
    vl_plain1 = R.vanilla[("steering", PLAIN, "interactive")]
    gpus = ", ".join(sorted(R.gpus))

    sections = [("best", "Best per engine"), ("table", "All rows"), ("changes", "What each realization is"),
                ("correctness", "Correctness"), ("scope", "Scope")]
    secs = "".join(f'<a href="#{sid}">{E(t)}</a>' for sid, t in sections)
    parts.append(f'<nav class="topbar" aria-label="Report"><div class="topbar-inner">'
                 f'<span class="brand">nnbench</span><div class="pages"><a aria-current="page">nnsight 0.8.1 · 7B realizations</a></div>'
                 f'<div class="sections">{secs}</div></div></nav><main class="frame">')
    parts.append(f'''<header class="hero">
<p class="eyebrow">nnbench · {E(gpus)} per job · {E(", ".join(sorted(R.dates)))}</p>
<h1>nnsight 0.8.1 on HF and on vLLM, at its fastest</h1>
<p class="lede">The same five workloads on Qwen2.5-7B-Instruct, bf16 on every engine. nnsight's HF path has one form, the general
cell. Its vLLM path is run three ways: the general cell on nnsight's synchronous <code>LLM</code> path, the same block installed once
as a named edit and selected per request, and that edit with CUDA-graph taps. Each column runs in its own job, with that engine's
no-trace request timed in the same job; correctness is scored against the HF column at the same precision.</p>
{provenance([
        ("Model", "Qwen2.5-7B-Instruct, bf16 on every timed backend"),
        ("GPU", f"{E(gpus)}, every job"),
        ("Reference", "the HF column: nnsight on HF transformers, bf16"),
        ("nnsight", f"{E(R.nnsight(GEN))}, the upstream release"),
        ("Runs", " ".join(f"<code>{E(i)}</code>" for i in R.ids)),
    ])}
</header>''')
    k_st, k_ll, k_gs, k_pa = ("steering", st), ("logit_lens", ll), ("gen_steering", gs), ("activation_patching", pa)
    parts.append(f'''<ul class="findings">
<li><b>General cell.</b> One steering forward: {fnum(hf[k_st])} ms on HF against {fnum(gen[k_st])} ms on vLLM
({fnum(r_gen[k_st], 2)}×). Over the single-forward rows vLLM's general cell runs at {rng(r_gen, one)}× HF, the logit lens at
{fnum(r_gen[k_ll], 2)}× and 8-token generation at {fnum(r_gen[k_gs], 2)}×.</li>
<li><b>Installed edits.</b> Installing the block once and selecting it per request takes the lens from {fnum(gen[k_ll])} to
{fnum(opt[k_ll])} ms, steering from {fnum(gen[k_st])} to {fnum(opt[k_st])} ms and patching from {fnum(gen[k_pa])} to
{fnum(opt[k_pa])} ms. Against plain compiled vLLM the general cell is {rng(gen_vs_plain, fwd)}× on the steering and ablation
rows and installed edits {rng(opt_vs_plain, fwd)}×.</li>
<li><b>Taps.</b> With installed edits under CUDA graphs vLLM runs at {rng(r_taps, one)}× HF on single forwards and
{fnum(r_taps[k_gs], 2)}× on generation ({fnum(taps[k_gs])} against {fnum(hf[k_gs])} ms), and at
{rng(taps_vs_plain, fwd + [k for k in R.order if k[0] == "gen_steering"])}× plain compiled vLLM on the steering, ablation and
generation rows ({fnum(taps_vs_plain[k_ll], 2)}× on the lens).</li>
<li><b>Plain engines.</b> A plain forward is {fnum(hf_plain1)} ms on HF and {fnum(vl_plain1)} ms on compiled vLLM; nnsight on HF
is {fnum(hf[k_st] / hf_plain1, 2)}× plain on steering and {fnum(hf[k_ll] / R.vanilla[("logit_lens", HF, "interactive")], 2)}× on
the lens, which reads and projects all {R.layers()} layers.</li>
<li><b>Correctness.</b> {E(R.correctness((GEN, OPT, TAPS)))}</li>
</ul>''')
    figs = []
    for fid, spec, label, title in HEADLINE:
        specs.append(headline_spec(R, fid, spec, label))
        figs.append(figure(fid, title, "Median of 7 timed calls after 3 warm-ups. Per engine: the plain engine with nothing attached, "
                           "nnsight's general cell, and the fastest documented realization (named under the engine)."))
    parts.append(f'''<section id="best"><h2>Best per engine</h2>
<div class="prose"><p>For each workload, the engine alone, nnsight's general cell on it, and the fastest form nnsight documents for that
engine. HF has one form; a missing bar says so.</p></div>
<div class="grid2">{"".join(figs[:4])}</div>{figs[4]}
</section>''')
    parts.append(f'''<section id="table"><h2>All rows</h2>
<div class="prose"><p>Every row of the five workloads: the plain engines, each nnsight column's median, and its ratio to its own engine
with nothing attached (plain HF for the HF column, plain compiled vLLM for the vLLM columns; two plain forwards for patching, which runs
a clean and a corrupted pass).</p></div>
{full_table(R)}</section>''')
    parts.append('''<section id="changes"><h2>What each realization is</h2>
<div class="audit">
<article><h3>General cell</h3><p>One trace per call. On vLLM it goes through nnsight's synchronous <code>LLM</code> path: the block is
built in the client, carried to the engine with the request, run against the live modules, and its saved values shipped back. Reads sit
at the documented sites (<code>lm_head.output[:, -1]</code> on HF, <code>model.logits</code> on vLLM; the lens projects through
<code>model.logits_processor</code>).</p></article>
<article><h3>Installed edits</h3><p>A block whose parameters are fixed per task is installed once with <code>model.edit(name=...)</code>;
every timed call is a plain request selecting it with <code>edits=[name]</code>, so nothing is built or serialized per call. Writes land in
place, readout takes the last row. Patching, whose clean value changes per call, stays two traces.</p></article>
<article><h3>Installed edits + taps</h3><p>The same edits on an engine started with <code>taps</code> naming only the touched sites, so vLLM
keeps CUDA graphs and breaks them at those sites (vLLM 0.27.1's breakable graphs). Decode steps replay the graph, which is where the gain
shows.</p></article>
<article><h3>HF transformers</h3><p>nnsight's <code>LanguageModel</code> on transformers 5.12.1, sdpa attention, bf16, one GPU; this
column is also the correctness reference. There is no faster documented form at batch one: the trace runs in-process, so there is
nothing to install or ship.</p></article>
</div></section>''')
    parts.append(f'''<section id="correctness"><h2>Correctness</h2>
<div class="prose"><p>Every cell against the HF column, bf16, on all prompts of its workload: the fraction of prompts whose top-1
token matches and the mean total-variation distance between next-token distributions. "Unresolved" is a row the oracle cannot settle:
the intervention's effect on the reference is too weak to tell apart from noise (shown without metrics), or the row's numbers differ
from the reference without a demonstrated contract violation (shown with them). Under the validation policy it stays in the timing
table.</p></div>
{state_table(R)}</section>''')
    parts.append(f'''<section id="scope"><h2>Scope</h2><div class="prose"><p>Batch one, short prompts, {E(gpus)}, Qwen2.5-7B. The
offline and concurrent request patterns, where a serving engine schedules many requests, are not on this page, nor is the two-GPU
comparison (HF tensor parallel against vLLM tensor parallel), which has its own page.</p></div></section>''')
    parts.append("</main>")
    return "\n".join(parts), specs


def main():
    run_dir, out = sys.argv[1], Path(sys.argv[2])
    body, specs = page(run_dir)
    out.write_text(page_head("nnsight 7B Realizations",
                             "nnsight 0.8.1 best realizations on Qwen2.5-7B, HF against vLLM, all bf16, one A100.")
                   + body + mounts(specs))
    print("wrote", out, len(specs), "charts")


if __name__ == "__main__":
    main()
