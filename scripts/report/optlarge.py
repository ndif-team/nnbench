"""nnsight 0.8.1 best realizations on a large model over two GPUs, HF against vLLM, all bf16: one report
page in the Delta report's format (same CSS/JS, hero, findings, per-workload bars, all-rows table,
correctness). Built for the Qwen2.5-72B run of 2026-10-22; the texts name that run.

usage: python optlarge.py RUN_DIR OUT.html
RUN_DIR holds the bench.py runs (report.json + experiments/**/result.json).
"""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build import E, SERIES, chip, figure, fnum, mounts, page_head, provenance, table  # noqa: E402

HF, HF_TP = "nnsight-hf/sdpa/0-8-1/gpu2", "nnsight-hf/sdpa/0-8-1/tp2"
GEN, OPT, TAPS = ("nnsight-vllm/sync/0-8-1/byref/tp2", "nnsight-vllm/opt/0-8-1/byref/tp2",
                  "nnsight-vllm/opt/0-8-1/taps/byref/tp2")
PLAIN = "vllm-plain/0-28-0/tp2"
COLS = [  # (backend, head, sub)
    (HF, "nnsight on HF", "general cell · device_map over two GPUs, sdpa"),
    (HF_TP, "nnsight on HF", "general cell · tensor parallel, transformers 5.19"),
    (GEN, "nnsight on vLLM", "general cell · vLLM 0.27.1 TP2, eager"),
    (OPT, "nnsight on vLLM", "installed edits · vLLM 0.27.1 TP2, eager"),
    (TAPS, "nnsight on vLLM", "installed edits + taps · vLLM 0.27.1 TP2, CUDA graphs"),
]
PLAIN_OF = {HF: HF, HF_TP: HF_TP, GEN: PLAIN, OPT: PLAIN, TAPS: PLAIN}
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
        self.cells, self.ids, self.vanilla, self.order, self.hosts = {}, [], {}, [], set()
        for path in sorted(glob.glob(f"{run_dir}/*/report.json")):
            d = json.load(open(path))
            self.ids.append(d["run"])
            for x in d["experiments"]:
                spec = x["spec"].split("/")[0]
                for c in x["cells"]:
                    self.cells[(spec, c["label"], c["backend"])] = c
                    if (spec, c["label"]) not in self.order:
                        self.order.append((spec, c["label"]))
                for res in glob.glob(f"{Path(path).parent}/experiments/{x['experiment_id'][:16]}*/**/result.json",
                                     recursive=True):
                    r = json.load(open(res))
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

    def regime(self, spec, label):
        return self.get(spec, label, HF)["workload"]

    def plain(self, spec, label, b):
        """The engine with nothing attached: each HF column's own no-trace call, plain compiled vLLM for a vLLM cell."""
        v = self.vanilla.get((spec, PLAIN_OF[b], self.regime(spec, label)))
        return v * 2 if v and spec == "activation_patching" else v   # patching is two forwards

    def vstate(self, spec, label, b):
        if b == HF:
            return "RAN"   # the reference column is what the others are scored against
        c = self.get(spec, label, b)
        return (c.get("validation_state") or c["state"]) if c else "not run"

    def metrics(self, spec, label, b):
        return ((self.get(spec, label, b) or {}).get("metrics") or {})


def short(label):
    return (label.replace("add toward ' Rome' at ", "").replace("clean ", "").replace(" into corrupt run", "")
            .replace("scaled by each token's norm", "per-token norm").replace("zero the ", ""))


def headline_spec(R, fid, spec, label):
    regime = R.regime(spec, label)
    vals = [[R.vanilla.get((spec, HF, regime)), R.ms(spec, label, HF), None],
            [R.vanilla.get((spec, HF_TP, regime)), R.ms(spec, label, HF_TP), None],
            [R.vanilla.get((spec, PLAIN, regime)), R.ms(spec, label, GEN),
             min(v for v in (R.ms(spec, label, OPT), R.ms(spec, label, TAPS)) if v)]]
    taps, opt = R.ms(spec, label, TAPS), R.ms(spec, label, OPT)
    fastest = "installed edits + taps" if taps and (not opt or taps <= opt) else "installed edits"
    return {"kind": "hbars", "id": fid,
            "categories": [{"label": "HF, device_map", "sub": "nnsight 0.8.1 · one general form"},
                           {"label": "HF, tensor parallel", "sub": "nnsight 0.8.1 · one general form"},
                           {"label": "vLLM 0.27.1, TP2", "sub": f"nnsight 0.8.1 · {fastest}"}],
            "values": vals,
            "missing": [[None, None, "no faster public form"], [None, None, "no faster public form"], [None, None, None]],
            "series": [{"label": "plain engine, nothing attached", "color": "var(--deemph)"},
                       {"label": "general cell", "color": SERIES[1]},
                       {"label": "fastest documented realization", "color": SERIES[0]}],
            "fmt": "ms", "tickFmt": "ms", "catHead": "Engine", "labelWidth": 190}


def full_table(R):
    head = ["Workload", 'plain HF<span class="sub">device_map, no trace</span>'] \
        + [f'{E(h)}<span class="sub">{E(s.split(" · ")[0])}{"" if b != HF_TP else ", TP"}</span>' for b, h, s in COLS] \
        + ['plain vLLM<span class="sub">0.28.0 TP2, compiled</span>']
    rows, cur = [], None
    for spec, label in R.order:
        if spec != cur:
            rows.append({"group": E(dict(SPEC_NAMES)[spec])})
            cur = spec
        regime = R.regime(spec, label)
        r = [E(short(label)), f"{fnum(R.vanilla.get((spec, HF, regime)))} ms"]
        for b, _, _ in COLS:
            ms, pl = R.ms(spec, label, b), R.plain(spec, label, b)
            r.append(f'<span class="sub">{E(R.vstate(spec, label, b).lower())}</span>' if ms is None
                     else f"{fnum(ms)} ms" + (f'<span class="sub">{ms / pl:.2f}× plain</span>' if pl else ""))
        r.append(f"{fnum(R.vanilla.get((spec, PLAIN, regime)))} ms")
        rows.append(r)
    return table(head, rows, num_cols=set(range(1, len(head))))


def state_table(R):
    head = ["Workload"] + [f'{E(h)}<span class="sub">{E(s.split(" · ")[0])}{"" if b != HF_TP else ", TP"}</span>' for b, h, s in COLS]
    rows, cur = [], None
    for spec, label in R.order:
        if spec != cur:
            rows.append({"group": E(dict(SPEC_NAMES)[spec])})
            cur = spec
        r = [E(short(label))]
        for b, _, _ in COLS:
            m = R.metrics(spec, label, b)
            extra = (f'<span class="cell-metrics">top-1 {m["top1_agree"]:.2f} · TV {m["tv"]:.3f}</span>'
                     if m.get("tv") is not None else "")
            r.append(vchip(R.vstate(spec, label, b)) + extra)
        rows.append(r)
    return table(head, rows)


def summary(R):
    """The numbers the findings quote, also printed so the findings file can be written from them."""
    st = "add toward ' Rome' at half depth"
    ll, gs, pa = HEADLINE[0][2], "add toward ' Rome' at half depth, every step", HEADLINE[3][2]
    keys = {"st": ("steering", st), "ll": ("logit_lens", ll), "gs": ("gen_steering", gs), "pa": ("activation_patching", pa)}
    col = lambda b: {k: R.ms(k[0], k[1], b) for k in R.order}  # noqa: E731
    hf, tp, gen, opt, taps = col(HF), col(HF_TP), col(GEN), col(OPT), col(TAPS)
    ratio = lambda a, b: {k: a[k] / b[k] for k in R.order if a[k] and b[k]}  # noqa: E731
    one = [k for k in R.order if k[0] in ("steering", "ablation", "activation_patching")]
    S = {"keys": keys, "hf": hf, "tp": tp, "gen": gen, "opt": opt, "taps": taps,
         "r_gen": ratio(gen, hf), "r_taps": ratio(taps, hf), "r_tp": ratio(tp, hf), "r_opt_gen": ratio(opt, gen),
         "taps_vs_plain": {k: taps[k] / R.plain(k[0], k[1], TAPS) for k in R.order if taps[k]},
         "gen_vs_plain": {k: gen[k] / R.plain(k[0], k[1], GEN) for k in R.order if gen[k]},
         "hf_vs_plain": {k: hf[k] / R.plain(k[0], k[1], HF) for k in R.order if hf[k]},
         "one": one,
         "hf_plain1": R.vanilla[("steering", HF, "interactive")], "tp_plain1": R.vanilla[("steering", HF_TP, "interactive")],
         "vl_plain1": R.vanilla[("steering", PLAIN, "interactive")],
         "passed": {b: sum(1 for k in R.order if R.vstate(k[0], k[1], b) == "NUMERICAL_CHECK_PASSED") for b in (HF_TP, GEN, OPT, TAPS)},
         "rows": len(R.order)}
    return S


def page(run_dir):
    R = Run(run_dir)
    S = summary(R)
    k_st, k_ll, k_gs, k_pa = S["keys"]["st"], S["keys"]["ll"], S["keys"]["gs"], S["keys"]["pa"]
    hf, tp, gen, opt, taps = S["hf"], S["tp"], S["gen"], S["opt"], S["taps"]
    r_gen, r_taps, r_tp, one = S["r_gen"], S["r_taps"], S["r_tp"], S["one"]
    tvp = S["taps_vs_plain"]
    rng = lambda d, ks: f"{fnum(min(d[k] for k in ks), 2)}–{fnum(max(d[k] for k in ks), 2)}"  # noqa: E731
    pa_rows = [k for k in R.order if k[0] == "activation_patching"]
    pa_mis = [short(k[1]) for k in pa_rows if R.vstate(k[0], k[1], OPT) != "NUMERICAL_CHECK_PASSED"]
    specs, parts = [], []

    sections = [("best", "Best per engine"), ("table", "All rows"), ("changes", "What each realization is"),
                ("correctness", "Correctness"), ("scope", "Scope")]
    secs = "".join(f'<a href="#{sid}">{E(t)}</a>' for sid, t in sections)
    parts.append(f'<nav class="topbar" aria-label="Report"><div class="topbar-inner">'
                 f'<span class="brand">nnbench · Delta</span><div class="pages"><a aria-current="page">nnsight 0.8.1 · 72B realizations</a>'
                 f'<a href="https://claude.ai/artifact/GRYnyVL3behEuvLtnLUCYH">7B realizations</a>'
                 f'<a href="https://claude.ai/artifact/RgnckstVBhvGuq3tsw4jyZ">2026-10-02 systems report</a></div>'
                 f'<div class="sections">{secs}</div></div></nav><main class="frame">')
    parts.append(f'''<header class="hero">
<p class="eyebrow">nnbench · NCSA Delta · two H200 (141 GB) per job · 2026-10-22</p>
<h1>nnsight 0.8.1 on HF and on vLLM at 72B, at its fastest</h1>
<p class="lede">The same five workloads on Qwen2.5-72B-Instruct, bf16 on every engine, over two GPUs. nnsight's HF path is run two
ways: accelerate's device_map, which splits the layers across the cards, and nnsight's transformers tensor parallelism. Its vLLM path
is run three ways under tensor parallelism: the general cell, the same block installed once as a named edit, and that edit with
CUDA-graph taps. The vLLM client uses envoys by reference. Every column is timed in the same job on the same cards, with each engine's
no-trace request beside it; correctness is scored against the HF device_map column at the same precision.</p>
{provenance([
        ("Model", "Qwen2.5-72B-Instruct, bf16 on every backend"),
        ("GPUs", "2 × H200 141 GB, gpue02, both jobs"),
        ("Reference", "nnsight 0.8.1 on HF transformers 5.12.1, sdpa, device_map over the two GPUs"),
        ("nnsight", "0.8.1 (de7fec4) with envoys by reference on vLLM; transformers 5.19 for the TP column"),
        ("Runs", " ".join(f"<code>{E(i)}</code>" for i in R.ids)),
    ])}
</header>''')
    parts.append(f'''<ul class="findings">
<li><b>At 72B the general cells favour vLLM.</b> One steering forward: {fnum(hf[k_st])} ms on HF with device_map against
{fnum(gen[k_st])} ms on vLLM ({fnum(r_gen[k_st], 2)}×). Over the single-forward rows vLLM's general cell runs at
{rng(r_gen, one)}× HF, 8-token generation at {fnum(r_gen[k_gs], 2)}×; the logit lens is again the exception at
{fnum(r_gen[k_ll], 2)}×.</li>
<li><b>Taps widen it.</b> Installed edits under CUDA graphs run at {rng(r_taps, one)}× HF on single forwards and
{fnum(r_taps[k_gs], 2)}× on generation ({fnum(taps[k_gs])} against {fnum(hf[k_gs])} ms), within
{rng(tvp, [k for k in R.order if k[0] not in ("logit_lens",)])}× of plain compiled vLLM on everything but the lens. On the lens
taps are slower than the general cell ({fnum(taps[k_ll])} against {fnum(gen[k_ll])} ms): the graph is broken at each of the 80 read
sites, and a single forward has no decode steps to replay it.</li>
<li><b>nnsight's HF tensor parallelism is slower than device_map at this size.</b> The TP column runs at {rng(r_tp, one)}× the
device_map column on single forwards and {fnum(r_tp[k_gs], 2)}× on generation, and its plain forward is {fnum(S["tp_plain1"])} ms
against {fnum(S["hf_plain1"])} ms: a single prompt gains nothing from splitting each layer while paying the collective per layer.
vLLM's TP forward is {fnum(S["vl_plain1"])} ms on the same cards.</li>
<li><b>nnsight's own cost on HF is small except on the lens.</b> {fnum(S["hf_vs_plain"][k_st], 2)}× plain on steering,
{fnum(S["hf_vs_plain"][k_gs], 2)}× on generation, {fnum(S["hf_vs_plain"][k_pa], 2)}× on patching (two forwards) and
{fnum(S["hf_vs_plain"][k_ll], 2)}× on the lens, which reads and projects all 80 layers.</li>
<li><b>Installed edits alone gain nothing here.</b> With envoys by reference the edit realization is level with or slower than the
general cell: the lens {fnum(gen[k_ll])} → {fnum(opt[k_ll])} ms, steering {fnum(gen[k_st])} → {fnum(opt[k_st])} ms, patching
{fnum(gen[k_pa])} → {fnum(opt[k_pa])} ms. The gain is the taps.</li>
<li><b>Correctness holds for the general cell; three installed-edit patching rows are wrong.</b> The lens and steering pass on every
column (top-1 0.94 to 1.00, TV at most 0.017); ablation is unresolved on every column, HF TP included (the effect is too weak to
score); the two alpha-0.5 generation rows are unresolved on the vLLM columns (top-1 0.90 to 0.94) and pass on HF TP. Patching passes
on the general cell on all {len(pa_rows)} rows, and on the installed-edit realizations only at "{E(short(pa_rows[0][1]))}"; the other
{len(pa_mis)} rows disagree with the reference outright (top-1 0.25 to 0.69, TV 0.26 to 0.62), the same on edits with and without
taps. The same cells pass at 7B on one GPU, so this is the installed-edit patching cell's in-place write under tensor parallelism: a
finding to classify, not a cell to patch here.</li>
</ul>''')
    figs = []
    for fid, spec, label, title in HEADLINE:
        specs.append(headline_spec(R, fid, spec, label))
        figs.append(figure(fid, title, "Median of 7 timed calls after 3 warm-ups. Per engine: the plain engine with nothing attached, "
                           "nnsight's general cell, and the fastest documented realization (named under the engine)."))
    parts.append(f'''<section id="best"><h2>Best per engine</h2>
<div class="prose"><p>For each workload, the engine alone, nnsight's general cell on it, and the fastest form nnsight documents for that
engine. HF has one form on either multi-GPU path; a missing bar says so.</p></div>
<div class="grid2">{"".join(figs[:4])}</div>{figs[4]}
</section>''')
    parts.append(f'''<section id="table"><h2>All rows</h2>
<div class="prose"><p>Every row of the five workloads: the plain engines, each nnsight column's median, and its ratio to its own engine
with nothing attached (each HF column's own no-trace call, plain compiled vLLM for the vLLM columns; two plain forwards for patching,
which runs a clean and a corrupted pass).</p></div>
{full_table(R)}</section>''')
    parts.append('''<section id="changes"><h2>What each realization is</h2>
<div class="audit">
<article><h3>HF, device_map</h3><p>nnsight's <code>LanguageModel</code> on transformers 5.12.1, sdpa attention, bf16, accelerate's
<code>device_map="auto"</code>: the 80 layers split across the two cards and run one card after the other. The trace runs in-process,
so there is nothing to install or ship; this column is also the correctness reference.</p></article>
<article><h3>HF, tensor parallel</h3><p>The same model under nnsight's transformers tensor parallelism
(<code>DistributedConfig(tp_size=2)</code>, transformers 5.19, one <code>torchrun</code> rank per card): every layer is sharded over
both cards and the ranks run it together. Parameters are DTensor shards; the cells read the steering direction from the checkpoint
instead of the sharded <code>lm_head.weight</code>.</p></article>
<article><h3>General cell</h3><p>One trace per call on vLLM 0.27.1 with <code>tensor_parallel_size=2</code>: the block is built in the
client, carried to the engine with the request, run against the live modules, and its saved values shipped back. Reads sit at the
documented sites (<code>model.logits</code>; the lens projects through <code>model.logits_processor</code>). The client runs with envoys
by reference, so a request carries the block and not the module tree.</p></article>
<article><h3>Installed edits</h3><p>A block whose parameters are fixed per task is installed once with <code>model.edit(name=...)</code>;
every timed call is a plain request selecting it with <code>edits=[name]</code>. Patching, whose clean value changes per call, stays two
traces.</p></article>
<article><h3>Installed edits + taps</h3><p>The same edits on an engine started with <code>taps</code> naming only the touched sites, so
vLLM keeps CUDA graphs and breaks them at those sites (vLLM 0.27.1's breakable graphs). Decode steps replay the graph, which is where
the gain shows; the all-layer lens breaks the graph 80 times and loses.</p></article>
</div></section>''')
    parts.append(f'''<section id="correctness"><h2>Correctness</h2>
<div class="prose"><p>Every cell against nnsight on HF with device_map, bf16, on all 16 prompts of its workload: the fraction of prompts
whose top-1 token matches and the mean total-variation distance between next-token distributions. "Unresolved" is a row the oracle
cannot settle (the intervention's effect is too weak to tell apart from noise, or a graded row forks at a near-tie) or a row whose
numbers differ from the reference without a demonstrated contract violation; under the validation policy it stays in the timing table.
The installed-edit patching rows at top-1 0.25 to 0.69 are wrong numbers, not noise, and are listed as such in the findings.</p></div>
{state_table(R)}</section>''')
    parts.append('''<section id="scope"><h2>Scope</h2><div class="prose"><p>Batch one, short prompts, two H200s, Qwen2.5-72B in bf16.
The offline and concurrent request patterns are not on this page, nor is stock nnsight without envoys by reference (measured at 7B
only). The 7B page shows the same five workloads on one A100; its HF column is one GPU, so the two pages are not a scaling
series.</p></div></section>''')
    parts.append("</main>")
    return "\n".join(parts), specs, S


def main():
    run_dir, out = sys.argv[1], Path(sys.argv[2])
    body, specs, S = page(run_dir)
    out.write_text(page_head("nnsight 72B Realizations",
                             "nnsight 0.8.1 best realizations on Qwen2.5-72B, HF against vLLM, all bf16, two H200s.")
                   + body + mounts(specs))
    print("wrote", out, len(specs), "charts")
    for name in ("r_gen", "r_taps", "r_tp", "r_opt_gen", "taps_vs_plain", "gen_vs_plain", "hf_vs_plain"):
        print(name, {f"{k[0]}:{short(k[1])[:28]}": round(v, 2) for k, v in S[name].items()})
    print("plain steering: hf", S["hf_plain1"], "hf_tp", S["tp_plain1"], "vllm", S["vl_plain1"], "passed", S["passed"], "rows", S["rows"])


if __name__ == "__main__":
    main()
