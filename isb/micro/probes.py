"""Micro tier — Level 0/1 primitive probes (design.md §3.7, §12.6).

One minimal probe per row of the primitive inventory
(`docs/interp-methods-catalog.md`), per backend. A probe measures ONE primitive —
a control op or its cross-edge data movement (iteration loop-carried saves /
barrier fork-join sharing / session cross-region flow / edit staging replay /
scan region mode) or a Level-1 site (boundary `.input`, engine `logits`/`samples`,
derived head/neuron, non-attention `.source`) — with a self-contained denotation
check, so its verdict uses the same applicability states as method cells:

  SUPPORTED       ran AND the denotation check passed
  SILENTLY_WRONG  ran, no error, denotation check failed (the dangerous cell)
  ERROR           raised a clean exception (runner catches; last line recorded)
  HANG            exceeded the per-probe timeout (runner-level)

Probes are explicit per (name, backend) like method cells — same §12 idiom, no
abstraction. Family: GPT-2 only for now (the micro tier needs ONE family; per-family
denotation differences like the fused residual stay in the method tier where the
family axis lives). vLLM probes own their async drain in the probe body because the
trace body must share a frame with `with model.trace(...)` (isb/backends/base.py).

Registration order per backend IS execution order: vLLM probes are ordered
safest-first so a HANG (which poisons the persistent event loop) aborts as little
as possible — the runner stops a backend's sweep at the first HANG.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..oracle.equivalence import compare, is_equivalent
from ..states import AppState

PROMPT = "The Eiffel Tower is in"
CLEAN, CORRUPT = "The Eiffel Tower is in", "The Colosseum is in"

PROBES = {}  # (name, backend) -> fn(be, model) -> (state, note)


def probe(name: str, backend: str):
    def deco(fn):
        PROBES[(name, backend)] = fn
        return fn

    return deco


def names_for(backend: str) -> list:
    """Probe names for a backend, in REGISTRATION order (= safe execution order)."""
    return [n for (n, b) in PROBES if b == backend]


def _untuple(x):
    return x[0] if isinstance(x, tuple) else x


def _cpu(t):
    return t.detach().float().cpu()


def _rel_dev(got, want) -> float:
    """Max deviation relative to the target's own scale — layout/denotation checks
    compare a CPU-fp32 reconstruction against a kernel-computed tensor, so the
    tolerance is relative, not absolute."""
    got, want = _cpu(got), _cpu(want)
    denom = want.abs().max().item() or 1.0
    return (got - want).abs().max().item() / denom


def _verdict_close(rel: float, tol: float, what: str):
    state = AppState.SUPPORTED if rel <= tol else AppState.SILENTLY_WRONG
    return state, f"{what}: rel-dev={rel:.2e} (tol {tol:.0e})"


def _verdict_compare(ref, got, what: str):
    m = compare(ref, got)
    state = AppState.SUPPORTED if is_equivalent(m) else AppState.SILENTLY_WRONG
    return state, f"{what}: top1={m['top1_agree']:.2f} tv={m['tv']:.3f}"


def _vllm_saves(out) -> dict:
    """Unwrap a finished async output's nested `.saves`, surfacing a worker
    exception payload as a clean error (mirrors VLLMAsyncBackend._extract, but
    returns the whole dict — micro probes save lists/scalars, not one tensor)."""
    if not hasattr(out, "saves"):
        raise RuntimeError("finished output carried no saves")
    saves = out.saves
    while (
        isinstance(saves, dict)
        and len(saves) == 1
        and isinstance(next(iter(saves.values())), dict)
        and not {"type_name", "message", "traceback"} <= set(saves)
    ):
        saves = next(iter(saves.values()))
    if isinstance(saves, dict) and {"type_name", "message", "traceback"} <= set(saves):
        msg = (saves.get("message") or "").strip().splitlines()
        raise RuntimeError(f"worker intervention: {msg[-1] if msg else saves['type_name']}")
    return saves


# ---------------------------------------------------------------------------
# HF probes
# ---------------------------------------------------------------------------

@probe("input_boundary", "hf")
def input_boundary_hf(be, model):
    """Level-1 site: boundary `.input`. Denotation: block[6].input is block[5]'s
    residual output (forward-order access within the trace)."""
    with model.trace(PROMPT):
        out5 = model.transformer.h[5].output[0].save()
        inp6 = model.transformer.h[6].input.save()
    return _verdict_close(_rel_dev(inp6, out5), 1e-5, "h[6].input vs h[5].output[0]")


@probe("engine_logits", "hf")
def engine_logits_hf(be, model):
    """Level-1 engine site: the model-root logits vs the lm_head boundary read."""
    with model.trace(PROMPT):
        lm = model.lm_head.output[:, -1, :].save()
        root = model.output.logits[:, -1, :].save()
    return _verdict_compare(_cpu(lm), _cpu(root), "model.output.logits vs lm_head.output")


@probe("engine_samples", "hf")
def engine_samples_hf(be, model):
    """Level-1 engine site: the sampled token (`generator.output` final sequence).
    Denotation under greedy: last generated id == argmax of the last-step logits."""
    with model.generate(PROMPT, max_new_tokens=1, do_sample=False) as tracer:
        lm = model.lm_head.output[:, -1, :].save()
        seq = model.generator.output.save()
    tok, arg = int(seq[0, -1].item()), int(_cpu(lm).argmax(-1).item())
    state = AppState.SUPPORTED if tok == arg else AppState.SILENTLY_WRONG
    return state, f"generated id {tok} vs greedy argmax {arg}"


@probe("derived_head", "hf")
def derived_head_hf(be, model):
    """Level-1 derived site: head slice = reshape of the o-proj input (READ ∘ view).
    Denotation: reconstructing c_proj from that input reproduces attn.output[0],
    proving the captured tensor (and so the head view) is the right one."""
    attn = model.transformer.h[6].attn
    with model.trace(PROMPT):
        cp_in = attn.c_proj.input.save()
        attn_out = attn.output[0].save()
    H = attn.num_heads
    x = _cpu(cp_in)
    head4 = x.view(*x.shape[:-1], H, x.shape[-1] // H)[..., 4, :]
    recon = x @ _cpu(attn.c_proj.weight) + _cpu(attn.c_proj.bias)  # HF Conv1D: y = x W + b
    state, note = _verdict_close(_rel_dev(recon, attn_out), 1e-3, "c_proj recon vs attn.output[0]")
    return state, f"head slice {tuple(head4.shape)}; {note}"


@probe("derived_neuron", "hf")
def derived_neuron_hf(be, model):
    """Level-1 derived site: neuron slice of the MLP activation. Denotation:
    act.output == gelu_new(c_fc.output) (GPT-2's NewGELU is the tanh approximation)."""
    mlp = model.transformer.h[6].mlp
    with model.trace(PROMPT):
        cfc = mlp.c_fc.output.save()
        act = mlp.act.output.save()
    neuron7 = _cpu(act)[..., 7]
    recon = F.gelu(_cpu(cfc), approximate="tanh")
    state, note = _verdict_close(_rel_dev(recon, act), 1e-3, "gelu_new(c_fc) vs act.output")
    return state, f"neuron slice {tuple(neuron7.shape)}; {note}"


@probe("source_mlp", "hf")
def source_mlp_hf(be, model):
    """Level-1 internal site, non-attention: `.source.self_c_fc_0` inside the MLP
    forward. Two traces (direct child read, then source-op read) so the probe never
    mixes source-rewritten and plain hooks in one pass; eval mode is deterministic."""
    mlp = model.transformer.h[6].mlp
    with model.trace(PROMPT):
        direct = mlp.c_fc.output.save()
    with model.trace(PROMPT):
        via_source = mlp.source.self_c_fc_0.output.save()
    return _verdict_close(_rel_dev(via_source, direct), 1e-5, "source.self_c_fc_0 vs c_fc.output")


def _iter_hf(be, model, iter_slice):
    """Per-step boundary reads over a 3-token generate; denotation: 3 steps collected
    and step 0 equals a plain single trace."""
    with model.trace(PROMPT):
        ref = model.lm_head.output[:, -1, :].save()
    with model.generate(PROMPT, max_new_tokens=3, do_sample=False) as tracer:
        rows = list().save()
        for step in tracer.iter[iter_slice]:
            rows.append(model.lm_head.output[:, -1, :])
    if len(rows) != 3:
        return AppState.SILENTLY_WRONG, f"expected 3 per-step reads, got {len(rows)}"
    state, note = _verdict_compare(_cpu(ref), _cpu(rows[0]), "step0 vs plain trace")
    return state, f"3 steps; {note}"


@probe("iteration_bounded", "hf")
def iteration_bounded_hf(be, model):
    """Level-0 construct, BOUNDED realization: `tracer.iter[0:3]` (explicit stop)."""
    return _iter_hf(be, model, slice(0, 3))


@probe("iteration_unbounded", "hf")
def iteration_unbounded_hf(be, model):
    """Level-0 construct, UNBOUNDED realization: `tracer.iter[:]` — the stop bound
    comes from `interleaver.default_all` (set from max_new_tokens on the HF path)."""
    return _iter_hf(be, model, slice(None))


@probe("scan", "hf")
def scan_hf(be, model):
    """Level-0 construct: shape-only execution. Denotation: the block-output shape
    matches (1, n_prompt_tokens, hidden) without running real kernels."""
    import nnsight

    n_tok = len(model.tokenizer(PROMPT)["input_ids"])
    hidden = model.config.n_embd
    with model.scan(PROMPT):
        shp = nnsight.save(tuple(model.transformer.h[0].output[0].shape))
    want = (1, n_tok, hidden)
    state = AppState.SUPPORTED if tuple(shp) == want else AppState.SILENTLY_WRONG
    return state, f"scanned shape {tuple(shp)} vs expected {want}"


@probe("edit", "hf")
def edit_hf(be, model):
    """Level-0 construct: persistent non-inplace edit (zero attn@6 — non-vacuous on
    GPT-2, it flips the top-1). Denotation: edited trace == in-trace ablation;
    isolation: the ORIGINAL model is unchanged."""
    def ablate_inline(m):
        out = m.transformer.h[6].attn.output
        m.transformer.h[6].attn.output = (torch.zeros_like(out[0]), *out[1:])
        return m.lm_head.output[:, -1, :]

    with model.trace(PROMPT):
        base = model.lm_head.output[:, -1, :].save()
    with model.trace(PROMPT):
        ref_abl = ablate_inline(model).save()
    with model.edit() as edited:
        out = edited.transformer.h[6].attn.output
        edited.transformer.h[6].attn.output = (torch.zeros_like(out[0]), *out[1:])
    with edited.trace(PROMPT):
        ed = edited.lm_head.output[:, -1, :].save()
    with model.trace(PROMPT):
        base_after = model.lm_head.output[:, -1, :].save()

    if is_equivalent(compare(_cpu(base), _cpu(ed))):
        return AppState.SILENTLY_WRONG, "edit had no effect (edited == unedited baseline)"
    if not is_equivalent(compare(_cpu(base), _cpu(base_after))):
        return AppState.SILENTLY_WRONG, "non-inplace edit leaked into the original model"
    return _verdict_compare(_cpu(ref_abl), _cpu(ed), "edited trace vs in-trace ablation")


@probe("session_saved", "hf")
def session_saved_hf(be, model):
    """Level-0 construct, SAVED flow: `.save()` inside a session trace, read after
    session exit. Denotation: equals the same read from a plain trace."""
    with model.trace(PROMPT):
        ref = model.transformer.ln_f.output[:, -1, :].save()
    with model.session():
        with model.trace(PROMPT):
            lg = model.transformer.ln_f.output[:, -1, :].save()
    return _verdict_close(_rel_dev(lg, ref), 1e-5, "session-saved read vs plain trace")


@probe("session_unsaved", "hf")
def session_unsaved_hf(be, model):
    """Level-0 construct, UN-SAVED cross-trace flow (the session contract): a variable
    produced in trace 1 WITHOUT .save() consumed in trace 2. Denotation: same prompt,
    deterministic eager → difference is 0."""
    with model.session():
        with model.trace(PROMPT):
            v = model.transformer.ln_f.output[:, -1, :]
        with model.trace(PROMPT):
            diff = (model.transformer.ln_f.output[:, -1, :] - v).abs().max().save()
    rel = float(diff.detach())
    state = AppState.SUPPORTED if rel < 1e-5 else AppState.SILENTLY_WRONG
    return state, f"cross-trace value reuse, |Δ|={rel:.2e}"


@probe("barrier", "hf")
def barrier_hf(be, model):
    """Level-0 construct: tracer.barrier(2) sharing a value across two invokes that
    touch the SAME module. Denotation: equals the two-trace patch of the same edit."""
    def capture():
        return model.transformer.h[5].output[0][:, -1, :]

    def patch_fn(clean_act):
        out = model.transformer.h[5].output
        hs = out[0].clone()
        hs[:, -1, :] = clean_act.to(hs.dtype).to(hs.device)
        model.transformer.h[5].output = (hs, *out[1:])
        return model.lm_head.output[:, -1, :]

    ref = be.patch(model, CLEAN, CORRUPT, capture, patch_fn)

    with model.trace() as tracer:
        barrier = tracer.barrier(2)
        with tracer.invoke(CLEAN):
            clean_hs = model.transformer.h[5].output[0][:, -1, :]
            barrier()
        with tracer.invoke(CORRUPT):
            barrier()
            out = model.transformer.h[5].output
            hs = out[0].clone()
            hs[:, -1, :] = clean_hs
            model.transformer.h[5].output = (hs, *out[1:])
            patched = model.lm_head.output[:, -1, :].save()
    return _verdict_compare(_cpu(ref), _cpu(patched), "barrier patch vs two-trace patch")


# ---------------------------------------------------------------------------
# vLLM-async probes (registration order = safest first; runner stops at a HANG)
# ---------------------------------------------------------------------------

def _vllm_trace_saves(be, model, body, **trace_kw):
    """One single-prompt async trace. The saves collector reads the TRACE-FRAME
    locals (var name = saves key), so a `.save()` inside a nested closure frame is
    silently lost. The shared-container idiom from `VLLMAsyncBackend.run_batched`:
    create+save ONE dict in the with-frame, let `body(tracer, out)` fill it by
    mutation, and return the collected dict."""
    kw = {"temperature": 0.0, "top_p": 1, "max_tokens": 1, **trace_kw}

    async def _go():
        with model.trace(PROMPT, **kw) as tracer:
            out = dict().save()  # noqa: F841 — with-frame var name IS the saves key
            body(tracer, out)
        last = None
        async for output in tracer.backend:
            last = output
        saves = _vllm_saves(last)
        return saves["out"] if isinstance(saves, dict) and "out" in saves else saves

    return be._run_coro(_go())


@probe("input_boundary", "vllm_async")
def input_boundary_vllm(be, model):
    def body(tracer, out):
        out["out5"] = _untuple(model.transformer.h[5].output)
        out["inp6"] = model.transformer.h[6].input

    s = _vllm_trace_saves(be, model, body)
    return _verdict_close(_rel_dev(s["inp6"], s["out5"]), 1e-5, "h[6].input vs h[5].output")


@probe("engine_logits", "vllm_async")
def engine_logits_vllm(be, model):
    """Engine site `model.logits` (pre-sampling eproperty) vs the portable unembed
    of the final block (the working logit-lens recipe). Vocab padding is aligned by
    the oracle compare."""
    def body(tracer, out):
        with torch.no_grad():
            normed = model.transformer.ln_f(_untuple(model.transformer.h[-1].output))
            out["manual"] = F.linear(normed, model.lm_head.weight)[-1:, :]
        out["eng"] = model.logits

    s = _vllm_trace_saves(be, model, body)
    return _verdict_compare(_cpu(s["manual"]), _cpu(s["eng"]), "model.logits vs portable unembed")


@probe("engine_samples", "vllm_async")
def engine_samples_vllm(be, model):
    """Engine site `model.samples` vs greedy argmax of `model.logits`."""
    def body(tracer, out):
        out["lg"] = model.logits
        out["smp"] = model.samples

    s = _vllm_trace_saves(be, model, body)
    tok = int(_cpu(s["smp"]).flatten()[-1].item())
    arg = int(_cpu(s["lg"]).argmax(-1).flatten()[-1].item())
    state = AppState.SUPPORTED if tok == arg else AppState.SILENTLY_WRONG
    return state, f"sampled id {tok} vs greedy logits argmax {arg}"


@probe("derived_head", "vllm_async")
def derived_head_vllm(be, model):
    attn = model.transformer.h[6].attn

    def body(tracer, out):
        # the client-side envoy is the META model — weights are real only in the
        # worker, so the c_proj reconstruction must run INSIDE the trace
        with torch.no_grad():
            cp_in = _untuple(attn.c_proj.input)
            # vLLM RowParallelLinear: y = x Wᵀ + b (bias added internally at TP=1)
            out["recon"] = F.linear(cp_in, attn.c_proj.weight, attn.c_proj.bias)
            out["cp_in"] = cp_in
            out["attn_out"] = _untuple(attn.output)

    s = _vllm_trace_saves(be, model, body)
    H = attn.num_heads
    x = _cpu(_untuple(s["cp_in"]))
    head4 = x.view(*x.shape[:-1], H, x.shape[-1] // H)[..., 4, :]
    state, note = _verdict_close(_rel_dev(s["recon"], _untuple(s["attn_out"])), 5e-2,
                                 "c_proj recon vs attn.output")
    return state, f"head slice {tuple(head4.shape)}; {note}"


@probe("derived_neuron", "vllm_async")
def derived_neuron_vllm(be, model):
    mlp = model.transformer.h[6].mlp

    def body(tracer, out):
        out["cfc"] = _untuple(mlp.c_fc.output)
        out["act"] = _untuple(mlp.act.output)

    s = _vllm_trace_saves(be, model, body)
    neuron7 = _cpu(_untuple(s["act"]))[..., 7]
    recon = F.gelu(_cpu(_untuple(s["cfc"])), approximate="tanh")
    state, note = _verdict_close(_rel_dev(recon, _untuple(s["act"])), 5e-2,
                                 "gelu_new(c_fc) vs act.output")
    return state, f"neuron slice {tuple(neuron7.shape)}; {note}"


@probe("source_mlp", "vllm_async")
def source_mlp_vllm(be, model):
    """Non-attention `.source` on the vLLM forward. vLLM GPT2MLP: `hidden_states, _
    = self.c_fc(...)` → op `self_c_fc_0`, output tuple element 0."""
    mlp = model.transformer.h[6].mlp

    def body_direct(tracer, out):
        out["direct"] = _untuple(mlp.c_fc.output)

    def body_source(tracer, out):
        out["via"] = mlp.source.self_c_fc_0.output[0]

    direct = _vllm_trace_saves(be, model, body_direct)["direct"]
    via = _vllm_trace_saves(be, model, body_source)["via"]
    return _verdict_close(_rel_dev(via, direct), 1e-5, "source.self_c_fc_0 vs c_fc.output")


def _iter_vllm(be, model, bounded: bool):
    """Per-step `model.logits` reads over max_tokens=3. The bounded/unbounded split is
    a realization (Level 1.5) distinction: bounded `iter[0:3]` carries its own stop;
    unbounded `iter[:]` relies on a stop bound the vLLM path never sets, so the loop
    overruns, blocks, and is unwound by Cancelation BEFORE the body's final push —
    all saves lost (root cause: nnsight docs/developing/vllm-construct-gaps.md §1)."""
    def body_ref(tracer, out):
        out["ref"] = model.logits

    ref = _vllm_trace_saves(be, model, body_ref)["ref"]

    async def _go():
        with model.trace(PROMPT, temperature=0.0, top_p=1, max_tokens=3) as tracer:
            rows = list().save()  # noqa: F841
            if bounded:
                for step in tracer.iter[0:3]:
                    rows.append(model.logits)
            else:
                for step in tracer.iter[:]:
                    rows.append(model.logits)
        last = None
        async for output in tracer.backend:
            last = output
        return _vllm_saves(last)

    form = "bounded iter[0:3]" if bounded else "unbounded iter[:] (the documented idiom)"
    try:
        rows = be._run_coro(_go())["rows"]
    except RuntimeError as e:
        if "no saves" in str(e):
            raise RuntimeError(
                f"per-step saves dropped under {form}: the finished output carries no "
                "saves; see nnsight docs/developing/vllm-construct-gaps.md §1"
            ) from e
        raise
    if len(rows) != 3:
        return AppState.SILENTLY_WRONG, f"{form}: expected 3 per-step reads, got {len(rows)}"
    state, note = _verdict_compare(_cpu(ref), _cpu(rows[0]), "step0 vs single-step trace")
    return state, f"{form}: 3 steps; {note}"


@probe("iteration_bounded", "vllm_async")
def iteration_bounded_vllm(be, model):
    return _iter_vllm(be, model, bounded=True)


@probe("iteration_unbounded", "vllm_async")
def iteration_unbounded_vllm(be, model):
    return _iter_vllm(be, model, bounded=False)


@probe("scan", "vllm_async")
def scan_vllm(be, model):
    """Shape-only execution on the vLLM path (vllm.md: 'no .scan() — not validated')."""
    import nnsight

    n_tok = len(model.tokenizer(PROMPT)["input_ids"])
    with model.scan(PROMPT):
        shp = nnsight.save(tuple(_untuple(model.transformer.h[0].output).shape))
    state = (AppState.SUPPORTED
             if len(shp) >= 2 and shp[-2] == n_tok or shp[0] == n_tok
             else AppState.SILENTLY_WRONG)
    return state, f"scanned shape {tuple(shp)} (prompt tokens={n_tok})"


@probe("edit", "vllm_async")
def edit_vllm(be, model):
    """Persistent edit on the vLLM path (vllm.md: 'no module editing — not
    validated'). The dangerous outcome is the edit being silently dropped:
    edited trace == unedited baseline → SILENTLY_WRONG."""
    def body_base(tracer, out):
        out["base"] = model.logits

    base = _vllm_trace_saves(be, model, body_base)["base"]

    with model.edit() as edited:
        out = edited.transformer.h[6].attn.output
        if isinstance(out, tuple):
            edited.transformer.h[6].attn.output = (torch.zeros_like(out[0]), *out[1:])
        else:
            edited.transformer.h[6].attn.output = torch.zeros_like(out)

    async def _go():
        with edited.trace(PROMPT, temperature=0.0, top_p=1, max_tokens=1) as tracer:
            ed = edited.logits.save()  # noqa: F841
        last = None
        async for output in tracer.backend:
            last = output
        return _vllm_saves(last)

    ed = be._run_coro(_go())["ed"]
    if is_equivalent(compare(_cpu(base), _cpu(ed))):
        return AppState.SILENTLY_WRONG, "edit silently dropped (edited == unedited baseline)"
    return AppState.SUPPORTED, "edit applied (edited logits diverge from baseline)"


@probe("barrier", "vllm_async")
def barrier_vllm(be, model):
    """tracer.barrier(2) across two invokes on the vLLM path. Two documented causes
    stack here: multi-invoke submission is gated upstream, and the Barrier object is
    not shared across invokes (nnsight docs/developing/barrier-vllm-not-shared.md) —
    whichever fires first is the measured failure mode. LAST: highest hang risk."""
    async def _go():
        with model.trace(temperature=0.0, top_p=1, max_tokens=1) as tracer:
            res = dict().save()  # noqa: F841 — parent-scope container (run_batched idiom)
            barrier = tracer.barrier(2)
            with tracer.invoke(CLEAN):
                clean_hs = _untuple(model.transformer.h[5].output)[-1:, :]
                barrier()
            with tracer.invoke(CORRUPT):
                barrier()
                out = model.transformer.h[5].output
                hs = _untuple(out).clone()
                hs[-1:, :] = clean_hs
                if isinstance(out, tuple):
                    model.transformer.h[5].output = (hs, *out[1:])
                else:
                    model.transformer.h[5].output = hs
                res["patched"] = model.logits
        last = None
        async for output in tracer.backend:
            last = output
        saves = _vllm_saves(last)
        return saves["res"] if isinstance(saves, dict) and "res" in saves else saves

    try:
        s = be._run_coro(_go())
    except RuntimeError as e:
        if "no saves" in str(e):
            raise RuntimeError(
                "multi-invoke trace returned no saves (two stacked upstream causes: the "
                "async multi-prompt submission gate, and Barrier not shared across invokes "
                "— nnsight docs/developing/barrier-vllm-not-shared.md)"
            ) from e
        raise
    if not (isinstance(s, dict) and "patched" in s):
        raise RuntimeError(f"post-barrier save dropped (saves: {s!r:.120})")
    return AppState.SUPPORTED, "barrier patch produced a value (compare in method tier)"


@probe("session_saved", "vllm_async")
def session_saved_vllm(be, model):
    """SAVED session flow: `.save()` in a session trace, read after session exit.
    Works on the SYNC engine (measured — vllm-construct-gaps.md §3); on the async
    engine there is no drain point inside a captured session body, so expect ERROR
    until session-owned draining exists."""
    def body_ref(tracer, out):
        out["ref"] = model.logits

    ref = _vllm_trace_saves(be, model, body_ref)["ref"]
    with model.session():
        with model.trace(PROMPT, temperature=0.0, top_p=1, max_tokens=1):
            lg = model.logits.save()
    return _verdict_compare(_cpu(ref), _cpu(lg.detach()), "session-saved read vs plain trace")


@probe("session_unsaved", "vllm_async")
def session_unsaved_vllm(be, model):
    """UN-SAVED cross-trace session flow (the session contract): a trace-1 variable
    without .save() consumed in trace 2. Broken on BOTH vLLM engines — only
    Globals.saves ship back from the worker (vllm-construct-gaps.md §3). Same shape
    as the HF probe. LAST: if session exit blocks, the watchdog HANG forfeits only
    this probe."""
    try:
        with model.session():
            with model.trace(PROMPT, temperature=0.0, top_p=1, max_tokens=1):
                v = model.logits
            with model.trace(PROMPT, temperature=0.0, top_p=1, max_tokens=1):
                diff = (model.logits - v).abs().max().save()
        val = float(_cpu(diff.detach()))
    except (NameError, UnboundLocalError) as e:
        raise RuntimeError(
            "un-saved trace-1 value never materializes client-side (only saves ship "
            "back from the worker), so trace 2 dies and its saved var never binds — "
            "vllm-construct-gaps.md §3"
        ) from e
    state = AppState.SUPPORTED if val < 1e-3 else AppState.SILENTLY_WRONG
    return state, f"cross-trace value reuse, |Δ|={val:.2e}"
