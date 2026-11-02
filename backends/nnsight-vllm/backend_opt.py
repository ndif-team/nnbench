"""nnsight on vLLM in its fastest documented realization (the optimal-performance axis).

Same nnsight release and engine as `nnsight-vllm/default`, but each workload uses the form nnsight
0.8 documents as cheapest for repeated calls (docs/models/vllm-editing.md, vllm.md):

- a block whose parameters are fixed per task is installed on the engine once with
  `model.edit(name=...)` and every timed call is a plain request that selects it with
  `edits=[name]`, so the block is not serialized per request;
- a block that needs a per-call value (patching's clean activation) stays a trace, bound to the
  layer handles it touches rather than the root model, which would ship the whole envoy tree;
- writes land in place, and readout reads only the last row.

With `taps` in the engine options (`"auto"` derives them from the spec) the engine replays CUDA
graphs and serves only those locations (vLLM's breakable graphs; vLLM 0.27 or newer): the
`*/opt/taps` configs. The cells are in cells_opt.py, registered for the `nnsight_opt` interface;
backend.py hands `mode: opt` here.
"""
import contextlib

from isb.backends.base import prompts_of
from isb.backends.vllm_sync import VLLMSyncBackend
from isb.runs import EngineConfig, RunConfig

EDIT = "isb-cell"                       # the one installed edit's name; every request names it


class NnsightOptBackend(VLLMSyncBackend):
    name = "nnsight_opt"

    def __init__(self, taps=(), **kwargs):
        super().__init__(**kwargs)
        self.taps = tuple(taps)
        self._edit = None               # (key, handle) of the edit installed on the engine

    def load(self, repo: str, gpu_memory_utilization: float | None = None):
        from nnsight.modeling.vllm import VLLM

        kw = self._engine_kwargs()
        if self.tensor_parallel_size > 1:
            kw["tensor_parallel_size"] = self.tensor_parallel_size
        if self.taps:
            kw["taps"] = list(self.taps)
        return VLLM(repo, mode="sync", dispatch=True,
                    gpu_memory_utilization=gpu_memory_utilization or self.gpu_memory_utilization, **kw)

    # -- installed edits: one at a time, keyed by the task's parameters -------------------------
    def edit_installed(self, key) -> bool:
        return self._edit is not None and self._edit[0] == key

    def remember_edit(self, key, handle) -> None:
        self._edit = (key, handle)

    def clear_edit(self) -> None:
        if self._edit is not None:
            self._edit[1].clear()
            self._edit = None

    def run_edit(self, model, prompts, *, max_tokens=1):
        """One plain request per prompt in one `generate` call, each running the installed edit;
        returns each request's saves in prompt order."""
        outputs = model.generate(list(prompts), max_tokens=max_tokens, min_tokens=max_tokens,
                                 temperature=0.0, top_p=1, edits=[EDIT])
        return [output.saves for output in outputs]

    def plain(self, model, units, *, new_tokens=1):
        """Plain requests with no edit selected (named edits run only when asked for)."""
        model.generate(prompts_of(units), max_tokens=new_tokens, min_tokens=new_tokens,
                       temperature=0.0, top_p=1, edits=[])

    def teardown(self, model) -> None:
        with contextlib.suppress(Exception):
            self.clear_edit()
        super().teardown(model)


def _layer_outputs(indices, n_layers):
    """Tap locations of the decoder layer outputs `indices` (negative indices count from the end)."""
    return sorted({f"model.layers.{i % n_layers}.output" for i in indices},
                  key=lambda tap: int(tap.split(".")[2]))


def _listed(p):
    """The layers one call's params name: `layers` when given, else `layer`."""
    if p.get("layers") is not None:
        return list(p["layers"])
    return [p["layer"]] if p.get("layer") is not None else []


def spec_taps(spec):
    """The locations a spec's cells read or write: the smallest tap set for its graphs. A call
    that reads and writes nothing at a layer (steering at alpha 0, a harvest with export off) adds
    no tap, so a case's taps are exactly the layers its rows, baseline and effect check touch."""
    params = [t.params for t in spec.tasks] + [spec.baseline.params]
    if spec.effect is not None:
        params += [spec.effect.baseline_params, spec.effect.perturbed_params]
    n = spec.n_layers
    if spec.methodology == "logit_lens":
        if any(p.get("layers", "all") == "all" for p in params):
            return ["model.layers.*.output"]
        return _layer_outputs([i for p in params for i in p["layers"]], n)
    if spec.methodology in ("steering", "gen_steering"):
        return _layer_outputs([i for p in params if p.get("alpha", 1) != 0 for i in _listed(p)], n)
    if spec.methodology in ("residual_readout", "activation_harvest", "gen_capture"):
        return _layer_outputs([i for p in params if p.get("export", True) for i in _listed(p)], n)
    if spec.methodology == "activation_patching":     # the readout is the engine's `model.logits`
        return sorted({f"model.layers.{p['layer']}.output" for p in params if "layer" in p})
    if spec.methodology == "ablation":
        sub = {"mlp": "mlp", "attn": "self_attn"}
        return sorted({f"model.layers.{p['layer']}.{sub[p['target']]}.output"
                       for p in params if p["target"] in sub})
    return []


def create(spec, params):
    """The opt backend and its run record for the engine options `params` (mode removed)."""
    from . import cells_opt  # noqa: F401  (registers the nnsight_opt cells)

    taps = params.pop("taps", ())
    if taps == "auto":
        taps = spec_taps(spec)

    # nnsight builds vllm.LLM with enforce_eager=not taps (modeling/vllm/vllm.py, _load_sync).
    run = RunConfig(EngineConfig("vllm", mode="nnsight-opt", params={
        **params, "enforce_eager": not taps, "taps": list(taps)}))
    return NnsightOptBackend(taps=taps, **params), run
