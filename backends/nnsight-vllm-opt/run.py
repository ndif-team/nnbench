"""nnsight on vLLM in its fastest documented realization (the optimal-performance axis).

Same nnsight release and engine as `nnsight-vllm`, but each workload uses the form nnsight 0.8
documents as cheapest for repeated calls (docs/models/vllm-editing.md, vllm.md):

- a block whose parameters are fixed per task is installed on the engine once with
  `model.edit(name=...)` and every timed call is a plain request that selects it with
  `edits=[name]`, so the block is not serialized per request;
- a block that needs a per-call value (patching's clean activation) stays a trace, bound to the
  layer handles it touches rather than the root model, which would ship the whole envoy tree;
- writes land in place, and readout reads only the last row.

With `taps` in ISB_ENGINE_OPTIONS (`"auto"` derives them from the spec) the engine replays CUDA
graphs and serves only those locations (vLLM's breakable graphs; vLLM 0.27 or newer). The cells
are in cells.py, registered for the `nnsight_opt` interface.
"""
import contextlib

from isb.backends.vllm_sync import VLLMSyncBackend
from isb.jobs.worker import main, merge_requirements, options
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

    def run_edit(self, model, prompt, *, max_tokens=1):
        """A plain request that runs the installed edit; returns that request's saves."""
        (out,) = model.generate([prompt], max_tokens=max_tokens, temperature=0.0, top_p=1,
                                edits=[EDIT])
        return out.saves

    def vanilla(self, model, prompt, *, new_tokens):
        """A plain request with no edit selected (named edits run only when asked for)."""
        model.generate([prompt], max_tokens=new_tokens, temperature=0.0, top_p=1, edits=[])

    def teardown(self, model) -> None:
        with contextlib.suppress(Exception):
            self.clear_edit()
        super().teardown(model)


def spec_taps(spec):
    """The locations a spec's cells read or write: the smallest tap set for its graphs."""
    params = [t.params for t in spec.tasks] + [spec.baseline.params]
    if spec.effect is not None:
        params += [spec.effect.baseline_params, spec.effect.perturbed_params]
    last = 27                          # Qwen2.5-7B: 28 decoder layers
    if spec.methodology == "logit_lens":
        return ["model.layers.*.output"]
    if spec.methodology in ("steering", "gen_steering"):
        return sorted({f"model.layers.{p['layer']}.output" for p in params if "layer" in p})
    if spec.methodology == "activation_patching":
        return sorted({f"model.layers.{p['layer']}.output" for p in params if "layer" in p}
                      | {f"model.layers.{last}.output"})
    if spec.methodology == "ablation":
        sub = {"mlp": "mlp", "attn": "self_attn"}
        return sorted({f"model.layers.{p['layer']}.{sub[p['target']]}.output"
                       for p in params if p.get("target") in sub})
    return []


def create_backend(spec):
    params = merge_requirements(spec.vllm_kwargs, options())
    taps = params.pop("taps", ())
    if taps == "auto":
        taps = spec_taps(spec)
    import cells  # noqa: F401  (registers the nnsight_opt cells)

    # nnsight builds vllm.LLM with enforce_eager=not taps (modeling/vllm/vllm.py, _load_sync).
    run = RunConfig(EngineConfig("vllm", mode="nnsight-opt", params={
        **params, "enforce_eager": not taps, "taps": list(taps)}))
    return NnsightOptBackend(taps=taps, **params), run


if __name__ == "__main__":
    main(create_backend)
