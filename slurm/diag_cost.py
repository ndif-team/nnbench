"""Client vs engine cost of an N-invoke nnsight trace on vLLM, and whether envoy-by-reference cuts it.

Usage (inside an nnsight-vllm image, /workspace on PYTHONPATH):
    python3 diag_cost.py MODE INPUTS.jsonl OUT.json      MODE = mp | inproc
On Qwen2.5-7B bf16 (sync engine, eager, prefix caching off), 16 CounterFact prompts, one invoke
each, every invoke saving its last-token logits (the batched baseline of the suite):

  mp      vLLM's default process layout (EngineCore in its own process). No profiler; wall-clock
          timers on the client's pieces: serialize each mediator (`vllm.dumps`), the engine call
          (`LLM.generate`, which includes the engine process's work), result collect.
  inproc  VLLM_ENABLE_V1_MULTIPROCESSING=0: the engine runs in this process, so the engine-side
          pieces are timed too: mediator unpickle (`Requests.add` -> `loads`), the model step
          (`execute_model`), sampling, result pickling (`collect_nnsight`). Then a cProfile of
          the whole call. Then `byref`: a prototype that pickles each Envoy as a reference to the
          engine's own envoy at the same path (persistent id `Envoy:<path>`), instead of
          rebuilding the 317-envoy tree per request -- timed, payload-sized, and checked for
          identical logits against the unpatched run.
"""
import cProfile
import functools
import io
import json
import os
import pstats
import statistics
import sys
import time
from collections import defaultdict

MODE = sys.argv[1]
if MODE == "inproc":
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"   # before vllm is imported

import torch  # noqa: E402

REPO = "Qwen/Qwen2.5-7B-Instruct"
N, WARMUP, TRIALS = int(os.environ.get("DIAG_N", "16")), 3, 10
# DIAG_BYREF=1 (inproc): time and profile the byref path itself, at DIAG_N invokes
BYREF_ONLY = os.environ.get("DIAG_BYREF") == "1"

ACC = defaultdict(float)     # accumulated ms per timer over the current measurement
CNT = defaultdict(int)
SIZES = []                   # bytes of each serialized mediator
BYREF = {"on": False}


def timer(name, fn):
    @functools.wraps(fn)
    def wrapped(*a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            ACC[name] += (time.perf_counter() - t0) * 1e3
            CNT[name] += 1
    return wrapped


def instrument(engine_side):
    import nnsight.modeling.vllm.vllm as vmod
    from nnsight.modeling.vllm import VLLM

    dumps = vmod.dumps

    def sized_dumps(obj, *a, **k):
        out = dumps(obj, *a, **k)
        SIZES.append(len(out))
        return out
    vmod.dumps = timer("client: serialize mediators", sized_dumps)
    VLLM._collect = staticmethod(timer("client: collect saves", VLLM._collect))
    import vllm
    vllm.LLM.generate = timer("client: LLM.generate (incl. engine)", vllm.LLM.generate)
    if engine_side:
        import nnsight.modeling.vllm.model_runners.GPUModelRunner as rmod
        rmod.loads = timer("engine: unpickle mediators", rmod.loads)
        R = rmod.NNsightGPUModelRunner
        R.execute_model = timer("engine: execute_model (forward + hooks)", R.execute_model)
        R.sample_tokens = timer("engine: sample_tokens", R.sample_tokens)
        R.collect_nnsight = timer("engine: collect_nnsight (pickle saves)", R.collect_nnsight)


def install_byref():
    """Envoys travel as `Envoy:<path>`; the engine resolves them to its own envoy tree."""
    from nnsight.intervention.envoy import Envoy
    from nnsight.intervention.serialization import CustomCloudPickler
    from nnsight.modeling.mixins.remotable import Remotable

    pid = CustomCloudPickler.persistent_id

    def persistent_id(self, obj):
        if BYREF["on"] and isinstance(obj, Envoy):
            return f"Envoy:{obj.path}"
        return pid(self, obj)
    CustomCloudPickler.persistent_id = persistent_id

    objects = Remotable._remoteable_persistent_objects

    def with_envoys(self):
        out = objects(self)
        for envoy in self.modules():
            out[f"Envoy:{envoy.path}"] = envoy
        return out
    Remotable._remoteable_persistent_objects = with_envoys


def main(inputs_path, out_path):
    prompts = [json.loads(line)["unit"] for line in open(inputs_path)]
    prompts = (prompts * (N // len(prompts) + 1))[:N]   # repeated past the file (prefix caching is off)
    if MODE == "inproc":
        install_byref()            # the engine-side map must hold envoys before load
    instrument(engine_side=(MODE == "inproc"))
    from isb.backends.vllm_sync import VLLMSyncBackend
    be = VLLMSyncBackend(dtype="bfloat16", gpu_memory_utilization=0.85,
                         enable_prefix_caching=False)
    model = be.load(REPO)
    from vllm import SamplingParams
    params = SamplingParams(temperature=0.0, top_p=1, max_tokens=1)

    def vanilla():
        model.vllm_entrypoint.generate(prompts, params, use_tqdm=False)
    # `prompts` is rebound by the mp batch-size loop below; the closures read it at call time

    def traced():
        with model.trace(temperature=0.0, top_p=1, max_tokens=1) as tracer:
            rows = [None] * len(prompts)
            rows = rows.save()
            for i, p in enumerate(prompts):
                with tracer.invoke(p):
                    rows[i] = model.logits[-1:, :]
        return torch.cat([r.float().cpu() for r in rows])

    def measure(fn):
        for _ in range(WARMUP):
            fn()
        ACC.clear(); CNT.clear(); SIZES.clear()
        samples = []
        for _ in range(TRIALS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1e3)
        return {"median_ms": statistics.median(samples), "std_ms": statistics.stdev(samples),
                "per_call_ms": {k: v / TRIALS for k, v in sorted(ACC.items())},
                "calls_per_call": {k: v // TRIALS for k, v in sorted(CNT.items())},
                "mediator_bytes": (statistics.median(SIZES) if SIZES else None)}

    from importlib.metadata import version
    import vllm
    result = {"mode": MODE, "nnsight": version("nnsight"), "vllm": vllm.__version__, "n": N}
    result["envoy_byref_env"] = os.environ.get("NNSIGHT_ENVOY_BYREF") == "1"
    BYREF["on"] = BYREF_ONLY
    result["byref_only"] = BYREF_ONLY
    result["vanilla"] = measure(vanilla)
    result["traced"] = measure(traced)
    print(json.dumps(result, indent=1), flush=True)
    if MODE == "mp":
        # batch-size scaling of the same trace, and its logits for an across-run identity check
        full = prompts
        result["by_n"] = {}
        for n in (1, 4, 16):
            prompts = full[:n]
            result["by_n"][n] = {"vanilla": measure(vanilla), "traced": measure(traced)}
            print(n, json.dumps({k: v["median_ms"] for k, v in result["by_n"][n].items()}),
                  flush=True)
        prompts = full
        torch.save(traced(), out_path.replace(".json", ".pt"))

    if MODE == "inproc":
        prof = cProfile.Profile()
        prof.enable()
        for _ in range(3):
            traced()
        prof.disable()
        buf = io.StringIO()
        pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(45)
        result["profile_cumulative"] = buf.getvalue()
        buf = io.StringIO()
        pstats.Stats(prof, stream=buf).sort_stats("tottime").print_stats(30)
        result["profile_tottime"] = buf.getvalue()

        if BYREF_ONLY:
            with open(out_path, "w") as f:
                json.dump(result, f, indent=1)
            be.teardown(model)
            return
        reference = traced()
        BYREF["on"] = True
        try:
            result["byref"] = measure(traced)
            patched = traced()
            result["byref"]["max_abs_diff_vs_unpatched"] = float((patched - reference).abs().max())
        except Exception as e:
            result["byref"] = {"error": f"{type(e).__name__}: {e}"[:500]}
        finally:
            BYREF["on"] = False
        print(json.dumps({k: result[k] for k in ("byref",)}, indent=1), flush=True)

    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    be.teardown(model)


if __name__ == "__main__":
    main(sys.argv[2], sys.argv[3])
