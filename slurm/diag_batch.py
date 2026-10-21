"""Where batched time goes on nnsight's sync vLLM engine (run inside an nnsight-vllm image).

Usage (inside the container, /workspace on PYTHONPATH): python3 diag_batch.py INPUTS.jsonl OUT.json
On Qwen2.5-7B bf16, eager, prefix caching off, for N = 1, 2, 4, 8, 16 CounterFact prompts:
  vanilla      vllm.LLM.generate of the N prompts, max_tokens=1, no trace
  empty        one trace, N invokes, empty bodies
  logits       one trace, N invokes, each saving its last-token logits (the batched baseline)
  list         one trace, ONE invoke over the list of N prompts, saving the logits
Then a cProfile of `logits` at N=16 (top functions by cumulative and by own time).
"""
import cProfile
import io
import json
import pstats
import statistics
import sys
import time

import torch

REPO = "Qwen/Qwen2.5-7B-Instruct"
WARMUP, TRIALS = 3, 10
SIZES = (1, 2, 4, 8, 16)


def timed(fn):
    for _ in range(WARMUP):
        fn()
    samples = []
    for _ in range(TRIALS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    return {"median_ms": statistics.median(samples), "std_ms": statistics.stdev(samples)}


def main(inputs_path, out_path):
    from vllm import SamplingParams

    from isb.backends.vllm_sync import VLLMSyncBackend

    prompts = [json.loads(line)["unit"] for line in open(inputs_path)][:max(SIZES)]
    be = VLLMSyncBackend(dtype="bfloat16", gpu_memory_utilization=0.85,
                         enable_prefix_caching=False)
    model = be.load(REPO)                   # nnsight's sync engine runs eager without taps
    params = SamplingParams(temperature=0.0, top_p=1, max_tokens=1)

    def vanilla(ps):
        model.vllm_entrypoint.generate(ps, params, use_tqdm=False)

    def empty(ps):
        with model.trace(temperature=0.0, top_p=1, max_tokens=1) as tracer:
            for p in ps:
                with tracer.invoke(p):
                    pass

    def logits(ps):
        with model.trace(temperature=0.0, top_p=1, max_tokens=1) as tracer:
            rows = [None] * len(ps)
            rows = rows.save()
            for i, p in enumerate(ps):
                with tracer.invoke(p):
                    rows[i] = model.logits[-1:, :]
        assert all(r is not None for r in rows)

    def as_list(ps):
        with model.trace(ps, temperature=0.0, top_p=1, max_tokens=1):
            out = model.logits.save()
        assert out.shape[0] == len(ps), out.shape

    from importlib.metadata import version
    import vllm
    result = {"nnsight": version("nnsight"), "vllm": vllm.__version__, "torch": torch.__version__,
              "sizes": {}}
    for n in SIZES:
        ps = prompts[:n]
        row = {}
        for name, fn in [("vanilla", vanilla), ("empty", empty), ("logits", logits),
                         ("list", as_list)]:
            try:
                row[name] = timed(lambda fn=fn: fn(ps))
            except Exception as e:                          # one form failing must not stop the rest
                row[name] = {"error": f"{type(e).__name__}: {e}"[:300]}
        result["sizes"][n] = row
        print(n, json.dumps(row), flush=True)

    prof = cProfile.Profile()
    prof.enable()
    for _ in range(3):
        logits(prompts[:16])
    prof.disable()
    for key in ("cumulative", "tottime"):
        buf = io.StringIO()
        pstats.Stats(prof, stream=buf).sort_stats(key).print_stats(30)
        result[f"profile_{key}"] = buf.getvalue()
        print(buf.getvalue(), flush=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    be.teardown(model)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
