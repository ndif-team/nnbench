"""How an nnsight trace's time scales with prompts (N) and intervention sites per prompt (K).

Usage (inside an nnsight 0.8.1 image, /workspace on PYTHONPATH):
    python3 diag_scaling.py vllm|hf OUT.json
Qwen2.5-7B bf16 on one GPU (vLLM: sync engine, eager, prefix caching off; HF: sdpa). Short
CounterFact prompts, one output token. For each N in (1, 4, 16, 64):

  plain        the engine with no trace (vLLM `generate` on N prompts; HF one padded forward)
  K = 0..28    a trace whose body, per prompt, reads the last-token residual of the first K blocks
               (a running sum, so every read is used) and saves the last-token logits
  empty        a trace that touches nothing and saves nothing (vLLM: N invokes)

vLLM has one invoke per prompt (its sync tracer takes one prompt per invoke). HF is run both ways:
`batch` (one invoke over N prompts, the padded batch the benchmark's HF cells use) and `invokes`
(N invokes of one prompt, which nnsight concatenates into one batch). Median of 10 after 3 warm-up.
With NNSIGHT_ENVOY_BYREF=1 and backends/nnsight-vllm/envoy_byref on PYTHONPATH the vLLM side runs
the envoy-by-reference prototype.
"""
import json
import os
import statistics
import sys
import time

import torch

REPO = "Qwen/Qwen2.5-7B-Instruct"
NS, KS = (1, 4, 16, 64), (0, 1, 4, 28)
PROMPTS = ["The mother tongue of Danielle Darrieux is", "The official religion of Edwin of Northumbria is",
           "Toko Yasuda, the", "Autonomous University of Madrid, which is located in"]


def timed(fn, trials=10, warmup=3):
    try:
        return _timed(fn, trials, warmup)
    except Exception as e:                       # record and keep going: one case must not end the sweep
        return f"{type(e).__name__}: {e}"[:300]


def _timed(fn, trials, warmup):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def vllm_cases(result):
    from vllm import SamplingParams

    from isb.backends.vllm_sync import VLLMSyncBackend
    from isb.profiles import _resid

    be = VLLMSyncBackend(dtype="bfloat16", gpu_memory_utilization=0.85, enable_prefix_caching=False)
    model = be.load(REPO)
    blocks = model.model.layers
    params = SamplingParams(temperature=0.0, top_p=1, max_tokens=1)

    def body(k):
        acc = None
        for i in range(k):
            h = _resid(blocks[i].output, "fused")[-1:]
            acc = h if acc is None else acc + h
        return model.logits[-1:, :].clone()

    for n in NS:
        prompts = (PROMPTS * n)[:n]
        result.append({"engine": "vllm", "n": n, "case": "plain",
                       "ms": timed(lambda: model.vllm_entrypoint.generate(prompts, params, use_tqdm=False))})

        def empty():
            with model.trace(temperature=0.0, top_p=1, max_tokens=1) as tracer:
                for p in prompts:
                    with tracer.invoke(p):
                        pass
        result.append({"engine": "vllm", "n": n, "case": "empty", "ms": timed(empty)})
        for k in KS:
            def traced():
                with model.trace(temperature=0.0, top_p=1, max_tokens=1) as tracer:
                    rows = [None] * len(prompts)
                    rows = rows.save()
                    for i, p in enumerate(prompts):
                        with tracer.invoke(p):
                            rows[i] = body(k)
            result.append({"engine": "vllm", "n": n, "case": f"K={k}", "ms": timed(traced)})
        print(json.dumps(result[-6:]), flush=True)
    be.teardown(model)


def hf_cases(result):
    from nnsight import LanguageModel

    model = LanguageModel(REPO, device_map="cuda:0", dispatch=True, dtype=torch.bfloat16,
                          attn_implementation="sdpa")
    tok = model.tokenizer
    tok.padding_side = "left"
    blocks = model.model.layers

    def body(k):
        acc = None
        for i in range(k):
            h = blocks[i].output
            h = (h[0] if isinstance(h, tuple) else h)[:, -1, :]
            acc = h if acc is None else acc + h
        return model.lm_head.output[:, -1, :].clone()

    for n in NS:
        prompts = (PROMPTS * n)[:n]
        inputs = tok(prompts, return_tensors="pt", padding=True).to("cuda:0")

        def plain():
            with torch.no_grad():
                model._model(**inputs)
        result.append({"engine": "hf", "n": n, "case": "plain", "ms": timed(plain)})

        def empty():
            with model.trace(prompts):
                pass
        result.append({"engine": "hf", "n": n, "case": "empty", "ms": timed(empty)})
        for k in KS:
            def batch():
                with model.trace(prompts):
                    out = body(k).save()

            def invokes():
                with model.trace() as tracer:
                    rows = [None] * len(prompts)
                    rows = rows.save()
                    for i, p in enumerate(prompts):
                        with tracer.invoke(p):
                            rows[i] = body(k)
            result.append({"engine": "hf", "n": n, "case": f"batch K={k}", "ms": timed(batch)})
            result.append({"engine": "hf", "n": n, "case": f"invokes K={k}", "ms": timed(invokes)})
        print(json.dumps(result[-10:]), flush=True)


def main(engine, out_path):
    result = []
    meta = {"engine": engine, "repo": REPO,
            "byref": os.environ.get("NNSIGHT_ENVOY_BYREF") == "1"}
    (vllm_cases if engine == "vllm" else hf_cases)(result)
    with open(out_path, "w") as f:
        json.dump({"meta": meta, "rows": result}, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
