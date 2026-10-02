"""Per-trace overhead decomposition for one nnsight install (run inside a backend image).

Usage (inside the container, /workspace on PYTHONPATH): python3 diag_overhead.py hf|vllm OUT.json
Times, on Qwen2.5-7B bf16 with one CounterFact-length prompt:
  vanilla   the backend's no-intervention call (HF: model._module; vLLM: AsyncLLM request)
  read1     a trace reading layer 14's last-token residual          (1 module event)
  read28    a trace reading every layer's last-token residual        (28 module events)
  readwrite a trace reading layer 14, writing it back, reading layer 27 (3 events, 1 swap)
HF also times a raw transformers model loaded without nnsight (`raw`) and counts installed hooks.
"""
import json
import statistics
import sys
import time

import torch

REPO = "Qwen/Qwen2.5-7B-Instruct"
PROMPT = "The Eiffel Tower is located in the city of"
WARMUP, TRIALS = 5, 30


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


def main(kind, out_path):
    from isb.profiles import PROFILES

    import nnsight
    m = PROFILES["llama"]
    if kind == "hf":
        from isb.backends.hf import HFBackend
        be = HFBackend(dtype="bfloat16")
        residual = "plain"
    else:
        from isb.backends.vllm_async import VLLMAsyncBackend
        be = VLLMAsyncBackend(dtype="bfloat16", gpu_memory_utilization=0.85,
                              enable_prefix_caching=False)
        residual = "fused"
    model = be.load(REPO)
    h = m.blocks(model)
    resid = m.resid
    last = be.last

    def read1():
        return last(resid(h[14].output, residual))

    def read28():
        return torch.stack([last(resid(h[i].output, residual)) for i in range(len(h))])

    def readwrite():
        out = h[14].output
        h[14].output = out                      # whole-tuple replacement with itself
        return last(resid(h[27].output, residual))

    from importlib.metadata import version
    result = {"kind": kind, "nnsight": version("nnsight"),
              "vllm": None, "torch": torch.__version__}
    try:
        import vllm
        result["vllm"] = vllm.__version__
    except ImportError:
        pass
    result["vanilla"] = timed(lambda: be.vanilla(model, PROMPT, new_tokens=1))
    for name, build in [("read1", read1), ("read28", read28), ("readwrite", readwrite)]:
        result[name] = timed(lambda build=build: be.run(model, [PROMPT], build))
    if kind == "hf":
        mods = list(model._module.modules())
        result["modules"] = len(mods)
        result["forward_hooks"] = sum(len(x._forward_hooks) for x in mods)
        result["forward_pre_hooks"] = sum(len(x._forward_pre_hooks) for x in mods)
        result["wrapped_forwards"] = sum(hasattr(x, "__nnsight_forward__") for x in mods)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        raw = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.bfloat16, device_map="cuda:0",
                                                   attn_implementation="eager")
        tok = AutoTokenizer.from_pretrained(REPO)
        inputs = tok(PROMPT, return_tensors="pt").to("cuda:0")

        def raw_call():
            with torch.no_grad():
                raw(**inputs)
        result["raw"] = timed(raw_call)
    print(json.dumps(result, indent=1), flush=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    be.teardown(model)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
