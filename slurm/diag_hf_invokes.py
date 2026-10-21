"""The same 16-prompt trace shapes on nnsight's HF backend (run inside an nnsight-hf image).

Usage (inside the container, /workspace on PYTHONPATH): python3 diag_hf_invokes.py INPUTS.jsonl OUT.json
Qwen2.5-7B bf16, sdpa: plain transformers forward of the 16-prompt padded batch; one trace with ONE
invoke over the 16 prompts (the suite's batched HF form); one trace with 16 invokes (the vLLM form).
Each traced form saves the last-token logits of every prompt.
"""
import json
import statistics
import sys
import time

import torch

REPO = "Qwen/Qwen2.5-7B-Instruct"
N, WARMUP, TRIALS = 16, 3, 10


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
    from isb.backends.hf import HFBackend

    prompts = [json.loads(line)["unit"] for line in open(inputs_path)][:N]
    be = HFBackend(dtype="bfloat16", attn_implementation="sdpa")
    model = be.load(REPO)
    tok = model.tokenizer
    inputs = tok(prompts, return_tensors="pt", padding=True).to("cuda:0")

    def plain():
        with torch.no_grad():
            model._module(**inputs)

    def one_invoke():
        with model.trace(prompts):
            out = model.lm_head.output[:, -1, :].save()
        return out

    def many_invokes():
        with model.trace() as tracer:
            rows = [None] * N
            rows = rows.save()
            for i, p in enumerate(prompts):
                with tracer.invoke(p):
                    rows[i] = model.lm_head.output[:, -1, :]
        return rows

    from importlib.metadata import version
    result = {"nnsight": version("nnsight"), "torch": torch.__version__, "n": N}
    for name, fn in [("plain", plain), ("one_invoke", one_invoke), ("many_invokes", many_invokes)]:
        try:
            result[name] = timed(fn)
        except Exception as e:
            result[name] = {"error": f"{type(e).__name__}: {e}"[:400]}
        print(name, result[name], flush=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
